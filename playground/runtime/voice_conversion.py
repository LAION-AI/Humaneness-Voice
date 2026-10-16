"""Optional Chatterbox stage, following LAION's wrapper and the old VC stream.

No built-in target, silent source fallback, SIDON or new TTS sampling policy.
Per-branch source context / held overlap can be forked without shared state.
Conversion runs before timestamp alignment, all rewards and delivery.
"""
from dataclasses import dataclass,field
import hashlib
import importlib.util
from pathlib import Path
import threading
import time
from collections import OrderedDict, defaultdict

import numpy as np
import soundfile as sf

import settings

SR=48000
NATIVE_SR=24000
CONTEXT=int(1.6*SR)
OVERLAP=int(.04*SR)
MIN_SOURCE=int(.8*SR)


@dataclass
class Stream:
    seed: int
    context: np.ndarray = field(default_factory=lambda:np.zeros(0,np.float32))
    tail: np.ndarray = field(default_factory=lambda:np.zeros(0,np.float32))
    pending: np.ndarray = field(default_factory=lambda:np.zeros(0,np.float32))
    chunks: int = 0
    done: bool = False

    def fork(self,seed):
        return Stream(seed,self.context.copy(),self.tail.copy(),self.pending.copy(),self.chunks,self.done)


def common_pcm_prefix(waves,rows,limit):
    """CFM is stochastic: shared TTS tokens alone do not prove shared VC PCM."""
    count=min([limit]+[len(waves[i]) for i in rows])
    if len(rows)>1 and count:
        same=np.ones(count,bool)
        for i in rows[1:]:same &= waves[i][:count]==waves[rows[0]][:count]
        different=np.flatnonzero(~same)
        if len(different):count=int(different[0])
    return count


class ConversionManager:
    def __init__(self):
        self.lock=threading.RLock()
        self.cb=None
        self.targets=OrderedDict()
        self.last_inference_report={}
        self.fp32_estimator=None
        self.bf16_estimator=None
        self.condition_scorer=None
        self.score_targets=OrderedDict()
        self.score_names=None

    def bind_condition_scorer(self,scorer):
        self.condition_scorer=scorer

    def describe(self):
        available=all(importlib.util.find_spec(m) is not None for m in ('chatterbox_vc','chatterbox'))
        return {'available':available,'loaded':self.cb is not None,'default':settings.DEFAULT_VOICE_CONVERSION,
            'device':settings.VC_DEVICE,'native_sample_rate':NATIVE_SR,'processing_sample_rate':SR,
            'reference_seconds_max':10,'repo':'https://github.com/LAION-AI/chatterbox-voice-conversion',
            'model_repo':settings.VC_FULL_REPO,'model_revision':settings.VC_FULL_REVISION,
            'checkpoint':settings.VC_FULL_CHECKPOINT,'checkpoint_sha256':settings.VC_FULL_SHA256,
            'step':40000,'arm':'merged_full',
            'condition_predictor':'Humaneness Ears Medium: 99 distilled native axes; source dynamic + target static',
            'precision_default':'bf16','precision_modes':['bf16','fp32'],
            'batch_size':settings.VC_BATCH_SIZE,'batch_method':'same-length GPU microbatches; per-candidate diffusion seeds',
            'streaming_method':'overlapping context windows; 40 ms held crossfade, not native causal decoding'}

    def load(self):
        with self.lock:
            if self.cb is not None:return
            if not self.describe()['available']:
                raise RuntimeError('Chatterbox VC dependencies are unavailable; disable voice conversion')
            from chatterbox_vc import VoiceConverter
            from chatterbox.vc import ChatterboxVC
            from chatterbox.models.s3gen import S3Gen
            from vc_full import install
            import torch
            with torch.random.fork_rng(devices=[]):
                model=S3Gen()
            names=install(model)
            model.to(settings.VC_DEVICE).eval()
            # Retain the existing wrapper API and watermark, but bypass its
            # vanilla from_local loader entirely. No built-in target reference.
            cb=VoiceConverter.__new__(VoiceConverter)
            cb.device=settings.VC_DEVICE;cb._model_path=None
            cb._model=ChatterboxVC(model,settings.VC_DEVICE)
            self.score_names=names
            if self.condition_scorer is None:
                from ears_reward import EarsModels
                self.condition_scorer=EarsModels(settings.VC_DEVICE)
            from vc_batch import install_seeded_decoder
            install_seeded_decoder(cb._model)
            self.fp32_estimator=cb._model.s3gen.flow.decoder.estimator
            if cb.sample_rate!=NATIVE_SR:raise RuntimeError('Unsupported Chatterbox sample rate')
            self.cb=cb

    def prepare_profile(self,reference_path,seed=777):
        return self.prepare(reference_path,seed,cache_target=True)

    def prepare_optimized(self,reference_path,seed=777,*,profile=False,precision='bf16'):
        return self.prepare(reference_path,seed,cache_target=profile,precision=precision)

    def prepare(self,reference_path,seed=777,*,cache_target=False,precision='fp32'):
        if not reference_path:raise ValueError('Voice conversion requires reference audio')
        if precision not in ('fp32','bf16'):raise ValueError('Unsupported VC precision')
        tick=time.perf_counter();self.load()
        import torch
        with self.lock,torch.inference_mode(),torch.cuda.device(settings.VC_DEVICE),torch.autocast('cuda',enabled=False):
            if precision=='bf16' and self.bf16_estimator is None:
                from copy import deepcopy
                # Keep the original FP32 weights intact for honest A/B switching.
                self.bf16_estimator=deepcopy(self.fp32_estimator).to(dtype=torch.bfloat16).eval()
            self.cb._model.s3gen.flow.decoder.estimator=self.bf16_estimator if precision=='bf16' else self.fp32_estimator
            path=Path(reference_path).resolve()
            cacheable=cache_target or path.is_relative_to((settings.HERE/'fixed-references/audio').resolve())
            key=hashlib.sha256(path.read_bytes()).hexdigest() if cacheable else None
            # This in-process cache only stores conditioning under a content
            # hash, never an accessible reference URL. Dynamic same-voice clips
            # can use it too; ownership is checked by the request route.
            cached=self.targets.get(key) if key else None
            if cached is None:
                self.cb.set_target_voice(str(reference_path))
                if key:
                    self.targets[key]=self.cb._model.ref_dict
                    while len(self.targets)>4:self.targets.popitem(last=False)
            else:
                self.targets.move_to_end(key)
                self.cb._model.ref_dict=cached
            info=sf.info(reference_path)
            # The conditioning acoustic prefix remains upstream's <=10 s.
            # Measure its own static/timbre axes once, never infer identity
            # attributes from the acting prompt or the generated source.
            score_key=key or hashlib.sha256(path.read_bytes()).hexdigest()
            target_scores=self.score_targets.get(score_key) if cacheable else None
            if target_scores is None:
                import torchaudio
                with sf.SoundFile(path) as audio:
                    rate=audio.samplerate
                    wave=audio.read(frames=rate*10,dtype='float32',always_2d=True).mean(1)
                target16=torchaudio.functional.resample(torch.as_tensor(wave,device=settings.VC_DEVICE),rate,16000)
                target_scores=self.condition_scorer.predict_conditioning([target16],self.score_names)[0]
                if cacheable:
                    self.score_targets[score_key]=target_scores.copy()
                    while len(self.score_targets)>4:self.score_targets.popitem(last=False)
            conditioning={**self.cb._model.ref_dict,'_grow_target_scores':target_scores.copy()}
            session=ConversionSession(self,seed,conditioning)
            session.report.update(prepare_seconds=time.perf_counter()-tick,
                precision=('BF16 flow estimator; FP32 tokenizer/encoder/vocoder; original 10 flow steps' if precision=='bf16' else 'FP32; original 10 flow steps'),
                precision_mode=precision,microbatch_limit=settings.VC_BATCH_SIZE,
                reference_cache_hit=cached is not None,
                reference_source_seconds=info.duration,reference_used_seconds=min(info.duration,10.),
                score_conditioning='99 Ears Medium audio-predicted axes; source emotion/style + 14 target timbre axes',
                score_conditioning_approximation='Distilled predictions of preparation targets, not the original corpus annotations',
                checkpoint=settings.VC_FULL_CHECKPOINT,checkpoint_sha256=settings.VC_FULL_SHA256,
                model_repo=settings.VC_FULL_REPO,model_revision=settings.VC_FULL_REVISION,step=40000,arm='merged_full')
            return session

    def infer_many(self,waves,conditioning,seeds):
        import torch
        from vc_batch import batch_native
        from scipy.signal import resample_poly
        from vc_grow import mix_scores
        devices=[torch.device(settings.VC_DEVICE).index or 0]
        output=[None]*len(waves)
        batch_sizes=[]
        groups=defaultdict(list)
        for index,wave in enumerate(waves):groups[len(wave)].append(index)
        with self.lock,torch.inference_mode(),torch.random.fork_rng(devices=devices),\
                torch.cuda.device(settings.VC_DEVICE),torch.autocast('cuda',enabled=False):
            for indices in groups.values():
                for start in range(0,len(indices),settings.VC_BATCH_SIZE):
                    batch=indices[start:start+settings.VC_BATCH_SIZE]
                    batch_sizes.append(len(batch))
                    sources=[resample_poly(waves[i],1,3).astype(np.float32) for i in batch]
                    predicted=self.condition_scorer.predict_conditioning(sources,self.score_names)
                    mixed=mix_scores(predicted,conditioning['_grow_target_scores'],self.score_names)
                    projector=self.cb._model.s3gen.flow.spk_embed_affine_layer
                    projector.condition=torch.as_tensor(mixed,device=settings.VC_DEVICE)
                    projector.present=torch.ones_like(projector.condition)
                    reference={k:v for k,v in conditioning.items() if not k.startswith('_grow')}
                    # Upstream projects a singleton reference BEFORE its batch
                    # broadcast. Distinct source-style rows need B CAMP inputs.
                    embedding=reference['embedding']
                    if len(embedding)==1:reference['embedding']=embedding.expand(len(batch),-1).contiguous()
                    try:
                        native=batch_native(self.cb._model,[waves[i] for i in batch],reference,
                            [seeds[i] for i in batch],settings.VC_DEVICE)
                    finally:
                        projector.condition=None;projector.present=None
                    for row,index in zip(native,batch):
                        marked=self.cb._model.watermarker.apply_watermark(row.reshape(-1),sample_rate=NATIVE_SR)
                        output[index]=np.asarray(resample_poly(marked,2,1),dtype=np.float32)
            self.last_inference_report={'gpu_microbatches':len(batch_sizes),'microbatch_sizes':batch_sizes}
        return output

    def infer(self,wave,conditioning,seed):
        return self.infer_many([wave],conditioning,[seed])[0]


class ConversionSession:
    def __init__(self,manager,seed,conditioning=None):
        self.manager,self.seed,self.conditioning=manager,seed,conditioning
        self.report={'enabled':True,'model':'Humaneness Voice Converter · merged 40k','native_sample_rate':NATIVE_SR,
            'processing_sample_rate':SR,'context_seconds':CONTEXT/SR,'crossfade_seconds':OVERLAP/SR,
            'minimum_initial_seconds':MIN_SOURCE/SR,'conversion_seconds':0.,'conversion_calls':0,
            'gpu_microbatches':0,'max_microbatch_size':0,'precision':'FP32; original 10 flow steps',
            'scoring_audio':'converted output, never raw TTS','timing_adjustments':[],
            'streaming_method':'overlapping source-context windows, not native causal S3Gen decoding',
            'device':settings.VC_DEVICE}

    def new_stream(self,index=0):return Stream((self.seed+104729*index)%(2**31-1))

    def _convert(self,wave,seed):
        return self._convert_many([wave],[seed])[0]

    def _convert_many(self,waves,seeds):
        padded=[];actual=[]
        for wave in waves:
            wave=np.asarray(wave,dtype=np.float32).reshape(-1)
            if not len(wave) or not np.isfinite(wave).all():raise ValueError('Invalid source audio for Chatterbox')
            actual.append(len(wave));padded.append(np.pad(wave,(0,max(0,MIN_SOURCE-len(wave)))))
        tick=time.perf_counter()
        if hasattr(self.manager,'infer_many'):
            converted=self.manager.infer_many(padded,self.conditioning,seeds)
            info=getattr(self.manager,'last_inference_report',{})
            self.report['gpu_microbatches']+=info.get('gpu_microbatches',0)
            self.report['max_microbatch_size']=max([self.report['max_microbatch_size']]+info.get('microbatch_sizes',[]))
        else:
            converted=[self.manager.infer(w,self.conditioning,s) for w,s in zip(padded,seeds)]
        self.report['conversion_seconds']+=time.perf_counter()-tick
        self.report['conversion_calls']+=len(waves)
        if len(converted)!=len(waves):raise RuntimeError('Missing Chatterbox batch outputs')
        return [self._fix_duration(w,p,n) for w,p,n in zip(converted,padded,actual)]

    def _fix_duration(self,converted,padded,actual):
        converted=np.asarray(converted,dtype=np.float32).reshape(-1)
        if not len(converted) or not np.isfinite(converted).all():raise RuntimeError('Invalid Chatterbox output; no raw-audio fallback')
        # Native content-token rounding must not accumulate time drift or shift
        # beam prefix indexes. Explicitly report small duration corrections.
        difference=(len(converted)-len(padded))/SR
        if abs(difference)>max(.2,len(padded)/SR*.1):
            raise RuntimeError('Chatterbox duration drift too large for safe streaming')
        if len(converted)!=len(padded):
            self.report['timing_adjustments'].append({'source_seconds':len(padded)/SR,
                'native_output_seconds':len(converted)/SR,'delta_seconds':difference})
            converted=np.interp(np.linspace(0,len(converted)-1,len(padded)),np.arange(len(converted)),converted).astype(np.float32)
        return converted[:actual].copy()

    def chunk(self,chunk,state,final=False):
        return self.chunk_many([chunk],[state],[final])[0]

    def chunk_many(self,chunks,states,finals):
        if not len(chunks)==len(states)==len(finals):raise ValueError('VC stream batch mismatch')
        jobs=[];output=[np.zeros(0,np.float32) for _ in chunks]
        for index,(chunk,state,final) in enumerate(zip(chunks,states,finals)):
            chunk=np.asarray(chunk,dtype=np.float32).reshape(-1)
            if state.done:
                if len(chunk):raise RuntimeError('Cannot append to a closed VC stream')
                continue
            if not np.isfinite(chunk).all():raise ValueError('Invalid VC chunk')
            state.pending=np.concatenate((state.pending,chunk))
            if not final and len(state.context)+len(state.pending)<MIN_SOURCE:continue
            if len(state.pending):
                joined=np.concatenate((state.context,state.pending))
                jobs.append((index,state,final,joined))
            elif final:
                output[index]=state.tail.copy();state.tail=np.zeros(0,np.float32);state.done=True
        if jobs:
            converted=self._convert_many([j[3] for j in jobs],[j[1].seed+10007*j[1].chunks for j in jobs])
            for (index,state,final,joined),out in zip(jobs,converted):
                output[index]=self._commit_chunk(out,joined,state,final)
        return output

    def _commit_chunk(self,out,joined,state,final):
        parts=[]
        if len(state.pending):
            state.chunks+=1
            overlap=min(len(state.tail),len(state.context),OVERLAP)
            # Re-render the held previous samples, not the first new samples:
            # the old implementation shortened speech by a crossfade per join.
            out=out[len(state.context)-overlap:].copy()
            if overlap:
                ramp=np.linspace(0,1,overlap,dtype=np.float32)
                out[:overlap]=state.tail[-overlap:]*(1-ramp)+out[:overlap]*ramp
            state.context=joined[-CONTEXT:].copy()
            state.pending=np.zeros(0,np.float32)
            held=min(OVERLAP,len(out)) if not final else 0
            state.tail=out[-held:].copy() if held else np.zeros(0,np.float32)
            parts.append(out[:-held] if held else out)
        elif final and len(state.tail):
            parts.append(state.tail.copy());state.tail=np.zeros(0,np.float32)
        if final:state.done=True
        return np.concatenate(parts) if parts else np.zeros(0,np.float32)

    def full(self,waves,streaming=False,chunk_frames=20):
        if not streaming:
            return self._convert_many(waves,[self.seed+104729*i for i in range(len(waves))])
        states=[self.new_stream(i) for i in range(len(waves))];parts=[[] for _ in waves]
        size=chunk_frames*3840
        for start in range(0,max(map(len,waves),default=0),size):
            converted=self.chunk_many([w[start:start+size] for w in waves],states,[False]*len(waves))
            for target,chunk in zip(parts,converted):target.append(chunk)
        for target,chunk in zip(parts,self.chunk_many([np.zeros(0,np.float32) for _ in waves],states,[True]*len(waves))):
            target.append(chunk)
        return [np.concatenate(p) for p in parts]

    def close(self):
        with self.manager.lock:
            self.conditioning=None
            if self.manager.cb is not None:
                self.manager.cb._model.ref_dict=None
                if getattr(self.manager,'fp32_estimator',None) is not None:
                    self.manager.cb._model.s3gen.flow.decoder.estimator=self.manager.fp32_estimator
