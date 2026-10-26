"""Loopback-only ASR model bank: CPU-resident weights, one extra GPU model.

Run with ROBO_ASR_DEPS pointing to isolated faster-whisper/CT2 dependencies.
The shared Nemotron/TYGER encoder and production TTS process are untouched.
"""
import os,sys
from pathlib import Path
if os.environ.get('ROBO_ASR_DEPS'):sys.path.insert(0,os.environ['ROBO_ASR_DEPS'])
import asyncio,gc,json,logging,re,tarfile,threading,time
import numpy as np
from aiohttp import web
from asr_catalog import MODELS,cache_root

log=logging.getLogger('asr.models')

def telephony_model(folder):
    """Strict in-memory NeMo conversion; no second 2.5-GB extracted checkpoint.

    Key-layout reference: Hugging Face's Apache-2.0 Nemotron 3.5 converter:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron3_5_asr/convert_nemotron3_5_asr_to_hf.py
    """
    import torch,yaml,sentencepiece,paths
    from transformers import AutoConfig,AutoModel,AutoProcessor,GenerationConfig
    proc=AutoProcessor.from_pretrained(paths.BUNDLE/'nemotron',local_files_only=True)
    mapping={
        r'prompt_kernel\.0\.':'prompt_projector.linear_1.',r'prompt_kernel\.2\.':'prompt_projector.linear_2.',
        r'encoder\.pre_encode\.conv\.0\.':'encoder.subsampling.conv_in.',
        r'encoder\.pre_encode\.conv\.2\.':'encoder.subsampling.layers.0.depthwise_conv.',
        r'encoder\.pre_encode\.conv\.3\.':'encoder.subsampling.layers.0.pointwise_conv.',
        r'encoder\.pre_encode\.conv\.5\.':'encoder.subsampling.layers.1.depthwise_conv.',
        r'encoder\.pre_encode\.conv\.6\.':'encoder.subsampling.layers.1.pointwise_conv.',
        r'encoder\.pre_encode\.out\.':'encoder.subsampling.linear.',r'encoder\.pos_enc\.':'encoder.encode_positions.',
        r'encoder\.layers\.(\d+)\.conv\.batch_norm\.':r'encoder.layers.\1.conv.norm.',
        r'linear_([kv])':r'\1_proj',r'linear_out':'o_proj',r'linear_q':'q_proj',
        r'pos_bias_([uv])':r'bias_\1',r'linear_pos':'relative_k_proj',
        r'decoder\.prediction\.embed\.':'decoder.embedding.',
        r'decoder\.prediction\.dec_rnn\.lstm\.':'decoder.lstm.',
        r'joint\.enc\.':'encoder_projector.',r'joint\.pred\.':'decoder.decoder_projector.',r'joint\.joint_net\.2\.':'joint.head.'}
    with tarfile.open(Path(folder)/'nemotron-3.5-asr-de-telephony.nemo') as archive:
        members=archive.getmembers()
        config=yaml.safe_load(archive.extractfile(next(m for m in members if m.name.endswith('model_config.yaml'))))
        if config['encoder']['n_layers']!=24 or config['encoder']['d_model']!=1024:raise ValueError('Unexpected telephony encoder')
        sp=sentencepiece.SentencePieceProcessor(model_proto=archive.extractfile(next(m for m in members if m.name.endswith('tokenizer.model'))).read())
        if any(proc.tokenizer.convert_tokens_to_ids(sp.id_to_piece(i))!=i for i in range(sp.get_piece_size())):
            raise ValueError('Telephony tokenizer differs from pinned Nemotron processor')
        if config['model_defaults']['prompt_dictionary']['de']!=proc.prompt_dictionary['de']:
            raise ValueError('Telephony German prompt mismatch')
        raw=torch.load(archive.extractfile(next(m for m in members if m.name.endswith('model_weights.ckpt'))),map_location='cpu',weights_only=True)
    weights={}
    for key,value in raw.items():
        if key.endswith(('featurizer.window','featurizer.fb')) or key.startswith('ctc_decoder.') or 'ctc_loss' in key:continue
        for pattern,replacement in mapping.items():key=re.sub(pattern,replacement,key)
        if key in weights:raise ValueError('Duplicate converted weight')
        weights[key]=value
    config=AutoConfig.from_pretrained(paths.BUNDLE/'nemotron',local_files_only=True)
    config.default_prompt_id=proc.prompt_dictionary['de']
    # Initialize non-persistent positional buffers on CPU as well; a meta
    # constructor leaves those outside state_dict and cannot later be moved.
    model=AutoModel.from_config(config)
    model.load_state_dict(weights,strict=True,assign=True)
    model.generation_config=GenerationConfig.from_pretrained(paths.BUNDLE/'nemotron',local_files_only=True)
    return model.eval(),proc

class ModelBank:
    def __init__(self):
        self.entries={key:{'phase':'pending','error':None} for key in MODELS if key!='nemotron'}
        self.models={};self.active=None;self.lock=threading.RLock();self.device=os.environ.get('ROBO_ASR_DEVICE','cuda:0')
    def status(self):
        return {'active_gpu_model':self.active,'device':self.device,'models':[
            {'id':key,**{k:v for k,v in spec.items() if k!='files'},
             **({'phase':'resident','ready':True} if key=='nemotron' else
                {**self.entries[key],'ready':key in self.models})} for key,spec in MODELS.items()],
            'strategy':'Original Nemotron/TYGER stays resident; one selected extra recognizer on GPU; others cached in CPU RAM.'}
    def preload(self):
        from prepare_asr_models import prepare
        import torch
        from transformers import AutoModel,AutoProcessor,MoonshineStreamingForConditionalGeneration
        torch.set_num_threads(4)
        for key,spec in MODELS.items():
            if key=='nemotron':continue
            self.entries[key]['phase']='downloading'
            try:
                _,folder=prepare(key);self.entries[key]['phase']='loading_cpu'
                if spec['kind']=='telephony':model,proc=telephony_model(folder)
                elif spec['kind']=='whisper':
                    from faster_whisper import WhisperModel
                    index=int(self.device.split(':')[-1])
                    with self.lock:
                        previous=self.active
                        if previous:self.activate('nemotron')
                        try:
                            model=WhisperModel(folder,device='cuda',device_index=index,compute_type='float16',cpu_threads=4,num_workers=1,local_files_only=True)
                            model.model.unload_model(to_cpu=True)
                        finally:
                            if previous:self.activate(previous)
                    proc=None
                else:
                    proc=AutoProcessor.from_pretrained(folder,local_files_only=True)
                    cls=MoonshineStreamingForConditionalGeneration if spec['kind']=='moonshine' else AutoModel
                    model=cls.from_pretrained(folder,dtype=torch.float32,local_files_only=True).eval()
                with self.lock:
                    self.models[key]=(model,proc)
                    self.entries[key].update(phase='cpu_ready',cpu_bytes=sum(p.numel()*p.element_size() for p in model.parameters()) if proc else 1617884929)
                log.info('CPU ready: %s',key)
            except Exception as exc:
                self.entries[key].update(phase='failed',error=type(exc).__name__+': '+str(exc)[:250]);log.exception('ASR preload failed: %s',key)
    def activate(self,key):
        import torch
        if key not in MODELS:raise ValueError('Unknown ASR model')
        with self.lock:
            start=time.perf_counter()
            if key!='nemotron' and key not in self.models:raise RuntimeError('Selected ASR is not CPU-ready yet')
            if self.active==key:return {'model':key,'switch_ms':0.,'ready':True}
            if self.active:
                old=self.active;model,proc=self.models[old]
                if proc:model.to('cpu')
                else:model.model.unload_model(to_cpu=True)
                self.entries[old]['phase']='cpu_ready';self.active=None
                torch.cuda.synchronize();torch.cuda.empty_cache()
            if key!='nemotron':
                model,proc=self.models[key]
                try:
                    if proc:model.to(self.device)
                    else:model.model.load_model(keep_cache=True)
                    torch.cuda.synchronize();self.active=key;self.entries[key]['phase']='gpu_ready'
                except Exception:
                    if proc:model.to('cpu')
                    else:model.model.unload_model(to_cpu=True)
                    torch.cuda.empty_cache();raise
            return {'model':key,'ready':True,'switch_ms':round((time.perf_counter()-start)*1000,2)}
    def transcribe(self,key,audio):
        import torch
        if key=='nemotron':raise ValueError('Original Nemotron uses the existing live engine')
        with self.lock,torch.inference_mode():
            start=time.perf_counter();switched=self.activate(key);model,proc=self.models[key]
            seconds=len(audio)/16000
            if proc is None:
                segments,info=model.transcribe(audio,language='de',beam_size=3,temperature=0.,vad_filter=False,
                    condition_on_previous_text=False,word_timestamps=False,no_speech_threshold=.6,
                    max_new_tokens=min(440,max(32,int(seconds*12))))
                text=' '.join(s.text.strip() for s in segments).strip()
            else:
                kwargs={'language':'de'} if key=='nemotron_de' else {}
                inputs=proc([audio],sampling_rate=16000,return_tensors='pt',padding=True,**kwargs)
                inputs={k:v.to(self.device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
                # In particular Moonshine MUST retain attention_mask for its
                # trained sliding windows and cap repetition-prone decoding.
                if key=='moonshine_de':limit=min(384,int(seconds*6.5)+4)
                # RNN-T/TDT generation also consumes blank/time-advance steps;
                # a text-only token rate cuts final words mid-utterance.
                else:limit=min(4096,max(128,int(seconds*50)+128))
                output=model.generate(**inputs,max_new_tokens=limit)
                ids=output.sequences if hasattr(output,'sequences') else output
                text=proc.batch_decode(ids,skip_special_tokens=True)[0].strip()
            text=re.sub(r'<[a-z]{2,3}-[A-Za-z]{2}>','',text).strip()
            return {'model':key,'text':text,'seconds':round(seconds,3),'transcribe_ms':round((time.perf_counter()-start)*1000,2),**switched}

async def startup(app):
    app['preload']=asyncio.create_task(asyncio.to_thread(app['bank'].preload))

async def status(request):return web.json_response(request.app['bank'].status())

async def activate(request):
    try:return web.json_response(await asyncio.to_thread(request.app['bank'].activate,request.match_info['model']))
    except ValueError as exc:raise web.HTTPBadRequest(text=str(exc))
    except RuntimeError as exc:raise web.HTTPServiceUnavailable(text=str(exc))

async def transcribe(request):
    key=request.match_info['model'];raw=await request.read()
    if key not in MODELS:raise web.HTTPBadRequest(text='Unknown ASR model')
    if not raw or len(raw)%4 or len(raw)>30*16000*4:raise web.HTTPBadRequest(text='Send 0–30 s mono 16-kHz float32 PCM')
    audio=np.frombuffer(raw,'<f4')
    if not np.isfinite(audio).all():raise web.HTTPBadRequest(text='Finite samples required')
    try:return web.json_response(await asyncio.to_thread(request.app['bank'].transcribe,key,audio.copy()))
    except Exception as exc:
        log.exception('ASR inference failed: %s',key)
        raise web.HTTPServiceUnavailable(text=type(exc).__name__+': selected recognizer failed')

def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)s %(message)s')
    app=web.Application(client_max_size=2_000_000);app['bank']=ModelBank()
    app.router.add_get('/models',status);app.router.add_post('/activate/{model}',activate)
    app.router.add_post('/transcribe/{model}',transcribe);app.on_startup.append(startup)
    web.run_app(app,host='127.0.0.1',port=int(os.environ.get('ROBO_ASR_MODELS_PORT','8801')))

if __name__=='__main__':main()
