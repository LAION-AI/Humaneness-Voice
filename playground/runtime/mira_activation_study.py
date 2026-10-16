"""Resumable offline Mira activation study. Never mutates the live engine.

One research TTS copy per assigned GPU, bounded four-row batches. Full FT only.
Collection states are selected only after scoring each entire untrimmed output.
Run separate process instances for layer/semantic search, not competing copies
of the live model. All audio, scores, failures, seeds and vectors are retained.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
from types import SimpleNamespace
import urllib.request

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

import settings
import mira_activation_analysis as analysis
from activation_steering import ActivationCollector, research_stream_frames
from prompting import build, parse_literal
from semantic_steering import SemanticVectorTransform

ROOT = settings.HERE / 'mira-activation-study'
REFERENCE = settings.HERE / 'fixed-references/audio/emolia_c1699.wav'
ANCHOR = Path('/mnt/nvme/moss-15-v2-assets/refs2/pilot/emolia_c1699/reference.wav')
EMOTION_KEYS = {'joy':'eiv:Elation', 'anger':'eiv:Anger', 'sadness':'eiv:Sadness', 'fear':'eiv:Fear'}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))


def read_results(phase=None):
    return [json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('*.json'))
            if phase is None or p.name.startswith(phase+'-')]


def freeze_design():
    data = analysis.design(100, 28)
    data.update(checkpoint={'repo':settings.REPO,'revision':settings.REVISION,
        'sha256':settings.FULL_FT_SHA256,'step':26774},
        actual_reference={'name':'Mira','voice':'emolia_c1699','file':str(REFERENCE),
            'sha256':hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
            'seconds':sf.info(REFERENCE).duration},
        batching='Fixed batches of up to four rows per condition, sorted block order; matched batch seeds across conditions. Each row has independently assigned design seed; actual batch RNG seed and companions are recorded. Streams have separate samples but share batch RNG, not per-row generators.',
        scoring='Untrimmed FP32 conditioning/output PCM; mono 16k anti-aliased resampling. Parakeet-v3 WER, pinned ECAPA cosine, Ears Medium CE/PQ/MOS, VoiceNet Aesthetics/Genuineness and EIV+ emotion proxies. No fades/alignment/VC or quality retries.',
        activation_points={'layer_0..layer_27':'Qwen block output, 1024D, before final norm',
            'semantic_global':'post-final-Qwen-RMSNorm, 1024D',
            'semantic_local':'proj_out output/local acoustic decoder seed, 2560D'},
        capture_detail='Only conditional last-position predictor states committed for actual emitted acoustic frames. Last prefill position included iff it predicts frame0; all other prompt/reference positions, EOS, already-finished rows and padded tails excluded. No full prompt averaging. Both CFG branches receive the same vector.',
        references=[{'name':'Activation Addition','url':'https://arxiv.org/abs/2308.10248'},
            {'name':'Representation Engineering','url':'https://arxiv.org/abs/2310.01405'}])
    path = ROOT/'design.json'
    if path.exists():
        assert json.loads(path.read_text()) == data, 'Frozen design changed; use another study directory'
    else:
        write_json(path, data)
    return data


class Quality:
    def __init__(self, device):
        from engine import Scorers
        from adaptive_reward import RewardPlan
        self.device = device
        self.base = Scorers()
        self.base.speakers.prepare('ecapa')
        self.target, _ = self.base.speakers.target(REFERENCE,'ecapa')
        self.anchor, _ = self.base.speakers.target(ANCHOR,'ecapa')
        self.plan = RewardPlan(terms=[{'dimension':v} for v in EMOTION_KEYS.values()])
        self.base.dimensions.prepare(self.plan)
        self.base.dimensions.prepare_keys({'vn:ESTH'})
        write_json(ROOT/'reference-controls.json', {'fixed_reference_sha256':hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
            'reference_to_original_anchor_cosine':float(self.target @ self.anchor),
            'ecapa_repo':settings.SPEAKER_REPO,'ecapa_revision':settings.SPEAKER_REVISION,
            'note':'Different files from the same corpus label, not independently verified recording-session/speaker ground truth.'})

    @torch.inference_mode()
    def score(self, waves, rows):
        from search_scoring import SearchScorer
        from metrics import word_errors
        from prompting import performance_direction
        with torch.cuda.device(self.device):
            audio = [torchaudio.functional.resample(torch.as_tensor(w,device=self.device),48000,16000) for w in waves]
            reader = SearchScorer.__new__(SearchScorer)
            reader.asr = self.base.asr
            hypotheses, stamps = reader.transcribe_batch(audio)
            ears = self.base.ears.predict(audio,'medium','ce')
            speaker, _ = self.base.speakers.embed(audio,'ecapa')
            units = F.normalize(speaker.float(),dim=-1)
            similarities = (units @ self.target.float()).cpu().tolist()
            anchors = (units @ self.anchor.float()).cpu().tolist()
            # VoiceCLAP has no true per-wave length mask. Encode each complete
            # waveform separately, so other candidates' durations cannot change
            # its aesthetics/genuineness score through batch padding.
            embeddings = torch.cat([self.base.retriever.model.encode_waveform(w.unsqueeze(0),
                sample_rate=16000).float() for w in audio],dim=0)
            aesthetics = self.base.dimensions.predict_voicenet(embeddings,('vn:ESTH',)).reshape(-1).cpu().tolist()
            emotions = self.base.dimensions.predict(audio,embeddings,self.plan)
            genuine = self.base.judge.gen._predict(embeddings).reshape(-1).cpu().tolist()
        result = []
        for i, row in enumerate(rows):
            raw = ears[i]['scores_raw']
            result.append({'ecapa':similarities[i],'anchor_ecapa':anchors[i],
                'content_enjoyment':raw['audiobox:CE'],'production_quality':raw['audiobox:PQ'],
                'mos':raw['dnsmos:P808_MOS'],'aesthetics':aesthetics[i],'genuineness':genuine[i],
                'emotion_score':emotions[i].get(EMOTION_KEYS.get(row['emotion'])),
                'emotion_scores':emotions[i],'speaker_embedding':units[i].cpu().tolist(),
                'hypothesis':hypotheses[i],'timestamps':stamps[i],**word_errors(row['text'],hypotheses[i]),
                'ears':ears[i],'audio_seconds':len(waves[i])/48000})
        return result


class RemoteQuality:
    """Share one local scorer copy between both GPU research workers."""
    def __init__(self, url):
        self.url = url.rstrip('/')

    def score(self, waves, rows):
        paths=[]
        folder=ROOT/'audio';folder.mkdir(exist_ok=True)
        for wave,row in zip(waves,rows):
            path=folder/(row['id']+'.wav')
            sf.write(path,np.asarray(wave,dtype=np.float32),48000,subtype='FLOAT')
            paths.append(str(path))
        data=json.dumps({'rows':rows,'paths':paths}).encode()
        request=urllib.request.Request(self.url+'/score',data=data,
            headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=180) as response:
            return json.load(response)['scores']


def save_audio(id_, wave):
    folder = ROOT/'audio';folder.mkdir(exist_ok=True)
    wav = folder/(id_+'.wav');mp3 = wav.with_suffix('.mp3')
    sf.write(wav,np.asarray(wave,dtype=np.float32),48000,subtype='FLOAT')
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(wav),'-map_metadata','-1',
        '-ac','1','-ar','48000','-c:a','libmp3lame','-b:a','96k',str(mp3)],check=True,timeout=30)
    return str(mp3.relative_to(ROOT))


def load_vectors(vector_file=None):
    path = Path(vector_file) if vector_file else ROOT/'vectors/selected.npz'
    if not path.is_absolute(): path = ROOT/path
    path = path.resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Research vector bank must be inside this study directory')
    meta = {'vector_file':str(path.relative_to(ROOT)),
        'vector_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    arrays = np.load(path,allow_pickle=False)
    return {key:torch.from_numpy(arrays[key].copy()) for key in arrays.files},meta


@torch.inference_mode()
def verify_instrumentation(engine):
    """GPU-code parity before selecting vectors, including both CFG branches."""
    specs = [build({'text':'We can talk about this tomorrow.','caption':'Calm conversation.'},
        'caption','en',reference_identity=True),
        build({'text':'Wir können morgen darüber sprechen.','caption':'Contained anger.'},
        'caption','de',reference_identity=True)]
    checks = []
    with torch.cuda.device(engine.device),torch.autocast('cuda',dtype=torch.bfloat16):
        reference=engine.reference(REFERENCE)
        branches=[engine.pack_many(specs,reference),engine.pack_many(specs,reference,neutral=True)]
        for temperature in (0.,1.):
            params={'max_new_tokens':32,'audio_temperature':temperature,'audio_top_p':.95,
                'audio_top_k':50,'audio_repetition_penalty':1.,'text_temperature':temperature,
                'text_top_p':1.,'text_top_k':50}
            outputs={};capture=None
            for mode in ('original','capture','zero_layer','zero_semantic'):
                torch.manual_seed(6701);torch.cuda.manual_seed_all(6701)
                collector=ActivationCollector(2) if mode=='capture' else None
                if mode=='original':
                    iterator=engine.fast.stream_frames(branches,params,33,guidance=2.,graph=False)
                else:
                    sem=SemanticVectorTransform(torch.ones(2560),point='semantic_local',alpha=0.) if mode=='zero_semantic' else None
                    iterator=research_stream_frames(engine,branches,params,33,guidance=2.,collector=collector,
                        layer_vectors={0:torch.ones(1024),27:torch.ones(1024)} if mode=='zero_layer' else {},
                        factors=0.,semantic_transform=sem,graph=False)
                frames=[]
                for frames in iterator:pass
                assert frames,'Parity probe produced no audio'
                outputs[mode]=torch.stack(frames,dim=1).cpu()
                if collector is not None:capture=collector.export()
            assert all(torch.equal(outputs['original'],codes) for codes in outputs.values())
            valid=outputs['original'][:,:,0].ne(engine.model.config.audio_pad_token_id).sum(-1)
            assert torch.equal(valid,capture['counts'])
            assert len(capture['means'])==30 and all(torch.equal(c,valid) for c in capture['point_counts'].values())
            checks.append({'temperature':temperature,'code_exact':True,'frame_counts':valid.tolist(),
                'points':{key:list(value.shape) for key,value in capture['means'].items()}})
    write_json(ROOT/'instrumentation-check.json',{'passed':True,'checks':checks,
        'scope':'Two matched bilingual 32-frame prefix tests, eager global/local, original vs capture vs zero layer vs zero projected semantic steering. Not full-distribution or production-compiled-path parity.'})
    print('GPU instrumentation and zero-factor parity passed',flush=True)


def run_rows(rows, device, score_device, capture=False, batch_size=4, score_url=None, vector_file=None):
    settings.DEVICE = device;settings.SCORE_DEVICE = score_device
    settings.COMPILE_GLOBAL = False;settings.FAST_INFERENCE = True
    torch.set_num_threads(4);torch.cuda.set_device(device)
    from engine import SmallEngine
    from dpo_adapters import AdapterSelection
    engine = SmallEngine()
    assert engine.checkpoint_info['sha256'] == settings.FULL_FT_SHA256
    engine.configure_adapters(AdapterSelection())
    if capture:verify_instrumentation(engine)
    quality = RemoteQuality(score_url) if score_url else Quality(score_device)
    arrays, meta = ({},None) if capture else load_vectors(vector_file)
    banks = {}
    groups = defaultdict(list)
    for row in rows:
        groups[row['condition']].append(row)
    for condition, group in groups.items():
        group.sort(key=lambda x:x['block'])
        declared_bank = group[0]['intervention'].get('vector_file')
        if declared_bank:
            if declared_bank not in banks: banks[declared_bank] = load_vectors(declared_bank)
            condition_arrays, condition_meta = banks[declared_bank]
        else:
            condition_arrays, condition_meta = arrays, meta
        for offset in range(0,len(group),batch_size):
            block = group[offset:offset+batch_size]
            paths = [ROOT/'results'/(r['id']+'.json') for r in block]
            if all(p.exists() for p in paths):continue
            if any(p.exists() for p in paths):
                raise RuntimeError('Partial matched batch; preserve results and explicitly recover before retry')
            collector = ActivationCollector(len(block)) if capture else None
            spec = [build(parse_literal(r['prompt'],'caption'),'caption',r['language'],
                literal=r['prompt'],reference_identity=True) for r in block]
            intervention = block[0]['intervention'];points = intervention['points']
            kind = intervention.get('vector_kind') or ('contrast' if 'contrast' in intervention['kind'] else 'raw_mean')
            layer_vectors = {int(p.removeprefix('layer_')):condition_arrays[p+'__'+kind] for p in points if p.startswith('layer_')}
            scale = intervention['alpha']/max(1,len(points))
            relative = intervention['kind'] == 'norm_matched_mean' or intervention.get('normalization') == 'rms_relative'
            semantic_points = [p for p in points if p.startswith('semantic')]
            assert len(semantic_points) <= 1, 'Two semantic sites simultaneously need explicit separate-dose protocol'
            semantic = SemanticVectorTransform(vector=condition_arrays[semantic_points[0]+'__'+kind],
                point=semantic_points[0],alpha=scale,normalization='rms_relative' if relative else 'raw') if semantic_points else None
            def stream(branches,p,chunk_frames,guidance=2.,min_frames=0,graph=False):
                return research_stream_frames(engine,branches,p,chunk_frames,guidance=guidance,
                    collector=collector,layer_vectors=layer_vectors,factors=scale,
                    normalization='rms_relative' if relative else 'raw',semantic_transform=semantic,
                    min_frames=min_frames,graph=False)
            engine.fast.stream_frames = stream
            started = time.monotonic()
            base_info = {'device':device,'score_device':score_device,'batch_seed':block[0]['seed'],
                'batch_ids':[r['id'] for r in block],'batch_size':len(block),
                'checkpoint':engine.checkpoint_info,'adapters':[], 'cfg':2.,'temperature':1.,
                'reference_sha256':hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
                'per_point_alpha':scale,'normalization':'rms_relative' if relative else 'raw',
                'max_frames':256,'global_eager':True,'local_cuda_graph':False}
            base_info['scoring_service']=score_url
            base_info['vector_bank']=condition_meta
            base_info['vector_kind']=kind if points else None
            try:
                waves = engine.generate_many(spec,guidance=2.,seed=block[0]['seed'],reference_path=REFERENCE,
                    temperature=1.,streaming=False,batch_size=len(block),max_frames=256)
                values = collector.export() if collector else None
                scored = quality.score(waves,block)
                for i,(row,path,wave,scores) in enumerate(zip(block,paths,waves,scored)):
                    count = len(engine.last_codes[i])
                    if capture:
                        assert count == int(values['counts'][i])
                        assert all(int(c[i]) == count for c in values['point_counts'].values())
                        directory = ROOT/'activations';directory.mkdir(exist_ok=True)
                        np.savez_compressed(directory/(row['id']+'.npz'),
                            **{key:t[i].numpy() for key,t in values['means'].items()},frame_count=np.array(count))
                    audio_path = save_audio(row['id'],wave)
                    write_json(path,{**row,**base_info,**scores,'audio_path':audio_path,
                        'status':'ok','failed':False,'generated_frames':count,
                        'acoustic_tokens':count*12,'hit_frame_limit':count>=min(256,spec[i]['frames']+60),
                        'batch_wall_seconds':time.monotonic()-started,
                        'performance':engine.last_performance})
            except Exception as exc:
                for row,path in zip(block,paths):
                    if not path.exists():write_json(path,{**row,**base_info,'status':'failed','failed':True,
                        'error':str(exc),'traceback':traceback.format_exc(),'batch_wall_seconds':time.monotonic()-started})
                print(json.dumps({'failed_batch':base_info['batch_ids'],'error':str(exc)}),flush=True)
                if isinstance(exc,torch.cuda.OutOfMemoryError):
                    raise
            print(json.dumps({'phase':block[0]['phase'],'condition':condition,
                'completed_in_condition':offset+len(block),'condition_total':len(group),
                'wall_seconds':round(time.monotonic()-started,2)}),flush=True)
    print('Completed requested rows',len(rows),flush=True)


def vectors():
    rows = read_results('collection')
    assert len(rows) == 100, 'Do not silently complete an incomplete collection'
    arrays = {}
    for row in rows:
        if row['failed']:continue
        data = np.load(ROOT/'activations'/(row['id']+'.npz'),allow_pickle=False)
        arrays[row['id']] = {k:data[k] for k in data.files if k!='frame_count'}
    pools = [analysis.aggregate_vectors(rows,arrays,threshold=t) for t in (.50,.25)]
    chosen = next((x for x in pools if x['metadata']['eligible']),None)
    if chosen is None:
        chosen = next((x for x in pools if x['metadata']['high_n']>=10),None)
    if chosen is None:
        raise RuntimeError('Fewer than ten >.25 samples; no threshold-qualified vector. Keep collection and report this limitation.')
    metadata = {'selected':chosen['metadata'],'threshold_pools':[x['metadata'] for x in pools],
        'collection_rows':len(rows),'collection_failures':sum(r['failed'] for r in rows),
        'created_utc':datetime.now(timezone.utc).isoformat(),'reference_sha256':hashlib.sha256(REFERENCE.read_bytes()).hexdigest()}
    folder=ROOT/'vectors';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'selected.npz',**{p+'__'+kind:value for p,kinds in chosen['vectors'].items() for kind,value in kinds.items()})
    write_json(folder/'metadata.json',metadata)
    print(json.dumps(metadata['selected'],indent=2),flush=True)


def rescore_collection_style(url):
    """Correct only unmasked VoiceCLAP padding, preserving the original scores.

    Selection is already frozen from ECAPA; no audio/states/thresholds change.
    Keep all original Aesthetics/Genuineness values in each result's ledger.
    """
    rows=sorted(read_results('collection'),key=lambda x:x['block'])
    amendment={'date':datetime.now(timezone.utc).isoformat(),
        'before_search_selection':True,'discovered_by':'independent layer agent audit',
        'reason':'Original collection VoiceCLAP encoded zero-padded batches without a true waveform-length mask, making Aesthetics/Genuineness depend on companion lengths.',
        'change':'Encode each complete waveform independently for Aesthetics/Genuineness; preserve original values in original_batch_padded_style. All subsequent search uses independent style encoding. Collection ECAPA/WER/Ears scores, activation arrays and frozen selection/threshold/IDs remain unchanged.',
        'scope':'Measurement correction only, no generation or vector resampling.',
        'language_confound':'Primary >.50 pool contains 22 German and 1 English sample; emotion balancing alone does not remove phonetic/language/session confounding. Secondary >.25 joint-balanced diagnostics must be labeled separately.'}
    path=ROOT/'measurement-amendment.json'
    if not path.exists():write_json(path,amendment)
    scorer=RemoteQuality(url)
    for offset in range(0,len(rows),4):
        block=rows[offset:offset+4]
        if all('original_batch_padded_style' in r or r['failed'] for r in block):continue
        good=[r for r in block if not r['failed']]
        waves=[sf.read(ROOT/'audio'/(r['id']+'.wav'),dtype='float32')[0] for r in good]
        scores=scorer.score(waves,good)
        for row,score in zip(good,scores):
            row['original_batch_padded_style']={k:row[k] for k in ('aesthetics','genuineness')}
            row['aesthetics']=score['aesthetics'];row['genuineness']=score['genuineness']
            row['style_encoding']='Each waveform independently, no companion-duration padding'
            row['rescore_ecapa_delta_diagnostic']=score['ecapa']-row['ecapa']
            write_json(ROOT/'results'/(row['id']+'.json'),row)
        print('Collection measurement correction',offset+len(block),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['design','collection','vectors','run','summary','stylefix'])
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--score-device',default='cuda:1')
    parser.add_argument('--plan',type=Path);parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--score-url',default=None)
    parser.add_argument('--vectors',type=Path,default=None)
    args=parser.parse_args();ROOT.mkdir(exist_ok=True)
    if args.action=='design':freeze_design()
    elif args.action=='collection':
        design=freeze_design();run_rows(design['collection_rows'],args.device,args.score_device,True,args.batch_size,args.score_url)
    elif args.action=='vectors':vectors()
    elif args.action=='stylefix':
        assert args.score_url
        rescore_collection_style(args.score_url)
    elif args.action=='run':
        assert args.plan is not None
        run_rows(json.loads(args.plan.read_text())['rows'],args.device,args.score_device,False,args.batch_size,args.score_url,args.vectors)
    else:
        write_json(ROOT/'summary.json',analysis.summarize(read_results()))


if __name__=='__main__':main()
