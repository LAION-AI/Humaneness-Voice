"""Durable collection of five-layer targets for all 500 synthetic profiles.

Independent research copies; never changes the live TTS model or steering bank.
All generation, scoring and capture uses one pass. Resume at atomic batch
boundaries; failed attempts are recorded, never replaced with extra good seeds.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import soundfile as sf

import settings
from catalogue_target_analysis import LAYERS, aggregate, speaker_split

ROOT = settings.HERE/'catalogue-target-study'
PROFILE_REPO = 'laion/moss-voice-profile-loras-500'
PROFILE_REVISION = 'ec77b7b476bd31ceb9c0c0581c9aef680d62d6f5'


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            json.dump(data, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.flush(); os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def freeze(path, data):
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError('Immutable study artifact differs: '+str(path))
    else:
        atomic_json(path, data)
    return data


def cells(phase='collection'):
    import mira_activation_analysis as prompts
    source = 'screen' if phase == 'collection' else 'validation'
    rows = []
    for i, (emotion, caption) in enumerate(prompts.EMOTIONS.items()):
        for language in ('en','de'):
            text = prompts.TEXTS[source][language][i]
            rows.append({'condition_id': emotion+'-'+language, 'emotion': emotion,
                'language': language, 'text': text, 'reference_identity': False,
                'prompt': 'CAPTION: '+caption+'\nTRANSCRIPT: '+json.dumps(text, ensure_ascii=False)})
    return rows


def batches(voice, repeats=100):
    """Each identical cell has 100 distinct actual seeds, 4/4/2 mixed batches."""
    definitions = cells()
    for repeat in range(repeats):
        for group, start in enumerate((0,4,8)):
            seed = 41000000+repeat*1009+group*4
            rows = [{**cell, 'repeat': repeat, 'block': repeat*10+start+i,
                     'voice': voice, 'seed': seed, 'batch_seed': seed,
                     'id': f'{voice}-{repeat*10+start+i:04d}'}
                    for i, cell in enumerate(definitions[start:start+4])]
            yield f'r{repeat:03d}-g{group}', rows


def prepare():
    """Download only 500 reference WAVs and the name catalog, NOT adapter weights."""
    from huggingface_hub import HfApi, hf_hub_download
    import dynamic_references as refs
    ROOT.mkdir(exist_ok=True)
    path = ROOT/'design.json'
    if path.exists():
        design = json.loads(path.read_text())
        for voice in design['voices']:
            if hashlib.sha256(Path(voice['reference_path']).read_bytes()).hexdigest() != voice['reference_sha256']:
                raise ValueError('Reference changed: '+voice['id'])
        return design
    names = json.loads(Path(hf_hub_download(PROFILE_REPO,'names.json',revision=PROFILE_REVISION)).read_text())
    info = HfApi().model_info(PROFILE_REPO, revision=PROFILE_REVISION)
    ids = sorted(s.rfilename.split('/')[1] for s in info.siblings
                 if s.rfilename.startswith('voices/') and s.rfilename.endswith('/reference.wav'))
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError('Expected exactly 500 published profiles')
    demo = refs.catalog()['voices']; demo_ids = [v['id'] for v in demo]
    if not set(demo_ids).issubset(ids):
        raise ValueError('Demo profiles missing from large catalog')
    order = demo_ids+[v for v in ids if v not in demo_ids]
    def fetch(voice):
        source = Path(hf_hub_download(PROFILE_REPO, f'voices/{voice}/reference.wav', revision=PROFILE_REVISION))
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if voice in demo_ids:
            reference, _ = refs.fixed_clip(voice)
            name = next(v['name'] for v in demo if v['id'] == voice)
            policy = 'unchanged fixed demo reference'
        else:
            reference = source; name = names[voice]['name']; policy = 'pinned published profile reference'
        audio = sf.info(reference)
        if audio.duration < .5:
            raise ValueError('Reference too short: '+voice)
        digest = hashlib.sha256(reference.read_bytes()).hexdigest()
        return {'id': voice, 'name': name, 'demo': voice in demo_ids,
            'reference_path': str(reference), 'reference_sha256': digest,
            'reference_source_seconds': audio.duration, 'reference_used_seconds': min(15., audio.duration),
            'reference_policy': policy, 'published_reference_sha256': source_hash,
            'split': speaker_split(digest),
            'source_url': f'https://huggingface.co/{PROFILE_REPO}/blob/{PROFILE_REVISION}/voices/{voice}/reference.wav'}
    voices = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for item in pool.map(fetch, order):
            voices.append(item)
            if len(voices)%25 == 0 or len(voices) <= len(demo_ids):
                print('Downloaded references',len(voices),'/',len(order),item['name'], flush=True)
                atomic_json(ROOT/'preparation.json', {'updated_utc': now(), 'references': len(voices), 'total': 500})
    design = {'version': 1, 'profile_repository': PROFILE_REPO, 'profile_revision': PROFILE_REVISION,
        'checkpoint_repo': settings.REPO, 'checkpoint_revision': settings.REVISION,
        'checkpoint_sha256': settings.FULL_FT_SHA256, 'voices': voices,
        'cells': cells(), 'n_per_cell': 100, 'top_percent': 2, 'bottom_percent': 25,
        'total_takes': 500000, 'layers': list(LAYERS), 'layer_indexing': 'zero-based Qwen block output before final global RMSNorm',
        'settings': {'cfg': 2., 'temperature': 1., 'max_frames': 256, 'batch_size': 4,
            'local_cuda_graphs': True, 'global_compile': False, 'adapters': [], 'steering': False,
            'voice_conversion': False, 'alignment_trim': False, 'reference_cap_seconds': 15},
        'selection': 'Within each fixed emotion/language/text cell, highest two of exactly 100 ECAPA-scored takes minus lowest 25; mean take vectors then equally mean the ten cell differences. The same high/low take IDs apply to all five layers.',
        'secondary_target': 'Nested first 50 seeds, highest one minus lowest 13 per cell; no additional generation or seed selection.',
        'seed_rule': '4/4/2 batches contain different cells. Every cell receives 100 distinct actual batch seeds, fixed before outcomes. Batch companions and RNG recorded; not independent per-row RNG.',
        'capture': 'Conditional final predictor position only, committed for emitted acoustic frames. Includes predictor of frame zero; excludes other prompt/reference positions, stop/EOS and padded finished rows. Equal sample weighting, not pooled-token weighting.',
        'storage': 'Atomic batch directories retain every MP3, per-take scores/frames, five-layer activation means and normalized output speaker embeddings. No large generated WAV archive; encode MP3 from memory after scoring uncompressed PCM.',
        'mlp_dataset': {'input': '192D pinned ECAPA embedding of actual reference prefix used for conditioning',
            'target': 'float32 delta[5,1024], also individual layer_14..18__contrast and cell_deltas[10,5,1024]',
            'split': 'Deterministic 80/10/10 hash split grouped by actual reference content hash; no training or test evaluation performed yet.'},
        'limitations': ['Top ECAPA is not Top Content Enjoyment; no aesthetic/WER/emotion gate in this large collection.',
            'These are unvalidated candidate targets, not automatically deployed steering directions.',
            'One fixed reference per synthetic profile; no cross-reference or real-speaker generalization claim.',
            'Identity, prosody, content and generic reference effects can remain entangled.',
            'Unknown aliases sharing a speaker but different reference hashes can still leak between dataset splits.',
            'Two extreme selected samples per cell remain noisy. Fresh held-out generation and quality checks are needed before UI replacement.']}
    return freeze(path, design)


def completed_batches(folder):
    return sorted(p for p in (folder/'batches').glob('r*-g*') if p.is_dir() and (p/'results.json').is_file())


def read_collection(voice):
    rows, means = [], {}
    for folder in completed_batches(ROOT/'voices'/voice):
        batch = json.loads((folder/'results.json').read_text())
        with np.load(folder/'captures.npz', allow_pickle=False) as data:
            values, counts = data['means'], data['frame_counts']
            if not np.array_equal(data['layers'],np.asarray(LAYERS)):
                raise ValueError('Incorrect capture layer ordering')
            if values.shape != (len(batch),5,1024) or not np.isfinite(values).all():
                raise ValueError('Invalid capture: '+str(folder))
            for i,row in enumerate(batch):
                if row['generated_frames'] != int(counts[i]):
                    raise ValueError('Capture/frame mismatch')
                means[row['id']] = values[i].copy()
        rows.extend(batch)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate take IDs')
    return rows, means


def export_dataset():
    """Compact input/target matrix, only after all declared speakers finish."""
    design=json.loads((ROOT/'design.json').read_text())
    xs,ys,ids,splits,hashes=[],[],[],[],[]
    for voice in design['voices']:
        folder=ROOT/'targets'/voice['id']
        if not (folder/'metadata.json').exists():
            raise ValueError('Cannot export complete dataset with missing speaker: '+voice['id'])
        with np.load(folder/'n100-top2.npz',allow_pickle=False) as data:
            xs.append(data['speaker_embedding'].copy());ys.append(data['delta'].copy())
        ids.append(voice['id']);splits.append(voice['split']);hashes.append(voice['reference_sha256'])
    temp=ROOT/'training-dataset.tmp.npz'
    np.savez_compressed(temp,speaker_embeddings=np.stack(xs),deltas=np.stack(ys),
        voice_ids=np.asarray(ids),splits=np.asarray(splits),reference_sha256=np.asarray(hashes),
        layers=np.asarray(LAYERS))
    temp.replace(ROOT/'training-dataset.npz')
    return len(ids)


def derive(voice):
    folder = ROOT/'voices'/voice
    rows, means = read_collection(voice)
    if len(rows) != 1000:
        raise ValueError('Require all 1000 takes: '+voice)
    arrays, meta = aggregate(rows, means, 100)
    secondary, secondary_meta = aggregate(rows, means, 50)
    with np.load(folder/'speaker-input.npz', allow_pickle=False) as data:
        speaker = data['speaker_embedding'].copy()
    if speaker.shape != (192,) or not np.isfinite(speaker).all() or not np.isclose(np.linalg.norm(speaker), 1., atol=1e-5):
        raise ValueError('Invalid speaker training input')
    design = json.loads((ROOT/'design.json').read_text())
    profile = next(v for v in design['voices'] if v['id'] == voice)
    output = ROOT/'targets'/voice; output.mkdir(parents=True, exist_ok=True)
    for name, content in (('n100-top2', arrays),('n50-top2',secondary)):
        destination = output/(name+'.npz')
        if destination.exists():
            with np.load(destination, allow_pickle=False) as prior:
                if not np.array_equal(prior['delta'],content['delta']):
                    raise ValueError('Immutable target differs')
        else:
            temp = output/(name+'.tmp.npz')
            np.savez_compressed(temp, **content, speaker_embedding=speaker)
            temp.replace(destination)
    meta.update(profile=profile, checkpoint_sha256=settings.FULL_FT_SHA256,
        speaker_model={'repo': settings.SPEAKER_REPO,'revision':settings.SPEAKER_REVISION},
        primary_file=str(output/'n100-top2.npz'),secondary=secondary_meta,
        cell_order=[c['cell'] for c in meta['cells']], updated_utc=now())
    # Completion is published only after both immutable NPZs exist.
    atomic_json(output/'metadata.json', meta)
    return meta


def save_mp3(path, wave):
    wave = np.asarray(wave, dtype='<f4')
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-f','f32le','-ar','48000','-ac','1',
        '-i','pipe:0','-map_metadata','-1','-c:a','libmp3lame','-b:a','96k',str(path)],
        input=wave.tobytes(),check=True,timeout=30,stdout=subprocess.DEVNULL)


def batch_commit(folder, key, rows, means, counts, embeddings, waves):
    destination = folder/'batches'/key
    if destination.exists():
        raise ValueError('Do not overwrite committed batch')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.partial-'+key+'-',dir=destination.parent))
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda item: save_mp3(staging/(item[0]['id']+'.mp3'),item[1]), zip(rows,waves)))
        np.savez_compressed(staging/'captures.npz',means=means,frame_counts=counts,
                            layers=np.asarray(LAYERS),output_speaker_embeddings=embeddings)
        atomic_json(staging/'results.json', rows)
        staging.replace(destination)
    except Exception:
        # Keep the exact owned partial directory for diagnosis, never consume it.
        atomic_json(staging/'failure.json', {'utc':now(),'traceback':traceback.format_exc()})
        raise


def run_worker(voice_ids, device, limit_batches=None):
    import torch
    import torchaudio
    from torch.nn import functional as F
    from types import SimpleNamespace
    from activation_steering import ActivationCollector, research_stream_frames
    from prompting import build, parse_literal
    from speaker_reward import SpeakerModels
    settings.DEVICE=device; settings.COMPILE_GLOBAL=False; settings.FAST_INFERENCE=True
    settings.REFERENCE_SECONDS=15.; settings.STAGE=settings.FULL_FT_STAGE
    torch.set_num_threads(4); torch.cuda.set_device(device)
    from engine import SmallEngine
    from dpo_adapters import AdapterSelection
    design=json.loads((ROOT/'design.json').read_text())
    engine=SmallEngine(); engine.configure_adapters(AdapterSelection())
    engine.configure_speaker_steering(None,{'active':False,'reason':'offline_training_target_collection'})
    if engine.checkpoint_info['sha256'] != design['checkpoint_sha256']:
        raise ValueError('Wrong checkpoint')
    speakers=SpeakerModels(SimpleNamespace(retriever=SimpleNamespace(device=device)))
    speakers.prepare('ecapa')
    batch_count=0
    with torch.inference_mode():
        # GPU preflight: prove the five passive capture hooks do not change the
        # generated codes/audio. Warm the same local graph before paired checks.
        preflight_profile=next(v for v in design['voices'] if v['id']==voice_ids[0])
        preflight_reference=Path(preflight_profile['reference_path'])
        probe=[build(parse_literal(r['prompt'],'caption'),'caption',r['language'],
                     literal=r['prompt'],reference_identity=False) for r in cells('validation')[:4]]
        checks=[]
        for capture in (False,False,True):
            collector=ActivationCollector(len(probe)) if capture else None
            def probe_stream(branches,p,chunk_frames,guidance=2.,min_frames=0,graph=False):
                return research_stream_frames(engine,branches,p,chunk_frames,guidance=guidance,
                    collector=collector,collect_layers=LAYERS,layer_vectors={},factors=0.,
                    min_frames=min_frames,graph=True)
            engine.fast.stream_frames=probe_stream
            audio=engine.generate_many(probe,guidance=2.,seed=39000000,reference_path=preflight_reference,
                temperature=1.,streaming=False,batch_size=4,max_frames=128)
            if capture:
                prior_codes,prior_audio=checks[-1]
                if any(not torch.equal(a,b) for a,b in zip(prior_codes,engine.last_codes)) or any(
                    not np.array_equal(a,b) for a,b in zip(prior_audio,audio)):
                    raise ValueError('GPU preflight failed: seeded capture changed output')
            checks.append(([c.clone() for c in engine.last_codes],[w.copy() for w in audio]))
            if collector is not None:
                values=collector.export()
                frame_counts=np.asarray([len(c) for c in engine.last_codes])
                if not np.array_equal(values['counts'].numpy(),frame_counts):
                    raise ValueError('GPU preflight: frame mask mismatch')
                for layer in LAYERS:
                    if values['means'][f'layer_{layer}'].shape!=(4,1024) or not torch.isfinite(values['means'][f'layer_{layer}']).all():
                        raise ValueError('GPU preflight: invalid layer capture')
        atomic_json(ROOT/('gpu-preflight-'+device.replace(':','-')+'.json'),
            {'passed':True,'updated_utc':now(),'device':device,'layers':list(LAYERS),
             'seed':39000000,'paired_codes_identical':True,'paired_audio_identical':True,
             'actual_frame_counts':frame_counts.tolist(),'outside_collection_pool':True})
        print('GPU PREFLIGHT PASSED',device,'five layers; codes/audio exact',flush=True)
        del checks
        for voice in voice_ids:
            profile=next(v for v in design['voices'] if v['id']==voice)
            folder=ROOT/'voices'/voice; folder.mkdir(parents=True,exist_ok=True)
            with (folder/'worker.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if (ROOT/'targets'/voice/'metadata.json').exists():
                    print('Already complete',voice,flush=True);continue
                reference=Path(profile['reference_path'])
                if hashlib.sha256(reference.read_bytes()).hexdigest()!=profile['reference_sha256']:
                    raise ValueError('Reference changed')
                target,_=speakers.target(reference,'ecapa')
                target_np=target.cpu().numpy().astype(np.float32)
                dest=folder/'speaker-input.npz'
                if dest.exists():
                    with np.load(dest,allow_pickle=False) as data:
                        if not np.allclose(data['speaker_embedding'],target_np,atol=1e-6):
                            raise ValueError('Speaker input changed')
                else:
                    np.savez_compressed(folder/'speaker-input.tmp.npz',speaker_embedding=target_np)
                    (folder/'speaker-input.tmp.npz').replace(dest)
                prior=completed_batches(folder); completed=sum(len(json.loads((p/'results.json').read_text())) for p in prior)
                print('Starting',profile['name'],voice,'already',completed,'device',device,flush=True)
                for key,batch in batches(voice):
                    path=folder/'batches'/key
                    if path.exists():
                        saved=json.loads((path/'results.json').read_text())
                        if [r['id'] for r in saved]!=[r['id'] for r in batch] or any(r.get('failed') for r in saved):
                            raise ValueError('Invalid committed batch')
                        continue
                    if shutil.disk_usage(ROOT).free < 20*1024**3:
                        raise RuntimeError('Stopping safely: less than 20 GiB free disk')
                    collector=ActivationCollector(len(batch))
                    def stream(branches,p,chunk_frames,guidance=2.,min_frames=0,graph=False):
                        return research_stream_frames(engine,branches,p,chunk_frames,guidance=guidance,
                            collector=collector,collect_layers=LAYERS,layer_vectors={},factors=0.,
                            min_frames=min_frames,graph=True)
                    engine.fast.stream_frames=stream
                    specs=[build(parse_literal(r['prompt'],'caption'),'caption',r['language'],literal=r['prompt'],reference_identity=False) for r in batch]
                    tick=time.monotonic()
                    try:
                        waves=engine.generate_many(specs,guidance=2.,seed=batch[0]['seed'],reference_path=reference,
                            temperature=1.,streaming=False,batch_size=len(batch),max_frames=256)
                        values=collector.export()
                        counts=np.asarray([len(c) for c in engine.last_codes],dtype=np.int64)
                        if not np.array_equal(values['counts'].numpy(),counts) or np.any(counts<=0):
                            raise ValueError('Empty take or actual-frame capture mismatch')
                        for layer in LAYERS:
                            if not np.array_equal(values['point_counts'][f'layer_{layer}'].numpy(),counts):
                                raise ValueError('Layer-specific predictor count mismatch')
                        means=np.stack([values['means'][f'layer_{layer}'].numpy() for layer in LAYERS],axis=1)
                        if means.shape != (len(batch),5,1024) or not np.isfinite(means).all():
                            raise ValueError('Invalid layer means')
                        audio16=[torchaudio.functional.resample(torch.as_tensor(w,device=device),48000,16000) for w in waves]
                        if any(len(w)<8000 for w in audio16):
                            raise ValueError('Generated take too short for speaker embedding; no automatic retry')
                        encoded,_=speakers.embed(audio16,'ecapa')
                        embeddings=F.normalize(encoded,dim=-1)
                        if not torch.isfinite(embeddings).all():
                            raise ValueError('Nonfinite output embedding')
                        scores=(embeddings@target).cpu().numpy()
                        reference_info=engine.reference_info[str(reference)]
                        rows=[{**r,'ecapa':float(scores[i]),'generated_frames':int(counts[i]),
                            'seconds':len(waves[i])/48000,'hit_frame_cap':int(counts[i])>=256,
                            'audio_file':r['id']+'.mp3','failed':False,'status':'ok',
                            'device':device,'batch_size':len(batch),'batch_ids':[t['id'] for t in batch],
                            'reference_sha256':profile['reference_sha256'],'reference_info':reference_info,
                            'checkpoint_sha256':design['checkpoint_sha256'],'batch_wall_seconds':time.monotonic()-tick,
                            'capture_layers':list(LAYERS),'local_cuda_graphs':True} for i,r in enumerate(batch)]
                        batch_commit(folder,key,rows,means,counts,embeddings.cpu().numpy(),waves)
                        completed+=len(batch); batch_count+=1
                        atomic_json(folder/'state.json',{'status':'collecting','voice':voice,'device':device,
                            'completed':completed,'expected':1000,'last_batch':key,'updated_utc':now(),
                            'last_batch_seconds':time.monotonic()-tick})
                        print(profile['name'],completed,'/1000',key,round(time.monotonic()-tick,2),'seconds',flush=True)
                    except Exception as error:
                        atomic_json(folder/'failure-'+key+'.json',{'failed':True,'rows':batch,'utc':now(),
                            'error':str(error),'traceback':traceback.format_exc()})
                        atomic_json(folder/'state.json',{'status':'failed','voice':voice,'device':device,
                            'completed':completed,'error':str(error),'updated_utc':now()})
                        raise
                    if limit_batches is not None and batch_count>=limit_batches:
                        print('Smoke limit complete',batch_count,flush=True);return
                metadata=derive(voice)
                atomic_json(folder/'state.json',{'status':'complete','voice':voice,'device':device,
                    'completed':1000,'updated_utc':now(),'high_ecapa':metadata['mean_high_ecapa']})
                engine.forget_reference(reference)
                print('TARGET COMPLETE',voice,flush=True)


def status():
    design=json.loads((ROOT/'design.json').read_text())
    profiles=[]
    for voice in design['voices']:
        state=ROOT/'voices'/voice['id']/'state.json'
        current=json.loads(state.read_text()) if state.exists() else {'status':'queued','completed':0}
        meta=ROOT/'targets'/voice['id']/'metadata.json'
        if meta.exists():
            target=json.loads(meta.read_text())
            current={**current,'status':'complete','completed':1000,
                'mean_high_ecapa':target['mean_high_ecapa'],'mean_low_ecapa':target['mean_low_ecapa']}
        profiles.append({**{k:voice[k] for k in ('id','name','demo','split')},**current})
    data={'updated_utc':now(),'total_takes':500000,'completed_takes':sum(v['completed'] for v in profiles),
        'total_voices':500,'completed_voices':sum(v['status']=='complete' for v in profiles),
        'profiles':profiles,'stage':'collection','live_demo_changed':False}
    atomic_json(ROOT/'progress.json',data)
    index=[{'voice':v['id'],'name':v['name'],'split':v['split'],
        'input_target_file':str(ROOT/'targets'/v['id']/'n100-top2.npz'),
        'metadata_file':str(ROOT/'targets'/v['id']/'metadata.json')}
        for v in profiles if v['status']=='complete']
    atomic_json(ROOT/'training-index.json',{'completed':len(index),'expected':500,
        'input_key':'speaker_embedding','target_key':'delta','target_shape':[5,1024],
        'layers':list(LAYERS),'targets':index,'validation':'Not yet tested on fresh generations.'})
    return data


def report(publish=False):
    data=status()
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in (
        v['name'],v['id'],'demo' if v['demo'] else 'catalog',v['status'],v['completed'],v['split'],
        f"{v['mean_high_ecapa']:.3f}" if 'mean_high_ecapa' in v else '—'))+'</tr>' for v in data['profiles'])
    section=f'''<section id="training-targets"><h2>500-profile Top-2% training-target collection</h2>
<p><strong>{data['completed_takes']:,} / 500,000 takes; {data['completed_voices']} / 500 voices complete.</strong>
Updated {html.escape(data['updated_utc'])}. Reload this page to see the latest snapshot.</p>
<p>100 seeds for each of ten fixed conditions (five emotions × English/German). Per condition:
exactly the highest two ECAPA cosines minus the lowest 25. Average actual-frame predictor states per take,
then equal-weight the ten condition differences. Capture layers 14, 15, 16, 17, 18 simultaneously
(zero-based block outputs, 1024D each). A nested 50-seed Top2% target is retained as a sensitivity check.</p>
<p>Full-FT base only, CFG 2.0, temperature 1.0, reference cap 15 seconds. All ten demo voices run first
with their unchanged fixed references; the remaining profiles use pinned published reference WAVs.
<a href="https://huggingface.co/{PROFILE_REPO}">500-profile source repository</a>.
Local CUDA graphs accelerate only the acoustic decoder; the hooked semantic backbone remains eager.</p>
<p>Training input: normalized 192D ECAPA reference embedding. Training target: five × 1024D raw high-minus-low
vectors, per-cell deltas and high/low means, with complete selection provenance and speaker-grouped split.
All generated MP3s, scores and per-take captures are retained. No MLP has been trained.</p>
<p><strong>Collection targets are not validated improvements.</strong> ECAPA selection does not screen
Content Enjoyment, MOS, emotion or WER. One reference per synthetic voice is insufficient to establish
reference-invariant identity. Exact duplicate reference hashes share a split; unknown aliases may remain.
The live UI defaults and deployed steering bank are unchanged.</p>
<div style="overflow-x:auto"><table><thead><tr><th>Name</th><th>Profile</th><th>Group</th><th>Status</th>
<th>Takes / 1000</th><th>MLP split</th><th>Selected ECAPA</th></tr></thead><tbody>{table}</tbody></table></div></section>'''
    atomic_json(ROOT/'report-state.json',{'updated_utc':data['updated_utc'],'completed_takes':data['completed_takes']})
    local='<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Speaker training targets</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px}td,th{padding:8px;text-align:left}tr:nth-child(even){background:#eee}</style>'+section+'</html>'
    (ROOT/'index.tmp.html').write_text(local); (ROOT/'index.tmp.html').replace(ROOT/'index.html')
    if publish:
        from seed_scaling_analysis import insert_report_extension
        existing=settings.HERE/'catalogue-steering-study/index.html'
        new=insert_report_extension(existing.read_text(),'training-targets',section)
        temp=existing.with_name('index.training-targets.tmp.html')
        temp.write_text(new);temp.replace(existing)
    return data


def pipeline(devices=('cuda:0',)):
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        design=prepare(); report(True)
        jobs=[]
        try:
            for shard,device in enumerate(devices):
                assigned=[v['id'] for v in design['voices'][shard::len(devices)]]
                log=(ROOT/f'worker-{shard}.log').open('ab')
                jobs.append(subprocess.Popen([sys.executable,__file__,'worker','--voices',','.join(assigned),'--device',device],
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
                log.close()
            atomic_json(ROOT/'processes.json',{'coordinator_pid':os.getpid(),'worker_pids':[p.pid for p in jobs],
                'devices':list(devices),'started_utc':now(),'commands':'catalogue_target_study.py pipeline / worker'})
            last_publish=0;last_count=-1
            while any(p.poll() is None for p in jobs):
                if any(p.poll() not in (None,0) for p in jobs):
                    raise RuntimeError('Collection worker failed; see owned worker logs. No failed samples silently replaced.')
                current=status()
                if time.monotonic()-last_publish>300 or current['completed_voices']!=last_count:
                    report(True);last_publish=time.monotonic();last_count=current['completed_voices']
                print(now(),current['completed_takes'],'/500000; targets',current['completed_voices'],'/500',flush=True)
                time.sleep(30)
            if any(p.returncode for p in jobs):raise RuntimeError('Worker failure')
            data=report(True)
            if data['completed_voices']!=500:raise RuntimeError('Incomplete target catalogue')
            export_dataset()
            atomic_json(ROOT/'completion.json',{'completed_utc':now(),'takes':500000,'voices':500,
                'deployed':False,'mlp_trained':False})
            print('CATALOGUE TARGET COLLECTION COMPLETE',flush=True)
        except Exception:
            atomic_json(ROOT/'pipeline-failure.json',{'utc':now(),'traceback':traceback.format_exc()})
            raise
        finally:
            # Terminate only processes this coordinator owns; never live TTS/tunnel.
            for job in jobs:
                if job.poll() is None:
                    job.terminate()
                    try:job.wait(timeout=15)
                    except subprocess.TimeoutExpired:job.kill();job.wait()
            if (ROOT/'design.json').exists():report(True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','worker','pipeline','report','derive','export'])
    parser.add_argument('--voices');parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--limit-batches',type=int);parser.add_argument('--publish',action='store_true')
    parser.add_argument('--devices',default='cuda:0',help='Research GPUs; default one GPU only')
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='pipeline':
        devices=tuple(args.devices.split(','))
        if not devices or len(devices)!=len(set(devices)) or any(d not in ('cuda:0','cuda:1') for d in devices):
            raise ValueError('Use distinct explicit cuda:0/cuda:1 devices')
        pipeline(devices)
    elif args.action=='report':report(args.publish)
    elif args.action=='export':export_dataset()
    elif args.action=='derive':
        for voice in args.voices.split(','):derive(voice)
    else:run_worker(args.voices.split(','),args.device,args.limit_batches)


if __name__=='__main__':main()
