"""Audited source + public-synthetic student checkpoints; no microphone data.

Default builds a local allowlisted release. --publish explicitly uploads to the
fixed LAION model repo, never deleting remote files or exposing credentials.
"""
import argparse,gzip,hashlib,json,re,shutil,subprocess,sys,tempfile,time,zipfile
from pathlib import Path
from speaker_v2.publish_pilot import configured_token,validate_write_identity,SECRET

ROOT=Path(__file__).resolve().parent
REPO='laion/tyger-speaker-predictor'
OUT=Path('/mnt/nvme/robo-speaker-studies/public-release')

def clean(text):
    return text.replace(str(ROOT),'${ROBBY_SOURCE}').replace(str(Path.home()),'${HOME}').replace('/mnt/nvme/robo-speaker-studies','${STUDY_VOLUME}')

def build_at(OUT):
    import torch
    from speaker_v4.common import OLD,WORK,atomic
    from speaker_v4.train import objective
    from export_source import main as export
    export();OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'releases/robby-source.zip'
    with zipfile.ZipFile(source) as archive:
        for name in archive.namelist():
            if name.startswith('/') or '..' in Path(name).parts:raise ValueError('Unsafe source ZIP path')
            raw=archive.read(name)
            if SECRET.search(raw):raise ValueError('Credential pattern in source')
            path=OUT/'code'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    results=json.loads((WORK/'results.json').read_text())
    if not results.get('finished'):raise RuntimeError('V4 incomplete; refusing final publication')
    selected=results['selected_factorial_arm'];rows=[r for r in results['factorial'] if r['arm']==selected]
    best=max(rows,key=lambda r:objective(r['validation']));model_dir=OUT/'models';model_dir.mkdir(exist_ok=True)
    published=[]
    for row in rows:
        saved=torch.load(WORK/'models'/row['checkpoint'],map_location='cpu',weights_only=False)
        name=f'v4-layer-mix-320ms-seed-{row["seed"]}.pt'
        torch.save({'model':saved['model'],'architecture':'query_gru','width':128,'mix_layers':True,'context_frames':4,
            'layer_ids_one_based':[4,8,12,15],'embedding_dim':192,'frame_ms':80,'seed':row['seed'],
            'validation':row['validation'],'format_version':4},model_dir/name)
        published.append({'file':'models/'+name,'seed':row['seed'],'validation_selected_default':row['seed']==best['seed']})
    old=torch.load(OLD/'models/query_gru-best.pt',map_location='cpu',weights_only=False)
    torch.save({'model':old['model'],'architecture':'query_gru','width':128,'mix_layers':False,'context_frames':25,
        'layer_ids_one_based':[15],'embedding_dim':192,'frame_ms':80,'format_version':3},model_dir/'v3-l15-2000ms-baseline.pt')
    long_models={};v5=Path.home()/'.cache/robo-v5';v5_results=None
    if (v5/'results.json').exists():
        v5_results=json.loads((v5/'results.json').read_text())
        if v5_results.get('finished'):
            for frames,key in [(25,'long'),(38,'long3')]:
                recipe=v5_results['selected_recipe_by_context'][str(frames*80)]
                row=max((r for r in v5_results['runs'] if r['frames']==frames and r['recipe']==recipe),key=lambda r:objective(r['validation']))
                checkpoint=torch.load(v5/'models'/row['checkpoint'],map_location='cpu',weights_only=False)
                name=f'v5-l15-{frames*80}ms-{recipe}-seed-{row["seed"]}.pt'
                torch.save({'model':checkpoint['model'],'architecture':'query_gru','width':128,'mix_layers':False,
                    'context_frames':frames,'layer_ids_one_based':[15],'embedding_dim':192,'frame_ms':80,
                    'seed':row['seed'],'recipe':recipe,'validation':row['validation'],'format_version':5},model_dir/name)
                long_models[key]={'file':'models/'+name,'frames':frames,'mix_layers':False,
                    'label':f'V5 optimized · {frames*80} ms · {recipe}'}
    atomic(OUT/'models/index.json',{'models':published,'default_320ms_seed':best['seed'],'long_models':long_models,
        'long_context':'Completed validation-selected V5 references; V3 baseline retained as fallback' if long_models else 'V3 two-second baseline'})
    reports=OUT/'reports';reports.mkdir(exist_ok=True)
    # Public-synthetic provenance + exact procedural annotation recipes, not
    # the 17-GB feature cache or audio. Every source remains hash-addressable.
    data=OUT/'data';data.mkdir(exist_ok=True)
    manifest=json.loads((OLD/'sources_manifest.json').read_text())
    if len(manifest)!=500 or len({r['voice'] for r in manifest})!=500:raise ValueError('Expected audited 500 public synthetic identities')
    for row in manifest:
        for clip in row['clips']:
            if not clip['uid'].startswith(row['voice']+'__'):raise ValueError('Unexpected non-public source UID')
            clip['path']='sources/'+row['voice']+'/'+Path(clip['path']).name
    for name,payload in [('sources_manifest',manifest),*((name,json.loads((OLD/(name+'.json')).read_text()))
            for name in ('global_split','plan','splits','array_schema'))]:
        raw=clean(json.dumps(payload,separators=(',',':'),allow_nan=False)).encode()
        if SECRET.search(raw):raise ValueError('Potential secret in public metadata')
        (data/(name+'.json.gz')).write_bytes(gzip.compress(raw,compresslevel=6,mtime=0))
    recipes=sorted((OLD/'conversations').glob('*.json.gz'))
    if len(recipes)!=500:raise ValueError('Incomplete public synthetic conversation recipe export')
    clocks=windows=0
    with zipfile.ZipFile(data/'conversation-recipes.zip','w',zipfile.ZIP_STORED) as archive:
        for path in recipes:
            raw=gzip.decompress(path.read_bytes());payload=json.loads(raw)
            clocks+=len(payload);windows+=sum(len(row['selected_320ms_steps']) for row in payload)
            raw=clean(json.dumps(payload,separators=(',',':'),allow_nan=False)).encode()
            if SECRET.search(raw):raise ValueError('Potential secret in generated recipes')
            archive.writestr(path.name,gzip.compress(raw,compresslevel=6,mtime=0))
    if (clocks,windows)!=(100000,500000):raise ValueError('Unexpected public conversation/window counts')
    atomic(data/'COUNTS.json',{'synthetic_identities':500,'source_clips':sum(len(row['clips']) for row in manifest),
        'conversation_clocks':clocks,'selected_decision_windows':windows,'private_microphone_recordings':0,
        'source_revision':'88ea857d6a6f20839674dfde30936d8fd6b23c62'})
    (data/'README.md').write_text('''# Public synthetic conversation recipes and provenance

These metadata/annotations derive from LAION annotated synthetic voice profiles,
revision `88ea857d6a6f20839674dfde30936d8fd6b23c62`:
https://huggingface.co/datasets/laion/laion-voice-profiles-annotated
The source declares **CC BY 4.0**; attribution and source hashes/UIDs are retained.
This directory's derived metadata/recipes are CC BY 4.0, not Apache-licensed
model code and not a relicensing of any third-party model weights.

`sources_manifest.json.gz` records selected public clip IDs, roles, word times,
language, qualities, exact HTTP archive ranges and audio SHA-256. Paths are
portable relative hints, not a ready-populated cache. `global_split`, `plan`
and `splits` preserve 480/10/10 identities and selected train/validation/test
window indices. `array_schema` describes the separately regenerable features.
`conversation-recipes.zip` holds 500 gzipped per-anchor annotation files: the
100,000 procedural conversation clocks, source events and 500,000 selected
320-ms decision steps. These are recipes/annotations, not 100,000 rendered WAVs.

Re-run `code/speaker_v3/overnight.py` with the documented pinned dataset and
encoder, compare IDs/hashes here, and then run V4/V5 reproduction. Do not copy
relative hint paths unchanged into an absolute-path local feature cache.
Processed feature tensors and raw source audio are not bundled; no full dataset
download is needed. There are no private recordings, enrolled users, teacher or
encoder weights. Public-synthetic benchmark limitations apply to all recipes.
''')
    (reports/'v4-results.json').write_text(clean(json.dumps(results,indent=2,allow_nan=False)))
    (reports/'v4-report.html').write_text(clean((WORK/'report.html').read_text()))
    (reports/'v3-results.json').write_text(clean((OLD/'models/results.json').read_text()))
    for name in ('protocol.json','progress.json'):
        p=(ROOT/'speaker_v5'/name) if name=='progress.json' else v5/name
        if p.exists():(reports/('v5-'+name)).write_text(clean(p.read_text()))
    if v5_results and v5_results.get('finished'):
        for name in ('results.json','report.html'):(reports/('v5-'+name)).write_text(clean((v5/name).read_text()))
    # Explicit audited test reports only, never arbitrary logs or microphone data.
    for name in ('desk-robot-checks.json','browser-capture-checks.json','browser-capture-desk.json','browser-capture-original.json','browser-lifecycle-checks.json','browser-checks.json','unit-checks.json','integration-results.json','sync-model-final.json','final-verification.json','strict-first-pass-attempt.json','host-gpu-incident.json','http-checks.json'):
        path=ROOT/'logs'/name
        if path.exists():(reports/name).write_text(clean(path.read_text()))
    live=Path('/mnt/nvme/robo-speaker-studies/speaker-live-results')
    for name in ('v4-long-live.json','multi-user-live.json','three-user-live.json','three-user-unshifted.json','fixture-clean-qa.json','streamed-echo-live.json'):
        path=live/name
        if path.exists():
            payload=json.loads(path.read_text())
            if not (payload.get('public_synthetic_only') or payload.get('synthetic_fixtures_only')):raise ValueError('Only public-synthetic live checks may be released')
            if name in ('multi-user-live.json','three-user-live.json','three-user-unshifted.json') and (payload.get('protocol_version',0)<3 or not payload.get('passed')):continue
            payload.pop('synthetic_audio_debug',None)
            (reports/name).write_text(clean(json.dumps(payload,indent=2)))
    card='''---
license: apache-2.0
tags:
- speaker-verification
- audio
- nemotron
- tyger
- experimental
datasets:
- laion/laion-voice-profiles-annotated
---
# TYGER speaker-embedding predictor — reproducible research snapshot

Predict normalized ECAPA-like 192-dimensional speaker cues from frozen Nemotron
ASR 3.5 hidden states. **This is not an ASR/TTS model or secure authentication.**
No private microphones, enrolled-user voiceprints, encoder, teacher or TTS
weights are included. Code is Apache 2.0 except separately MIT robot assets;
third-party model/data licenses remain separate (see code/NOTICE).

## Latest completed result: 320-ms layer mixing

The frozen encoder supplies four new 1024D states every 320 ms. The best V4
recipe learns a mixture of layers 4/8/12/15 (one-based), followed by a small
Query-GRU (648,071 trainable parameters). Three seeds: test macro speaker
identification **71.07%**, EER **14.28%**, versus matched baseline 67.38% and
15.48%. Hard pairs alone do not improve identification; layer+hard reaches
70.67% and 13.96% EER. [Complete report](reports/v4-report.html),
[machine-readable results](reports/v4-results.json).

480 training / 10 validation / 10 test synthetic identities. No identity overlap
between splits. The ten test voices were benchmarked in earlier studies, so this
is a repeated synthetic benchmark, not an independent human validation set.
Layer mixing's paired identity-bootstrap accuracy delta is +3.69 pp with
interval [+1.77,+5.77] pp. Correlated windows are not independent people.

Each hidden state has causal history from an eight-second input window: 320 ms
is the output pooling span, **not a cold-start 320-ms-only audio claim**.
Teacher/prototype thresholds are not calibrated for predicted-to-predicted
real-user enrollment. Larger genuine cosine can coincide with larger impostor
cosine. Background noise, overlaps and real-room microphones remain limitations.

## Weights and inference

`models/index.json` identifies the single checkpoint selected using validation,
not test. All three 320-ms seeds are retained; quoted aggregate metrics average
seeds, not an ensemble. `v3-l15-2000ms-baseline.pt` is a completed two-second
baseline. The completed V5 comparison covers 45 runs (five recipes, three seeds
per duration). Validation selected the utterance-target recipe at all durations.
Matched V5 test macro identification is 67.76% / 70.63% / 71.05% for
320 / 2000 / 3040 ms; EER is 15.60% / 14.25% / 14.18%. The three-second advantage
over two seconds is small (+0.43 pp, identity-bootstrap interval −0.28 to +1.07
pp). V4 and V5 recipes are not identical comparisons. `long_models` in the index
selects the completed V5 two-/three-second checkpoints by validation; both are
included without optimizer state. [V5 report](reports/v5-report.html).

```python
import sys, torch
sys.path.insert(0, "code")
from speaker_v4.train import MixedStudent
saved = torch.load("models/v4-layer-mix-320ms-seed-2023.pt", map_location="cpu", weights_only=False)
model = MixedStudent(mix_layers=True).eval()
model.load_state_dict(saved["model"])
# features: [batch, 4 layers, 4 frames, 1024]; energy_db: [batch, 4]
with torch.inference_mode():
    speaker_embedding, activity_logits = model(features, energy_db)
```

Input features must come from the pinned TYGER/Nemotron frontend and matching
normalization; random features do not identify speakers. Read [code/README.md](code/README.md)
for bootstrap pins, UI/API setup, GPU/CPU dependencies, frozen teacher revision,
causal extraction, data selection, losses, split rules, and reproduction commands.
The source is a working-host snapshot, not a verified clean-machine installer.

## Conversation and CPU-rendered desk robot

The integration includes timestamped browser capture/playback, Nemotron streaming
diarization, V4/V5 speaker cues, full-utterance ECAPA checks, session-local name
memory and explicit validated Luna name metadata (not hidden reasoning). The
microphone panel separately shows capture, server receipt, recognition and
acceptance. An explicit unfiltered mode is available; no hidden bypass is used.
Read [the current source README](code/README.md) for tests and limitations.

First-pass recognition is not guaranteed: similar synthetic male voices can
produce ambiguous teacher scores and require repetition. The final QA receipt
explicitly records whether it used strict first-pass mode or the separately
labelled one-repeat recovery protocol. Recovery repeats only after nine seconds
without acceptance; queued, already accepted TTS is never duplicated. Reports
retain first-attempt counts/reasons and the failed strict stress receipt. A
recovery pass must not be interpreted as perfect first-pass speaker accuracy.

The linked `/desk-robot/` page uses the same conversation and GPU-loaded TTS,
not a second speech model. Its original small Canvas-2D software renderer models
the supplied two-axis desk-robot proposal, with 40 interpolated expressions,
camera-follow pan/tilt, orbit/manual controls and an exploded inspection view.
It needs no WebGL or large mesh download. Original user-supplied design images,
PDFs and dense mesh archives are not republished or relicensed. This is a visual
prototype, not fabrication CAD, a calibrated gaze tracker or a servo controller.

## Data / reproduction / privacy

Public synthetic sources: [LAION annotated voice profiles](https://huggingface.co/datasets/laion/laion-voice-profiles-annotated),
revision `88ea857d6a6f20839674dfde30936d8fd6b23c62`, declared CC BY 4.0.
Source attribution, source ranges/UIDs and per-voice splits are created by
`code/speaker_v3/overnight.py`. The exact selected-source provenance, immutable
splits, word timestamps and 100,000 conversation-clock / 500,000-window recipes
are included as compact [CC BY 4.0 metadata](data/README.md). These are not
rendered conversation WAVs or the large feature cache; both are regenerable.
Full audio archives are not bundled. Frozen
ECAPA teacher: `speechbrain/spkrec-ecapa-voxceleb`, revision
`0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`. TYGER bundle is separately licensed
OpenMDW 1.1; no upstream weights are relicensed here.

`code/speaker_v4/` reproduces specialized-target and layer/hard-pair ablations;
`code/speaker_v5/` contains the preregistered completed duration comparison.
Preserve source manifests and use the documented bounded
downloads/feature caches. No private live recordings may enter a release.
Speaker embeddings are approximate identity cues; do not use them for access
control, surveillance, sensitive inference or as proof of a person's identity.
'''
    (OUT/'README.md').write_text(card)
    shutil.copyfile(ROOT/'LICENSE',OUT/'LICENSE');shutil.copyfile(ROOT/'NOTICE',OUT/'NOTICE')
    entries={}
    for p in OUT.rglob('*'):
        if p.is_symlink():raise ValueError('Release symlink forbidden')
        if not p.is_file() or p.name=='RELEASE_MANIFEST.json':continue
        raw=p.read_bytes()
        if SECRET.search(raw):raise ValueError('Potential secret in release')
        rel=p.relative_to(OUT).as_posix()
        if any(x in {'logs','.runtime','.env','__pycache__','sessions','microphone','enrolled'} for x in Path(rel).parts):raise ValueError('Private/cache path in release')
        entries[rel]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    if sum(v['bytes'] for v in entries.values())>60_000_000:raise ValueError('Unexpected release growth')
    atomic(OUT/'RELEASE_MANIFEST.json',{'files':entries,'private_microphone_recordings':0,'enrolled_user_embeddings':0,'upstream_weights':0})
    return entries

def build():
    # Fresh staging prevents a previously passed but now failed check from
    # surviving as a stale public report. Retain the old audited release as a
    # recoverable backup; no user/research data are deleted.
    stage=Path(tempfile.mkdtemp(prefix='.public-release-stage-',dir=OUT.parent))
    entries=build_at(stage)
    if OUT.exists():OUT.rename(OUT.with_name('public-release-previous-'+str(time.time_ns())))
    stage.rename(OUT)
    return entries

def main(publish=False):
    if publish:
        from final_verify import runtime_signature
        proof=json.loads((ROOT/'logs/final-verification.json').read_text())
        if not proof.get('passed') or not proof.get('source_unchanged') or proof.get('runtime_source_sha256')!=runtime_signature():
            raise RuntimeError('Final verification must pass on the current unchanged runtime before publication')
    entries=build();print(json.dumps({'audited_files':len(entries),'bytes':sum(v['bytes'] for v in entries.values()),'repo':REPO}),flush=True)
    if not publish:return
    from huggingface_hub import HfApi
    token=configured_token();api=HfApi(endpoint='https://huggingface.co',token=token)
    validate_write_identity(api.whoami(),[REPO])
    # No wildcard upload outside this newly built, audited release directory.
    api.create_repo(REPO,repo_type='model',private=False,exist_ok=True)
    result=api.upload_folder(repo_id=REPO,repo_type='model',folder_path=str(OUT),
        commit_message='Publish audited Robby source and completed Nemotron speaker-prediction experiments')
    print(json.dumps({'published':True,'url':'https://huggingface.co/'+REPO,'commit':result.oid}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true')
    try:main(parser.parse_args().publish)
    except Exception as exc:
        print(json.dumps({'ok':False,'error_type':type(exc).__name__,'http_status':getattr(getattr(exc,'response',None),'status_code',None),'message':'Release failed; credentials and remote response details suppressed'}),file=sys.stderr)
        raise SystemExit(1)
