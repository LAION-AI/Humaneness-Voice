"""Build public documentation/metadata from the checked-in October snapshot.

No credentials, weights, user uploads or source audio are exported. Hub calls
read public metadata only; a moving main revision is never a runtime pin.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import sys
from urllib.parse import quote

import httpx

ROOT=Path(__file__).resolve().parents[1]
PLAY=ROOT/'playground'
RUNTIME=PLAY/'runtime'
sys.path.insert(0,str(RUNTIME))
import settings
from acting_models import MODELS


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def build():
    stamp=datetime.now(timezone.utc).isoformat()
    legacy=json.loads((ROOT/'manifests/agent-default-v1.json').read_text())
    catalog=json.loads((RUNTIME/'reward_catalog.json').read_text())
    models=[{k:v for k,v in item.items() if k!='cache_dir'} for item in MODELS.values()]
    extra=[settings.REPO,settings.LORA_REPO,settings.DPO_REPO,settings.REF15_REPO,
        'laion/Humaneness-Voice-Small-Acting','laion/Humaneness-Voice-Large-Acting',
        'laion/Humaneness-Voice-Small-1.1-Alpha','laion/Humaneness-Voice-Large-1.1-Alpha',
        settings.VC_FULL_REPO,settings.VC_GROW_REPO,settings.VC_GROW_PARENT_REPO,
        'laion/humaneness-ears-base-medium',catalog['voicenet_repo'],catalog['eiv_repo'],
        catalog['encoder_repo'],'laion/vocalburst-locator','laion/vocal-burst-detector-commercial',
        settings.SPEAKER_REPO,'laion/moss-voice-profile-loras-500']
    repos=sorted(set(extra+[m['repo'] for m in models]+[v['repo'] for v in legacy['assets'].values() if v['type']=='model']))
    def inspect(repo):
        with httpx.Client(timeout=60,follow_redirects=True) as client:
            r=client.get('https://huggingface.co/api/models/'+repo,params={'blobs':'true'})
            r.raise_for_status();data=r.json();revision=data['sha']
        files=[]
        for f in data.get('siblings',[]):
            name=f['rfilename']
            if not name.endswith(('.safetensors','.pt','.pth','.bin','.tar','.tar.gz','.gguf')):continue
            lfs=f.get('lfs') or {}
            files.append({'path':name,'bytes':f.get('size',lfs.get('size')),'sha256':lfs.get('sha256'),
                'url':f'https://huggingface.co/{repo}/resolve/{revision}/{quote(name,safe="/")}'})
        card=data.get('cardData') or {}
        return {'repo':repo,'inspected_revision':revision,'card_license':card.get('license'),
            'note':'Public inventory at inspection time; use a runtime manifest pin for generation.',
            'files':files}
    with ThreadPoolExecutor(max_workers=4) as pool:inventory=list(pool.map(inspect,repos))
    write_json(ROOT/'manifests/checkpoint-inventory-2026-10-16.json',
        {'observed_utc':stamp,'repositories':inventory,'scope':'Current UI models, earlier Small stages, relevant adapters, perception and historical SFT3 sources; not every LAION model.'})
    prompt=next(ast.literal_eval(n.value) for n in ast.parse((RUNTIME/'actor_prompt.py').read_text()).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SYSTEM' for t in n.targets))
    digest=hashlib.sha256(prompt.encode()).hexdigest()
    (ROOT/'docs/LUNA_SYSTEM_PROMPT.md').write_text(
        '# Exact current Luna director system prompt\n\n'
        'Exported from [`playground/runtime/actor_prompt.py`](../playground/runtime/actor_prompt.py), '
        'not reconstructed from historical notes.\n\n'
        f'SHA-256 of the UTF-8 SYSTEM string: `{digest}`.\n\n'
        'The model names inside this prompt retain their historical Acting wording. '
        'The selected TTS model is chosen by the server/UI, not the director. '
        'This does not add voice/LoRA tool calls.\n\n'
        'The complete per-request message construction, history, reference flag, exact-text mode, '
        'JSON validation, retry and default-delivery enforcement are in '
        '[`director.py`](../playground/runtime/director.py). Its separate `reward_plan()` '
        'system instructions append the machine-readable dimension catalog from '
        '[`reward_catalog.json`](../playground/runtime/reward_catalog.json).\n\n'
        '```text\n'+prompt+'\n```\n')
    names=['torch','torchaudio','torchvision','transformers','huggingface-hub','safetensors',
        'fastapi','uvicorn','pydantic','soundfile','numpy','scipy','httpx','python-multipart',
        'speechbrain','librosa','tokenizers','accelerate','einops','omegaconf','tqdm','diffusers']
    versions={}
    for name in names:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    write_json(PLAY/'environment-2026-10-16.json',{'observed_utc':stamp,'python':sys.version,
        'packages':versions,'meaning':'Observed working-host environment, not a newly validated lockfile; optional editable VC packages need their own source revision.'})
    defaults={k:getattr(settings,k) for k in ['DEFAULT_MODEL','DEFAULT_VOICE','DEFAULT_CHUNK_FRAMES',
        'DEFAULT_GENERATION','DEFAULT_N','DEFAULT_STREAMING','DEFAULT_SPEAKER_STEERING',
        'DEFAULT_DYNAMIC_REFERENCE','DEFAULT_TEMPERATURE','DEFAULT_GUIDANCE','DEFAULT_FORMAT',
        'DEFAULT_VOICE_CONVERSION','REFERENCE_SECONDS','MAX_FRAMES']}
    write_json(ROOT/'manifests/playground-2026-10-16.json',{'system':'Humaneness Voice Playground',
        'snapshot_date':'2026-10-16','models':models,'defaults':defaults,'luna_system_sha256':digest,
        'legacy_manifest':'agent-default-v1.json','checkpoint_inventory':'checkpoint-inventory-2026-10-16.json',
        'rl_precision':{'base':'BF16-rounded then FP32','adapter':'FP32 residual, not merged','tf32':False,
            'single_and_best_n_gpu_verified':True,'chunk_search_rl_precision_verified':False},
        'privacy':'No source reference recordings, uploads, credentials or model weights in this snapshot.',
        'portability':'Working-host integration source with disclosed absolute asset paths; not a clean-machine installer. See docs/PLAYGROUND_INSTALL.md.'})
    entries=[]
    credential=re.compile(rb'hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}')
    for path in sorted(PLAY.rglob('*')):
        if not path.is_file() or any(p in path.parts for p in ('__pycache__','.pytest_cache')) or path.name=='source-manifest.json':continue
        relative=path.relative_to(PLAY)
        if relative.parts[:2] in {('runtime','runtime'),('runtime','model-cache'),('runtime','logs')}:
            continue
        if path.suffix not in {'.py','.js','.cjs','.html','.json','.md','.txt','.sh','.example'} and path.name!='.gitignore':
            continue
        if relative.parts[:3]==('runtime','fixed-references','audio') or path.name.startswith('.env'):
            continue
        raw=path.read_bytes()
        if credential.search(raw):raise ValueError('Credential-like value in publication: '+str(path.relative_to(ROOT)))
        entries.append({'path':path.relative_to(ROOT).as_posix(),'bytes':len(raw),
            'sha256':hashlib.sha256(raw).hexdigest()})
    write_json(PLAY/'source-manifest.json',{'snapshot_date':'2026-10-16','files':entries,
        'note':'Hashes describe the published snapshot, not untracked runtime state or missing audio. No model or upload directories are included.'})
    print(f'Built {len(models)} runtime model pins, {len(inventory)} public inventories, exact Luna prompt, {len(entries)} source/evidence hashes.')


if __name__=='__main__':build()
