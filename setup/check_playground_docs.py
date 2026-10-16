"""Offline consistency, links, source integrity and publication-boundary checks."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
PLAY=ROOT/'playground'
sys.path.insert(0,str(PLAY/'runtime'))
import settings
from acting_models import MODELS


def main():
    manifest=json.loads((ROOT/'manifests/playground-2026-10-16.json').read_text())
    for key,value in manifest['defaults'].items():
        assert getattr(settings,key)==value, 'Default drift: '+key
    assert manifest['models']==[{k:v for k,v in row.items() if k!='cache_dir'} for row in MODELS.values()]
    prompt=next(ast.literal_eval(n.value) for n in ast.parse((PLAY/'runtime/actor_prompt.py').read_text()).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SYSTEM' for t in n.targets))
    assert hashlib.sha256(prompt.encode()).hexdigest()==manifest['luna_system_sha256']
    assert '```text\n'+prompt+'\n```' in (ROOT/'docs/LUNA_SYSTEM_PROMPT.md').read_text()
    checked=0
    for path in [ROOT/'README.md',PLAY/'README.md',*sorted((ROOT/'docs').glob('PLAYGROUND_*.md')),ROOT/'docs/LUNA_SYSTEM_PROMPT.md']:
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',path.read_text()):
            if target.startswith(('http:','https:','#','mailto:')):continue
            local=target.split('#')[0]
            assert (path.parent/local).exists(),f'Broken local link in {path.name}: {target}'
            checked+=1
    sources=json.loads((PLAY/'source-manifest.json').read_text())['files']
    credential=re.compile(rb'hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}')
    for row in sources:
        path=ROOT/row['path'];raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==row['sha256'], 'Source drift: '+row['path']
        assert not credential.search(raw),'Credential-like content: '+row['path']
        assert path.suffix.lower() not in {'.wav','.mp3','.pt','.npz','.safetensors'},'Unexpected media/weights: '+row['path']
        if path.suffix=='.py':ast.parse(raw,filename=str(path))
    assert len(manifest['models'])==6 and manifest['defaults']['DEFAULT_MODEL']=='small_rl'
    print(f'Passed: {checked} local links, exact Luna prompt/default/model agreement, {len(sources)} file hashes, Python syntax and credential/media boundaries.')


if __name__=='__main__':main()
