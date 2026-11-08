"""Validated custom scripts and fixed reference routing; never rewrite literal words."""
import copy,json
from pathlib import Path
from recipe import HERE,sha
from languages import LANGUAGES

def load_inputs(a):
    refpath=Path(a.references).resolve()
    refs=json.loads(refpath.read_text())
    custom=refpath!=HERE/'references.json'
    for speaker,ref in refs.items():
        if custom:
            ref['path']=str((refpath.parent/ref['path']).resolve())
            if ref.get('codes'):ref['codes']=str((refpath.parent/ref['codes']).resolve())
        ref.setdefault('sha256',sha(HERE/ref['path']))
    names={'legacy':'script','sentence':'script-speaker','spontaneous':'script-spontaneous'}
    scripts=[Path(x).resolve() for x in a.script] if a.script else [HERE/f"{names[a.prompt_style]}-{l}.json" for l in ('en','de')]
    jobs=[];seen=set()
    for path in scripts:
        script=json.loads(path.read_text());lang=script['language']
        if lang not in LANGUAGES:raise ValueError('Supported languages: en, de, nl, fr, es')
        for i,source in enumerate(script['turns'][:a.limit or None]):
            t=copy.deepcopy(source);t.setdefault('index',i)
            if (lang,t['index']) in seen:raise ValueError('Duplicate language/turn index')
            seen.add((lang,t['index']))
            if not t.get('text','').strip() or t['speaker'] not in refs:raise ValueError('Missing text or speaker reference')
            t.setdefault('direction','warm and sincere')
            if a.prompt_style=='spontaneous' and not t.get('delivery_chunks'):
                t['delivery_chunks']=[dict(text=t['text'],cue=t['direction']+'; relaxed, spontaneous, gently spoken')]
            if t.get('delivery_chunks'):
                if not 1<=len(t['delivery_chunks'])<=2 or ' '.join(c['text'] for c in t['delivery_chunks'])!=t['text']:
                    raise ValueError('Use at most two delivery chunks, preserving the exact literal text')
            jobs.append((lang,t))
    if not jobs:raise ValueError('Empty script')
    return refs,scripts,jobs
