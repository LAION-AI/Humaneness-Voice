"""Build a bounded, offline, audited HF-ready pilot release; never publishes.

Only explicit source allowlists are used. No microphone/session directory,
Robo reference voiceprint, raw audio, third-party model weights or caches enter
the release. Existing release directories are verified, never overwritten.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

import numpy as np

try:
    from .publish_pilot import audit_release
except ImportError:
    from publish_pilot import audit_release

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUNDLE = Path('/home/c4r33u19/.cache/robo-tyger/hub/models--nineninesix--tyger/snapshots/a5d4d73bb74fa4a7082dcd4387e44a1e5ac151b6')
CODE = ['__init__.py','models.py','inference.py','train.py','evaluate.py','select.py',
        'compile_report.py','prepare_pilot.py','prepare_emotion_pilot.py','audit_pilot.py',
        'test_dataset_pipeline.py','test_predictor.py']
REPORTS = ['report.html','context_ablation_test.json','ood_centered_v2.json','ood_v1_v2.json']
RUNS = ['results','results_centered','results_v2_raw','results_v2_centered','results_v2_gru']
MODEL_KEYS = {'model','config','seed','epoch','validation','activity_threshold','format_version',
              'feature_source','teacher','warning','verification_threshold','test',
              'selection_validation','selection_protocol'}
NPZ_KEYS = {'features','frame_end_s','speaker_idx','valid_mask','speech_fraction','ambiguity'}


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def dump(path, value):
    write(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def clean(value):
    """Retain public source provenance but strip machine-local cache locations."""
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in {'source_cache'}:
                continue
            if key == 'path' and isinstance(item, str) and item.startswith('/'):
                out['source_audio_not_included'] = True
                continue
            out[key] = clean(item)
        return out
    if isinstance(value, str):
        value = value.replace(str(ROOT)+'/', '').replace(str(HERE)+'/', '')
        value = re.sub(r'/home/[^/\s"<>]+/[^\s"<>]*', '[local-cache]', value)
        value = re.sub(r'/mnt/nvme/[^\s"<>]*', '[local-workspace]', value)
    return value


def copy_text(source, destination):
    if source.suffix == '.json':
        dump(destination, clean(json.loads(source.read_text())))
    else:
        write(destination, clean(source.read_text()))


def add_licenses(destination, model=False):
    # Existing grants are copied, not invented. Code and weights/data are distinct.
    cc = ROOT.parent/'repository-update'/'Humaneness-Voice'/'LICENSE'
    shutil.copy2(cc, destination/'LICENSE-CC-BY-4.0.txt')
    shutil.copy2(ROOT/'LICENSE', destination/'LICENSE-APACHE-2.0.txt')
    third = destination/'third_party'
    third.mkdir()
    for name in ('LICENSE.md','NOTICE.md'):
        shutil.copy2(BUNDLE/name, third/('TYGER-'+name))
    shutil.copy2(BUNDLE/'licenses'/'OpenMDW-1.1.txt', third/'OpenMDW-1.1.txt')
    write(destination/'NOTICE', '''TYGER V2 public-synthetic research pilot, 2026.

Original integration and predictor code: Apache License 2.0.
Selected public-synthetic derived features, provenance and our original trained
student weights: Creative Commons Attribution 4.0 International. Credit LAION,
Humaneness Labs, and Christoph Schuhmann; these are new derived research artifacts.

Source speech: Christoph Schuhmann and LAION, laion/laion-voice-profiles-annotated,
revision 88ea857d6a6f20839674dfde30936d8fd6b23c62, CC BY 4.0. Audio is not included.
Frozen feature extractor: nineninesix/tyger, revision
a5d4d73bb74fa4a7082dcd4387e44a1e5ac151b6; Nemotron source revision
ea30d66debe3740a08b573244286791d423d6b3e; OpenMDW 1.1.
The original TYGER/NVIDIA license and origin notices are retained under third_party.
OpenMDW 1.1 explicitly leaves sharing model outputs unrestricted. No TYGER or
Nemotron parameters/configuration are relicensed under CC BY 4.0 or included here.
Teacher: speechbrain/spkrec-ecapa-voxceleb, revision
0f99f2d0ebe89ac095bcc5903c4dd8f72b367286, separate upstream Apache 2.0 model/code
terms. No teacher or classifier weights are included. Public synthetic teacher
outputs are included for reproducibility, not private enrolled-user voiceprints.

No original TYGER GRU, TTS weights, microphone recordings, camera frames,
uploaded reference audio, session identity memory, or enrolled-user embeddings
are included. Student cues are not secure authentication. Pilot reliability
checks failed; default identity filtering remains OFF.
''')


def add_code(destination):
    code = destination/'code'/'speaker_v2'
    code.mkdir(parents=True)
    for name in CODE:
        shutil.copy2(HERE/name, code/name)
    write(destination/'code'/'LICENSE', (ROOT/'LICENSE').read_text())
    versions = {name: importlib.metadata.version(name) for name in
                ['numpy','torch','torchaudio','safetensors','huggingface_hub','requests','scipy','pyarrow','speechbrain']}
    dump(destination/'environment.json', versions)
    write(destination/'requirements.txt',
          '# Exact observed CUDA build suffixes are recorded in environment.json.\n'
          '# Select an appropriate official PyTorch CPU/CUDA package index locally.\n'
          +'\n'.join(f'{k}=={v.split("+")[0]}' for k,v in versions.items())+'\n')


def add_reports(destination):
    reports = destination/'reports'
    reports.mkdir()
    for name in REPORTS:
        copy_text(HERE/name, reports/name)
    for folder in RUNS:
        for name in ['results.json','report.html']:
            copy_text(HERE/folder/name, reports/folder/name)


def dataset_release(destination):
    destination.mkdir()
    add_code(destination)
    add_licenses(destination)
    copy_text(HERE/'DATASET_CARD.md', destination/'METHODS.md')
    counts = {}
    for source_name, release_name in [('pilot','v1'),('pilot_v2','v2')]:
        source, target = HERE/source_name, destination/'data'/release_name
        target.mkdir(parents=True)
        original = json.loads((source/'manifest.json').read_text())
        assert len(original['conversations']) == 64 and len(original['speakers']) == 32
        for record in original['conversations']:
            relative = Path(record['path'])
            assert not relative.is_absolute() and '..' not in relative.parts
            src, dest = source/relative, target/relative
            with np.load(src, allow_pickle=False) as f:
                assert set(f.files) == NPZ_KEYS
                assert f['features'].shape == (300,1024) and np.isfinite(f['features']).all()
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copy_text(src.with_suffix('.json'), dest.with_suffix('.json'))
        for name in ['manifest.json','speakers.json','audit.json','teacher_quality.json']:
            copy_text(source/name, target/name)
        for name in ['teacher_embeddings.npy','probe_teacher_embeddings.npz']:
            shutil.copy2(source/name, target/name)
        counts[release_name] = {'conversations':64, 'decision_hops':4800,
            'audio_seconds':1536, 'identities':32, 'feature_dimensions':1024}
    copy_text(HERE/'pilot'/'global_split.json', destination/'global_split.json')
    copy_text(HERE/'pilot_v2'/'retrieval.json', destination/'v2_retrieval.json')
    dump(destination/'summary.json', counts)
    write(destination/'README.md', '''---
license: cc-by-4.0
language: [de, en]
tags: [speaker-verification, synthetic, tyger, nemotron, experimental, features]
pretty_name: TYGER V2 public-synthetic speaker predictor pilot
---
# TYGER V2: public-synthetic frozen-feature pilot

**Research artifact; failed reliability gate. Not an authentication dataset.**
Two 32-identity corpora each contain 64 procedural 24-second conversations and
4,800 decision hops of 320 ms. Each has 24 train, 4 validation and 4 test identities,
with no speaker overlap. The global 480/10/10 split is retained for future work.
Only public synthetic source-derived features/provenance are released: no raw
audio, microphone recordings, private uploads or enrolled-user voiceprints.

## Files and use

`data/v1/` and `data/v2/` contain the training-compatible manifests, compact
conversation NPZ arrays, normalized public-synthetic ECAPA teacher targets and
full-utterance teacher probes. Every NPZ has Nemotron layer 15 features `[T,1024]`
at 80 ms, frame times, identity/activity labels, confidence masks and overlap flags.
See `METHODS.md`, the audit JSONs, source hashes and timestamped event provenance.
This custom NPZ feature collection is not an automatically playable audio viewer.

Before upload, the Hugging Face URLs below are **intended publication targets**,
not evidence that repositories already exist. Use local paths from the unpacked
offline ZIP. Only after actual publication should `snapshot_download` be used.

```python
from pathlib import Path
import numpy as np
root = Path('./dataset')  # run from the unpacked ZIP root
# After publication only:
# from huggingface_hub import snapshot_download
# root = Path(snapshot_download('laion/tyger-v2-speaker-pilot', repo_type='dataset'))
example = np.load(root/'data/v2/conversations/test-028-0.npz', allow_pickle=False)
```

Run training with the included Apache 2.0 code from the unpacked ZIP root:
`PYTHONPATH=./dataset/code python -m speaker_v2.train --data ./dataset/data/v1 --out ./results --epochs 30 --device cpu`.
For the broader-emotion corpus use `--data ./dataset/data/v2`. Model/library versions are
recorded in `environment.json`; rebuild source data using the bounded-download
scripts and an independently installed frozen TYGER service, not bundled weights.

## Controls and limitations

Two complete long enrollment utterances per identity are disjoint from all
student source utterances. V2 student recordings are also disjoint from V1 while
the teacher targets are bitwise identical. Procedural scenes include 1–5
speakers, non-hop-aligned boundaries, pauses, overlaps, noise, reverb, telephone
filtering and known-delay synthetic echo. No third-party music is used.

V1's 192 source clips are balanced at 98 DE / 94 EN but cover only fatigue/background
conditions. V2's 383 source clips include 338 DE / 45 EN and explicit source conditions
anger, amusement, elation and sadness; this is not balanced English emotion validation.
The extra BKGN/very_high condition is not a clean neutral control. Whispering is
not covered. Synthetic identity drift and machine alignments are imperfect labels.
The packaged speaker student still has 25.3% held-out EER; large-scale training
was deliberately not started after 15 exploratory runs failed the quality gate.

## Attribution and distinct licenses

Derived dataset artifacts: CC BY 4.0, credit **Christoph Schuhmann, LAION and
Humaneness Labs**. Source: [LAION annotated voice profiles](https://huggingface.co/datasets/laion/laion-voice-profiles-annotated),
revision 88ea857d6a6f20839674dfde30936d8fd6b23c62, CC BY 4.0. Integration/training
code: Apache 2.0. Feature extractor: nineninesix/tyger and NVIDIA Nemotron,
OpenMDW 1.1; its outputs are unrestricted by that agreement. Teacher: SpeechBrain
ECAPA, separate upstream Apache 2.0 terms. **No third-party model weights are
included or relicensed**. Read `NOTICE`, both LICENSE files and `third_party/`.
''')


def model_release(destination):
    import torch
    from safetensors.torch import save_file, load_file
    destination.mkdir()
    add_code(destination)
    add_licenses(destination, model=True)
    add_reports(destination)
    copy_text(HERE/'README.md', destination/'METHODS.md')
    source = HERE/'artifacts'/'best.pt'
    state = torch.load(source, map_location='cpu', weights_only=False)
    assert set(state) <= MODEL_KEYS, 'Unexpected checkpoint fields; manual privacy review required'
    assert all(isinstance(v,torch.Tensor) and torch.isfinite(v).all() for v in state['model'].values())
    assert sum(v.numel() for v in state['model'].values()) < 1_000_000
    save_file({k:v.detach().cpu().contiguous() for k,v in state['model'].items()},
              str(destination/'model.safetensors'))
    assert set(load_file(str(destination/'model.safetensors'))) == set(state['model'])
    # Original trusted research checkpoint is retained for the existing local
    # loader; safetensors + explicit config is preferred for downloaded weights.
    torch.save(state, destination/'best.pt')
    selection = json.loads((HERE/'artifacts'/'selection.json').read_text())
    dump(destination/'selection.json', clean(selection))
    dump(destination/'config.json', {**state['config'], 'input_dim':1024, 'output_dim':192,
        'frame_ms':80, 'hop_ms':320, 'activity_threshold':state['activity_threshold'],
        'verification_threshold':state['verification_threshold'], 'recommended_enabled':False,
        'pilot_pass':False, 'format':'safetensors state_dict; original custom PyTorch architecture'})
    write(destination/'README.md', '''---
license: cc-by-4.0
tags: [speaker-verification, tyger, nemotron, experimental, pytorch, failed-quality-gate]
language: [de, en]
---
# TYGER V2 experimental speaker-embedding student

**Do not enable identity filtering by default. This pilot failed reliability
checks and is not authentication.** The original trained student has 470,147
parameters: centered MLP, 640-ms feature context, 1024D frozen Nemotron layer 15
inputs and 192D normalized ECAPA-like outputs, with an auxiliary speech/silence/
overlap head. It runs alongside—not instead of—the original TYGER turn-taking GRU.

## Measured results and nonclaims

Chosen on the broader corpus's **validation** identities: EER 18.0%. Unseen test
identities: EER 25.3%; the frozen conservative threshold rejects about 90% of genuine
short probes. Warmed CPU inference is about 0.32 ms/update. Fifteen actual training
runs were completed; no 100,000-conversation expansion was started. The long-clean
ECAPA teacher's 2.5% EER is not an equal-context comparison. V2 is German-heavy;
results do not establish real microphones, broad English emotion robustness,
reliable overlap detection or secure multi-user filtering.

`reports/report.html` contains the combined English report, and linked per-run
reports/JSONs contain every comparison and its limitations. `selection.json`
retains the validation-only choice and original thresholds, with `pilot_pass:false`.

## Load without pickle

Before upload, Hugging Face links and repository IDs in this card are **intended
publication targets**, not a claim that they are online. The example works with
the offline ZIP; enable its commented download lines only after publication.

```python
import json, sys, torch
from pathlib import Path
from safetensors.torch import load_file
sys.dont_write_bytecode = True  # keep the audited offline release unchanged
root = Path('./model')  # run from the unpacked ZIP root
# After publication only:
# from huggingface_hub import snapshot_download
# root = Path(snapshot_download('laion/tyger-v2-speaker-predictor-pilot'))
sys.path.insert(0, str(root/'code'))  # review this small custom architecture first
from speaker_v2.models import SpeakerPredictor
c = json.loads((root/'config.json').read_text())
m = SpeakerPredictor(c['architecture'], c['width'], centered=c['centered_target'])
m.load_state_dict(load_file(str(root/'model.safetensors')))
m.eval()
with torch.inference_mode():
    embedding, activity = m(torch.zeros(1,c['context'],1024))  # shape example only
```

Supply only frozen Nemotron ASR 3.5 **layer 15** states, four newly produced frames
per 320 ms; do not repeat the entire rolling ASR window. Install TYGER/Nemotron and
its separate OpenMDW 1.1 weights independently. The included original `best.pt`
supports the existing `speaker_v2.inference.Predictor`; it uses a custom PyTorch
checkpoint and should only be loaded after verifying provenance/checksums.
Safetensors avoids pickle for the weight-loading path above.

## Reproduction and licensing

The paired offline features are in `./dataset/data/v1` and `./dataset/data/v2`.
Their [intended Hugging Face dataset target](https://huggingface.co/datasets/laion/tyger-v2-speaker-pilot)
is usable only after publication. Together with `code/speaker_v2/`, these reproduce training and evaluation. Library versions
are in `environment.json`. See `METHODS.md` and dataset source provenance.

Our original student weights: CC BY 4.0, LAION/Humaneness Labs. Original code:
Apache 2.0. Source speech: Christoph Schuhmann and LAION, CC BY 4.0. Frozen TYGER/
NVIDIA Nemotron: separate OpenMDW 1.1; frozen SpeechBrain ECAPA: separate upstream
Apache 2.0 terms. These teachers/encoders produced training targets/features but
**none of their weights are distributed or relicensed here**. Their original
license and origin notices are retained separately. No microphone recordings,
private references, enrolled-user embeddings, or Robo-reference voiceprints are
included. Speaker cues require consent and are never secure authentication.
''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE/'release')
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists():
        print(json.dumps(audit_release(output), indent=2))
        return
    if shutil.disk_usage(output.parent).free < 500_000_000:
        raise RuntimeError('Need500MBfree for bounded offline export')
    output.mkdir()
    dataset_release(output/'dataset')
    model_release(output/'model')
    for name in ('export_pilot.py','publish_pilot.py'):
        shutil.copy2(HERE/name, output/name)
    write(output/'PUBLISHING.md', '''# Offline audited release

This directory is prepared locally. **Nothing has been uploaded to Hugging Face.**
The configured credential previously failed authentication; this build never
reads credentials or probes the network.

`dataset/` and `model/` are separate HF-ready repositories. Data/student weights
are CC BY 4.0; code is Apache 2.0; upstream model notices remain separate. No raw
audio, third-party model weights, private microphone recordings, uploaded
references, Robo voiceprint, or session enrollment memory are included.

Local audit only, no network or token access:
`python publish_pilot.py --release .`

After a properly scoped configured HF credential and explicit publication
authorization are available:
`python publish_pilot.py --release . --publish`

Publication validates `whoami` LAION write/admin membership and token repository
write scope before creating or uploading either repository. It fails closed if
metadata is insufficient. Existing targets need explicit `--allow-existing`;
remote files are never deleted. Credentials come only from normal HF config or
the Git credential helper, remain in memory, and are never printed or stored.

The ZIP contains only the manifest-checked allowlist. A server may expose this
specific ZIP path; never serve the workspace, source cache, logs or runtime.
The research student failed its quality gate; identity filtering remains OFF.
''')
    files = {}
    for p in output.rglob('*'):
        if p.is_file():
            files[p.relative_to(output).as_posix()] = {'bytes':p.stat().st_size,
                'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    dump(output/'MANIFEST.json', {'schema_version':1, 'public_synthetic_only':True,
        'private_recordings':0, 'third_party_weights':False, 'files':files})
    audit_release(output)
    with zipfile.ZipFile(output/'tyger-v2-public-pilot.zip', 'w',
            compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name in sorted([*files, 'MANIFEST.json']):
            archive.write(output/name, arcname=name)
    result = audit_release(output)
    result['network_used'] = False
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
