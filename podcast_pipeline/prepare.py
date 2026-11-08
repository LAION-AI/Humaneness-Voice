"""Download only the published assets used by this recipe; no private tokens needed."""
import argparse, json, shutil
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
from recipe import HERE, MODEL_REPO, MODEL_REVISION, references
from run import setup

def main():
    p=argparse.ArgumentParser();p.add_argument('--references-only',action='store_true');a=p.parse_args()
    release=json.loads((HERE/'references/source.json').read_text())
    for name in ['de_s1.mp3','de_s2.mp3','de_anchors.npz']:
        origin=hf_hub_download(release['repo'],release['directory']+'/'+name,revision=release['revision'])
        shutil.copyfile(origin,HERE/'references'/name)
    references(json.loads((HERE/'references.json').read_text()))
    if a.references_only:return
    setup('cpu');import settings
    model=snapshot_download(MODEL_REPO,revision=MODEL_REVISION,allow_patterns=['*.py','*.json','*.safetensors','*.txt','*.model','LICENSE','NOTICE.md'])
    snapshots={
      'codec':snapshot_download('OpenMOSS-Team/MOSS-Audio-Tokenizer-v2',revision='f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169'),
      'asr':snapshot_download('nvidia/parakeet-tdt-0.6b-v3',revision='541d1f99c6b0c3cd0b11a95167540bb8edefd82b',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.py']),
      'aligner':snapshot_download('Qwen/Qwen3-ForcedAligner-0.6B-hf',revision='c07281df297b9905d24a508279258cccf987a064',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.py']),
    }
    snapshot_download('speechbrain/spkrec-ecapa-voxceleb',revision='0f99f2d0ebe89ac095bcc5903c4dd8f72b367286',cache_dir=str(Path(settings.HF_CACHE)/'hub'),allow_patterns=['hyperparams.yaml','*.ckpt','label_encoder.txt'])
    from ears_reward import EarsModels
    EarsModels('cpu').load('medium')
    (HERE/'assets.local.json').write_text(json.dumps(dict(model=model,**snapshots),indent=2))
    print('Assets prepared. Policy snapshot:',model)
if __name__=='__main__':main()
