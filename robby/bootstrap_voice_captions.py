"""Fetch pinned caption rules and 389-MB Medium weights, not a second ASR."""
import argparse
import os
from pathlib import Path
import subprocess
from voice_captioning import MODEL_REVISION,CAPTION_REVISION

ROOT=Path(__file__).resolve().parent
def sources():
    root=Path(os.environ.get('ROBO_CAPTION_SOURCE',ROOT/'vendor/procedural-voice-captions'))
    if not root.exists():
        subprocess.run(['git','clone','https://github.com/LAION-AI/procedural-voice-captions.git',str(root)],check=True)
    if subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip():
        raise RuntimeError('Preserve local caption-source edits; cannot change its revision')
    subprocess.run(['git','-C',str(root),'checkout','--detach',CAPTION_REVISION],check=True)
    return root

def weights():
    from huggingface_hub import snapshot_download
    return snapshot_download('laion/humaneness-ears-base-medium',revision=MODEL_REVISION,token=False,
        cache_dir=os.environ.get('ROBO_CAPTION_CACHE'),allow_patterns=['model.py','training_normalization.json',
        'classes.json','medium/config.json','medium/model.safetensors','medium/preprocessor_config.json'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-only',action='store_true');args=parser.parse_args()
    print(sources())
    if not args.source_only:print(weights())
