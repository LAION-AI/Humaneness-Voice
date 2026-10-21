"""Fetch only pinned TYGER EARS sources and public inference files, not vLLM/TTS."""
import argparse
from pathlib import Path
import shutil
import subprocess
import paths


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-only',action='store_true');args=parser.parse_args()
    if not (paths.TYGER/'.git').exists():
        if paths.TYGER.exists() and any(paths.TYGER.iterdir()):raise SystemExit('Refuse to overwrite a nonempty TYGER source directory')
        paths.TYGER.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','https://github.com/nineninesix-ai/tyger-s2s.git',str(paths.TYGER)],check=True)
    status=subprocess.check_output(['git','-C',str(paths.TYGER),'status','--porcelain'],text=True)
    if status.strip():raise SystemExit('TYGER has local changes; preserve them and resolve the source pin manually')
    subprocess.run(['git','-C',str(paths.TYGER),'checkout','--detach',paths.TYGER_REVISION],check=True)
    if not args.source_only:
        paths.CACHE.mkdir(parents=True,exist_ok=True)
        if not (paths.BUNDLE/'model.safetensors').exists() and shutil.disk_usage(paths.CACHE).free<6*1024**3:
            raise SystemExit('Need at least 6 GiB free before downloading the 2.6 GB inference bundle')
        from huggingface_hub import snapshot_download
        result=snapshot_download('nineninesix/tyger',revision=paths.BUNDLE_REVISION,cache_dir=paths.CACHE,token=False,
            allow_patterns=['model.safetensors','config.json','manifest.json','nemotron/*','LICENSE*','NOTICE*','licenses/*'])
        print('Pinned EARS bundle:',result)
    print('Pinned TYGER EARS source:',paths.TYGER_REVISION)


if __name__=='__main__':main()
