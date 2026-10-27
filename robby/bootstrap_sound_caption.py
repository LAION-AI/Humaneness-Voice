"""Download only pinned inference assets, never training data or optimizer state."""
import os
from pathlib import Path
import shutil

REPO = 'laion/sound-effect-captioning-whisper'
REVISION = 'bb36f9c6534037a8d1d78cff97d119dd37da478a'
PROCESSOR_REVISION = '973afd24965f72e36ca33b3055d56a652f456b4d'

def assets():
    from huggingface_hub import snapshot_download
    root = Path(os.environ.get('ROBO_SOUND_CACHE', '/mnt/nvme/robby-sound-caption'))
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 2 * 1024**3:
        raise RuntimeError('Need 2 GiB free before preparing the sound captioner')
    model = snapshot_download(REPO, revision=REVISION, cache_dir=str(root / 'hub'),
        allow_patterns=['config.json', 'generation_config.json', 'model.safetensors'])
    processor = snapshot_download('openai/whisper-small', revision=PROCESSOR_REVISION,
        cache_dir=str(root / 'hub'), allow_patterns=['*.json', 'merges.txt', 'vocab.json'],
        ignore_patterns=['*model*', '*generation*'])
    return model, processor

if __name__ == '__main__':
    for path in assets(): print(path)
