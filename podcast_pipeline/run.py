"""Runtime configuration shared by generation and export; no service secrets."""
import os,sys
from pathlib import Path
DEFAULTS={'HVS_CHUNK_RAMP': '1', 'HVS_CODEC_FUSED': '1', 'HVS_CODEC_GRAPH': '0', 'HVS_FAST_SAMPLE': '1', 'HVS_LARGE17_LAYERWISE': '1', 'HVS_LOCAL_COMPILE': '0', 'HVS_LOCAL_TRITON': '1', 'HVS_RAMP_START': '3', 'HVS_RL_PRECISION': 'bf16_merged', 'HVS_SEMANTIC_BANKS': '1', 'HVS_SEMANTIC_BUCKET': '256', 'HVS_SEMANTIC_COMPILE': '1', 'HVS_SEMANTIC_CUDA_GRAPH': '1'}
def setup(device):
    for k,v in DEFAULTS.items():os.environ.setdefault(k,v)
    os.environ.update(ACTING_DEVICE=device)
    sys.path.insert(0,str(Path(__file__).resolve().parent/"runtime"))
    import json
    index=Path(__file__).resolve().parent/'assets.local.json'
    if index.exists():
        assets=json.loads(index.read_text())
        for key in ('codec','asr','aligner'):
            os.environ.setdefault('PODCAST_'+key.upper(),assets[key])
