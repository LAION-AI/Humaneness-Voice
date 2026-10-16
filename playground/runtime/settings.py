"""Small-demo settings; existing credentials stay outside this directory."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
REPO = "laion/Humaneness-Voice-Small"
REVISION = "88faa078581276a16d69172b8a89fee509266b7b"
FULL_FT_STAGE = "S3-Ref15-FullFT"
FULL_FT_SHA256 = "e1b2c4613545d1db21c6f60a5ab2c27b5320aa05c177717ae268b1ff870d406e"
LORA_REPO = "laion/Humaneness-Voice-Small-Rank1-LoRAs"
LORA_REVISION = "df46cf6c98c1ec42d9275327d59fae1a039b645e"
DPO_REPO = "laion/Humaneness-Voice-Small-DPO-LoRAs"
DPO_REVISION = "1a3fc77df1a420710ed521f59d4467efe0ff57d6"
PREFERRED_DPO_ADAPTER = "subset_pitch_quality_p2_rank016"
DEFAULT_ADAPTER = None  # Full FT is the base; all adapters are opt-in.
REF15_REPO = "laion/Humaneness-Voice-Small-Ref15-LoRA-r128"
REF15_REVISION = "0c6a96cf00780a76809ba557897d2f3ee18e8033"
REF15_SHA256 = "a64656d078cf0233b1d59567c4ee75daa27e81341b78e4ba32e6e8e1d6047d9f"
DEFAULT_MODEL = 'small_rl'
DEFAULT_VOICE = 'emolia_c1699'  # Mira; verified against the local profile catalog.
DEFAULT_CHUNK_FRAMES = 10  # 800 ms
DEFAULT_GENERATION = 'single'
DEFAULT_N = 1
DEFAULT_STREAMING = True
DEFAULT_SPEAKER_STEERING = False
DEFAULT_DYNAMIC_REFERENCE = False
ACTING_DEVICE = os.environ.get('ACTING_DEVICE', 'cuda:1')
ALIGN_DEVICE = os.environ.get('SMALL_ALIGN_DEVICE', ACTING_DEVICE)
SPEAKER_REPO = 'speechbrain/spkrec-ecapa-voxceleb'
SPEAKER_REVISION = '0f99f2d0ebe89ac095bcc5903c4dd8f72b367286'
STAGE = os.environ.get("SMALL_STAGE", FULL_FT_STAGE)
if STAGE == 'default':
    STAGE = FULL_FT_STAGE
ADAPTERS_ENABLED = STAGE in ('S3', FULL_FT_STAGE)
HF_CACHE = os.environ.get("HF_HOME", "/mnt/nvme/hf_cache")
EXPERIMENTAL_LARGE_CACHE = Path(os.environ.get('HUMANENESS_EXPERIMENTAL_LARGE_CACHE',
    str(Path.home()/'.cache/humaneness-alpha/hub')))
# This host's disks are almost full. An explicit tmpfs cache avoids deleting
# unrelated data; weights are re-downloaded after a host reboot if necessary.
RL_CACHE = Path(os.environ.get('HUMANENESS_RL_CACHE', '/dev/shm/humaneness-rl/hub'))
DEVICE = os.environ.get("SMALL_DEVICE", "cuda:0")
SCORE_DEVICE = os.environ.get("SMALL_SCORE_DEVICE", "cuda:1")
VC_DEVICE = os.environ.get('SMALL_VC_DEVICE', SCORE_DEVICE)
VC_BATCH_SIZE = max(1,min(16,int(os.environ.get('SMALL_VC_BATCH_SIZE','8'))))
VC_REPO = 'ResembleAI/chatterbox'
VC_REVISION = '5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18'
VC_SHA256 = '2b78103c654207393955e4900aac14a12de8ef25f4b09424f1ef91941f161d4e'
DEFAULT_VOICE_CONVERSION = False
VC_FULL_REPO = 'laion/Humaneness-Voice-Converter'
VC_FULL_REVISION = '952ef56c945763be50432738e31a32c30b359619'
VC_FULL_CHECKPOINT = 'model.safetensors'
VC_FULL_SHA256 = '45bbf058ea2a82f1840dc6161f01be929217d28843003f5354336c81b11ea3dd'
ACTING_COMPILE_GLOBAL = os.environ.get('ACTING_COMPILE_GLOBAL', '1') == '1'
VC_GROW_REPO = 'laion/chatterbox-s3gen-vc-grow-ce-speaker'
VC_GROW_REVISION = 'ad1161233f38f95496008f014d960e6607e6aa0c'
VC_GROW_CHECKPOINT = 'checkpoints/step-0024000.pt'
VC_GROW_SHA256 = '974b45fd47a49df79ea545c50ddde3330c95657222c02d6cd339c962028f95af'
VC_GROW_PARENT_REPO = 'laion/chatterbox-s3gen-vc-grow'
VC_GROW_PARENT_REVISION = 'ede42a5aa9ded910457272fde36b11b8eec7f749'
VC_GROW_PARENT_SHA256 = 'ca6a8a7b3f50658da99ca3ce243de6a1244b8dd6f2ecead2fb40a381136b78e6'
TTL = int(os.environ.get("SMALL_TTL_SECONDS", "3600"))
MAX_UPLOAD = 20 * 1024 * 1024
# Supported reference context cap, independent of the selected adapter.
REFERENCE_SECONDS = float(os.environ.get("SMALL_REFERENCE_SECONDS", "15"))
if not 0.24 <= REFERENCE_SECONDS <= 20:
    raise ValueError("SMALL_REFERENCE_SECONDS must be between 0.24 and 20 seconds")
MAX_REFERENCES = 2
MAX_FRAMES = 750
FAST_INFERENCE = os.environ.get("SMALL_FAST_INFERENCE", "1") == "1"
COMPILE_GLOBAL = os.environ.get("SMALL_COMPILE_GLOBAL", "1") == "1"
BATCH_SIZE = int(os.environ.get("SMALL_BATCH_SIZE", "12"))
DEFAULT_FORMAT = "structured"
DEFAULT_TEMPERATURE = 0.7  # Published RL adapter inference recommendation.
DEFAULT_GUIDANCE = 2.0
DATA = HERE / "runtime"
LUNA_BASE = os.environ.get("SMALL_LUNA_BASE", os.environ.get("MOSS_LUNA_BASE", "https://api.hyprlab.io"))
LUNA_MODEL = os.environ.get("SMALL_LUNA_MODEL", os.environ.get("MOSS_LUNA_MODEL", "gpt-5.6-luna"))
KEY_FILE = Path(os.environ.get("SMALL_LUNA_KEY_FILE", os.environ.get("MOSS_LUNA_KEY_FILE", "/home/c4r33u19/moss15v2/.hyprlab_key")))


def luna_key():
    key = os.environ.get("SMALL_LUNA_API_KEY") or os.environ.get("HYPRLAB_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    return KEY_FILE.read_text().strip() if KEY_FILE.is_file() else ""
