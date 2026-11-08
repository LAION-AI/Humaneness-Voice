import os
ALIGN_QWEN_DEVICE=os.environ.get("ACTING_DEVICE","cuda:0")
ALIGN_QWEN_REPO=os.environ.get("PODCAST_ALIGNER","Qwen/Qwen3-ForcedAligner-0.6B-hf")
ALIGN_QWEN_MAX_WORD_S=3.0
