import os
from pathlib import Path
HERE=Path(os.environ.get("PODCAST_CACHE",Path.home()/".cache/humaneness-podcast"))
HF_CACHE=os.environ.get("HF_HOME",str(Path.home()/".cache/huggingface"))
ACTING_DEVICE=os.environ.get("ACTING_DEVICE","cuda:0")
SCORE_DEVICE=ACTING_DEVICE
