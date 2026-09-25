#!/usr/bin/env python3
"""Download the two pinned perceptual heads used by best-of-N selection.

Each repository includes the VoiceCLAP encoder it needs. This uses slightly
more disk than a shared symlink but works on a fresh machine without an
undocumented local predictor directory.
"""
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
ASSETS = json.loads((ROOT / "manifests/agent-default-v1.json").read_text())["assets"]
DATA = Path(os.environ.get("HUMANENESS_DATA_DIR") or os.environ.get(
    "MOSS_DATA_DIR", "~/.cache/humaneness-voice")).expanduser()


def main():
    for key, folder in (("genuineness_head", "genuineness"),
                        ("blend_head", "blend")):
        spec = ASSETS[key]
        dest = DATA / "scorers" / folder
        snapshot_download(repo_id=spec["repo"], revision=spec["revision"],
                          local_dir=dest)
        print(f"{key}: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
