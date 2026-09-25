#!/usr/bin/env python3
"""Check release metadata and, optionally, the installed full-stack layout."""
import argparse
import json
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFEST = json.loads((ROOT / "manifests/agent-default-v1.json").read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", action="store_true", help="also require the full default stack on disk")
    args = ap.parse_args()
    errors = []

    def require(condition, message):
        print(("ok  " if condition else "FAIL") + " " + message)
        if not condition:
            errors.append(message)

    require(MANIFEST["system"] == "Humaneness Voice", "system name")
    require(MANIFEST["default_profile"] in MANIFEST["pilot_profiles"],
            "default profile belongs to pilot set")
    for key, item in MANIFEST["assets"].items():
        require(bool(re.fullmatch(r"[0-9a-f]{40}", item["revision"])),
                f"{key} has a pinned 40-character revision")
        require(item["type"] in ("model", "dataset"), f"{key} has a Hub type")
        require("/" in item["repo"] and " " not in item["repo"],
                f"{key} has a repository ID")

    import config
    require(config.SFT3, "current base family uses compatible adapter keys")
    require(config.DEFAULT_BRAIN == "local", "local director is the key-free default")
    for key, actual in (("sft3_dpo:p2", config.SFT3_DPO_LAM),
                        ("sft3_qdpo:quality_dpo", config.QDPO_LORAS["sft3_qdpo:quality_dpo"]),
                        *config.QUALITY_LORAS.items()):
        expected = MANIFEST["default_adapter_weights"][key]
        require(actual == expected, f"{key} weight matches manifest ({expected})")
    require(config.SFT3_VN_MAX == 1, "one delivery adapter by default")
    require(config.SPEAK_BEST_OF == MANIFEST["selection"]["speak_endpoint_n"],
            "speak candidate count matches manifest")
    require(config.BON_N == MANIFEST["selection"]["chat_n_when_enabled"],
            "chat candidate count matches manifest")

    if args.assets:
        data = Path(config.DATA_DIR)
        loras = Path(config.ASSETS)
        paths = {
            "preference p2": loras / "sft3_dpo/p2/adapter_model.safetensors",
            "selected identity": loras / f"sft3_voice/{config.DEFAULT_PROFILE}/adapter_model.safetensors",
            "quality DPO": loras / "sft3_qdpo/quality_dpo/adapter_model.safetensors",
            "genuineness LoRA": loras / "sft3_quality/adapters/genuineness_high/adapter_model.safetensors",
            "blend LoRA": loras / "sft3_quality/adapters/blend_high/adapter_model.safetensors",
            "aesthetics LoRA": loras / "sft3_quality/adapters/esthetics_high/adapter_model.safetensors",
            "voice anchor": data / f"refs/pilot/{config.DEFAULT_PROFILE}/reference.wav",
            "voice index": data / f"refs/index_profile_{config.DEFAULT_PROFILE}.json",
            "retrieval index": data / "retrieval/index.npz",
            "genuineness head": data / "scorers/genuineness/genuineness_head.pt",
            "blend head": data / "scorers/blend/blend_head_commercial.pt",
        }
        for key, path in paths.items():
            require(path.is_file(), f"{key}: {path}")
        for kind in ("sft3_emotion", "sft3_voicenet", "burst"):
            root = Path(config.LORA_ROOTS.get(kind) or "")
            count = len(list(root.glob("*/adapter_model.safetensors"))) if root.is_dir() else 0
            require(count > 0, f"{kind} has discoverable adapter files")
    else:
        print("Asset-file checks skipped. Pass --assets after downloading weights and references.")
    print("GPU audio generation is not checked by this script.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
