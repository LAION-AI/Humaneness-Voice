"""Evaluate a frozen pilot checkpoint on a separate feature corpus, no refitting."""
import argparse
import json
from pathlib import Path

import torch

from .models import SpeakerPredictor,load_state
from .train import Windows,predictions,evaluate,grouped_metrics,accumulation_metrics,teacher_oracle,stable_context_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--data",required=True)
    parser.add_argument("--out",required=True)
    args = parser.parse_args()
    torch.set_num_threads(6)
    state = torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    cfg = state["config"]
    model = SpeakerPredictor(cfg["architecture"],cfg["width"],centered=cfg.get("centered_target",False))
    load_state(model,state["model"])
    model.eval()
    data = Windows(args.data,"test",cfg["context"])
    validation = Windows(args.data,"val",cfg["context"])
    pred = predictions(model,data,"cpu")
    threshold = state.get("verification_threshold",state["validation"]["far05_threshold"])
    payload = {"checkpoint":args.checkpoint,"data":args.data,"config":cfg,
        "threshold_source":"Frozen original checkpoint validation threshold; no adaptation to new corpus",
        "test":evaluate(pred,data.teachers,threshold),
        "validation":evaluate(predictions(model,validation,"cpu"),validation.teachers,threshold),
        "test_by_scenario":grouped_metrics(pred,data,threshold),
        "test_stable_contexts_oracle":stable_context_metrics(pred,data,threshold),
        "accumulation_test_oracle_resets":accumulation_metrics(pred,data,threshold),
        "long_clean_teacher_oracle":teacher_oracle(args.data,Windows(args.data,"val",12),data)}
    Path(args.out).write_text(json.dumps(payload,indent=2))
    print(json.dumps({"test":payload["test"]}))


if __name__ == "__main__":main()
