"""Select the integration checkpoint by the broader V2 validation split only."""
import json
from pathlib import Path

import torch


def main():
    root = Path(__file__).resolve().parent
    candidates = []
    for report,artifact,name in (
        ("ood_v1_v2.json","results/best.pt","Frozen V1 raw MLP"),
        ("ood_centered_v2.json","artifacts_centered/best.pt","Frozen V1 centered MLP"),
        ("results_v2_raw/results.json","artifacts_v2_raw/best.pt","V2 raw balanced MLP"),
        ("results_v2_centered/results.json","artifacts_v2_centered/best.pt","V2 centered balanced MLP"),
        ("results_v2_gru/results.json","artifacts_v2_gru/best.pt","V2 centered balanced GRU")):
        if not (root/report).exists(): continue
        payload = json.loads((root/report).read_text())
        validation = payload.get("validation",payload.get("selected",{}).get("validation"))
        candidates.append({"name":name,"report":report,"artifact":artifact,"validation":validation,"test":payload["test"]})
    if len(candidates)!=5: raise ValueError("All five predefined candidate comparisons must finish before selecting")
    chosen = min(candidates,key=lambda c:c["validation"]["eer"])
    state = torch.load(root/chosen["artifact"],map_location="cpu",weights_only=False)
    state["selection_validation"] = chosen["validation"]
    state["test"] = chosen["test"]
    state["selection_protocol"] = "Minimum V2 validation EER among five predefined candidates; test does not choose weights or thresholds"
    destination = root/"artifacts"
    destination.mkdir(exist_ok=True)
    torch.save(state,destination/"best.pt")
    params = sum(v.numel() for k,v in state["model"].items() if k not in ("teacher_mean","teacher_radius"))
    selection = {"checkpoint":"best.pt","name":chosen["name"],"config":state["config"],
        "parameters":params,"verification_threshold":state["verification_threshold"],
        "activity_threshold":state["activity_threshold"],"recommended_enabled":False,"pilot_pass":False,
        "validation":chosen["validation"],"test":chosen["test"],
        "selection_protocol":state["selection_protocol"],
        "threshold_source":"Frozen selected checkpoint's original validation-only calibration; no test threshold adjustment",
        "activity_calibration_note":"Speech activity probability/context coverage is not speaker identity probability. Original validation nonspeech scores calibrate abstention, microphone RMS adds a separate silence gate.",
        "note":"Experimental synthetic-voice pilot: insufficient held-out verification and overlap robustness. Keep default identity gate OFF; do not use for authentication. Full-scale training was not started because the pilot failed reliability checks.",
        "candidates":[{"name":c["name"],"validation_eer":c["validation"]["eer"],"report":c["report"]} for c in candidates],
        "feature_source":state["feature_source"],"teacher":state["teacher"]}
    folder = {"Frozen V1 raw MLP":"results","Frozen V1 centered MLP":"results_centered",
              "V2 raw balanced MLP":"results_v2_raw","V2 centered balanced MLP":"results_v2_centered",
              "V2 centered balanced GRU":"results_v2_gru"}[chosen["name"]]
    original = json.loads((root/folder/"results.json").read_text())
    selection["cpu_batch1_ms"] = original["selected"]["cpu_batch1_ms"]
    (destination/"selection.json").write_text(json.dumps(selection,indent=2))
    print(json.dumps({"selected":chosen["name"],"validation_eer":chosen["validation"]["eer"],"test_eer":chosen["test"]["eer"],"recommended_enabled":False}))


if __name__ == "__main__": main()
