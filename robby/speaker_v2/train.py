"""Reproducible pilot ablations on cached frozen-encoder features.

Run from robo-integration: python -m speaker_v2.train --data speaker_v2/data
Only a validation split selects architecture/thresholds; held-out test is read
once after selection. Window-level metrics share speakers/utterances and are
not independent trials. This is a small synthetic-voice pilot, not biometrics.
"""
from __future__ import annotations

import argparse
import html
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .models import SpeakerPredictor, parameter_count


class Windows(Dataset):
    def __init__(self, root, split, context, max_windows=12000):
        self.context = context
        self.arrays = []
        self.speaker_frames = []
        self.records = []
        self.index = []
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        entries = manifest["conversations"] if isinstance(manifest, dict) else manifest
        self.teachers = np.load(root / "teacher_embeddings.npy").astype(np.float32)
        self.teachers /= np.maximum(np.linalg.norm(self.teachers, axis=-1, keepdims=True), 1e-8)
        for entry in entries:
            entry_split = "val" if entry["split"] == "validation" else entry["split"]
            if entry_split != split:
                continue
            with np.load(root / entry["path"]) as f:
                x = np.asarray(f["features"], dtype=np.float16)
                ids = np.asarray(f["speaker_idx"], dtype=np.int64)
                speech = np.asarray(f.get("speech_fraction", (ids >= 0).astype(float)))
                ambiguity = np.asarray(f.get("ambiguity", (ids == -2).astype(float)))
                valid = np.asarray(f.get("valid_mask", ids >= 0))
            arr_id = len(self.arrays)
            self.arrays.append(x)
            self.speaker_frames.append(ids)
            self.records.append(entry)
            for end in range(4, len(x) + 1, 4):
                tail = ids[end - 4:end]
                if ambiguity[end - 4:end].mean() > .25:
                    cls, target_id = 2, -1
                elif speech[end - 4:end].mean() < .25:
                    cls, target_id = 1, -1
                else:
                    positive = tail[tail >= 0]
                    if len(positive) == 0 or valid[end - 4:end].mean() < .5:
                        cls, target_id = 2, -1
                    else:
                        target_id = int(np.bincount(positive).argmax())
                        cls = 0 if (tail == target_id).mean() >= .5 else 2
                        if cls != 0:
                            target_id = -1
                self.index.append((arr_id, end, cls, target_id))
        if len(self.index) > max_windows:
            rng = random.Random(71)
            self.index = sorted(rng.sample(self.index, max_windows))
        if not self.index:
            raise ValueError(f"No {split} windows found in {root}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        arr_id, end, cls, target_id = self.index[i]
        x = self.arrays[arr_id][max(0, end-self.context):end].astype(np.float32)
        if len(x) < self.context:
            x = np.concatenate([np.repeat(x[:1], self.context-len(x), axis=0), x])
        y = self.teachers[target_id] if target_id >= 0 else np.zeros(192, np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), cls, target_id


def roc_metrics(positive, negative, threshold=None):
    positive, negative = np.asarray(positive), np.asarray(negative)
    if len(positive) == 0 or len(negative) == 0:
        return {}
    thresholds = np.unique(np.r_[positive, negative])
    # Avoid allocating [millions of pairs x thresholds].
    thresholds = np.quantile(thresholds, np.linspace(0, 1, min(1001, len(thresholds))))
    far = np.array([(negative >= t).mean() for t in thresholds])
    frr = np.array([(positive < t).mean() for t in thresholds])
    at = int(np.abs(far-frr).argmin())
    result = {"eer": float((far[at]+frr[at])/2), "eer_threshold": float(thresholds[at]),
              "far05_threshold": float(np.quantile(negative, .95)),
              "positive_cosine": float(positive.mean()), "impostor_cosine": float(negative.mean()),
              "genuine_trials": len(positive), "impostor_trials": len(negative)}
    if threshold is not None:
        result.update(threshold=float(threshold), false_accept_rate=float((negative >= threshold).mean()),
                      false_reject_rate=float((positive < threshold).mean()))
    return result


@torch.inference_mode()
def predictions(model, dataset, device):
    embeddings, targets, classes, identities, probabilities = [], [], [], [], []
    for x, target, cls, speaker in DataLoader(dataset, batch_size=128, shuffle=False):
        emb, logits = model(x.to(device))
        embeddings.append(emb.cpu().numpy())
        probabilities.append(logits.softmax(-1).cpu().numpy())
        targets.append(target.numpy())
        classes.append(cls.numpy())
        identities.append(speaker.numpy())
    return tuple(np.concatenate(x) for x in (embeddings, targets, classes, identities, probabilities))


def evaluate(pred, teachers, threshold=None):
    embeddings, targets, classes, identities, probabilities = pred
    valid = classes == 0
    e, ids = embeddings[valid], identities[valid]
    speakers = np.unique(ids)
    if len(speakers) < 2:
        raise ValueError("Verification needs at least two distinct speakers per split")
    gallery = teachers[speakers]
    sim = e @ gallery.T
    own = speakers[None, :] == ids[:, None]
    result = roc_metrics(sim[own], sim[~own], threshold)
    result.update(closed_set_accuracy=float((speakers[sim.argmax(1)] == ids).mean()),
                  activity_accuracy=float((probabilities.argmax(1) == classes).mean()),
                  speech_windows=int(valid.sum()), total_windows=len(classes),
                  speakers=int(len(speakers)))
    result["per_speaker"] = {str(int(s)): {"windows":int((ids==s).sum()),
        "teacher_cosine":float(sim[own][ids==s].mean()),
        "closed_set_accuracy":float((speakers[sim.argmax(1)][ids==s]==s).mean())}
        for s in speakers}
    result["balanced_closed_set_accuracy"] = float(np.mean([v["closed_set_accuracy"] for v in result["per_speaker"].values()]))
    result["majority_class_accuracy_baseline"] = float(max((ids==s).mean() for s in speakers))
    for cls, name in enumerate(("speech", "silence", "overlap")):
        mask = classes == cls
        result[name+"_recall"] = float((probabilities.argmax(1)[mask] == cls).mean()) if mask.any() else None
        result[name+"_windows"] = int(mask.sum())
    if len(speakers) >= 2 and threshold is not None:
        enrolled = speakers[::2]
        known = np.isin(ids, enrolled)
        score = (e @ teachers[enrolled].T).max(1)
        result["unknown_rejection"] = float((score[~known] < threshold).mean())
        result["known_acceptance"] = float((score[known] >= threshold).mean())
        result["unknown_protocol"] = "alternating held-out speaker IDs enrolled; disjoint remaining IDs unknown"
    return result


def loss_value(embedding, target, classes, logits, loss, class_weights=None, target_mean=None):
    mask = classes == 0
    if mask.any():
        e,t = embedding[mask],target[mask]
        if target_mean is not None:
            e,t = F.normalize(e-target_mean,dim=-1),F.normalize(t-target_mean,dim=-1)
        cosine = (1 - (e*t).sum(-1)).mean()
        mse = F.mse_loss(e,t) * 192
        regression = {"cosine": cosine, "mse": mse, "mixed": .5*cosine+.5*mse}[loss]
    else:
        regression = embedding.sum() * 0
    return regression + .15 * F.cross_entropy(logits, classes,weight=class_weights)


def train_one(cfg, train, validation, out, device, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = SpeakerPredictor(cfg["architecture"], cfg["width"],centered=cfg.get("centered_target",False)).to(device)
    target_mean = None
    if cfg.get("centered_target",False):
        ids = sorted(set(i[3] for i in train.index if i[3]>=0))
        teacher = torch.from_numpy(train.teachers[ids]).to(device)
        target_mean = teacher.mean(0)
        radius = (teacher-target_mean).norm(dim=-1).mean()
        model.teacher_mean.copy_(target_mean)
        model.teacher_radius.copy_(radius)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=64, shuffle=True, generator=generator)
    class_weights = None
    if cfg.get("balanced_activity",False):
        counts = torch.bincount(torch.tensor([i[2] for i in train.index]),minlength=3).float().clamp_min(1)
        class_weights = counts.rsqrt()
        class_weights = (class_weights/class_weights.mean()).to(device)
    best, best_epoch, best_state = -float("inf"), 0, None
    history = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        running = 0
        for x, target, classes, _ in loader:
            x, target, classes = x.to(device), target.to(device), classes.to(device)
            optimizer.zero_grad(set_to_none=True)
            embedding, logits = model(x)
            loss = loss_value(embedding, target, classes, logits, cfg["loss"],class_weights,target_mean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            running += float(loss.detach()) * len(x)
        model.eval()
        metrics = evaluate(predictions(model, validation, device), validation.teachers)
        # Identity ranking matters, not the regression objective alone.
        objective = 1 - metrics["eer"] + .1 * metrics["positive_cosine"]
        history.append({"epoch": epoch+1, "loss": running/len(train), **metrics})
        if objective > best:
            best, best_epoch = objective, epoch+1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch >= 14 and epoch+1 - best_epoch >= 8:
            break
    elapsed = time.perf_counter()-start
    model.load_state_dict(best_state)
    model.eval()
    val = evaluate(predictions(model, validation, device), validation.teachers)
    path = out / (f"{cfg['architecture']}-w{cfg['width']}-c{cfg['context']}-{cfg['loss']}.pt")
    state = {"model": best_state, "config": cfg, "seed": seed, "epoch": best_epoch,
             "validation": val, "activity_threshold": .6,
             "format_version": 1, "feature_source": "frozen Nemotron ASR 3.5 layer15 1024D/80ms",
             "teacher": "speechbrain/spkrec-ecapa-voxceleb normalized192",
             "warning": "Synthetic-voice pilot; not biometric authentication; no microphone recordings"}
    torch.save(state, path)
    # Batch1 CPU latency, warmed; avoid asynchronous GPU timing ambiguity.
    cpu_model = SpeakerPredictor(cfg["architecture"], cfg["width"],centered=cfg.get("centered_target",False))
    cpu_model.load_state_dict(best_state)
    cpu_model.eval()
    x = torch.randn(1, cfg["context"], 1024)
    with torch.inference_mode():
        for _ in range(10): cpu_model(x)
        t = time.perf_counter()
        for _ in range(100): cpu_model(x)
        latency_ms = (time.perf_counter()-t)*10
    return {"config": cfg, "parameters": parameter_count(model), "validation": val,
            "epochs": len(history), "best_epoch": best_epoch, "train_seconds": elapsed,
            "training_windows_per_second": len(train)*len(history)/elapsed,
            "cpu_batch1_ms": latency_ms, "checkpoint": str(path), "history": history}, model


def accumulation_metrics(pred, dataset, threshold):
    e, targets, cls, identities, _ = pred
    result = {}
    for updates in (1, 2, 3, 6, 12):
        out = e.copy()
        for i in range(len(e)):
            if cls[i] != 0:
                continue
            used = [i]
            for j in range(i-1, max(-1, i-updates), -1):
                # Consecutive windows only, same conversation and same ground truth.
                if cls[j] != 0 or identities[j] != identities[i] or dataset.index[j][0] != dataset.index[i][0] or dataset.index[j+1][1]-dataset.index[j][1] != 4:
                    break
                used.append(j)
            value = e[used].mean(0)
            out[i] = value/max(np.linalg.norm(value), 1e-8)
        result[str(updates*320)+"ms"] = evaluate((out, targets, cls, identities, pred[4]), dataset.teachers, threshold)
    return result


def baseline_predictions(train, dataset):
    """Frozen-feature ridge and constant-embedding controls, train speakers only."""
    def collect(data):
        features, targets, classes, identities = [], [], [], []
        for x,y,c,s in DataLoader(data,batch_size=256):
            features.append(F.layer_norm(x,(1024,)).mean(1))
            targets.append(y)
            classes.append(c)
            identities.append(s)
        return tuple(torch.cat(z) for z in (features,targets,classes,identities))
    x,y,c,s = collect(train)
    tx,ty,tc,ts = collect(dataset)
    keep = c == 0
    x = torch.cat([x[keep],torch.ones(int(keep.sum()),1)],-1)
    tx = torch.cat([tx,torch.ones(len(tx),1)],-1)
    regularizer = torch.eye(x.shape[1])*10
    weights = torch.linalg.solve(x.T@x+regularizer,x.T@y[keep])
    ridge = F.normalize(tx@weights,dim=-1).numpy()
    mean = F.normalize(y[keep].mean(0),dim=-1).numpy()
    constant = np.repeat(mean[None],len(tx),axis=0)
    probabilities = F.one_hot(tc,num_classes=3).float().numpy()
    # Ground-truth activity supplied only to isolate embedding performance; not deployable gates.
    rest = ty.numpy(),tc.numpy(),ts.numpy(),probabilities
    return {"ridge":(ridge,*rest),"constant_train_teacher_mean":(constant,*rest)}


def grouped_metrics(pred, dataset, threshold):
    result = {}
    scenarios = sorted(set(r.get("scenario","unspecified") for r in dataset.records))
    for scenario in scenarios:
        mask = np.array([dataset.records[i[0]].get("scenario","unspecified")==scenario for i in dataset.index])
        subset = tuple(x[mask] for x in pred)
        speakers = set(subset[3][subset[2]==0].tolist())
        if len(speakers) >= 2:
            result[scenario] = evaluate(subset,dataset.teachers,threshold)
        else:
            result[scenario] = {"windows":int(mask.sum()),"speakers":len(speakers),"note":"Too few identities for verification"}
    return result


def stable_context_metrics(pred,dataset,threshold):
    """Diagnostic only: ground-truth-pure contexts exclude boundary contamination."""
    selected = []
    for arr,end,cls,speaker in dataset.index:
        ids = dataset.speaker_frames[arr][max(0,end-dataset.context):end]
        selected.append(cls==0 and len(ids)>=dataset.context and (ids==speaker).mean()>=.9)
    mask = np.array(selected)
    if len(set(pred[3][mask].tolist()))<2:
        return {"windows":int(mask.sum()),"note":"Too few identities"}
    value = evaluate(tuple(x[mask] for x in pred),dataset.teachers,threshold)
    value["note"] = "Oracle-selected contexts with >=90% frames from same target speaker; diagnostic, not deployable filtering"
    return value


def teacher_oracle(root, validation, test):
    path = Path(root)/"probe_teacher_embeddings.npz"
    if not path.exists(): return {"available":False}
    with np.load(path,allow_pickle=False) as f:
        vectors, ids = f["embeddings"].astype(np.float32),f["speaker_idx"].astype(np.int64)
    vectors /= np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-8)
    def subset(dataset):
        speakers = set(i[3] for i in dataset.index if i[3]>=0)
        mask = np.isin(ids,list(speakers))
        n = int(mask.sum())
        probs = np.zeros((n,3),np.float32)
        probs[:,0] = 1
        return vectors[mask],dataset.teachers[ids[mask]],np.zeros(n,np.int64),ids[mask],probs
    val_metrics = evaluate(subset(validation),validation.teachers)
    test_metrics = evaluate(subset(test),test.teachers,val_metrics["far05_threshold"])
    return {"available":True,"validation":val_metrics,"test":test_metrics,
            "note":"Frozen ECAPA on complete separate clean probe source utterances vs enrollment-only long utterances. More audio than a320ms student window: a quality ceiling/domain-shift control, not an equal-context speed comparison."}


def write_report(results, destination):
    rows = "".join("<tr>"+"".join(f"<td>{html.escape(str(value))}</td>" for value in (
        r["config"]["architecture"], r["config"]["width"], r["config"]["context"]*80,
        r["config"]["loss"], r["parameters"], round(r["validation"]["eer"]*100,2),
        round(r["validation"]["positive_cosine"],3), round(r["cpu_batch1_ms"],2),
        round(r["train_seconds"],1)))+"</tr>" for r in results["ablations"])
    payload = html.escape(json.dumps({k:v for k,v in results.items() if k != "ablations"}, indent=2))
    test,selected = results["test"],results["selected"]
    comparison = [("Selected student",test)]
    comparison += [(name,value["test"]) for name,value in results.get("embedding_controls",{}).items()]
    if results.get("long_clean_teacher_oracle",{}).get("available"):
        comparison.append(("Long clean ECAPA teacher",results["long_clean_teacher_oracle"]["test"]))
    controls = "".join(f"<tr><td>{html.escape(name)}</td><td>{v['eer']*100:.1f}</td><td>{v['positive_cosine']:.3f}</td><td>{v['closed_set_accuracy']*100:.1f}</td><td>{v['false_accept_rate']*100:.1f}</td><td>{v['false_reject_rate']*100:.1f}</td></tr>" for name,v in comparison)
    summary = f"""<h2>Measured finding</h2><p><strong>Keep automatic identity filtering off by default.</strong>
    The validation-selected {html.escape(selected['config']['architecture'].upper())} has {selected['parameters']:,} trainable parameters,
    {selected['config']['context']*80} ms of feature context and {selected['cpu_batch1_ms']:.3f} ms CPU batch-one inference.
    Held-out EER is {test['eer']*100:.1f}%; at the validation-calibrated approximately 5% false-accept operating point,
    the test false-reject rate is {test['false_reject_rate']*100:.1f}%. Speed is promising; this is not yet a reliable enrollment gate.</p>
    <table><tr><th>Held-out control</th><th>EER % ↓</th><th>Teacher cosine ↑</th><th>Closed-set % ↑</th><th>False accept % ↓</th><th>False reject % ↓</th></tr>{controls}</table>
    <p>The long-clean teacher sees full clean source utterances, while the student sees short noisy contextual windows: its result is an identity-quality control, not an equal-context benchmark. The constant embedding control demonstrates why high average cosine alone is not proof of speaker discrimination.</p>"""
    destination.write_text("""<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width, initial-scale=1'><title>TYGER V2 speaker-predictor pilot</title><style>body{max-width:1100px;margin:3rem auto;padding:0 1rem;font:17px/1.6 system-ui;color:#20313b;background:#fbfaf7}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:.6rem;border:1px solid #ccc;text-align:left}pre{white-space:pre-wrap;background:#eee;padding:1rem}a{color:#255b83}.warning{border-left:5px solid #b07020;padding:1rem;background:#fff1d8}</style><h1>TYGER V2: held-out speaker-embedding pilot</h1><p class=warning>Exploratory synthetic-voice pilot, not a validated biometric system. Architecture and thresholds are selected on validation identities, never on test identities. Window-level trials are correlated; small speaker counts do not support population-level claims.</p><p>The original TYGER GRU and turn-taking rules are unchanged. This extra network consumes frozen Nemotron ASR layer-15 features (1024 dimensions / 80 ms), predicts normalized SpeechBrain ECAPA-192 embeddings, and emits speech/silence/ambiguous-overlap indicators. Confidence is not authentication.</p>"""+summary+"<h2>Architecture ablations (validation only)</h2><table><tr><th>Model</th><th>Width</th><th>Context ms</th><th>Loss</th><th>Parameters</th><th>EER %</th><th>Teacher cosine</th><th>CPU batch1 ms</th><th>Train s</th></tr>"+rows+"</table><h2>Selected model, untouched held-out test and protocols</h2><details><summary>Complete machine-readable results and protocols</summary><pre>"+payload+"</pre></details><p>Accumulation metrics use ground-truth speaker-change resets and therefore are an optimistic analysis, not deployment results. Unknown rejection enrolls alternating held-out speakers and treats the remainder as unseen. Enrollment uses clean long-utterance teacher embeddings, not the probe prediction.</p></html>")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="speaker_v2/pilot")
    p.add_argument("--out", default="speaker_v2/results")
    p.add_argument("--artifacts", default="speaker_v2/artifacts")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--balanced-activity", action="store_true", help="Inverse-square-root class-frequency auxiliary loss")
    p.add_argument("--max-windows", type=int, default=12000, help="Per-split pilot cap, not a scalable full-corpus loader")
    p.add_argument("--centered-target",action="store_true",help="Train-only mean/residual direction targets, reconstruct standard ECAPA output")
    p.add_argument("--only-config",help='One predefined config JSON, no adaptive extra ablations')
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = {}
    def dataset(split, context):
        key = split, context
        if key not in cache: cache[key] = Windows(args.data, split, context,args.max_windows)
        return cache[key]
    speaker_sets = {s:set(x[3] for x in dataset(s,12).index if x[3]>=0) for s in ("train","val","test")}
    for a,b in (("train","val"),("train","test"),("val","test")):
        if speaker_sets[a] & speaker_sets[b]:
            raise ValueError(f"Speaker leakage between {a} and {b}")
    configs = [{"architecture": a, "width": w, "context": 12, "loss": "mixed", "balanced_activity":args.balanced_activity,"centered_target":args.centered_target}
               for a in ("mlp", "gru", "tcn") for w in ((96,) if args.quick else (96,192))]
    if args.only_config:
        configs = [{**json.loads(args.only_config),"balanced_activity":args.balanced_activity,"centered_target":args.centered_target}]
    results = []
    models = {}
    def run(cfg):
        print(json.dumps({"starting": cfg}), flush=True)
        r, model = train_one(cfg, dataset("train",cfg["context"]), dataset("val",cfg["context"]),
                             out, args.device, args.epochs, 1729)
        results.append(r)
        models[r["checkpoint"]] = model.cpu()
        (out / "progress.json").write_text(json.dumps(results, indent=2))
        print(json.dumps({k:v for k,v in r.items() if k != "history"}), flush=True)
    for cfg in configs: run(cfg)
    chosen = min(results, key=lambda r:(r["validation"]["eer"], -r["validation"]["positive_cosine"]))["config"]
    extra = [{**chosen,"context": c} for c in (4,8,25)] + [{**chosen,"loss": l} for l in ("cosine","mse")]
    if args.only_config: extra = []
    for cfg in extra: run(cfg)
    best = min(results, key=lambda r:(r["validation"]["eer"], -r["validation"]["positive_cosine"]))
    model = models[best["checkpoint"]]
    test = dataset("test",best["config"]["context"])
    pred = predictions(model,test,"cpu")
    threshold = best["validation"]["far05_threshold"]
    payload = {"protocol_version": 1, "ablations": results, "selected": {k:v for k,v in best.items() if k != "history"},
               "threshold_source": "validation held-out identity impostor 95th percentile (~5% FAR); test labels never calibrate",
               "test": evaluate(pred,test.teachers,threshold),
               "test_by_scenario": grouped_metrics(pred,test,threshold),
               "test_stable_contexts_oracle":stable_context_metrics(pred,test,threshold),
               "accumulation_test_oracle_resets": accumulation_metrics(pred,test,threshold),
               "dataset": {s: {"windows":len(dataset(s,12)),"speakers":len(set(x[3] for x in dataset(s,12).index if x[3]>=0))} for s in ("train","val","test")},
               "cost_estimate": {"500000_windows_one_epoch_seconds": 500000/best["training_windows_per_second"],
                                 "excludes": "download, ASR/teacher extraction, augmentation, validation; CPU pilot throughput is not GPU full-scale throughput"}}
    baseline_val = baseline_predictions(dataset("train",12),dataset("val",12))
    baseline_test = baseline_predictions(dataset("train",12),dataset("test",12))
    payload["embedding_controls"] = {}
    for name in baseline_val:
        v = evaluate(baseline_val[name],dataset("val",12).teachers)
        t = evaluate(baseline_test[name],dataset("test",12).teachers,v["far05_threshold"])
        payload["embedding_controls"][name] = {"validation":v,"test":t,"activity_note":"oracle activity; compare embedding metrics only"}
    payload["loss_note"] = "For normalized vectors, 192*MSE = 2*(1-cosine). These losses change relative auxiliary-activity weighting, not embedding geometry."
    quality_path = Path(args.data)/"teacher_quality.json"
    if quality_path.exists(): payload["teacher_enrollment_quality"] = json.loads(quality_path.read_text())
    payload["long_clean_teacher_oracle"] = teacher_oracle(args.data,dataset("val",12),dataset("test",12))
    val_pred = predictions(model,dataset("val",best["config"]["context"]),"cpu")
    nonspeech = val_pred[4][val_pred[2]!=0,0]
    activity_threshold = float(max(.5,min(.95,np.quantile(nonspeech,.95)))) if len(nonspeech) else .9
    payload["activity_calibration"] = {"threshold":activity_threshold,"source":"validation nonspeech speech-probability95thpercentile, bounded.5–.95",
        "test_false_speech_rate":float((pred[4][pred[2]!=0,0]>=activity_threshold).mean()),
        "test_speech_acceptance":float((pred[4][pred[2]==0,0]>=activity_threshold).mean()),
        "warning":"Activity confidence is not identity probability; microphone RMS gate is separate"}
    manifest = json.loads((Path(args.data)/"manifest.json").read_text())
    payload["dataset_limitations"] = ["Only 32 synthetic identities and 4 held-out test identities; no real-microphone robustness claim.",
        "Alignment occupancy may mislabel untranscribed vocal bursts.",*manifest.get("limitations",[])]
    payload["source_metadata"] = {k:manifest.get(k) for k in ("dataset_repo","dataset_revision","source_license","attribution","teacher_repo","teacher_revision","pilot_revision")}
    records = manifest["conversations"]
    seconds = sum(r.get("duration_s",0) for r in records)
    feature_wall = sum(r.get("feature_seconds",0) for r in records)
    payload["feature_scaling"] = {"audio_seconds":seconds,"feature_call_wall_seconds":feature_wall,
        "feature_call_wall_rtf":feature_wall/max(seconds,1),
        "100000_x24s_feature_call_hours":100000*24*feature_wall/max(seconds,1)/3600,
        "all_frames_100000_x24s_float16_GB":100000*24/.08*1024*2/1e9,
        "500000_context25_windows_float16_GB":500000*25*1024*2/1e9,
        "note":"Feature call wall time includes local HTTP and serialization, excludes waveform construction/download; not isolated on-device ASR timing. Storage estimates exclude metadata and teacher caches."}
    (out / "results.json").write_text(json.dumps(payload,indent=2))
    write_report(payload, out / "report.html")
    selected = torch.load(best["checkpoint"],map_location="cpu",weights_only=False)
    selected["activity_threshold"] = activity_threshold
    selected["verification_threshold"] = threshold
    selected["test"] = payload["test"]
    torch.save(selected,out / "best.pt")
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True,exist_ok=True)
    torch.save(selected,artifacts / "best.pt")
    test_result = payload["test"]
    passed = (test_result["eer"] <= .2 and test_result["false_accept_rate"] <= .1
              and test_result.get("known_acceptance",0) >= .7
              and (test_result.get("silence_recall") or 0) >= .8)
    selection = {"checkpoint":"best.pt","config":best["config"],"parameters":best["parameters"],
                 "verification_threshold":threshold,"activity_threshold":activity_threshold,
                 "threshold_source":payload["threshold_source"],"recommended_enabled":False,
                 "pilot_pass":bool(passed),"test":test_result,"validation":best["validation"],
                 "cpu_batch1_ms":best["cpu_batch1_ms"],
                 "note":"Experimental synthetic-voice pilot only. Default identity gate remains off; held-out speakers are too few for real-user deployment claims. Confidence is activity/context, not calibrated identity probability.",
                 "feature_source":selected["feature_source"],"teacher":selected["teacher"]}
    (artifacts / "selection.json").write_text(json.dumps(selection,indent=2))
    print(json.dumps({"complete":str(out),"test":payload["test"]}),flush=True)


if __name__ == "__main__": main()
