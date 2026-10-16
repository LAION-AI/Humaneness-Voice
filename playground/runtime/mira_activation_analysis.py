"""Protocol, vector aggregation and exhaustive analysis for Mira steering.

No GPU/model loading or live-server mutation. A cosine is not a percentage of
speaker identity. Raw activation means contain text, emotion and other factors;
they are candidate steering directions, not isolated speaker representations.
"""
from __future__ import annotations

import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import itertools
import json
from pathlib import Path
import random

import numpy as np
from prompting import REFERENCE_IDENTITY

EMOTIONS = {
    "neutral": "Neutral conversational delivery, clear and natural phrasing.",
    "joy": "Joyful relief, lively smiling delivery and warm emphasis.",
    "anger": "Intense contained anger, forceful emphasis and controlled tension.",
    "sadness": "Restrained sadness, gentle emphasis and a brief emotional hesitation.",
    "fear": "Fear and uncertainty, hesitant starts and restrained urgency.",
}
IDENTITY = REFERENCE_IDENTITY
METRICS = ("ecapa", "anchor_ecapa", "content_enjoyment", "production_quality", "mos", "aesthetics", "genuineness", "wer", "emotion_score")
GATES = {
    "wer_delta_max": .03,
    "wer_mean_max": .15,
    "bad_wer_rate_delta_max": .10,
    "content_enjoyment_delta_min": -.30,
    "production_quality_delta_min": -.30,
    "mos_delta_min": -.20,
    "aesthetics_delta_min": -.30,
    "emotion_score_delta_min": -.20,
    "ecapa_delta_min": .01,
    "generation_failures_max": 0,
}
TEXTS = {
    "collection": {
        "en": ["The parcel arrived this morning. I put it beside the kitchen door.",
               "There is a small notebook on the table. Please leave it where it is.",
               "I checked the timetable again. We have enough time to walk to the station.",
               "Your message was waiting when I returned. I read it before making dinner.",
               "The lights are still on downstairs. I will turn them off before we leave.",
               "I remember the afternoon we spent there. The streets were almost empty.",
               "Please tell me what you decided. We can work out the details together.",
               "The window was open when I came in. I closed it and moved the chair.",
               "There is something I need to explain. Let me finish before you answer.",
               "We have everything we need for tomorrow. I will meet you outside."],
        "de": ["Das Paket ist heute Morgen angekommen. Ich habe es neben die Küchentür gestellt.",
               "Auf dem Tisch liegt ein kleines Notizbuch. Bitte lass es dort liegen.",
               "Ich habe den Fahrplan noch einmal geprüft. Wir können zum Bahnhof laufen.",
               "Deine Nachricht war da, als ich zurückkam. Ich las sie vor dem Abendessen.",
               "Unten brennt noch Licht. Ich mache es aus, bevor wir gehen.",
               "Ich erinnere mich an unseren Nachmittag dort. Die Straßen waren fast leer.",
               "Bitte sag mir, was du entschieden hast. Die Einzelheiten klären wir gemeinsam.",
               "Als ich hereinkam, stand das Fenster offen. Ich schloss es und rückte den Stuhl.",
               "Ich muss dir etwas erklären. Lass mich ausreden, bevor du antwortest.",
               "Für morgen haben wir alles vorbereitet. Ich treffe dich draußen."],
    },
    "screen": {
        "en": ["Our appointment is on Thursday. I have already written down the address.",
               "We crossed the bridge before sunset. Nobody else was waiting on the other side.",
               "I left the keys in your coat pocket. You should find them next to the receipt.",
               "The garden looks different this year. I think the old tree has finally grown.",
               "Please take a seat beside me. There is one more thing we should discuss."],
        "de": ["Unser Termin ist am Donnerstag. Die Adresse habe ich bereits aufgeschrieben.",
               "Wir überquerten die Brücke vor Sonnenuntergang. Auf der anderen Seite wartete niemand.",
               "Die Schlüssel stecken in deiner Manteltasche. Du findest sie neben der Quittung.",
               "Der Garten sieht dieses Jahr anders aus. Der alte Baum ist endlich gewachsen.",
               "Bitte setz dich neben mich. Wir sollten noch eine Sache besprechen."],
    },
    "refine": {
        "en": ["The shop closes in an hour. Let us get what we need before going home.",
               "I found the picture behind the bookshelf. I had forgotten how young we looked.",
               "Someone called while you were away. They said they would try again tonight.",
               "The room was quiet after everyone left. I stayed a little longer by myself.",
               "We can follow the path along the river. It should take us back to the village."],
        "de": ["Der Laden schließt in einer Stunde. Wir besorgen alles und gehen dann nach Hause.",
               "Das Bild lag hinter dem Bücherregal. Ich hatte vergessen, wie jung wir aussahen.",
               "Während du weg warst, hat jemand angerufen. Heute Abend versucht die Person es erneut.",
               "Als alle weg waren, wurde es still im Zimmer. Ich blieb noch ein wenig allein.",
               "Wir können dem Weg am Fluss folgen. Er sollte uns zum Dorf zurückführen."],
    },
    "validation": {
        "en": ["I saved a place for you near the stage. You will be able to see everything from there.",
               "The first snow fell during the night. This morning the whole hillside was white.",
               "Your brother asked about the journey. I told him we would explain when we arrived.",
               "I moved the boxes into the hallway. Be careful when you come through the door.",
               "We should keep a copy of this letter. It might become important later."],
        "de": ["Ich habe dir einen Platz an der Bühne freigehalten. Von dort kannst du alles sehen.",
               "In der Nacht fiel der erste Schnee. Heute Morgen war der ganze Hang weiß.",
               "Dein Bruder fragte nach der Reise. Wir erklären es ihm, sobald wir angekommen sind.",
               "Die Kisten stehen jetzt im Flur. Pass auf, wenn du durch die Tür kommst.",
               "Wir sollten diesen Brief kopieren. Vielleicht wird er später noch wichtig."],
    },
}


def condition(layers=(), alpha=0., kind="raw_mean", name=None):
    """Serializable intervention specification; baseline has no hooks."""
    points = tuple(str(x) if str(x).startswith(("layer_", "global_", "semantic")) else f"layer_{int(x)}" for x in layers)
    if alpha == 0 or not points:
        return {"id": "baseline", "points": [], "alpha": 0., "kind": "none", "name": "Matched no-steering control"}
    if kind not in {"raw_mean", "contrast", "norm_matched_mean", "semantic_raw_mean", "semantic_contrast"}:
        raise ValueError(f"Unknown vector kind: {kind}")
    if len(set(points)) != len(points) or len(points) > 3 or alpha <= 0:
        raise ValueError("Require 1–3 distinct points and positive alpha")
    key = f'{kind}-{"+".join(points)}-a{float(alpha):g}'
    return {"id": key, "points": list(points), "alpha": float(alpha), "kind": kind, "name": name or key}


def blocks(phase, count=None):
    """Every phase uses new text and new independently assigned seed ranges."""
    source_phase = "screen" if phase == "screen_semantic" else phase
    if source_phase not in TEXTS:
        raise ValueError(phase)
    count = count or (100 if phase == "collection" else 20 if phase == "validation" else 10)
    emotions = list(EMOTIONS)
    offsets = {"collection": 1000000, "screen": 2000000, "refine": 3000000, "validation": 4000000}
    out = []
    for i in range(count):
        emotion = emotions[(i // 2) % len(emotions)]
        language = ("en", "de")[i % 2]
        text = TEXTS[source_phase][language][(i // 2) % len(TEXTS[source_phase][language])]
        # Collection repeats each text under diverse requested emotions across cycles.
        if phase == "collection":
            emotion = emotions[((i // 2) + (i // 20)) % len(emotions)]
        prompt = "CAPTION: " + IDENTITY + " " + EMOTIONS[emotion] + "\nTRANSCRIPT: " + json.dumps(text, ensure_ascii=False)
        out.append({"block": i, "phase": phase, "voice": "mira", "seed": offsets[source_phase] + 1009 * i,
                    "emotion": emotion, "language": language, "text": text, "prompt": prompt})
    return out


def rows_for(phase, conditions, count=None):
    rows = [{**b, "id": f'{phase}-{c["id"]}-{b["block"]}', "condition": c["id"], "intervention": c}
            for b in blocks(phase, count) for c in conditions]
    random.Random({"collection": 101, "screen": 102, "screen_semantic": 102, "refine": 103, "validation": 104}[phase]).shuffle(rows)
    return rows


def design(collection_n=100, n_layers=28):
    if collection_n not in (50, 100):
        raise ValueError("Use 50 or 100 collection samples")
    screening = [condition()] + [condition([i], .25) for i in range(n_layers)]
    semantic_conditions = [condition()] + [condition([point], alpha, "semantic_raw_mean")
        for point in ("semantic_global", "semantic_local") for alpha in (.25, .5, .75, 1.)]
    return {
        "version": 1, "created_utc": "frozen-before-generation", "voice": "mira", "collection_n": collection_n,
        "collection_rows": rows_for("collection", [condition()], collection_n),
        "screen_conditions": screening, "screen_rows": rows_for("screen", screening, 10),
        "semantic_conditions": semantic_conditions, "semantic_rows": rows_for("screen_semantic", semantic_conditions, 10),
        "refine_blocks": blocks("refine", 8), "validation_blocks": blocks("validation"),
        "sampling": {"temperature": 1., "cfg": 2., "duration": "automatic", "max_semantic_frames": 256,
                     "reference_cap_seconds": 15, "adapters": [], "best_of_n": 1, "quality_retry": False,
                     "fixed_reference": True, "conversion": False, "alignment_trim": False},
        "activation_rule": "Mean only backbone/semantic states for positions used to predict generated audio frames, excluding prompt/reference prefill and post-EOS padding. Counts recorded. Each sample then receives equal weight; each available emotion receives equal weight.",
        "vector_rule": "Report strict ECAPA cosine >0.25 and >0.50. Use >0.50 only if >=10 samples and >=2 in each of five emotions; otherwise >0.25 with the same coverage. If neither qualifies, label exploratory and use the >=10 high-sim pool with emotion omissions disclosed; never lower the threshold silently.",
        "search_rule": "All 28 single global layers screen at alpha .25 on ten matched blocks (five emotions x two languages; exploratory evidence). Refine up to four promising single points across .25/.5/.75/1 on eight fresh matched blocks; explore pairs/triples from the top 2–3 points at .25/.5. Semantic_global and semantic_local are distinct additional raw-mean points, tested at .25/.5/.75/1 with matched controls on the same ten screen task/seed blocks, using independent no-steering controls on that GPU. Contrasts/RMS-matched interventions are optional diagnostics and must be labeled separately. Freeze up to three gate-qualified candidates before twenty fresh validation blocks; always include no-steering control. If nothing passes gates, validate best diagnostic candidates but make no improvement claim.",
        "combination_rule": "Total alpha budget shared across selected points (alpha/number_of_points per hook), preventing a 3-layer condition from silently tripling perturbation strength. Explicit fixed-budget naming required.",
        "selection_gates": GATES,
        "selection_score": "Maximize matched ECAPA gain among candidates passing all observed quality/WER/emotion/failure gates; then CE gain. Aesthetics is gated if measured for any matched take in that phase and must then be complete. Final claim requires held-out validation positive ECAPA paired-bootstrap lower bound, not discovery-only evidence.",
        "limitations": ["One voice, one conditioning reference and finite text/emotion coverage.",
                        "Raw activation means mix speaker, text, acoustic timing and emotion; no claim of an isolated speaker vector.",
                        "Semantic_global is the post-Qwen-normalization 1024-dimensional vector; semantic_local is its 2560-dimensional bias-free projection used as the acoustic decoder seed. Their means are approximately linearly related, so raw additions at these two points are not independent mechanisms; BF16 rounding can still produce autoregressive divergence.",
                        "ECAPA cosine is not a calibrated probability or percentage of identity.",
                        "CE/PQ/MOS/emotion are learned proxies, not blind human listening.",
                        "Threshold selection on collection outcomes induces selection bias; discovery/validation generations are independent.",
                        "Exploratory bootstrap intervals are not family-wise multiple-testing corrections.",
                        "Failures, truncated outputs, regressions and unsupported conditions are retained; no resampling until a good take."]}


def aggregate_vectors(records, activation_arrays, threshold=.25, min_per_emotion=2):
    """Equal-sample then equal-emotion candidate means; preserve raw amplitudes.

    activation_arrays: id -> {'layer_0': np.array[H], 'semantic': array[H], ...}.
    Return ndarray vectors for torch.save/NPZ plus JSON-compatible metadata.
    High-minus-low uses the same contributing emotions, otherwise is withheld.
    """
    good = [r for r in records if not r.get("failed") and r.get("status", "ok") != "failed"
            and r["id"] in activation_arrays and _metric(r, "ecapa") is not None]
    high = [r for r in good if _metric(r, "ecapa") > threshold]
    low = [r for r in good if _metric(r, "ecapa") <= threshold]
    hcounts = Counter(r["emotion"] for r in high)
    lcounts = Counter(r["emotion"] for r in low)
    expected = set(EMOTIONS)
    eligible = len(high) >= 10 and all(hcounts[e] >= min_per_emotion for e in expected)
    points = set.intersection(*(set(activation_arrays[r["id"]]) for r in high)) if high else set()
    vectors, info = {}, {}
    for point in sorted(points):
        def emotion_mean(pool, emotions):
            return np.mean([np.mean([np.asarray(activation_arrays[r["id"]][point], dtype=np.float64)
                                   for r in pool if r["emotion"] == emotion], axis=0)
                            for emotion in sorted(emotions)], axis=0).astype(np.float32)
        high_emotions = set(hcounts)
        mean = emotion_mean(high, high_emotions)
        if not np.all(np.isfinite(mean)):
            raise ValueError(f"Nonfinite activation: {point}")
        vectors[point] = {"raw_mean": mean}
        point_low = [r for r in low if point in activation_arrays[r["id"]]]
        point_low_counts = Counter(r["emotion"] for r in point_low)
        common = high_emotions & set(point_low_counts)
        if len(common) == len(expected) and all(point_low_counts[e] >= min_per_emotion for e in expected):
            contrast = emotion_mean(high, common) - emotion_mean(point_low, common)
            if not np.all(np.isfinite(contrast)):
                raise ValueError(f"Nonfinite contrast activation: {point}")
            vectors[point]["contrast"] = contrast
        info[point] = {"dimension": int(mean.size), "raw_mean_l2": float(np.linalg.norm(mean)),
                       "raw_mean_rms": float(np.sqrt(np.mean(mean * mean))),
                       "contrast_l2": float(np.linalg.norm(vectors[point]["contrast"])) if "contrast" in vectors[point] else None}
    return {"vectors": vectors, "metadata": {"threshold": threshold, "threshold_comparison": "strictly greater than",
            "eligible": eligible, "high_n": len(high), "low_n": len(low), "high_emotion_counts": dict(hcounts),
            "low_emotion_counts": dict(lcounts), "missing_high_emotions": sorted(expected-set(hcounts)),
            "high_ids": [r["id"] for r in high], "low_ids": [r["id"] for r in low], "points": info,
            "weighting": "equal sample within emotion, equal contributing emotion; generated-position means, not sum pooled by length"}}


def refinement_conditions(selected_points, include_contrast=True):
    points = list(dict.fromkeys(selected_points))[:3]
    result = [condition()]
    for size in range(1, min(3, len(points)) + 1):
        for combination in itertools.combinations(points, size):
            for alpha in (.25, .5, .75, 1.):
                result.append(condition(combination, alpha))
                if include_contrast:
                    result.append(condition(combination, alpha, "contrast"))
    return result


def _metric(row, key):
    aliases = {"ecapa": ("canonical_cosine", "speaker_similarity"), "content_enjoyment": ("ce",),
               "production_quality": ("pq",), "mos": ("dnsmos",), "emotion_score": ("emotion_target_raw",)}
    value = row.get(key)
    if value is None:
        value = next((row[a] for a in aliases.get(key, ()) if row.get(a) is not None), None)
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def paired_interval(values, repetitions=4000, seed=203):
    """Resample matched text/seed blocks, never unpaired independent takes."""
    if not values:
        return None
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (repetitions, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "low": float(np.quantile(means, .025)),
            "high": float(np.quantile(means, .975)), "n_pairs": len(values)}


def summarize(rows, baseline="baseline", bootstrap_repetitions=4000):
    """Keep every failure visible. Missing mandatory metrics prevent eligibility."""
    phases = sorted(set(r.get("phase", "unknown") for r in rows))
    output = {"conditions": [], "selection_gates": GATES, "bootstrap_repetitions": bootstrap_repetitions}
    for phase in phases:
        phase_rows = [r for r in rows if r.get("phase", "unknown") == phase]
        controls = {r["block"]: r for r in phase_rows if r["condition"] == baseline}
        for key in sorted(set(r["condition"] for r in phase_rows)):
            group = [r for r in phase_rows if r["condition"] == key]
            failures = [r for r in group if r.get("failed") or r.get("status") == "failed"]
            successful = [r for r in group if r not in failures]
            entry = {"phase": phase, "condition": key, "n": len(group), "failures": len(failures),
                     "failure_ids": [r["id"] for r in failures], "matched": {}, "means": {}, "missing": {}}
            for metric in METRICS:
                measured = [_metric(r, metric) for r in successful]
                valid = [v for v in measured if v is not None]
                entry["means"][metric] = float(np.mean(valid)) if valid else None
                entry["missing"][metric] = len(group) - len(valid)
                diffs = []
                for r in successful:
                    control = controls.get(r["block"])
                    if control is None or control.get("failed") or control.get("status") == "failed":
                        continue
                    a, b = _metric(r, metric), _metric(control, metric)
                    if a is not None and b is not None:
                        diffs.append(a-b)
                entry["matched"][metric] = paired_interval(diffs, bootstrap_repetitions)
            bad = lambda rs: sum(bool(r.get("failed") or r.get("status") == "failed" or (_metric(r, "wer") is not None and _metric(r, "wer") > .25)) for r in rs) / len(rs) if rs else None
            entry["bad_wer_rate"] = bad(group)
            matched_controls = [controls[r["block"]] for r in group if r["block"] in controls]
            entry["bad_wer_rate_delta"] = entry["bad_wer_rate"] - bad(matched_controls) if matched_controls else None
            reasons = []
            if failures: reasons.append("generation_failure")
            required = ("ecapa", "content_enjoyment", "production_quality", "mos", "wer")
            if any(_metric(r, "aesthetics") is not None for r in phase_rows):
                required += ("aesthetics",)
            for metric in required:
                pair = entry["matched"][metric]
                if not pair or pair["n_pairs"] != len(group): reasons.append("incomplete_matched_" + metric)
            delta = lambda metric: entry["matched"][metric]["mean"] if entry["matched"][metric] else None
            checks = [("wer", "max", GATES["wer_delta_max"]),
                      ("content_enjoyment", "min", GATES["content_enjoyment_delta_min"]),
                      ("production_quality", "min", GATES["production_quality_delta_min"]),
                      ("mos", "min", GATES["mos_delta_min"]),
                      ("aesthetics", "min", GATES["aesthetics_delta_min"])]
            # Neutral cases have no emotion target; require complete non-neutral scoring separately.
            nonneutral = [r for r in group if r.get("emotion") != "neutral"]
            if any(_metric(r, "emotion_score") is None for r in nonneutral): reasons.append("incomplete_emotion_metric")
            if delta("emotion_score") is not None and delta("emotion_score") < GATES["emotion_score_delta_min"]:
                reasons.append("emotion_drop")
            for metric, direction, limit in checks:
                d = delta(metric)
                if d is not None and ((direction == "max" and d > limit) or (direction == "min" and d < limit)):
                    reasons.append(metric + "_regression")
            if entry["means"]["wer"] is not None and entry["means"]["wer"] > GATES["wer_mean_max"]: reasons.append("absolute_wer_limit")
            if entry["bad_wer_rate_delta"] is not None and entry["bad_wer_rate_delta"] > GATES["bad_wer_rate_delta_max"]: reasons.append("catastrophic_wer_regression")
            if key != baseline and (delta("ecapa") is None or delta("ecapa") < GATES["ecapa_delta_min"]): reasons.append("insufficient_speaker_gain")
            entry["gate_reasons"] = sorted(set(reasons))
            entry["eligible"] = not reasons and key != baseline
            entry["heldout_positive_interval"] = bool(phase == "validation" and entry["eligible"] and entry["matched"]["ecapa"]["low"] > 0)
            output["conditions"].append(entry)
    return output


def select_conditions(rows, max_selected=3, phase=None):
    candidates = summarize(rows)["conditions"]
    if phase is not None: candidates = [c for c in candidates if c["phase"] == phase]
    candidates = [c for c in candidates if c["eligible"]]
    candidates.sort(key=lambda c: (c["matched"]["ecapa"]["mean"], c["matched"]["content_enjoyment"]["mean"]), reverse=True)
    return list(dict.fromkeys(c["condition"] for c in candidates))[:max_selected]


def build_refinement_plan(coarse_rows, available_vector_keys=(), max_points=3,
                          max_conditions=24, coarse_phase="screen"):
    """Bounded adaptive grid, frozen before the new refinement takes.

    Three raw-mean points produce 21 conditions incl. control: twelve singles,
    six pairs and two triples. Two best-point contrast diagnostics plus one
    RMS-matched diagnostic bring the maximum to 24. No post-hoc deletion of
    awful coarse conditions: complete summaries accompany the chosen grid.
    """
    if max_conditions < 2 or not 1 <= max_points <= 3:
        raise ValueError("Use 1–3 points and budget >=2")
    summary = summarize(coarse_rows)
    specs = {r["condition"]: r.get("intervention", {}) for r in coarse_rows}
    candidates = []
    for c in summary["conditions"]:
        points = specs.get(c["condition"], {}).get("points", [])
        if c["phase"] != coarse_phase or len(points) != 1 or not points[0].startswith("layer_"):
            continue
        if c["matched"]["ecapa"] is None:
            continue
        candidates.append(c)
    def rank(c):
        return (c["matched"]["ecapa"]["mean"],
                c["matched"]["content_enjoyment"]["mean"] if c["matched"]["content_enjoyment"] else -100.)
    qualified = sorted((c for c in candidates if c["eligible"]), key=rank, reverse=True)
    safe = sorted((c for c in candidates if not (set(c["gate_reasons"])-{"insufficient_speaker_gain"})), key=rank, reverse=True)
    measured = sorted(candidates, key=rank, reverse=True)
    chosen = []
    # Strong candidates first; quality-safe but noisy directions are worth a
    # fresh small grid. If all are bad, diagnostics are explicitly labeled.
    for c in qualified + safe + measured:
        point = specs[c["condition"]]["points"][0]
        if point not in chosen: chosen.append(point)
        if len(chosen) == max_points: break
    conditions = [condition()]
    for point in chosen:
        conditions.extend(condition([point], a) for a in (.25, .5, .75, 1.))
    for size in (2, 3):
        for combo in itertools.combinations(chosen, size):
            conditions.extend(condition(combo, a) for a in (.25, .5))
    available = set(available_vector_keys)
    if chosen and chosen[0] + "__contrast" in available:
        conditions.extend(condition([chosen[0]], a, "contrast") for a in (.25, .5))
    if chosen:
        conditions.append(condition([chosen[0]], .25, "norm_matched_mean"))
    conditions = conditions[:max_conditions]
    chosen_cells = [c for c in candidates if specs[c["condition"]]["points"][0] in chosen]
    return {"phase": "refine", "conditions": conditions, "rows": rows_for("refine", conditions, 8),
            "selected_points": chosen, "selection_details": chosen_cells,
            "coarse_summary": summary, "condition_budget": max_conditions,
            "diagnostic_only_if_no_qualified": not qualified,
            "reason": "Explore up to three single-layer points, prioritizing observed gate-qualified gains, then quality-safe noisy gains; fallback picks are diagnostic only. Fixed total alpha is shared across points. New refinement text/seeds were frozen before collection."}


def build_validation_plan(rows, max_selected=3, candidates_phases=("refine", "screen_semantic")):
    """Freeze at most three candidates; prefer fresh refinement over coarse.

    If no candidate passes safeguards, choose diagnostic candidates for a
    falsification check, explicitly not a positive recommendation.
    """
    summary = summarize(rows)
    cells = [c for c in summary["conditions"] if c["phase"] in candidates_phases and c["condition"] != "baseline"]
    specs = {r["condition"]: r.get("intervention", {}) for r in rows}
    refined = {c["condition"] for c in cells if c["phase"] == "refine"}
    cells = [c for c in cells if c["phase"] == "refine" or c["condition"] not in refined]
    measured = [c for c in cells if c["matched"]["ecapa"] is not None and specs.get(c["condition"], {}).get("points")]
    qualified = [c for c in measured if c["eligible"]]
    pool = qualified or measured
    pool.sort(key=lambda c: (c["matched"]["ecapa"]["mean"],
                            c["matched"]["content_enjoyment"]["mean"] if c["matched"]["content_enjoyment"] else -100.), reverse=True)
    selected = list(dict.fromkeys(c["condition"] for c in pool))[:max_selected]
    conditions = [condition()] + [specs[k] for k in selected]
    return {"phase": "validation", "conditions": conditions, "rows": rows_for("validation", conditions, 20),
            "selected_conditions": selected, "diagnostic_only": not qualified,
            "source_statistics": [c for c in pool if c["condition"] in selected],
            "interpretation": "Fresh text/seeds, same speaker/reference. Improvement requires quality gates and positive ECAPA lower paired-bootstrap bound on this phase; diagnostic fallback is not evidence of benefit."}


def append_log(path, event, **payload):
    """Append-only exhaustive JSONL event ledger; atomic JSON snapshots optional."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    value = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def frozen_json(path, value):
    """Refuse accidental redesign/resampling once a study snapshot is frozen."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value: raise ValueError(f"Frozen study artifact changed: {path.name}")
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def language_diagnostics(rows, baseline="baseline", repetitions=4000, by_language=True):
    """Additional language/anchor analysis, not a retrospective selection rule.

    Preserve the frozen row-bootstrap analysis and additionally resample whole
    generation batches where provenance permits. This avoids treating sibling
    rows sharing a batch RNG as unquestionably independent experimental runs.
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("phase", "unknown"), row["condition"], row.get("language", "unknown") if by_language else "pooled")].append(row)
    controls = {(r.get("phase", "unknown"), r.get("language", "unknown") if by_language else "pooled", r["block"]): r
                for r in rows if r["condition"] == baseline}
    output = []
    for (phase, condition_id, language), group in sorted(grouped.items()):
        item = {"phase": phase, "condition": condition_id, "language": language, "n": len(group),
                "failures": sum(bool(r.get("failed") or r.get("status") == "failed") for r in group),
                "means": {}, "matched": {}, "batch_clustered": {}}
        for metric in ("ecapa", "anchor_ecapa", "content_enjoyment", "wer", "emotion_score"):
            diffs, clusters, measured = [], [], []
            for r in group:
                if r.get("failed") or r.get("status") == "failed" or (metric == "emotion_score" and r.get("emotion") == "neutral"):
                    continue
                value = _metric(r, metric)
                if value is not None: measured.append(value)
                control = controls.get((phase, language, r["block"]))
                if control is None or control.get("failed") or control.get("status") == "failed": continue
                target = _metric(control, metric)
                if value is None or target is None: continue
                diffs.append(value-target)
                clusters.append(str(tuple(r["batch_ids"])) if r.get("batch_ids")
                                else str(r.get("batch_seed", f'row-{r["block"]}')))
            item["means"][metric] = float(np.mean(measured)) if measured else None
            item["matched"][metric] = paired_interval(diffs, repetitions)
            if diffs:
                cluster_values = defaultdict(list)
                for cluster, diff in zip(clusters, diffs): cluster_values[cluster].append(diff)
                cluster_sums = np.array([sum(v) for v in cluster_values.values()])
                cluster_sizes = np.array([len(v) for v in cluster_values.values()])
                rng = np.random.default_rng(405)
                indices = rng.integers(0, len(cluster_sizes), (repetitions, len(cluster_sizes)))
                samples = cluster_sums[indices].sum(axis=1)/cluster_sizes[indices].sum(axis=1)
                item["batch_clustered"][metric] = {"mean": float(np.mean(diffs)), "low": float(np.quantile(samples, .025)),
                    "high": float(np.quantile(samples, .975)), "n_pairs": len(diffs), "n_clusters": len(cluster_sizes)}
            else:
                item["batch_clustered"][metric] = None
        output.append(item)
    return output


def audit_validation(rows, validation_plan, prior_rows, repetitions=20000):
    """Independent raw-score protocol audit; not a GPU replay or new selection.

    Recomputes intervals directly from saved numeric fields rather than using
    summarize() gate output. The caller may freeze this diagnostic artifact.
    """
    expected = {r["id"]:r for r in validation_plan["rows"]}
    if len(rows) != len(expected) or {r["id"] for r in rows} != set(expected):
        raise ValueError("Validation results missing, duplicated or outside frozen plan")
    if not set(r["text"] for r in rows).isdisjoint(r["text"] for r in prior_rows):
        raise ValueError("Validation text overlaps prior experiments")
    if not set(r["seed"] for r in rows).isdisjoint(r["seed"] for r in prior_rows):
        raise ValueError("Validation design seed overlaps prior experiments")
    controls = {r["block"]:r for r in rows if r["condition"]=="baseline"}
    groups = defaultdict(list)
    for row in rows:
        for field in ("phase", "condition", "block", "seed", "language", "emotion", "text", "prompt", "intervention"):
            if row[field] != expected[row["id"]][field]: raise ValueError("Frozen field mismatch: "+field)
        if row.get("failed") or row.get("status")=="failed": raise ValueError("Validation includes a retained failure")
        if row.get("adapters") != [] or row.get("cfg") != 2. or row.get("temperature") != 1.:
            raise ValueError("Unexpected model/sampling configuration")
        if row.get("checkpoint", {}).get("stage") != "S3-Ref15-FullFT" or row["checkpoint"].get("step") != 26774:
            raise ValueError("Unexpected Full-FT checkpoint metadata")
        control = controls[row["block"]]
        for field in ("batch_seed", "batch_size", "checkpoint", "reference_sha256"):
            if row[field] != control[field]: raise ValueError("Matched control configuration mismatch: "+field)
        if row["acoustic_tokens"] != 12*row["generated_frames"]:
            raise ValueError("Generated frame/acoustic-token count mismatch")
        groups[row["condition"]].append(row)
    result = {"audit":"PASS", "validation_rows":len(rows), "prior_rows":len(prior_rows),
        "fresh_text_and_design_seeds":True, "frozen_labels_and_interventions_match":True,
        "all_matched_batch_seed_size_checkpoint_reference_match":True,
        "full_ft_no_adapters_cfg2_temperature1":True, "frame_token_count_consistent":True,
        "failures":0, "bootstrap_repetitions":repetitions, "bootstrap_seed":901, "conditions":[],
        "notes":["Metadata/protocol and independently recomputed score audit; no fresh GPU inference or hook replay.",
                 "Four prospective candidates, twenty matched takes each plus twenty controls. Only five four-row generation batches per condition.",
                 "Exploratory 95% and 99% intervals; no universal identity claim or formal family-wise corrected proof.",
                 "Different-file anchor shares corpus label and is not independently verified identity ground truth.",
                 "Positive emotion/aesthetics/WER mean changes do not imply individually significant perceptual improvements."]}
    for key, group in sorted(groups.items()):
        group.sort(key=lambda r:r["block"])
        item = {"condition":key, "n":len(group), "mean_ecapa":float(np.mean([r["ecapa"] for r in group])),
                "mean_wer":float(np.mean([r["wer"] for r in group])), "deltas":{}, "languages":{}}
        chunks = defaultdict(list)
        for row in group: chunks[row["batch_seed"]].append(row["ecapa"]-controls[row["block"]]["ecapa"])
        cluster = np.array([np.mean(v) for v in chunks.values()])
        rng = np.random.default_rng(901)
        means = cluster[rng.integers(0,len(cluster),(repetitions,len(cluster)))].mean(axis=1)
        item["cluster_ecapa_CI95"] = np.quantile(means,[.025,.975]).tolist()
        item["cluster_ecapa_means"] = cluster.tolist()
        item["cluster_count"] = len(cluster)
        for metric in ("ecapa","anchor_ecapa","content_enjoyment","production_quality","mos","aesthetics","wer","emotion_score"):
            diffs = np.array([r[metric]-controls[r["block"]][metric] for r in group
                if r.get(metric) is not None and controls[r["block"]].get(metric) is not None])
            if not len(diffs): item["deltas"][metric]=None; continue
            samples = diffs[rng.integers(0,len(diffs),(repetitions,len(diffs)))].mean(axis=1)
            item["deltas"][metric] = {"mean":float(diffs.mean()),"n_pairs":len(diffs),
                "CI95":np.quantile(samples,[.025,.975]).tolist(), "CI99":np.quantile(samples,[.005,.995]).tolist()}
        for lang in ("en","de"):
            subset = [r for r in group if r["language"]==lang]
            if subset:
                item["languages"][lang]={m:float(np.mean([r[m]-controls[r["block"]][m] for r in subset]))
                    for m in ("ecapa","anchor_ecapa","content_enjoyment","wer")}
        result["conditions"].append(item)
    return result


def render_report(path, plan, rows, summary=None, vectors_metadata=None, audio_root=None,
                  evidence=None, narrative=None, reference_audio=None, downloads=None):
    """English offline report with held-out results first and every take retained.

    Optional evidence maps names to JSON-compatible supplementary artifacts;
    narrative is a string or list of factual paragraphs, escaped as plain text.
    reference_audio maps labels to MP3 Paths (or path/caption dictionaries);
    downloads maps link labels to explicitly provided relative/HTTPS URLs.
    This function never modifies the frozen design or any experiment result.
    """
    summary = summary or summarize(rows)
    audio_root = Path(audio_root) if audio_root is not None else Path(path).parent
    def audio_file(row):
        filename = row.get("audio_path") or row.get("audio")
        if not filename:
            return None
        candidate = Path(filename)
        if not candidate.is_absolute(): candidate = audio_root / candidate
        return candidate if candidate.is_file() else None
    esc = lambda value: html.escape(str(value), quote=True)
    f = lambda value, digits=3: "—" if value is None else f"{value:.{digits}f}"
    evidence = dict(evidence or {})
    # Known measurement evidence, never arbitrary reference/upload directories.
    for filename in ("measurement-amendment.json", "reference-controls.json", "semantic-bridge-diagnostic.json"):
        artifact = audio_root/filename
        if filename not in evidence and artifact.is_file():
            evidence[filename] = json.loads(artifact.read_text())
    diagnostics = language_diagnostics(rows)
    validation_rows = [r for r in rows if r.get("phase") == "validation"]
    pooled_validation = language_diagnostics(validation_rows, by_language=False)
    validation = [c for c in summary["conditions"] if c["phase"] == "validation"]
    winners = [c["condition"] for c in validation if c.get("heldout_positive_interval")]
    failures = sum(bool(r.get("failed") or r.get("status") == "failed") for r in rows)
    def interval(d):
        return "—" if d is None else f'{d["mean"]:+.3f} [{d["low"]:+.3f}, {d["high"]:+.3f}]'
    def table(cells):
        result = ['<div class="scroll" tabindex="0" aria-label="Scrollable results table"><table><tr><th>Phase / condition</th><th>N / failures</th><th>ECAPA</th><th>Δ ECAPA [95% paired CI]</th><th>CE</th><th>PQ / MOS</th><th>Aesthetics</th><th>Emotion intensity</th><th>WER</th><th>Δ WER</th><th>Gate</th></tr>']
        for c in cells:
            m = c["means"]
            wd = c["matched"]["wer"]
            gate = "Matched control" if c["condition"] == "baseline" else "PASS" if c["eligible"] else "FAIL: " + ", ".join(c["gate_reasons"])
            result.append(f'<tr><td>{esc(c["phase"])} / {esc(c["condition"])}</td><td>{c["n"]} / {c["failures"]}</td><td>{f(m["ecapa"])}</td><td>{interval(c["matched"]["ecapa"])}</td><td>{f(m["content_enjoyment"],2)}</td><td>{f(m["production_quality"],2)} / {f(m["mos"],2)}</td><td>{f(m["aesthetics"],2)}</td><td>{f(m["emotion_score"],2)}</td><td>{f(100*m["wer"],1) if m["wer"] is not None else "—"}%</td><td>{f(wd["mean"] if wd else None)}</td><td>{esc(gate)}</td></tr>')
        result.append('</table></div>')
        return result
    def language_table(cells):
        result = ['<div class="scroll" tabindex="0" aria-label="Scrollable language diagnostic table"><table><tr><th>Phase / condition / language</th><th>N / failures</th><th>ECAPA Δ</th><th>Anchor Δ</th><th>CE Δ</th><th>WER Δ</th><th>Emotion Δ</th><th>Batch-cluster ECAPA Δ</th></tr>']
        for c in cells:
            b = c["batch_clustered"]["ecapa"]
            batch_text = interval(b) + (f' ({b["n_clusters"]} batches)' if b else '')
            result.append(f'<tr><td>{esc(c["phase"])} / {esc(c["condition"])} / {esc(c["language"])}</td><td>{c["n"]} / {c["failures"]}</td>' + ''.join(f'<td>{interval(c["matched"][m])}</td>' for m in ("ecapa", "anchor_ecapa", "content_enjoyment", "wer", "emotion_score")) + f'<td>{batch_text}</td></tr>')
        result.append('</table></div>')
        return result
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Mira activation steering — exploratory study</title><style>*{box-sizing:border-box}html{max-width:100%}body{font:16px system-ui;width:100%;max-width:1300px;margin:auto;padding:clamp(12px,3vw,20px);background:#f5f5f6;color:#20232a}h1{font-size:1.9em}h1,h2,h3,p,summary{overflow-wrap:anywhere}.scroll{overflow:auto;max-width:100%;min-width:0}table{border-collapse:collapse;font-size:14px;width:100%;min-width:1050px}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left;white-space:nowrap}td:first-child,td:last-child{white-space:normal;overflow-wrap:anywhere;min-width:180px;max-width:320px}article,.callout{min-width:0;background:white;border:1px solid #ddd;border-radius:10px;padding:16px;margin:12px 0}article[hidden]{display:none}audio{width:100%;max-width:600px;min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-width:100%}.filters{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(240px,100%),1fr));gap:10px}.filters label{min-width:0}select{display:block;width:100%;max-width:100%;min-width:0;padding:8px;margin:5px 0}p{line-height:1.5}.bad{color:#a02020}.details{font-size:14px}summary{cursor:pointer}</style>',
             '<h1>Mira activation-mean steering</h1><p>Exploratory study of one fixed Mira reference with the newest Full-FT base model, no DPO or other LoRAs, CFG 2.0 and temperature 1.0. Collection, discovery and held-out validation use disjoint text pools and seed ranges. No winner-only listening gallery: every take and failure is retained.</p>',
             f'<p><strong>{len(rows)} recorded takes · {failures} generation failures.</strong> Quality scores are automated proxies, not a blind listening panel.</p>',
             '<section class="callout"><h2>Held-out conclusion</h2>']
    if downloads:
        from urllib.parse import urlsplit
        links = []
        for label, href in downloads.items():
            href = str(href).strip()
            if urlsplit(href).scheme not in ("", "http", "https"):
                raise ValueError("Download links must be relative or HTTP(S) URLs")
            links.append(f'<a href="{esc(href)}" download>{esc(label)}</a>')
        parts.insert(-1, '<nav aria-label="Study downloads"><p>Download: ' + ' · '.join(links) + '</p></nav>')
    if winners:
        parts.append('<p>' + esc(", ".join(winners)) + ' passed the declared safeguards with positive conditioning-reference ECAPA intervals on fresh validation text/seeds. This is a result for one voice/reference, not a universal speaker-cloning improvement. Check the independent-file anchor and language diagnostics below for reference-specific overfitting.</p>')
    elif validation:
        parts.append('<p><strong>No validated improvement has been established.</strong> The held-out candidates did not jointly clear the speaker-gain and quality/reliability safeguards. Do not enable these steering vectors as production defaults on this evidence.</p>')
    else:
        parts.append('<p><strong>No validated improvement has been established.</strong> Held-out validation is not yet available; discovery results alone cannot justify a default change.</p>')
    if narrative:
        for paragraph in ([narrative] if isinstance(narrative, str) else narrative):
            parts.append('<p>' + esc(paragraph) + '</p>')
    parts.append('</section>')
    reference_count = 0
    if reference_audio:
        parts.append('<section class="callout reference-comparison"><h2>Compare the reference recordings</h2><p>The conditioning clip is the fixed synthesis reference. The different-file anchor was not used for synthesis; it has the same corpus label, but its true speaker/session identity is not independently verified. These comparison players are not experimental takes.</p>')
        for label, item in reference_audio.items():
            caption = item.get("caption", "") if isinstance(item, dict) else ""
            filename = item["path"] if isinstance(item, dict) else item
            candidate = Path(filename)
            if not candidate.is_absolute(): candidate = audio_root/candidate
            parts.append('<h3>' + esc(label) + '</h3>')
            if caption: parts.append('<p>' + esc(caption) + '</p>')
            if candidate.is_file():
                mime = {".mp3":"audio/mpeg", ".wav":"audio/wav", ".ogg":"audio/ogg"}.get(candidate.suffix.lower())
                if mime is None: raise ValueError("Reference comparison audio must be MP3, WAV or OGG")
                encoded = base64.b64encode(candidate.read_bytes()).decode("ascii")
                parts.append(f'<audio controls preload="none" src="data:{mime};base64,{encoded}"></audio>')
                reference_count += 1
            else:
                parts.append('<p class="bad">Reference comparison audio is unavailable.</p>')
        parts.append('</section>')
    parts.append('<h2>Fresh held-out validation</h2>')
    parts.extend(table(validation) if validation else ['<p>Pending / not present in this report.</p>'])
    parts.append('<p>Gate: ECAPA gain ≥0.01; CE and PQ drop ≤0.30; MOS drop ≤0.20; aesthetics drop ≤0.30 when measured; target-emotion drop ≤0.20; WER rise ≤0.03 absolute, mean WER ≤0.15, WER&gt;0.25 outlier-rate rise ≤0.10; no generation failures or missing required measurements. A positive 95% paired ECAPA interval is additionally required for the held-out improvement statement. All intervals are exploratory, without multiple-comparison correction.</p>')
    parts.append('<h2>Validation robustness: whole-batch bootstrap</h2>')
    parts.extend(language_table(pooled_validation) if pooled_validation else ['<p>Pending / not present in this report.</p>'])
    parts.append('<p>The final column resamples actual recorded generation batches as clusters, retaining all their paired rows together. Twenty validation takes generated in five four-row batches have five resampling units, not twenty. This conservative supplementary diagnostic does not silently replace the frozen row-bootstrap selection criterion.</p>')
    parts.append('<h2>Validation by language and independent-file anchor</h2>')
    heldout_diagnostics = [d for d in diagnostics if d["phase"] == "validation"]
    parts.extend(language_table(heldout_diagnostics) if heldout_diagnostics else ['<p>Pending / not present in this report.</p>'])
    parts.append('<p>Each delta uses the same phase, language, text block and no-steering control. WER deltas are absolute fractions, not relative percentages; emotion excludes neutral trials. The anchor is a different file never used to condition generation, but belongs to the same corpus label: it is not independently verified speaker/session ground truth. An increase against the conditioning reference alone may reflect recording-specific imitation rather than stable identity.</p>')
    parts.append('<h2>Method and known confounds</h2><p><strong>Precise capture:</strong> a hidden state is counted only when its prediction actually emits an acoustic frame. The last conditional prefill position is included only if it predicts the first emitted frame; all other prompt/reference prefill positions, EOS, already-finished rows and padded tails are excluded. Per-sample prediction counts are saved. Means weight samples equally within emotion, then weight contributing emotions equally. Both CFG branches receive the intervention. Multi-layer conditions share one total alpha budget across hooks.</p>')
    parts.append('<p><strong>Not an isolated speaker representation:</strong> raw means can contain text, phonetics, language, emotion, acoustic timing and generic residual-stream bias. A high-minus-low contrast is only a better-controlled candidate direction, not proof of speaker specificity. ECAPA cosine is neither an identity probability nor a percentage of cloning quality.</p>')
    selected_meta = (vectors_metadata or {}).get("selected", vectors_metadata or {})
    high_ids = set(selected_meta.get("high_ids", []))
    high_languages = Counter(r.get("language", "unknown") for r in rows if r["id"] in high_ids)
    if high_languages:
        counts = ", ".join(f'{n} {lang.upper()}' for lang, n in sorted(high_languages.items()))
        parts.append('<p><strong>Selection-language confound:</strong> the selected high-similarity pool contains ' + esc(counts) + f' at strict cosine &gt;{esc(selected_meta.get("threshold", "unknown"))}. Emotion balancing does not remove this language/phonetic/session imbalance. A secondary joint-balanced &gt;0.25 diagnostic, if run, is a separate exploratory intervention and must not silently replace the frozen primary vector.</p>')
    else:
        parts.append('<p>High-similarity pool counts and language coverage must be checked before interpreting an averaged activation as speaker-specific. Emotion balancing alone does not balance languages or recording sessions.</p>')
    parts.append('<p><strong>Semantic-site equivalence:</strong> semantic_global is the post-Qwen-normalization 1024-dimensional vector; semantic_local is its 2560-dimensional bias-free projection into the acoustic decoder. Their raw means are approximately linearly related, so the two additions are not independent mechanisms. BF16 projection/addition rounding can still produce diverging autoregressive trajectories; RMS-relative doses change intervention amplitude.</p>')
    parts.append('<p><strong>Batch RNG and uncertainty:</strong> rows within a generated batch share one RNG, despite separate design seed labels. Actual batch seeds and companion IDs are recorded, and matched conditions preserve grouping and batch seed. These are not 100 separately reseeded generator invocations. The declared intervals resample matched rows; supplementary diagnostics additionally resample whole generation batches when batch provenance exists. Few batches and repeated task templates limit precision and generalization.</p>')
    if evidence:
        amendment = evidence.get("measurement-amendment.json", evidence.get("measurement_amendment"))
        if amendment:
            parts.append('<h3>Measurement amendment — no generation/vector resampling</h3><p>' + esc(amendment.get("reason", "Recorded measurement correction.")) + ' ' + esc(amendment.get("change", "")) + '</p><p>The original padded-batch style scores remain in the ledger. This correction did not alter ECAPA, WER, Ears, captured activation arrays, selected threshold or sample IDs.</p>')
        for name, value in evidence.items():
            parts.append('<details><summary>' + esc(name) + '</summary><pre>' + esc(json.dumps(value, ensure_ascii=False, indent=2)) + '</pre></details>')
    parts.append('<h2>Full grid — including negative conditions</h2>')
    parts.extend(table(summary["conditions"]))
    parts.append('<details><summary>All phases: language-specific paired and anchor diagnostics</summary>')
    parts.extend(language_table(diagnostics))
    parts.append('</details>')
    parts.append('<details><summary>Frozen protocol, limits and selection gates</summary><pre>' + esc(json.dumps(plan, ensure_ascii=False, indent=2)) + '</pre></details>')
    if vectors_metadata:
        parts.append('<details><summary>Vector selection, threshold counts and amplitudes</summary><pre>' + esc(json.dumps(vectors_metadata, ensure_ascii=False, indent=2)) + '</pre></details>')
    parts.append('<h2>Listen to every take — embedded Base64 MP3</h2><p>Compare matched blocks/seeds rather than picking only attractive outputs. Audio is embedded in this page and remains playable offline. Choose a phase and block to compare the same task across conditions.</p><div class="filters">')
    for key in ("phase", "condition", "emotion", "language", "block"):
        options = sorted(set(str(r.get(key, "unknown")) for r in rows))
        parts.append(f'<label>{esc(key.title())} <select data-filter="{key}"><option value="">All</option>' + ''.join(f'<option value="{esc(o)}">{esc(o)}</option>' for o in options) + '</select></label>')
    parts.append('</div><p id="count"></p>')
    for row in rows:
        attrs = ' '.join(f'data-{k}="{esc(row.get(k,"unknown"))}"' for k in ("phase", "condition", "emotion", "language", "block"))
        parts.append(f'<article {attrs}><h3>{esc(row["id"])}</h3><p>{esc(row.get("text",""))}</p><p class="details">' + ' · '.join(f'{esc(m)}: {f(_metric(row,m))}' for m in METRICS) + '</p>')
        audio = audio_file(row)
        if audio is not None:
            data = base64.b64encode(audio.read_bytes()).decode("ascii")
            parts.append(f'<audio controls preload="none" src="data:audio/mpeg;base64,{data}"></audio>')
        else:
            parts.append('<p class="bad">No playable audio: ' + esc(row.get("error", "missing audio artifact")) + '</p>')
        if row.get("failed") or row.get("status") == "failed":
            parts.append('<p class="bad">Generation failure retained: ' + esc(row.get("error", "unknown")) + '</p>')
        parts.append('<details><summary>Prompt and intervention</summary><pre>' + esc(row.get("prompt", "")) + '\n' + esc(json.dumps(row.get("intervention", {}), indent=2)) + '</pre></details></article>')
    parts.append('<script>const selects=[...document.querySelectorAll("select[data-filter]")],takes=[...document.querySelectorAll("article")];function filter(){let n=0;for(const t of takes){t.hidden=selects.some(s=>s.value&&t.dataset[s.dataset.filter]!==s.value);if(!t.hidden)n++;}document.querySelector("#count").textContent=n+" of "+takes.length+" takes shown";}selects.forEach(s=>s.addEventListener("change",filter));filter();</script></html>')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(parts), encoding="utf-8")
    return {"takes": len(rows), "audio_embedded": sum(audio_file(r) is not None for r in rows),
            "validated_candidates": winners, "language_diagnostic_cells": len(diagnostics),
            "supplementary_evidence": sorted(evidence), "reference_audio_embedded": reference_count}
