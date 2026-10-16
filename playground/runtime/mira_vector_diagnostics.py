"""CPU-only, independent diagnostics for generated-position Mira means.

These diagnostics do NOT change the frozen intervention study or its selected
pool. Selection here is collection-only. A strict ECAPA cosine >.25 is not a
probability or percentage of speaker identity. All amplitudes describe saved
per-sample token means, not instantaneous frame activations.

Optional quality-filtered directions and shuffled/null controls are additional
exploratory hypotheses, never substitutes for held-out audio-generation tests.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np


def cosine(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator > 1e-12 else None


def auc(labels, scores):
    """Rank AUC, with ties half credit; withheld if either class is absent."""
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    high, low = scores[labels], scores[~labels]
    if not len(high) or not len(low):
        return None
    difference = high[:, None] - low[None, :]
    return float(((difference > 0) + .5 * (difference == 0)).mean())


def bridge_diagnostic(weight, global_vectors, local_vectors):
    """Audit the bias-free global->local projection on CPU saved means.

    FP32 linear reconstruction is intentionally not asserted bitwise identical
    to tokenwise BF16 projection followed by averaging; report rounding errors.
    Input rows can be individual-sample means or aggregated candidate vectors.
    """
    weight = np.asarray(weight, dtype=np.float32)
    left, right = np.asarray(global_vectors, dtype=np.float32), np.asarray(local_vectors, dtype=np.float32)
    if left.ndim == 1: left = left[None]
    if right.ndim == 1: right = right[None]
    if weight.ndim != 2 or left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("Expected projection [local, global] and matching vector rows")
    if weight.shape != (right.shape[1], left.shape[1]):
        raise ValueError("Projection dimensions do not match captured global/local points")
    reconstructed = left @ weight.T
    error = reconstructed.astype(np.float64) - right
    relative = np.linalg.norm(error, axis=-1) / np.maximum(np.linalg.norm(right, axis=-1), 1e-12)
    cosines = [cosine(a, b) for a, b in zip(reconstructed, right)]
    valid = [c for c in cosines if c is not None]
    return {"n": len(left), "global_dim": left.shape[1], "local_dim": right.shape[1],
            "cosine_mean": float(np.mean(valid)) if valid else None,
            "cosine_min": float(np.min(valid)) if valid else None,
            "relative_l2_mean": float(np.mean(relative)), "relative_l2_max": float(np.max(relative)),
            "maximum_absolute_error": float(np.abs(error).max()),
            "note": "Global/local raw additions are linearly related; BF16 projection/averaging rounding can cause autoregressive divergence."}


def _balanced_mean(values, records, mask):
    emotions = sorted({r["emotion"] for r, keep in zip(records, mask) if keep})
    if not emotions:
        return None
    return np.mean([values[np.array([keep and r["emotion"] == emotion for r, keep in zip(records, mask)])].mean(0)
                    for emotion in emotions], axis=0)


def _metric(row, key):
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None


def _loo_emotion_auc(values, records, labels):
    """High-minus-low direction trained outside each requested emotion.

    Report equally averaged held-out emotion AUCs. This is feature diagnostic
    discrimination on collected sample means, NOT proof that injection helps.
    """
    emotions = sorted({r["emotion"] for r in records})
    results = {}
    for emotion in emotions:
        holdout = np.array([r["emotion"] == emotion for r in records])
        high = _balanced_mean(values, records, labels & ~holdout)
        low = _balanced_mean(values, records, ~labels & ~holdout)
        if high is None or low is None:
            results[emotion] = None
            continue
        direction = high - low
        results[emotion] = auc(labels[holdout], values[holdout] @ direction)
    valid = [v for v in results.values() if v is not None]
    return {"per_emotion": results, "mean": float(np.mean(valid)) if valid else None,
            "tested_emotions": len(valid)}


def _count_groups(records, mask, keys):
    counts = {}
    for record, keep in zip(records, mask):
        if keep:
            name = "/".join(str(record.get(key, "unknown")) for key in keys)
            counts[name] = counts.get(name, 0) + 1
    return counts


def _language_emotion_balanced_mean(values, records, mask, min_per_cell=2):
    """Withhold full-cell balanced mean if any observed cell lacks coverage."""
    cells = sorted({(r.get("language", "unknown"), r["emotion"]) for r in records})
    subsets = [np.array([keep and (r.get("language", "unknown"), r["emotion"]) == cell
                         for r, keep in zip(records, mask)]) for cell in cells]
    if any(subset.sum() < min_per_cell for subset in subsets):
        return None
    return np.mean([values[subset].mean(0) for subset in subsets], axis=0)


def balanced_secondary_pool(records, activation_arrays, *, threshold=.25, min_per_cell=2,
                            languages=("en", "de"), emotions=("neutral", "joy", "anger", "sadness", "fear")):
    """Explicit secondary direction pool, never changes the primary .50 pool.

    Weight each language×requested-emotion cell equally, then each qualifying
    sample within its cell equally. Withhold high-minus-low unless ALL expected
    high AND low cells contain at least ``min_per_cell`` examples. A missing
    low cell is not imputed with all-sample, another language, or another emotion.
    """
    rows = [r for r in records if r.get("phase", "collection") == "collection"
            and not r.get("failed") and r.get("status", "ok") != "failed"
            and r["id"] in activation_arrays and _metric(r, "ecapa") is not None]
    if not rows:
        raise ValueError("No successful collection samples with activations")
    expected = {(language, emotion) for language in languages for emotion in emotions}
    observed = {(r.get("language", "unknown"), r["emotion"]) for r in rows}
    if observed - expected:
        raise ValueError("Unexpected language/emotion cells; declare another secondary pool")
    high = np.array([r["ecapa"] > threshold for r in rows])
    low = ~high
    all_mask = np.ones(len(rows), bool)
    def counts(mask):
        return {language + "/" + emotion: sum(bool(keep) and r.get("language") == language and r["emotion"] == emotion
                                              for r, keep in zip(rows, mask))
                for language, emotion in sorted(expected)}
    all_counts, high_counts, low_counts = counts(all_mask), counts(high), counts(low)
    eligible_high = all(n >= min_per_cell for n in high_counts.values())
    eligible_low = all(n >= min_per_cell for n in low_counts.values())
    metadata = {"diagnostic_only": True, "primary_selected_vectors_unchanged": True,
                "threshold": threshold, "comparison": "strict ECAPA cosine > threshold; not percent identity",
                "minimum_samples_per_cell": min_per_cell, "high_n": int(high.sum()), "low_n": int(low.sum()),
                "joint_cell_counts": {"all": all_counts, "high": high_counts, "low": low_counts},
                "eligible_high": eligible_high, "eligible_low": eligible_low,
                "raw_mean_available": eligible_high, "high_minus_all_available": eligible_high,
                "contrast_available": eligible_high and eligible_low,
                "missing_high_cells": [k for k, n in high_counts.items() if n < min_per_cell],
                "missing_low_cells": [k for k, n in low_counts.items() if n < min_per_cell],
                "high_ids": [r["id"] for r, keep in zip(rows, high) if keep],
                "low_ids": [r["id"] for r, keep in zip(rows, low) if keep],
                "weighting": "Equal expected language×emotion cells; equal sample within each cell; generated-predictor token means",
                "points": {},
                "caveats": ["Additional exploratory diagnostic: primary frozen mean is not changed.",
                            "High-minus-all has a smaller natural magnitude than the raw mean; raw-alpha effects are not norm-matched.",
                            "A balanced language/requested-emotion mean still contains linguistic/acoustic/recording biases.",
                            "Withheld contrast is not silently reconstructed from incomplete low-cell coverage."]}
    if not eligible_high:
        return {}, metadata
    points = sorted(set.intersection(*(set(activation_arrays[r["id"]]) for r in rows)))
    vectors = {}
    for point in points:
        if point == "frame_count": continue
        values = np.stack([np.asarray(activation_arrays[r["id"]][point], dtype=np.float64) for r in rows])
        if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
            raise ValueError(f"Invalid point matrix: {point}")
        high_mean = _language_emotion_balanced_mean(values, rows, high, min_per_cell)
        all_mean = _language_emotion_balanced_mean(values, rows, all_mask, min_per_cell)
        vectors[point + "__raw_mean"] = high_mean.astype(np.float32)
        vectors[point + "__high_minus_all"] = (high_mean - all_mean).astype(np.float32)
        vectors[point + "__all_mean"] = all_mean.astype(np.float32)
        if eligible_low:
            low_mean = _language_emotion_balanced_mean(values, rows, low, min_per_cell)
            vectors[point + "__contrast"] = (high_mean - low_mean).astype(np.float32)
        metadata["points"][point] = {"dimension": values.shape[1], "raw_mean_rms": float(np.sqrt(np.mean(high_mean ** 2))),
                                     "high_minus_all_rms": float(np.sqrt(np.mean((high_mean - all_mean) ** 2))),
                                     "raw_vs_all_cosine": cosine(high_mean, all_mean)}
    return vectors, metadata


def layer_contrast_plan(root):
    """Freezeable 64-take follow-up to discovery-only raw-layer screening.

    No GPU work. The primary strict >.50 high-minus-low bank is retained,
    including its disclosed 22-DE/1-EN selection imbalance. Conditions reuse
    the study's eight fresh refinement task/seed blocks, not collection/screen
    tasks. Eight blocks cover four emotions; held-out final validation must
    restore the complete five-emotion coverage.
    """
    import hashlib
    import mira_activation_analysis as analysis
    root = Path(root).resolve()
    bank = root / "vectors" / "selected.npz"
    metadata = json.loads((bank.parent / "metadata.json").read_text())
    conditions = [analysis.condition()]
    for layer in (20, 16, 17):
        for alpha in (.25, .75):
            conditions.append(analysis.condition([layer], alpha, "contrast"))
    conditions.append(analysis.condition([20, 16], .5, "contrast"))
    with np.load(bank, allow_pickle=False) as archive:
        for layer in (20, 16, 17):
            if f"layer_{layer}__contrast" not in archive:
                raise ValueError(f"Primary bank lacks layer_{layer} contrast")
    blocks = analysis.blocks("refine", 8)
    rows = []
    for condition in conditions:
        condition["vector_file"] = str(bank)
        condition["vector_kind"] = "contrast"
        for block in blocks:
            rows.append({**block, "phase": "layer_contrast", "condition": condition["id"],
                         "id": f'layer_contrast-{condition["id"]}-{block["block"]}', "intervention": dict(condition)})
    return {"version": 1, "phase": "layer_contrast", "created_utc": "frozen-before-generation",
            "rows": rows, "conditions": conditions, "planned_takes": len(rows),
            "vector_file": str(bank), "vector_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
            "source_vector_metadata": metadata,
            "rationale": "Coarse raw-addition discovery finds layer20 and16 speaker gains but emotion losses; layer17 shows gain with WER/emotion costs. Test whether removal of generic common activation via high-minus-low helps. A shared-dose20+16 pair tests multi-layer distribution without doubling total alpha.",
            "sampling": {"cfg": 2., "temperature": 1., "duration": "automatic", "max_frames": 256,
                         "reference": "fixed Mira", "adapters": [], "batch_size": 4, "global_eager": True,
                         "local_cuda_graph": False, "quality_retries": False},
            "combination_rule": "Total alpha divided equally across selected layers; pair alpha.5 means .25 per layer.",
            "selection_gates": analysis.GATES,
            "limitations": ["Exploratory diagnostic, not held-out validation or claim of a speaker-only vector.",
                            "Source >.50 pool is 22 German and one English; selected directions may encode language.",
                            "Same eight fresh-refinement task/seed blocks as other refinement diagnostics, never discovery or collection blocks.",
                            "Eight blocks cover neutral/joy/anger/sadness in both languages; fear absent here and required in final validation.",
                            "Raw contrast has different norm than raw mean: equal alpha does not mean equal perturbation magnitude.",
                            "All failures, regressions, audio and scores retained; no selective retry."]}


def diagnostics(records, activation_arrays, *, threshold=.25, ce_floor=5.5,
                wer_max=.15, pq_floor=5., seed=83419):
    """Return JSON diagnostics and extra CPU FP32 candidate/control vectors.

    ``activation_arrays`` maps sample ID to {point: np.ndarray[hidden_width]}.
    The diagnostic quality pool requires all listed metrics; absent values do
    not silently pass. Its thresholds are absolute and separately labeled.
    """
    records = [r for r in records if r.get("phase", "collection") == "collection"
               and not r.get("failed") and r.get("status", "ok") != "failed"
               and r["id"] in activation_arrays and _metric(r, "ecapa") is not None]
    if not records:
        raise ValueError("No successful collection rows with ECAPA and activations")
    labels = np.array([r["ecapa"] > threshold for r in records])
    quality = np.array([keep and _metric(r, "content_enjoyment") is not None
                        and r["content_enjoyment"] >= ce_floor
                        and _metric(r, "production_quality") is not None
                        and r["production_quality"] >= pq_floor
                        and _metric(r, "wer") is not None and r["wer"] <= wer_max
                        for r, keep in zip(records, labels)])
    points = sorted(set.intersection(*(set(activation_arrays[r["id"]]) for r in records)))
    output = {"diagnostic_only": True, "n": len(records), "threshold": threshold,
              "threshold_comparison": "strict ECAPA cosine > threshold; not calibrated probability",
              "high_n": int(labels.sum()), "low_n": int((~labels).sum()),
              "high_ids": [r["id"] for r, keep in zip(records, labels) if keep],
              "quality_high_n": int(quality.sum()),
              "quality_high_ids": [r["id"] for r, keep in zip(records, quality) if keep],
              "high_language_counts": _count_groups(records, labels, ("language",)),
              "low_language_counts": _count_groups(records, ~labels, ("language",)),
              "high_language_emotion_counts": _count_groups(records, labels, ("language", "emotion")),
              "quality_diagnostic_gates": {"content_enjoyment_min": ce_floor, "production_quality_min": pq_floor, "wer_max": wer_max},
              "points": {}, "ranking": [],
              "caveats": ["Collection feature discrimination is not evidence of causal steering benefit.",
                          "Directions also contain content, emotion, duration and generic model activation.",
                          "Perturbation ratios use per-sample token means, not instantaneous activation RMS.",
                          "Quality filtering is an additional diagnostic, not a replacement for the frozen pool.",
                          "Requested-emotion holdout is not an independent speaker/session test.",
                          "Cross-emotion discrimination may instead recover language; report within-language diagnostics and pool coverage.",
                          "Raw global and projected-local vectors are related by the linear proj_out bridge."]}
    languages = sorted({r.get("language", "unknown") for r in records})
    output["metrics_by_language"] = {}
    for language in languages:
        subset = [r for r in records if r.get("language", "unknown") == language]
        output["metrics_by_language"][language] = {}
        for metric in ("ecapa", "anchor_ecapa", "content_enjoyment", "production_quality", "wer"):
            measured = [_metric(row, metric) for row in subset]
            valid = [v for v in measured if v is not None]
            output["metrics_by_language"][language][metric] = float(np.mean(valid)) if valid else None
    rng = np.random.default_rng(seed)
    vectors = {}
    for point in points:
        if point == "frame_count":
            continue
        values = np.stack([np.asarray(activation_arrays[r["id"]][point], dtype=np.float64) for r in records])
        if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
            raise ValueError(f"Invalid per-sample activation matrix: {point}")
        generic = _balanced_mean(values, records, np.ones(len(records), bool))
        high, low = _balanced_mean(values, records, labels), _balanced_mean(values, records, ~labels)
        sample_rms = np.sqrt(np.mean(values ** 2, axis=1))
        stats = {"dimension": values.shape[1], "sample_mean_rms_p10_median_p90": np.quantile(sample_rms, [.1, .5, .9]).tolist(),
                 "generic_mean_rms": float(np.sqrt(np.mean(generic ** 2))),
                 "loo_emotion_auc": _loo_emotion_auc(values, records, labels),
                 "loo_emotion_auc_within_language": {},
                 "raw_high_mean_rms": None, "raw_high_vs_generic_cosine": None,
                 "contrast_rms": None, "contrast_vs_generic_cosine": None,
                 "high_emotion_centroid_pair_cosines": {}, "raw_mean_dose_ratios": {}}
        for language in languages:
            mask = np.array([r.get("language", "unknown") == language for r in records])
            stats["loo_emotion_auc_within_language"][language] = _loo_emotion_auc(
                values[mask], [r for r, keep in zip(records, mask) if keep], labels[mask])
        vectors[point + "__generic_mean"] = generic.astype(np.float32)
        if high is not None:
            vectors[point + "__diagnostic_high_mean"] = high.astype(np.float32)
            stats["raw_high_mean_rms"] = float(np.sqrt(np.mean(high ** 2)))
            stats["raw_high_vs_generic_cosine"] = cosine(high, generic)
            stats["high_minus_generic_rms"] = float(np.sqrt(np.mean((high - generic) ** 2)))
            stats["raw_mean_dose_ratios"] = {str(alpha): float(alpha * stats["raw_high_mean_rms"] / max(float(np.median(sample_rms)), 1e-12))
                                             for alpha in (.25, .5, .75, 1.)}
            high_emotions = sorted({r["emotion"] for r, keep in zip(records, labels) if keep})
            centroids = {emotion: values[np.array([keep and r["emotion"] == emotion for r, keep in zip(records, labels)])].mean(0)
                         for emotion in high_emotions}
            stats["high_emotion_centroid_pair_cosines"] = {left + "/" + right: cosine(centroids[left], centroids[right])
                                                           for left, right in combinations(high_emotions, 2)}
            shuffled = high[rng.permutation(len(high))]
            random_null = rng.normal(size=len(high))
            random_null *= np.linalg.norm(high) / max(np.linalg.norm(random_null), 1e-12)
            vectors[point + "__shuffled_high_mean"] = shuffled.astype(np.float32)
            vectors[point + "__random_norm_matched_null"] = random_null.astype(np.float32)
        if high is not None and low is not None:
            contrast = high - low
            vectors[point + "__diagnostic_contrast"] = contrast.astype(np.float32)
            stats["contrast_rms"] = float(np.sqrt(np.mean(contrast ** 2)))
            stats["contrast_vs_generic_cosine"] = cosine(contrast, generic)
            residual = contrast - (contrast @ generic / max(generic @ generic, 1e-12)) * generic
            vectors[point + "__contrast_orthogonal_to_generic"] = residual.astype(np.float32)
            stats["contrast_orthogonal_rms"] = float(np.sqrt(np.mean(residual ** 2)))
        quality_mean = _balanced_mean(values, records, quality)
        if quality_mean is not None:
            vectors[point + "__quality_filtered_high_mean"] = quality_mean.astype(np.float32)
            stats["quality_filtered_vs_high_cosine"] = cosine(quality_mean, high)
            stats["quality_filtered_mean_rms"] = float(np.sqrt(np.mean(quality_mean ** 2)))
        language_balanced = _language_emotion_balanced_mean(values, records, labels)
        stats["language_emotion_balanced_mean_available"] = language_balanced is not None
        if language_balanced is not None:
            vectors[point + "__language_emotion_balanced_high_mean"] = language_balanced.astype(np.float32)
            stats["language_balanced_vs_primary_high_cosine"] = cosine(language_balanced, high)
        output["points"][point] = stats
    output["ranking"] = sorted(
        [{"point": point, "loo_emotion_auc": stats["loo_emotion_auc"]["mean"],
          "tested_emotions": stats["loo_emotion_auc"]["tested_emotions"]}
         for point, stats in output["points"].items()],
        key=lambda item: -1. if item["loo_emotion_auc"] is None else item["loo_emotion_auc"], reverse=True)
    output["suggested_diagnostic_experiments"] = [
        "Compare raw high mean with generic/all-sample mean at the same RMS: isolates specific selection from generic activation increase.",
        "Compare high-minus-low and its generic-orthogonal residual with raw high mean, using the frozen quality/WER gates.",
        "Use a coordinate-shuffled mean and random norm-matched null; improvement in these controls would weaken the speaker-direction interpretation.",
        "Compare additive h + alpha*v with interpolation (1-alpha)*h + alpha*v, carefully labeled because replacement can erase content.",
        "Prioritize late layers only if held-out-emotion separation supports them; confirm with fresh generated audio and never infer efficacy from AUC alone."]
    return output, vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--threshold", type=float, default=.25)
    parser.add_argument("--ce-floor", type=float, default=5.5)
    parser.add_argument("--pq-floor", type=float, default=5.)
    parser.add_argument("--wer-max", type=float, default=.15)
    parser.add_argument("--output-subdir", default="vector-diagnostics")
    args = parser.parse_args()
    paths = sorted((args.root / "results").glob("collection-*.json"))
    records = [json.loads(path.read_text()) for path in paths]
    arrays = {}
    for row in records:
        path = args.root / "activations" / (row["id"] + ".npz")
        if path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                arrays[row["id"]] = {key: archive[key].copy() for key in archive.files if key != "frame_count"}
    report, vectors = diagnostics(records, arrays, threshold=args.threshold,
                                  ce_floor=args.ce_floor, pq_floor=args.pq_floor, wer_max=args.wer_max)
    folder = args.root / args.output_subdir; folder.mkdir(exist_ok=True)
    (folder / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(folder / "diagnostic-vectors.npz", **vectors)
    print(json.dumps({"n": report["n"], "high_n": report["high_n"], "quality_high_n": report["quality_high_n"],
                      "top_points": report["ranking"][:8]}, indent=2))


if __name__ == "__main__":
    main()
