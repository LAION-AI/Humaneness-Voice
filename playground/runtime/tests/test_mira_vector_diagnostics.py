from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mira_vector_diagnostics import auc, balanced_secondary_pool, bridge_diagnostic, diagnostics, layer_contrast_plan


def fixture():
    rows, arrays = [], {}
    for emotion_index, emotion in enumerate(("neutral", "joy", "anger", "sadness", "fear")):
        for index in range(4):
            high = index >= 2
            id_ = f"{emotion}-{index}"
            rows.append({"id": id_, "phase": "collection", "emotion": emotion,
                         "language": "en",
                         "ecapa": .5 if high else .2, "content_enjoyment": 6., "production_quality": 6., "wer": .01})
            # Generic direction and requested emotion occupy unrelated axes.
            arrays[id_] = {"layer_0": np.array([10., float(high), float(emotion_index), 0.]),
                           "layer_1": np.array([10., 0., float(emotion_index), 0.])}
    return rows, arrays


def test_emotion_holdout_identifies_reproducible_high_low_separation():
    rows, arrays = fixture()
    report, vectors = diagnostics(rows, arrays)
    assert report["n"] == 20 and report["high_n"] == 10
    assert report["ranking"][0]["point"] == "layer_0"
    assert report["points"]["layer_0"]["loo_emotion_auc"]["mean"] == 1.
    assert report["points"]["layer_1"]["loo_emotion_auc"]["mean"] == .5
    assert report["quality_high_n"] == 10
    assert np.isclose(vectors["layer_0__generic_mean"] @ vectors["layer_0__contrast_orthogonal_to_generic"], 0, atol=1e-7)


def test_null_controls_preserve_vector_norm_but_not_coordinates():
    rows, arrays = fixture()
    _, vectors = diagnostics(rows, arrays)
    high = vectors["layer_0__diagnostic_high_mean"]
    for key in ("layer_0__shuffled_high_mean", "layer_0__random_norm_matched_null"):
        assert np.linalg.norm(vectors[key]) == pytest.approx(np.linalg.norm(high), rel=1e-6)
        assert vectors[key].dtype == np.float32
    assert not np.array_equal(vectors["layer_0__random_norm_matched_null"], high)


def test_strict_threshold_and_missing_quality_metrics_cannot_pass():
    rows, arrays = fixture()
    rows[0]["ecapa"] = .25
    del rows[2]["content_enjoyment"]
    rows[3]["wer"] = .20
    report, _ = diagnostics(rows, arrays)
    assert rows[0]["id"] not in report["high_ids"]
    assert report["high_n"] == 10
    assert report["quality_high_n"] == 8


def test_empty_pool_withheld_and_failures_not_imputed():
    rows, arrays = fixture()
    rows[0]["failed"] = True
    report, vectors = diagnostics(rows, arrays, threshold=.9)
    assert report["n"] == 19
    assert report["high_n"] == 0
    assert report["points"]["layer_0"]["raw_high_mean_rms"] is None
    assert "layer_0__diagnostic_high_mean" not in vectors
    with pytest.raises(ValueError):
        diagnostics([], {})


def test_rank_auc_ties_and_absent_class():
    assert auc([True, False], [1., 1.]) == .5
    assert auc([True, False], [2., 1.]) == 1.
    assert auc([True, True], [2., 1.]) is None


def test_linear_bridge_projection_matches_same_global_local_saved_means():
    weight = np.arange(12, dtype=np.float32).reshape(4, 3)
    global_ = np.array([[1., 2., 3.], [4., 5., 6.]])
    local = global_ @ weight.T
    result = bridge_diagnostic(weight, global_, local)
    assert result["cosine_min"] == pytest.approx(1.)
    assert result["relative_l2_max"] == 0.
    with pytest.raises(ValueError):
        bridge_diagnostic(weight.T, global_, local)


def test_language_skew_reported_and_full_cell_balancing_withheld():
    rows, arrays = fixture()
    for row in rows:
        row["language"] = "de" if row["ecapa"] > .25 else "en"
    report, vectors = diagnostics(rows, arrays)
    assert report["high_language_counts"] == {"de": 10}
    assert report["low_language_counts"] == {"en": 10}
    assert report["points"]["layer_0"]["loo_emotion_auc_within_language"]["de"]["mean"] is None
    assert not report["points"]["layer_0"]["language_emotion_balanced_mean_available"]
    assert "layer_0__language_emotion_balanced_high_mean" not in vectors


def test_secondary_pool_joint_balancing_and_missing_low_cell_contrast():
    rows, arrays = fixture()
    extra = []
    for row in rows:
        new = {**row, "id": row["id"] + "-de", "language": "de"}
        arrays[new["id"]] = {k: value + 100 for k, value in arrays[row["id"]].items()}
        extra.append(new)
    rows += extra
    vectors, meta = balanced_secondary_pool(rows, arrays)
    assert meta["eligible_high"] and meta["eligible_low"]
    assert vectors["layer_0__raw_mean"][0] == 60.
    assert "layer_0__contrast" in vectors
    for row in rows:
        if row["language"] == "de" and row["emotion"] == "sadness": row["ecapa"] = .6
    vectors, meta = balanced_secondary_pool(rows, arrays)
    assert meta["raw_mean_available"] and meta["high_minus_all_available"]
    assert not meta["contrast_available"]
    assert meta["missing_low_cells"] == ["de/sadness"]
    assert "layer_0__contrast" not in vectors


def test_secondary_pool_does_not_silently_ignore_missing_high_cells():
    rows, arrays = fixture()
    vectors, meta = balanced_secondary_pool(rows, arrays)
    assert not vectors and not meta["eligible_high"]
    assert len(meta["missing_high_cells"]) == 5


def test_layer_contrast_followup_is64_unique_matched_rows(tmp_path):
    import json
    folder = tmp_path / "vectors"; folder.mkdir()
    np.savez_compressed(folder / "selected.npz", **{f"layer_{i}__contrast": np.ones(4) for i in (20, 16, 17)})
    (folder / "metadata.json").write_text(json.dumps({"selected": {"threshold": .5}}))
    plan = layer_contrast_plan(tmp_path)
    assert len(plan["rows"]) == 64
    assert len({row["id"] for row in plan["rows"]}) == 64
    assert all(row["phase"] == "layer_contrast" for row in plan["rows"])
    pair = next(c for c in plan["conditions"] if len(c["points"]) == 2)
    assert pair["alpha"] == .5
    assert all(row["intervention"]["vector_kind"] == "contrast" for row in plan["rows"])
