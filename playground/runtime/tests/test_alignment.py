from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment import AudioFinisher, edge_cues


def finisher(score=1.0):
    result = AudioFinisher.__new__(AudioFinisher)
    result.aligner = SimpleNamespace(ok=True, align=lambda wave, words:
        [(word, 0.5 + i * 0.5, 1.0 + i * 0.5, score) for i, word in enumerate(words)])
    return result


def test_legacy_fades_leave_words_untouched_and_remove_tail():
    wave = np.ones(3 * 48000, dtype=np.float32)
    edited, reports = finisher().finish([wave], {"text": "Hello world", "tagged": "Hello world"})
    out, report = edited[0], reports[0]
    assert report["lead_s"] == 0.5 and report["cut_s"] == 1.62
    assert len(out) == int(1.62 * 48000) + int(0.15 * 48000)
    assert np.all(out[:int(0.4 * 48000)] == 0)
    assert out[int(0.45 * 48000)] == pytest.approx(0.5, abs=0.001)
    assert np.all(out[int(0.5 * 48000):int(1.5 * 48000)] == 1)
    assert out[-1] == 0
    assert np.all(wave == 1), "Do not mutate the raw generated waveform"
    assert report["spans"] == [("Hello", 0.5, 1.0, 1.0), ("world", 1.0, 1.5, 1.0)]


@pytest.mark.parametrize("prefix", ["(sigh, 0.5 seconds)", "(chuckle)"])
def test_requested_leading_bursts_are_protected(prefix):
    wave = np.ones(3 * 48000, dtype=np.float32)
    edited, reports = finisher().finish([wave], {"text": "Hello world", "tagged": prefix + " Hello world"})
    assert reports[0]["lead_s"] is None
    assert np.all(edited[0][:48000] == 1)


def test_only_edge_bursts_protect_the_edges_even_without_durations():
    allowed, tail = edge_cues("(quietly amused) Hello (chuckle, 0.3 seconds) world.")
    assert allowed and "requested edge" not in tail
    _, tail = edge_cues("Hello (chuckle) world. (sigh, 0.5 seconds) [0.4 seconds pause]")
    assert "0.5 seconds" in tail and "0.4 seconds" in tail and "0.28 seconds" not in tail


def test_automatic_renderer_padding_does_not_disable_structured_lead_fade():
    from prompting import build
    spec = build({"text": "Hello world", "script": "(quietly) Hello world", "general": "quiet"}, "structured")
    assert spec["tagged"].startswith("[0.3 seconds pause]")
    wave = np.ones(3 * 48000, dtype=np.float32)
    _, reports = finisher().finish([wave], spec)
    assert reports[0]["lead_s"] == 0.5
    spec = build({"text": "Hello world", "script": "[0.5 seconds pause] Hello world", "general": "quiet"}, "structured")
    _, reports = finisher().finish([wave], spec)
    assert reports[0]["lead_s"] == 0.5, "Fading keeps requested leading silence without shortening it"


def test_closing_burst_is_kept_with_legacy_slack():
    wave = np.ones(3 * 48000, dtype=np.float32)
    edited, reports = finisher().finish([wave], {"text": "Hello world", "tagged": "Hello world (sigh, 0.5 seconds)"})
    assert reports[0]["cut_s"] == 2.74
    assert len(edited[0]) > int(2.7 * 48000)


@pytest.mark.parametrize("disabled,score", [(True, 1.0), (False, 0.0)])
def test_disabled_and_implausible_alignments_do_not_edit(disabled, score):
    wave = np.ones(3 * 48000, dtype=np.float32)
    edited, reports = finisher(score).finish([wave], {"text": "Hello world"}, not disabled)
    assert np.array_equal(edited[0], wave)
    assert not reports[0]["applied"]


@pytest.mark.parametrize("enabled", [True, False])
def test_every_candidate_is_finished_before_scoring_and_download(tmp_path, monkeypatch, enabled):
    import small_app
    from fastapi.testclient import TestClient
    from storage import AudioStore
    import soundfile as sf
    import io

    scored_lengths = []
    def score(waves, spec):
        scored_lengths.extend(len(w) for w in waves)
        return [{"rank": i, "sec": len(w) / 48000, "reward": 1} for i, w in enumerate(waves)]
    monkeypatch.setattr(small_app, "store", AudioStore(tmp_path))
    monkeypatch.setattr(small_app, "STATE", {"phase": "ready", "busy": False,
        "engine": SimpleNamespace(generate=lambda *args: [np.ones(3 * 48000, dtype=np.float32) * 0.1 for _ in range(3)]),
        "scorers": SimpleNamespace(score=score), "finisher": finisher()})
    client = TestClient(small_app.app)
    response = client.post("/api/turn", json={"message": 'CAPTION: Neutral.\nTRANSCRIPT: "Hello world"',
        "input_mode": "manual", "format": "caption", "generation": "all", "n": 3, "align": enabled,"streaming":False,
        'adapters': {'id': None}, 'voice_mode':'none','speaker':{'method':'off'},'ranking':{'mode':'legacy'}})
    import json
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert not [e for e in events if e["event"] == "error"]
    result = next(e for e in events if e["event"] == "result")
    assert len(result["candidates"]) == 3 and len(scored_lengths) == 3
    for candidate, count in zip(result["candidates"], scored_lengths):
        wave, sr = sf.read(io.BytesIO(client.get(candidate["url"]).content))
        assert sr == 48000 and len(wave) == count
        assert candidate["alignment"]["enabled"] == enabled
        assert candidate["alignment"]["applied"] == enabled
        assert count < 3 * 48000 if enabled else count == 3 * 48000
