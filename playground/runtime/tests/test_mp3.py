import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_mp3_transport_is_mono_96kbps(tmp_path):
    from small_app import encode_mp3
    wave = (0.1 * np.sin(2 * np.pi * 220 * np.arange(96000) / 48000)).astype(np.float32)
    path = tmp_path / "take.mp3"
    encode_mp3(wave, path)
    output = subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)])
    stream = json.loads(output)["streams"][0]
    assert stream["codec_name"] == "mp3" and stream["channels"] == 1
    assert stream["bit_rate"] == "96000" and stream["sample_rate"] == "48000"
    assert path.stat().st_size < wave.nbytes / 5


def test_mp3_reference_sidecar_and_orphan_expire(tmp_path):
    from storage import AudioStore
    store = AudioStore(tmp_path, ttl=10)
    ref = store.new("owner", "reference", "reference", "A")
    ref.path.touch()
    ref.playback.touch()
    take = store.new("owner", "take", "take")
    take.path.touch()
    assert take.path.suffix == ".mp3"
    assert {item.id for item in store.cleanup(max(ref.touched, take.touched) + 11)} == {take.id}
    store.remove(ref)
    assert not list(tmp_path.iterdir())
    orphan = tmp_path / "orphan.mp3"
    orphan.touch()
    store.cleanup(orphan.stat().st_mtime + 11)
    assert not orphan.exists()


def test_zero_temperature_uses_argmax_without_tiny_temperature_hack():
    from engine import zero_temperature_sampler
    sampler = zero_temperature_sampler(lambda **kwargs: kwargs)
    args = {"temperature": 0, "do_sample": True, "logits": "original", "repetition_penalty": 1.1}
    assert sampler(**args) == {**args, "do_sample": False}
    assert sampler(**{**args, "temperature": 1})["do_sample"] is True
