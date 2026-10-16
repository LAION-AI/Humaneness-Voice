import io
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompting import build, parse_literal
from storage import AudioStore


def test_cfg_preserves_exact_words_timing_and_bursts():
    turn = {"general": "furious, high arousal", "script": "(overwhelmingly furious) [3.2 seconds duration] You knew, [0.4 seconds pause] (sharp inhale, 0.3 seconds) you knew all along.",
            "text": "You knew, you knew all along."}
    spec = build(turn, "structured")
    assert "furious" in spec["prompt"] and "furious" not in spec["unconditional"]
    for token in ["[0.4 seconds pause]", "(sharp inhale, 0.3 seconds)", "you knew all along."]:
        assert token in spec["prompt"] and token in spec["unconditional"]


@pytest.mark.parametrize("fmt", ["caption", "tags", "transcript"])
def test_caption_cfg_and_quotes(fmt):
    text = 'Er sagte: "Es ist geschafft."'
    spec = build({"text": text, "general": "delighted", "tags": ["joyful"], "caption": "Bright delight."}, fmt, "de")
    assert parse_literal(spec["prompt"], fmt)["text"] == text
    assert spec["prompt"].split("TRANSCRIPT:")[1] == spec["unconditional"].split("TRANSCRIPT:")[1]
    if fmt == "transcript":
        assert spec["prompt"] == spec["unconditional"]
    else:
        assert spec["unconditional"].startswith("CAPTION:")


def test_manual_prompt_not_rewritten():
    prompt = 'CAPTION: Whisper, then let the voice bloom.\nTRANSCRIPT: "Das bleibt unter uns."'
    turn = parse_literal(prompt, "caption")
    spec = build(turn, "caption", "de", literal=prompt)
    assert spec["prompt"] == prompt
    with pytest.raises(ValueError):
        parse_literal("Hello, no transcript", "caption")


def test_manual_untimed_burst_survives_cfg():
    prompt = "GENERAL: warmly amused\nSCRIPT:\n(smiling) I did not expect that. (chuckle) It worked."
    spec = build(parse_literal(prompt, "structured"), "structured", literal=prompt)
    assert spec["prompt"] == prompt
    assert "(chuckle)" in spec["unconditional"]
    assert "(smiling)" not in spec["unconditional"]


def test_custom_reference_persists_until_removed_and_takes_expire(tmp_path):
    store = AudioStore(tmp_path, ttl=10)
    item = store.new("owner", "voice", "reference", "A")
    item.path.touch()
    with pytest.raises(KeyError):
        store.get(item.id, "another-owner")
    before = item.touched
    store.get(item.id, "owner")
    assert item.touched >= before
    item.active = 1
    assert store.cleanup(item.touched + 11) == []
    item.active = 0
    assert store.cleanup(item.touched + 11) == []
    assert store.get(item.id, 'owner') is item
    take = store.new('owner', 'take', 'take')
    take.path.touch()
    assert store.cleanup(take.touched + 11) == [take]
    store.remove(item)
    assert not item.path.exists()


def test_upload_and_audio_are_scoped_to_browser_session(tmp_path, monkeypatch):
    import small_app
    from fastapi.testclient import TestClient
    monkeypatch.setattr(small_app, "store", AudioStore(tmp_path))
    a, b = TestClient(small_app.app), TestClient(small_app.app)
    data = io.BytesIO()
    wave = 0.1 * np.sin(2 * np.pi * 220 * np.arange(48000) / 48000)
    sf.write(data, wave, 48000, format="WAV")
    response = a.post("/api/reference?slot=A", files={"file": ("sample.wav", data.getvalue(), "audio/wav")})
    assert response.status_code == 200
    id_ = response.json()["id"]
    audio = a.get("/api/audio/" + id_)
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/mpeg"
    assert b.get("/api/audio/" + id_).status_code == 404
    assert b.delete("/api/reference/" + id_).status_code == 404
    assert a.delete("/api/reference/" + id_).status_code == 200
    assert a.get("/api/audio/" + id_).status_code == 404
    assert not list(tmp_path.iterdir())


def test_empty_and_invalid_uploads_leave_no_files(tmp_path, monkeypatch):
    import small_app
    from fastapi.testclient import TestClient
    monkeypatch.setattr(small_app, "store", AudioStore(tmp_path))
    client = TestClient(small_app.app)
    assert client.post("/api/reference", files={"file": ("bad.wav", b"not audio")}).status_code == 400
    assert client.post("/api/reference", files={"file": ("empty.wav", b"")}).status_code == 400
    assert not list(tmp_path.iterdir())


def test_requested_n_and_guidance_are_bounded():
    from small_app import Turn
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Turn(message="Hello", n=100)
    with pytest.raises(ValidationError):
        Turn(message="Hello", guidance=100)
    assert Turn(message="Hello").guidance == 2.0
    assert Turn(message="Hello").temperature == 0.7
    assert Turn(message="Hello").format == "structured"
    assert Turn(message="Hello", temperature=0).temperature == 0
    with pytest.raises(ValidationError):
        Turn(message="Hello", temperature=-0.01)
    assert Turn(message="Hello").align is True


def test_chat_transcript_is_derived_from_actual_script(monkeypatch):
    import asyncio
    import httpx
    import settings
    from director import Director
    monkeypatch.setattr(settings, "luna_key", lambda: "test-only")
    response = {"text": "We made it!", "general": "relieved", "script": "(softly relieved) We made it.",
                "caption": "Quiet relief.", "tags": ["relieved"], "language": "en"}
    async def run():
        agent = Director()
        await agent.client.aclose()
        def respond(req):
            payload = __import__("json").loads(req.content)
            assert "tools" not in payload and "tool_choice" not in payload
            system = payload["messages"][0]["content"]
            assert "AN ACTING BRIEF IS NOT A TRANSCRIPT" in system
            assert "MOST PAUSES BELONG BETWEEN TWO WORDS" in system
            assert "choose_generation_mode" not in system and "select_reference_voice" not in system
            return httpx.Response(200, json={"choices": [{"message": {"content": __import__("json").dumps(response)}}]})
        agent.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            actual, _ = await agent.direct("Encourage me", "structured", "en", "chat", [])
            assert actual["text"] == "We made it."
            with pytest.raises(ValueError):
                await agent.direct("We made it!", "structured", "en", "perform", [])
        finally:
            await agent.close()
    asyncio.run(run())


def test_luna_timeout_has_a_useful_message(monkeypatch):
    import asyncio
    import httpx
    import settings
    from director import Director
    monkeypatch.setattr(settings, "luna_key", lambda: "test-only")
    async def run():
        agent = Director()
        await agent.client.aclose()
        def timeout(req):
            raise httpx.ReadTimeout("", request=req)
        agent.client = httpx.AsyncClient(transport=httpx.MockTransport(timeout))
        try:
            with pytest.raises(RuntimeError, match="Luna timed out"):
                await agent.direct("Encourage me", "structured", "en", "chat", [])
        finally:
            await agent.close()
    asyncio.run(run())


def test_failed_direction_releases_server_without_generating(monkeypatch):
    import small_app
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    async def failure(*args, **kwargs):
        raise RuntimeError("Luna hat nicht rechtzeitig geantwortet.")
    monkeypatch.setattr(small_app, "STATE", {"phase": "ready", "busy": False, "engine": None})
    monkeypatch.setattr(small_app, "director", SimpleNamespace(direct=failure))
    response = TestClient(small_app.app).post("/api/turn", json={"message": "Encourage me"})
    assert '"event": "error"' in response.text and "Luna hat nicht rechtzeitig" in response.text
    assert not small_app.STATE["busy"]
    assert small_app.STATE["job_phase"] is None and small_app.STATE["job_started"] is None
