"""Real Luna/TTS/alignment regression, including deterministic on/off comparison."""
import io
import json
from pathlib import Path
import time

import httpx
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "test-results"


def main():
    records = []
    with httpx.Client(base_url="http://127.0.0.1:8792", timeout=240) as client:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            state = client.get("/api/state").json()
            if state["ready"]:
                break
            assert not state["error"], state["error"]
            time.sleep(1)
        assert state["ready"] and state["alignment_available"] and state["alignment_default"]
        source = next(Path("/mnt/nvme/moss-15-v2-assets/refs3/audio/emolia_c1699").rglob("*.mp3"))
        with source.open("rb") as audio:
            upload = client.post("/api/reference?slot=A", files={"file": (source.name, audio, "audio/mpeg")})
        upload.raise_for_status()
        ref = upload.json()["id"]

        def turn(name, **options):
            events = []
            request = {"reference_id": ref, "seed": 777, **options}
            with client.stream("POST", "/api/turn", json=request) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        events.append(json.loads(line[6:]))
            assert not [e for e in events if e["event"] == "error"], events
            direction = next(e for e in events if e["event"] == "direction")
            result = next(e for e in events if e["event"] == "result")
            assert result["guidance"] == 1.5 and result["alignment_ms"] >= 0
            waves = []
            for c in result["candidates"]:
                audio = client.get(c["url"])
                audio.raise_for_status()
                wave, sr = sf.read(io.BytesIO(audio.content))
                assert sr == 48000 and np.isfinite(wave).all() and np.max(np.abs(wave)) > 0.001
                assert abs(len(wave) / sr - c["sec"]) < 0.01
                assert c["alignment"]["available"]
                waves.append(wave)
                assert audio.headers["content-type"] == "audio/mpeg"
                (OUT / f"{name}-take{c['index']+1}.mp3").write_bytes(audio.content)
            best = result["candidates"][result["winner"]]
            records.append({"name": name, "request": request, "direction": direction, "result": result})
            (OUT / "acting-alignment.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
            print(json.dumps({"name": name, "text": direction["text"], "wer": best["wer"],
                "alignment_ms": result["alignment_ms"], "alignment": best["alignment"]}, ensure_ascii=False), flush=True)
            return direction, result, waves

        try:
            fixed = {"message": 'CAPTION: Warm, quiet conversation.\nTRANSCRIPT: "Ich habe dir das noch nie gesagt."',
                "input_mode": "manual", "format": "caption", "language": "de", "seconds": 6}
            _, raw_result, raw = turn("alignment_off", **fixed, align=False)
            _, finished_result, finished = turn("alignment_on", **fixed)
            report = finished_result["candidates"][0]["alignment"]
            assert report["spans"] and report["enabled"]
            assert not raw_result["candidates"][0]["alignment"]["enabled"]
            assert len(finished[0]) <= len(raw[0])
            if report["lead_s"] is not None:
                before = int(max(0, report["lead_s"] - report["lead_ramp_s"] - 0.04) * 48000)
                assert np.max(np.abs(finished[0][:before]), initial=0) < 0.005
            if report["cut_s"] is not None:
                assert abs(finished[0][-1]) < 0.02
            start = int((report["lead_s"] or 0) * 48000) + 200
            end = min(len(finished[0]), int((report["cut_s"] or len(raw[0])/48000) * 48000)) - 200
            # The transport is now lossy; exact PCM equality is tested before
            # encoding in tests/test_alignment.py, not between two MP3 streams.
            assert end > start and np.sqrt(np.mean((raw[0][start:end] - finished[0][start:end]) ** 2)) < 0.02

            acting = [
                ("actor_confession", "Du willst jemandem gestehen, dass du ihn vermisst, traust dich aber kaum. Spiele die Szene in zwei kurzen deutschen Sätzen, ohne Einleitung.", "structured", "de", 3),
                ("actor_anger", "Spiele einen Menschen, der gerade einen Verrat entdeckt hat. Beginne gefährlich ruhig und werde dann kalt. Zwei kurze deutsche Sätze, keine Erklärung.", "structured", "de", 1),
                ("actor_comfort_caption", "I had an awful day. Comfort me warmly in one short sentence, as a close friend would.", "caption", "en", 1),
            ]
            for name, message, fmt, language, n in acting:
                direction, result, waves = turn(name, message=message, input_mode="chat", format=fmt,
                    language=language, generation="best" if n > 1 else "single", n=n)
                text = direction["text"]
                assert text != message and len(text.split()) <= 70
                assert not any(marker in text.lower() for marker in ("hier ist", "ich werde die szene", "sure, here", "spie­le"))
                assert len(waves) == n
                if fmt == "structured":
                    assert "seconds pause]" in direction["prompt"]
                else:
                    assert "(" not in direction["text"] and "[" not in direction["text"]
            direction, _, _ = turn("actor_exact", message="You knew what this meant to me, and you did it anyway.", input_mode="perform", format="caption", language="en")
            assert direction["text"] == "You knew what this meant to me, and you did it anyway."
        finally:
            client.delete("/api/reference/" + ref).raise_for_status()
        print("PASS: real Luna acting briefs, exact words, all-candidate alignment, CFG default, reproducible fade on/off", flush=True)


if __name__ == "__main__":
    main()
