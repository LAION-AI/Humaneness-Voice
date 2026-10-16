"""Real-model checks, writes exact prompts/scores and downloaded MP3 artifacts."""
import hashlib
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
    OUT.mkdir(exist_ok=True)
    records = []
    with httpx.Client(base_url="http://127.0.0.1:8792", timeout=240) as client:
        deadline = time.monotonic() + 90
        state = {}
        while time.monotonic() < deadline:
            try:
                state = client.get("/api/state").json()
                if state.get("ready") or state.get("error"):
                    break
            except httpx.ConnectError:
                pass
            time.sleep(0.5)
        assert state["ready"] and state["director_ready"] and state["adapters"] == []
        # Distinct preexisting corpus samples exercise uploads; not target takes.
        references = []
        for slot, voice in [("A", "emolia_c1699"), ("B", "k325_age3_bg1")]:
            source = next(Path(f"/mnt/nvme/moss-15-v2-assets/refs3/audio/{voice}").rglob("*.mp3"))
            with source.open("rb") as file:
                response = client.post("/api/reference", params={"slot": slot}, files={"file": (source.name, file, "audio/mpeg")})
            response.raise_for_status()
            info = response.json()
            assert 0.24 <= info["seconds"] <= 2.89
            references.append(info["id"])
        cases = [
            ("english_caption_cfg1", {"message": 'CAPTION: Warm, natural conversation.\nTRANSCRIPT: "The train was late, but we made it home."', "input_mode": "manual", "format": "caption", "language": "en", "guidance": 1, "seconds": 4}),
            ("english_caption_cfg3", {"message": 'CAPTION: Warm, natural conversation.\nTRANSCRIPT: "The train was late, but we made it home."', "input_mode": "manual", "format": "caption", "language": "en", "guidance": 3, "seconds": 4}),
            ("german_luna_reference_best5", {"message": "Ich hätte wirklich nicht gedacht, dass wir es schaffen.", "input_mode": "perform", "format": "caption", "language": "de", "generation": "best", "n": 5, "reference_id": references[0], "seconds": 4}),
            ("german_structured_reference_all2", {"message": 'GENERAL: tender, vulnerable, very soft, genuine\nSCRIPT:\n(overwhelmingly tender, almost whispered) [4.0 seconds duration] Ich habe [0.5 seconds pause] dir das noch nie gesagt.', "input_mode": "manual", "format": "structured", "language": "de", "generation": "all", "n": 2, "reference_id": references[1]}),
            ("luna_chat_joy", {"message": "We finally got the good news. Tell me something joyfully encouraging, in one short sentence.", "input_mode": "chat", "format": "structured", "language": "en", "generation": "single"}),
            ("luna_chat_reference", {"message": "Sag mir etwas sanft Tröstendes nach einem schweren Tag, in einem kurzen Satz.", "input_mode": "chat", "format": "structured", "language": "de", "generation": "best", "n": 3, "reference_id": references[0]}),
            ("tags_reference", {"message": 'CAPTION: Tags: warm, conversational. Caption: Quiet genuine relief.\nTRANSCRIPT: "We made it home before the rain started."', "input_mode": "manual", "format": "tags", "language": "en", "reference_id": references[0], "seconds": 4}),
            ("transcript_reference", {"message": 'TRANSCRIPT: "Wir sind vor dem Regen nach Hause gekommen."', "input_mode": "manual", "format": "transcript", "language": "de", "reference_id": references[1], "seconds": 4}),
        ]
        hashes = {}
        for name, body in cases:
            events = []
            request = {"guidance": 3, "seed": 777, **body}
            with client.stream("POST", "/api/turn", json=request) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        events.append(json.loads(line[6:]))
            errors = [x for x in events if x["event"] == "error"]
            assert not errors, (name, errors)
            result = next(x for x in events if x["event"] == "result")
            direction = next(x for x in events if x["event"] == "direction")
            expected_n = 1 if request.get("generation", "single") == "single" else request["n"]
            assert len(result["candidates"]) == expected_n
            assert sorted(c["rank"] for c in result["candidates"]) == list(range(expected_n))
            assert result["candidates"][result["winner"]]["rank"] == 0
            assert np.isfinite([c["reward"] for c in result["candidates"]]).all()
            for c in result["candidates"]:
                audio = client.get(c["url"])
                audio.raise_for_status()
                wave, sr = sf.read(io.BytesIO(audio.content))
                assert sr == 48000 and len(wave) > 1000 and np.isfinite(wave).all()
                assert np.max(np.abs(wave)) > 0.001
                assert audio.headers["content-type"] == "audio/mpeg"
                (OUT / f"{name}-take{c['index']+1}.mp3").write_bytes(audio.content)
                hashes.setdefault(name, []).append(hashlib.sha256(audio.content).hexdigest())
            best = result["candidates"][result["winner"]]
            if name in {"german_luna_reference_best5", "german_structured_reference_all2", "tags_reference", "transcript_reference", "luna_chat_reference"}:
                assert best["wer"] <= 0.3, (name, "Reference take is not intelligible", best)
            records.append({"name": name, "request": request, "direction": direction, "result": result, "sha256": hashes[name]})
            (OUT / "smoke.json").write_text(json.dumps({"tested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "state": state, "cases": records}, indent=2, ensure_ascii=False))
            print(json.dumps({"case": name, "n": expected_n, "ms": result["total_ms"], "wer": best["wer"], "hyp": best["hyp"], "cfg": result["cfg_effective"]}, ensure_ascii=False), flush=True)
        assert hashes["english_caption_cfg1"] != hashes["english_caption_cfg3"], "CFG has no effect"
        assert len(set(hashes["german_luna_reference_best5"])) == 5, "Candidates are identical"
        assert not records[-1]["result"]["cfg_effective"]
        # Produced MP3 also exercises microphone/ASR decoding.
        data = (OUT / "english_caption_cfg1-take1.mp3").read_bytes()
        response = client.post("/api/asr", content=data)
        response.raise_for_status()
        assert "train" in response.json()["text"].lower()
        with httpx.Client(base_url="http://127.0.0.1:8792") as stranger:
            assert stranger.get(records[0]["result"]["candidates"][0]["url"]).status_code == 404
        for ref in references:
            assert client.delete("/api/reference/" + ref).status_code == 200
        print("PASS: all formats, Luna, reference A/B, CFG, 5-way chunking, ranking, ASR, isolation, delete", flush=True)


if __name__ == "__main__":
    main()
