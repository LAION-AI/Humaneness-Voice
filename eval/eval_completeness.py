#!/usr/bin/env python3
"""Does the model actually say the whole line?

Generates a set of lines, transcribes the audio back with Gemma's audio encoder,
and reports how much of the intended text survived.  Word-level recall against
the intended text is the metric that matters here: a take that stops halfway
scores ~0.5 no matter how good the first half sounded.

  python eval_completeness.py [tag]
"""
import base64, json, os, re, subprocess, sys, tempfile

import httpx
import numpy as np
import soundfile as sf

API = os.environ.get("MOSS_API", "http://127.0.0.1:8792")
LLM = os.environ.get("MOSS_LLM_BASE", "http://127.0.0.1:8790")
TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
OUT = f"/tmp/moss_eval/complete_{TAG}"

LINES = [
    ("English", "The old lighthouse keeper climbed the spiral stairs every single "
                "evening without fail, carrying his small brass lantern and humming "
                "a tune that nobody else in the village had ever heard before."),
    ("English", "I told you three times already that the package was never delivered "
                "to the correct address, and now you are asking me to prove something "
                "that your own records should have shown you from the beginning."),
    ("German",  "Der alte Leuchtturmwaerter stieg jeden einzelnen Abend die "
                "gewundene Treppe hinauf, trug seine kleine Messinglaterne und summte "
                "eine Melodie, die sonst niemand im Dorf jemals gehoert hatte."),
    ("German",  "Ich habe dir schon dreimal gesagt, dass das Paket niemals an die "
                "richtige Adresse geliefert wurde, und jetzt soll ich auch noch "
                "beweisen, was deine eigenen Unterlagen laengst zeigen muessten."),
    ("English", "Listen very carefully because I will only explain this once: turn "
                "left at the broken fence, walk past the dry well, and do not under "
                "any circumstances open the red door at the end of the corridor."),
]

INSTRUCTION = ("GENERAL: {ident} speaking plainly and clearly, unhurried and "
               "even, genuine and spontaneous, like a real person in a real "
               "moment, not acted. pristine high-quality studio recording, no "
               "background noise.\nSCRIPT:\n(calm, even, unhurried) {line}")

def norm(s):
    """Fold umlauts both ways so "gehoert" and "gehört" compare equal — otherwise
    German scores look like truncation when they are only spelling."""
    s = s.lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        s = s.replace(a, b)
    return re.findall(r"[a-z]+", s)


def transcribe(path):
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", path, "-ac", "1",
                        "-ar", "16000", f.name], capture_output=True)
        b64 = base64.b64encode(open(f.name, "rb").read()).decode()
    r = httpx.post(f"{LLM}/v1/chat/completions", timeout=300, json={
        "model": "gemma-4-12b-it-qat", "max_tokens": 400, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Transcribe this audio verbatim. Output only "
                                     "the transcription."},
            {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}]})
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def recall(intended, heard):
    want, got = norm(intended), set(norm(heard))
    if not want:
        return 0.0
    return sum(1 for w in want if w in got) / len(want)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for i, (lang, line) in enumerate(LINES):
        path = f"{OUT}/{i}.wav"
        body = {"text": line, "instruction": INSTRUCTION.format(
                    ident="a man's voice in his early fifties, a warm baritone,",
                    line=line),
                "language": lang}
        if os.environ.get("MOSS_NO_TOKENS"):     # baseline: let the model choose
            body["tokens"] = -1
            body["max_new_tokens"] = 1600
        r = httpx.post(f"{API}/api/say", json=body, timeout=600)
        r.raise_for_status()
        d = r.json()
        pcm = np.frombuffer(base64.b64decode(d["pcm"]), "<i2").astype(np.float32) / 32768
        sf.write(path, pcm, d["sr"])
        heard = transcribe(path)
        rc = recall(line, heard)
        rows.append((lang, len(norm(line)), d["audio_sec"], d.get("tokens"), rc))
        print(f"{i} {lang:8s} words={len(norm(line)):3d} audio={d['audio_sec']:6.2f}s "
              f"tokens={d.get('tokens')} recall={rc:.3f}")
        if rc < 0.9:
            print(f"    intended: {line[:110]}")
            print(f"    heard   : {heard[:110]}")
    rc = [r[4] for r in rows]
    print(f"\nmean recall {np.mean(rc):.3f}   min {min(rc):.3f}   "
          f"complete (>=0.95): {sum(1 for x in rc if x >= 0.95)}/{len(rc)}")


if __name__ == "__main__":
    main()
