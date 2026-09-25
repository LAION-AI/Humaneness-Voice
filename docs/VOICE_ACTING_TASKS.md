# Getting acted speech out of the server with curl

A cookbook for the one thing most people want: hand it a line, get back a
performance as an MP3.

Everything below is one HTTP request. There is no SDK, no auth, and no key —
the director runs locally on this box.

---

## Set the host once

```bash
export MOSS_HOST="https://your-server-here"     # or http://localhost:8792
```

Ask whoever runs the server for the address. Everything below uses
`$MOSS_HOST`.

Check it is up and see what is loaded:

```bash
curl -s $MOSS_HOST/api/state | python -m json.tool | head -30
```

---

## 1. The best take, as an MP3

The default settings are the demo's own: **ten candidates, guidance 3.0**, the
winner ranked by a reward built from naturalness, vocal-burst blend, how well
the audio matches the direction, and word error.

```bash
curl -s $MOSS_HOST/api/speak \
  -H 'content-type: application/json' \
  -d '{
    "text": "I nearly packed my office badge out of habit this morning. Today we can stay on this train until the sea appears.",
    "bitrate": "128k"
  }' -o take.mp3
```

128 kbit/s, 48 kHz, mono. About 60–90 seconds for ten candidates at guidance 3.

**Everything the performance did travels in the headers**, so you can see the
script without asking for JSON:

```bash
curl -sD headers.txt $MOSS_HOST/api/speak -H 'content-type: application/json' \
  -d '{"text": "Say something kind.", "bitrate": "128k"}' -o take.mp3
grep -i '^x-' headers.txt
```

```
x-script: "[0.3 seconds pause] (clearly warm, letting it out, unhurried) [3.4 seconds duration] …"
x-reply: "…"
x-audio-sec: 9.07
x-language: English
x-best-of: 10
x-guidance: 3.0
x-total-ms: 71204.3
```

## 2. The same, spelled out

Identical to the above — the defaults written down, so you can change one:

```bash
curl -s $MOSS_HOST/api/speak -H 'content-type: application/json' -d '{
  "text": "I nearly packed my office badge out of habit this morning.",
  "best_of": 10,
  "best_of_guidance": 3.0,
  "format": "mp3",
  "bitrate": "128k"
}' -o take.mp3
```

Guidance 3.0 is the measured optimum: **+2.33 and +1.92 points of 15** against
no guidance across two runs, and **4.0 is worse than 3.0** on all three rubrics.
See [`GUIDANCE.md`](GUIDANCE.md). It costs about 1.93× the wall clock and is
worth it.

## 3. All ten candidates, ranked, as MP3s

One request, ten files, with the score that ranked each one. Use this when you
want to pick by ear — the reward is a model judging another model, and it does
not always agree with a listener.

```bash
curl -s $MOSS_HOST/api/speak -H 'content-type: application/json' -d '{
  "text": "I nearly packed my office badge out of habit this morning.",
  "best_of": 10, "best_of_guidance": 3.0,
  "format": "json", "best_of_audio": true, "bitrate": "128k"
}' -o run.json

python - <<'PY'
import json, base64
d = json.load(open("run.json"))
print(d["script"], "\n")
for c in sorted(d["candidates"], key=lambda c: c["rank"]):
    open(f"rank{c['rank']}.mp3", "wb").write(base64.b64decode(c["audio"]))
    print(f"rank {c['rank']}  reward {c['reward']:.3f}  wer {c['wer']:.3f}  "
          f"genuine {c['genuineness']:.2f}  blend {c['blend']:.2f}  {c['sec']}s")
PY
```

```
rank 0  reward 2.675  wer 0.130  genuine 1.42  blend 3.90  9.07s
rank 1  reward 2.489  wer 0.130  genuine 1.31  blend 3.55  8.83s
…
rank 9  reward 0.454  wer 0.130  genuine 0.61  blend 2.10  8.69s
```

With `format: "json"` the candidates come back as MP3 by default; add
`"candidate_format": "wav"` or `"pcm"` if you would rather have those.

## 4. A benchmark item, spoken verbatim

Paste the JSON item straight in as `text`. The words come back **unchanged** —
the server checks, and falls back to a plain correct rendering if the director
drifts. See [`BENCHMARK.md`](BENCHMARK.md).

```bash
curl -s $MOSS_HOST/api/speak -H 'content-type: application/json' \
  -d "{\"text\": $(jq -Rs . < item.json), \"bitrate\": \"128k\"}" -o item.mp3
```

## 5. German

The director picks the language from what you write, or force it:

```bash
curl -s $MOSS_HOST/api/speak -H 'content-type: application/json' -d '{
  "text": "Erzähl mir von jemandem, den du sehr vermisst.",
  "language": "German", "bitrate": "128k"
}' -o de.mp3
```

Stage directions are rewritten into English before the reference is retrieved —
German ones are off-distribution for the speech model and cost a median word
error of 0.267 against 0.000. That happens automatically; see
[`BABBLE.md`](BABBLE.md).

## 6. A named character

```bash
curl -s $MOSS_HOST/api/personas          # host, dracula, orc, cookie, counselor
curl -s $MOSS_HOST/api/speak -H 'content-type: application/json' -d '{
  "text": "Come in. I have been expecting you.",
  "persona": "dracula", "bitrate": "128k"
}' -o dracula.mp3
```

## 7. Choosing the voice

```bash
curl -s $MOSS_HOST/api/state | python -c \
  'import json,sys; [print(p["id"], "-", p["short"], p["accent"]) for p in json.load(sys.stdin)["profiles"]]'
```

```bash
curl -s $MOSS_HOST/api/speak -H 'content-type: application/json' -d '{
  "text": "This is a different voice entirely.",
  "profile": "emolia_c1682", "bitrate": "128k"
}' -o other.mp3
```

## 8. One take, still guided

Guidance applies at any `best_of`, including one. This is the cheapest request
that still gets the setting the measurements favour — about 20 seconds:

```bash
curl -s $MOSS_HOST/api/speak -H 'content-type: application/json' -d '{
  "text": "Today we can stay on this train until the sea appears.",
  "best_of": 1, "best_of_guidance": 3.0, "bitrate": "128k"
}' -o one.mp3
```

## 9. Faster still, when you do not need the guidance

```bash
curl -s $MOSS_HOST/api/speak -H 'content-type: application/json' -d '{
  "text": "Just one take, quickly.",
  "best_of": 1, "best_of_guidance": 1.0, "bitrate": "128k"
}' -o quick.mp3
```

One candidate, no guidance: a few seconds instead of twenty, and measurably
worse — guidance is worth about two points of fifteen. Use it for iterating on
wording, not for the final take.

| request | guidance | wall clock |
|---|--:|--:|
| `best_of: 1, best_of_guidance: 1.0` | off | ~6 s |
| `best_of: 1` (guidance 3.0 by default) | 3.0 | ~23 s |
| default (`best_of: 10`) | 3.0 | ~55 s |

---

## The web page

The same engine with a chat window, every adapter and reference clip shown per
turn, a best-of-N player list you can compare by ear, and a CFG sweep:

| page | what it is |
|---|---|
| `$MOSS_HOST/` | the voice-acting demo |
| `$MOSS_HOST/studio` | speech-recognition input, emotion scoring of *your* voice, persona switching |
| `$MOSS_HOST/report` | a static summary of the measurements |

---

## Every parameter

[`SERVER.md`](SERVER.md) lists all of them — output format, what gets generated,
who speaks, and how it is performed — with the defaults each one falls back to.
[`DEFAULTS.md`](DEFAULTS.md) says what those defaults are and which measurement
set them.

## If something goes wrong

| symptom | look at |
|---|---|
| `503 model still loading` | the server is starting; the speech model takes about eight minutes |
| a 400 with a token count in it | the prompt did not fit the context window — [`CONTEXT.md`](CONTEXT.md) |
| the audio is unintelligible | [`BABBLE.md`](BABBLE.md); check the burst adapter weights in the response |
| it sounds read rather than acted | [`PROMPTING.md`](PROMPTING.md) §Pacing |
