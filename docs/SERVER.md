# Using the server over HTTP

Everything the web page can do is an HTTP request. This page is the whole
surface: one endpoint for "say this and give me an MP3", one for the full
streaming turn, and the smaller ones underneath them.

Nothing here needs an API key **except** a hosted director. With the local
`gemma-4-12B` there is no outside dependency at all.

---

## The short version

```bash
curl -s localhost:8792/api/speak \
  -H 'content-type: application/json' \
  -d '{"text": "Tell me the worst thing that happened this week."}' \
  -o reply.mp3
```

That is the whole demo in one request: the language model writes the reply and
directs it, the reference recording is retrieved, the adapters are chosen,
**ten candidates** are generated at **guidance 3.0** and ranked, and the winner
comes back as an MP3.

It takes about 30–60 seconds. There is no streaming — best-of and restoration
both need the whole take before anything can be ranked.

The performance travels in the response headers, so you can see what was done
without asking for JSON:

```bash
curl -sD - localhost:8792/api/speak -H 'content-type: application/json' \
  -d '{"text":"Say something kind."}' -o out.mp3 | grep -i '^x-'
```

| header | what it is |
|---|---|
| `x-script` | the timed script the director wrote, with its cues |
| `x-reply` | the words alone |
| `x-audio-sec` | length of the returned audio |
| `x-language` | the language the director chose |
| `x-best-of` / `x-guidance` | what was actually run |
| `x-total-ms` | wall clock for the whole request |

---

## `POST /api/speak`

Text in, audio out. Every field except `text` is optional and defaults to what
the web UI ships with.

### Output

| field | default | values |
|---|---|---|
| `format` | `"mp3"` | `mp3`, `wav`, `json` (base64 PCM plus every field of the metadata) |
| `bitrate` | `"192k"` | any ffmpeg MP3 bitrate |

### What gets generated

| field | default | what it does |
|---|--:|---|
| `best_of` | `10` | generate N candidates and return the highest-ranked. `1` disables it |
| `best_of_guidance` | `3.0` | classifier-free guidance strength. `1.0` turns it off |
| `best_of_audio` | `false` | `true` returns every candidate's score in `candidates` |
| `seed` | `1234` | fixed seed; candidates vary by seed internally |
| `sidon` | `false` | restore before ranking. Off by default — it distorts on screams, see [`SIDON.md`](SIDON.md) |
| `align` | `true` | trim silence and over-run off the end |

### Who speaks

| field | default | what it does |
|---|---|---|
| `profile` | `emolia_c1699` | which voice. `GET /api/state` lists them |
| `persona` | none | a character brief prepended to the director's prompt. `GET /api/personas` lists them |
| `brain` | `local` | `local` (gemma-4-12B on this box), or `luna` / `glm` / `gemini-flash` / `gemini-flash-lite` with a key. `glm` is `glm-5.3`, a 1M-context model with three quirks the server handles: it takes `max_tokens` rather than `max_completion_tokens`, it returns an occasional retryable 400, and on roughly one turn in six it puts its whole chain of thought in `content` instead of `reasoning_content` |
| `prompt_style` | `prose` | `prose` or the compact `codes` style (see [`CONTEXT.md`](CONTEXT.md)) |
| `language` | the director decides | force with `"English"` / `"German"` |
| `history` | `[]` | `[{"role":"user","content":"…"}, …]` for a continuing conversation |
| `session` | none | any string; keeps the previous turns' audio tails as context so the voice does not reset |

### How it is performed

| field | default | what it does |
|---|--:|---|
| `pure_mode` | `false` | base model plus reference clips only — no expressive adapters |
| `char_lora` | `true` | the speaker's own identity adapter |
| `emotion_nuance` | `true` | let retrieval pick an emotion adapter |
| `quality_lams` | `{genuineness_high: 0.25, blend_high: 0.5, esthetics_high: 0.5}` | the three perceptual adapters |
| `adapter_overrides` | `{}` | `{"sft3_voicenet:S_DRAM_high": 1.5}` — force any adapter by name. `GET /api/adapters` lists all of them |
| `burst_lam_max` | `1.25` | ceiling for one burst adapter |
| `skills` | `true` | use the measured burst recipes rather than the original prompt block |
| `english_cues` | `true` | rewrite German stage directions into English before retrieval — worth a median 0.267 word error on German turns, see [`BABBLE.md`](BABBLE.md) |
| `prompt_extra` | none | an extra block appended to the director's system prompt |
| `retrieval` | `true` | match the director's prose against the corpus. Off falls back to the decoded label |

### Examples

Ten candidates, restored, every score returned:

```bash
curl -s localhost:8792/api/speak -H 'content-type: application/json' -d '{
  "text": "Tell me about the worst day you ever had.",
  "best_of": 10, "sidon": true, "best_of_audio": true, "format": "json"
}' | python -c 'import json,sys; d=json.load(sys.stdin); print(d["script"]); \
    [print(c["rank"], round(c["reward"],3), round(c["wer"],3)) for c in d["candidates"]]'
```

A named character, in German, one take, no guidance:

```bash
curl -s localhost:8792/api/speak -H 'content-type: application/json' -d '{
  "text": "Erzaehl mir eine Geschichte.",
  "persona": "dracula", "language": "German",
  "best_of": 1, "best_of_guidance": 1.0
}' -o dracula.mp3
```

Forcing a delivery axis and a louder burst budget:

```bash
curl -s localhost:8792/api/speak -H 'content-type: application/json' -d '{
  "text": "This is the last warning I am going to give you.",
  "adapter_overrides": {"sft3_voicenet:S_DRAM_high": 1.5},
  "burst_lam_max": 1.0
}' -o warned.mp3
```

A performance benchmark item, spoken verbatim — paste the JSON straight in and
the words come back unchanged; see [`BENCHMARK.md`](BENCHMARK.md):

```bash
curl -s localhost:8792/api/speak -H 'content-type: application/json' \
  -d "{\"text\": $(jq -Rs . < item.json)}" -o item.mp3
```

---

## `POST /api/turn` — the streaming version

The same pipeline, as a stream of events and PCM, which is what the browser
uses. Same fields as `/api/speak` except that `text` is called `message` and
there is no `format`.

The body is a sequence of frames: one byte of type (`0` = JSON event,
`1` = PCM), four bytes of big-endian length, then the payload. Events arrive in
order: `llm` (the script and every choice made), `best_of` (candidates and
scores, when N > 1), `start`, then PCM, then `end`.

```bash
curl -sN localhost:8792/api/turn -H 'content-type: application/json' \
  -d '{"message":"Say hello.","best_of":1,"sidon":false}' -o turn.bin
```

With `best_of: 1` and `sidon: false` the audio streams as it is generated and
the first PCM frame arrives in about a second. Anything else has to finish
generating first.

---

## Speaking a script you wrote yourself

`POST /api/say` runs no language model. You supply the timed script and the
adapters, and nothing is invented:

```bash
curl -s localhost:8792/api/say -H 'content-type: application/json' -d '{
  "text": "[0.3 seconds pause] (clearly amused, letting it out) [3.4 seconds duration] I still cannot believe the cat opened that door by herself.",
  "instruction": "GENERAL: a woman'\''s voice, in their thirties, Standard American; genuine, not acted. the same speaker continues without interruption: identical voice, identical person, same microphone and same room. genuine and spontaneous, like a real person in a real moment, not acted. pristine high-quality studio recording, no background noise. 3.7s, EN.\nSCRIPT:\n[0.3 seconds pause] (clearly amused, letting it out) [3.4 seconds duration] I still cannot believe the cat opened that door by herself.",
  "tokens": 46, "language": "English", "seed": 1234,
  "loras": [["sft3_dpo:p2", 1.0], ["sft3_voice:emolia_c1699", 1.0]]
}'
```

Returns base64 PCM and the sample rate. `tokens` must be the sum of every
number in the script × 12.5 — [`PROMPTING.md`](PROMPTING.md) explains the
format, and `timed_script.render()` computes it for you.

`POST /api/say_batch` takes `items: [ … ]` of the same shape and returns
`pcm: [ … ]`: one forward pass, so ten utterances cost roughly what two cost
one after another. Add `guidance` plus `instruction_unc` and `text_unc` per
item to run classifier-free guidance.

`POST /api/cfg_sweep` renders one script at several guidance values for
comparison. Nothing autoplays; the point is to hear where guidance stops
paying.

---

## Looking at what is loaded

| endpoint | what it tells you |
|---|---|
| `GET /api/state` | voices, personas, which models are up, SIDON's health, the defaults in force |
| `GET /api/adapters` | every adapter that can be named in `adapter_overrides`, grouped, with its default and ceiling |
| `GET /api/voices` | the reference corpus catalogue |
| `POST /api/asr` | raw audio bytes in the body, transcript out |

---

## Running it

```bash
./run.sh both      # language model + voice model + web UI
./run.sh sidon     # optional: the restoration service on port 8793
./run.sh stop
```

| process | port | GPU |
|---|--:|---|
| `llama-server` (gemma-4-12B-it-qat) | 8790 | `MOSS_LLM_GPU`, default 0 |
| the app (MOSS 4.55B + scorers + aligner) | 8792 | `MOSS_TTS_GPU`, default 1 |
| SIDON restoration | 8793 | `MOSS_LLM_GPU` |

Two 24 GB cards is what this is tuned for. On one card, run the director
elsewhere (`MOSS_LLM_BASE`) or use a hosted one.

### Without a hosted model

The default is already local: `brain: "local"` uses `gemma-4-12B-it-qat`
through llama.cpp on port 8790, and no key is read. The hosted models
(`luna`, `gemini-flash`, `gemini-flash-lite`) are only reached when you ask for
one by name, with one exception — rewriting German stage directions into
English uses the fastest hosted model **when a key is present**, and falls back
to the local one otherwise (15.6 s against 1.3 s, so it is worth having, but it
is not required). Set `MOSS_ENGLISH_CUES=0` to skip it entirely.

The context window matters on the local model: the prose system prompt is 6,917
tokens and a persona adds about 1,100, so `--ctx-size` is 16384 in `run.sh`.
On a smaller card use `"prompt_style": "codes"`, which carries the same rules in
about 2,400 tokens. See [`CONTEXT.md`](CONTEXT.md).
