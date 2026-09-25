# Using the server without the web page

The pages at `/` and `/studio` are one client of an HTTP API, not the product.
Everything they do is available directly, and two of the endpoints do not involve
a language model at all.

Start it with `./run.sh app` (port 8792). `./run.sh llm` additionally starts a
local language model on port 8790; you only need that for `/api/turn`, and only
if you are not using a hosted one.

| endpoint | language model | streams | what it is for |
|---|:-:|:-:|---|
| `POST /api/say` | no | no | speak an exact script with an exact adapter set |
| `POST /api/say_batch` | no | no | the same, several at once in one forward pass |
| `POST /api/cfg_sweep` | no | no | one line at several guidance values, for comparison |
| `POST /api/turn` | **yes** | yes | a whole conversational turn: write it, direct it, speak it |
| `POST /api/asr` | no | no | transcribe an uploaded recording |
| `GET /api/state` | no | — | what is loaded: voices, personas, adapters, levers |
| `GET /api/adapters` | no | — | every adapter that can be dialled, grouped, with defaults |
| `GET /api/voices` | no | — | the reference corpus catalogue |

---

## Speaking without a language model

`POST /api/say` takes the script you have already written. Nothing is invented
and nothing is chosen for you: the adapters you name are the adapters that merge.

```bash
curl -s localhost:8792/api/say -H 'content-type: application/json' -d '{
  "text": "[0.3 seconds pause] (clearly amused, letting it out, warm and unguarded) [3.4 seconds duration] I still cannot believe the cat opened that door by herself.",
  "instruction": "GENERAL: a woman'\''s voice, in their thirties, Standard American; close conversational volume, unforced; genuine, not acted; reads as amusement; 3.7s, EN.\nSCRIPT:\n[0.3 seconds pause] (clearly amused, letting it out, warm and unguarded) [3.4 seconds duration] I still cannot believe the cat opened that door by herself.",
  "tokens": 46,
  "language": "English",
  "seed": 1234,
  "loras": [["sft3_dpo:p2", 1.0], ["sft3_voice:emolia_c1699", 0.5]],
  "align": true
}' | python -c 'import json,sys,base64; d=json.load(sys.stdin); open("out.pcm","wb").write(base64.b64decode(d["pcm"])); print(d["sr"], "Hz")'
```

| field | required | what it does |
|---|:-:|---|
| `text` | ✅ | the rendered script; goes into the `Text:` field verbatim |
| `instruction` | ✅ | the `GENERAL:` / `SCRIPT:` block |
| `tokens` | ✅ | length budget in frames — seconds × 12.5, and the numbers in the script must add up to it |
| `language` | | `English` or `German` |
| `loras` | | `[[name, weight], …]`; names as `GET /api/adapters` reports them |
| `seed` | | fixed seed makes a call reproducible |
| `anchor`, `anchor_path` | | reference recording: the corpus anchor by default, or a path |
| `reference`, `ref_lang`, `speed` | | pick a conditioning clip from the corpus by condition instead |
| `align` | | trim to the script (see `ALIGNMENT.md`) |
| `stop_bias`, `audio_temperature`, `max_new_tokens`, `chunk_frames` | | sampling |

Returns `{"sr": 48000, "pcm": "<base64 int16 mono>", …}` plus the generation
metadata — `rtf`, `tokens`, `loras`, and `align` when trimming ran.

**The script format is not optional.** `timed_script.render()` builds it for you
if you would rather write plain text with cues; the rules are in
[`PROMPTING.md`](PROMPTING.md) and the arithmetic has to add up to `tokens`.

### Several at once

`POST /api/say_batch` takes `items: [ … ]` of the same shape and returns
`pcm: [ … ]`. One forward pass, so ten utterances cost about what two cost
sequentially. Add `guidance: 3.0` and give each item an `instruction_unc` and
`text_unc` — the same script with its delivery directions removed — to run
classifier-free guidance.

---

## A whole turn, with a language model

`POST /api/turn` writes the reply, directs it, picks the reference recording and
the adapters, and streams the audio. It needs a language model: either a hosted
one (`brain: "luna" | "gemini-flash" | "gemini-flash-lite"`, with
`$HYPRLAB_API_KEY` set) or the local one from `./run.sh llm` (`brain: "local"`).

```bash
curl -N -s localhost:8792/api/turn -H 'content-type: application/json' -d '{
  "message": "Tell me something that made you laugh this week.",
  "persona": "host",
  "session": "demo-1"
}' > turn.bin
```

### The wire format

A stream of frames: **1 byte tag, 4 bytes big-endian length, payload**. Tag `0`
is a JSON event, tag `1` is raw int16 mono PCM at the sample rate the `start`
event reports.

```python
import json, struct
def frames(buf):
    i = 0
    while i < len(buf):
        tag = buf[i]
        n = struct.unpack(">I", buf[i+1:i+5])[0]
        yield tag, buf[i+5:i+5+n]
        i += 5 + n

audio = bytearray()
for tag, payload in frames(open("turn.bin","rb").read()):
    if tag == 1:
        audio += payload
    else:
        ev = json.loads(payload)
        if ev["type"] == "llm":
            print(ev["reply"])          # the words
            print(ev["script"])         # with its cues
            print(ev["loras"])          # what was merged, and at what weight
        elif ev["type"] == "start":
            print("first audio after", ev["ttfa_server_ms"], "ms")
        elif ev["type"] == "end":
            print(ev["audio_sec"], "s, realtime factor", ev["rtf"])
```

Event types: `llm` (the director's output and everything it decided), `start`
(first audio, latency figures), `best_of` (the candidate ranking, when best-of-N
ran), `end` (totals, `align` report, `prompt_unc` when guidance ran), `error`.

The `llm` event also carries `prompt` on `start`: the `<user_inst>` block
**exactly as the voice model received it**. That is the field to read when a take
sounds wrong — it is the ground truth about what was asked for.

### What you can override per request

Everything the page's controls do. The full list is in
[`DEFAULTS.md`](DEFAULTS.md); the ones that matter most:

| field | default | |
|---|---|---|
| `persona`, `persona_custom` | `host` | character brief |
| `profile` | `emolia_c1699` | which of the ten voices |
| `brain`, `prompt_style` | `luna`, `prose` | which language model, prose or codes |
| `language` | follows the user | `English` / `German` |
| `seed` | random | fixes sampling |
| `skills` | `true` | measured burst recipes vs the older hand-written rules |
| `align` | `true` | trim to the script at both ends |
| `best_of`, `best_of_guidance` | `1`, `3.0` | generate N and rank them |
| `gen_mode`, `guidance` | `adapter`, — | steering / classifier-free guidance |
| `quality_lams`, `qdpo_lams`, `adapter_overrides` | see `DEFAULTS.md` | any adapter, any weight |
| `burst_set`, `burst_lam_max` | `recipe`, `1.5` | which burst adapters, and their ceiling |

`adapter_overrides` is the general escape hatch: `{"sft3_voicenet:S_RANT_high": 1.5}`
forces any adapter at any weight, whatever the director decided.

---

## Reproducing this server elsewhere

1. **Hardware.** Two 24 GB GPUs. The speech model takes ~10 GB in bfloat16 on
   one; the aligner, scorers, speech recognition and the local language model sit
   on the other. One card works if you use a hosted language model and skip
   best-of-N.
2. **Install.** Python 3.12, `pip install -r requirements.txt`. `flash-attn 2` is
   **not** compatible with this architecture — the model loads with
   `attn_implementation="sdpa"`. `ffmpeg` must be on `PATH`.
3. **Assets**, in this order:
   ```bash
   export HF_HOME=/path/with/space
   python setup/fetch_profile_refs3.py      # reference corpus, ~3.5 GB kept
   python setup/build_retrieval_index.py    # condition centroids + emotion anchors
   python setup/profile_traits.py           # measured gender/age/timbre per voice
   python setup/fetch_scorers.py            # the two perceptual scorers
   ```
   `fetch_scorers.py` is easy to skip and then hard to diagnose: `best_of_n`
   imports `genuineness_scorer` and `blend_model`, which are Python files that
   ship inside the two model repositories rather than on PyPI. Each also expects
   a copy of the VoiceCLAP encoder beside it; the script links the one
   `build_retrieval_index.py` already fetched instead of downloading it twice,
   which is why it runs after that one.
   The adapters download from the Hub into the directories in
   `config.LORA_ROOTS`; `wikiskills/` is committed here, so the burst recipes
   need no download at all. The two lever assets are release assets on this
   repository; without them the levers degrade to `adapter` and say so.
4. **Check.** `python setup/check_levers.py` (48 checks, no GPU needed) and
   `python -m pytest tests` (burst vocabulary).
5. **Run.** `./run.sh both`, then `http://localhost:8792`.

Every setting is an environment variable as well as a config entry, so nothing
above needs a code edit to change. `MOSS_` prefixes throughout —
[`DEFAULTS.md`](DEFAULTS.md) lists all of them with their current values.

---

## Restoration

Every endpoint that returns finished audio takes a `sidon` boolean, defaulting
to `SIDON_ON` (**on**):

| endpoint | effect |
|---|---|
| `POST /api/turn` | the reply is generated offline instead of streamed, restored, and returned; every candidate carries `pcm` (restored) and `pcm_raw` (as generated) |
| `POST /api/cfg_sweep` | each guidance value is restored; the take carries `pcm` and `pcm_raw` |

```bash
curl -s localhost:8792/api/turn -H 'content-type: application/json' -d '{
  "message": "Tell me something that made you laugh this week.",
  "best_of": 3, "sidon": true }' --output turn.bin
```

With `best_of > 1` the candidates are restored **before** they are scored, so
the reward ranks the audio that will actually be heard. `sidon: false` restores
streaming.

The service is separate: `./run.sh sidon` (port 8793). If it is not running the
server still works and returns unrestored takes; `GET /api/state` reports
`sidon: {on, ok, clips, ms_per_clip}`.
