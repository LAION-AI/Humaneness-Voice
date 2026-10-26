# October Playground API

These examples target `playground/runtime/small_app.py`, **not** the older root
app's `/api/speak`. See FastAPI `/docs` on your server for the current validated
schemas. The examples assume your own provisioned, idle local app at port 8792.
Keep one cookie jar across upload, generation and audio download; references
and audio are session-private, not public URLs independent of the session.

## State, assets and switching

```bash
curl -fsS http://127.0.0.1:8792/api/state
curl -fsS http://127.0.0.1:8792/api/models/catalog
curl -fsS http://127.0.0.1:8792/api/ranking/catalog
curl -fsS http://127.0.0.1:8792/api/voices/catalog
curl -fsS http://127.0.0.1:8792/api/reward/catalog
curl -fsS http://127.0.0.1:8792/api/models/select \
  -H 'content-type: application/json' -d '{"model":"small_rl"}'
```

Wait for `ready: true`. `model_default` and `model_id` are different concepts:
one is the configured startup choice; the other is the currently loaded model.
Only one TTS model is resident in VRAM. Switching/generation is serialized;
busy requests return 409 rather than starting competing GPU jobs. CPU prefetch
and first-use GPU compilation are not instant. Read checkpoint/base/adapter
hashes and actual batch caps from the catalog/state, not a stale browser label.

## One manual take, no LLM and no reference

```bash
curl -N -fsS -c cookies.txt -b cookies.txt \
  http://127.0.0.1:8792/api/turn -H 'content-type: application/json' \
  -d '{"model":"small_rl","input_mode":"manual","format":"structured",
       "language":"en","voice_mode":"none","voice_conversion":false,
       "message":"GENERAL: Gentle, natural and spontaneous delivery.\nSCRIPT:\n(warmly relieved) [3 seconds duration] I am glad you came back.",
       "generation":"single","n":1,"streaming":true,"chunk_frames":10,
       "temperature":0.7,"guidance":2.0,"align":true,"seed":777}'
```

`/api/turn` returns **Server-Sent Events**, not one MP3 response. Parse `data:`
JSON events: model/checkpoint/reference information, director/spec/status,
`audio_chunk`, `result` or `error`, and end-of-stream. Inspect event keys in
the [actual smoke records](../playground/evidence/rl-live.json) and UI consumer
in `index.html`. Audio chunk/result URLs must be fetched with the same session.
A successful HTTP 200 starting SSE does **not** imply generation succeeded;
check error events and the final result.

`chunk_frames: 10` means 800 ms; `20` means 1600 ms. Smaller single-take codec
chunks are also supported by the API. Chunk duration is not time-to-audible:
codec, optional VC, ASR buffering, fades and network each add latency.

## Reference upload and lifecycle

```bash
curl -fsS -c cookies.txt -b cookies.txt \
  -F 'file=@/path/to/consented-speaker.wav' \
  'http://127.0.0.1:8792/api/reference?slot=A'
```

The response includes `id`, normalized seconds and current references. In the
turn JSON set `"voice_mode":"custom"` and `"reference_id":"RETURNED_ID"`;
do not supply a catalog profile simultaneously. Uploads are capped at 20 MB,
normalized, checked and cropped before codec encoding; default maximum context
is 15 s. Custom references remain until removed, rather than expiring after
one hour. Generated output audio has a separate default one-hour TTL.

```bash
curl -fsS -b cookies.txt -X DELETE \
  http://127.0.0.1:8792/api/reference/RETURNED_ID
curl -fsS -b cookies.txt http://127.0.0.1:8792/api/audio/RESULT_ID -o take.mp3
```

Generated browser audio is mono MP3 at 96 kbps. Alignment and scoring operate
on PCM before MP3, not on a recompressed playback file.

## Parallel Best-of-N and alternative controls

Use `"generation":"best","n":8,"streaming":false`. `"generation":"all"`
returns all candidates. API N is bounded at 12; actual simultaneous execution
uses the selected model's cap. For example Small N=3 was one GPU batch `[3]`;
Large RL N=3 was `[2,1]`, not three independent serial calls. The requested
manual `batch_size:1` does not force serial Best-of-N. Actual executed batches,
OOM retries and on-device timing are in `result.performance`.

Default ranking can be sent explicitly:

```json
{"ranking":{"mode":"ears_mix","model":"medium","metric":"ce","weight":0.5}}
```

Use `ears_only` for raw-quality-only sorting. For adaptive/speaker/burst
combinations select `ranking.mode: "legacy"`; the schema rejects contradictory
hidden mixtures. See [formulas/settings](PLAYGROUND_PROMPTING.md).

Chat or exact-text direction uses `input_mode: "chat"` or `"perform"` and
requires the [director configuration](PLAYGROUND_LLM.md). Manual mode bypasses
it. External legacy adapters and old hidden-state vectors are rejected for
these checkpoints, even if an old client sends their fields.

Experimental search uses `generation: "search_winner"` or `"search_beam"`,
`search_n`, 800/1600-ms `search_chunk_frames` and a default 30-s search cap.
Its published historical study and current model-specific limits must not be
conflated. In particular, do not claim the RL precision path is verified for
chunk search; see [validation boundary](PLAYGROUND_EXPERIMENTS.md#current-rl-validation-boundary).

## Timing and production boundaries

`aggregate_rtf = TTS/codec device seconds / sum(generated audio seconds)` for
single/Best-of-N. It excludes LLM, reference encoding, ranking, alignment, VC,
MP3 and network. Search timing has a different scope that includes its judges/
selection. Aggregate batch RTF below 1 does not promise first-audio latency or
one interactive stream below 1. Cold graph capture is reported separately.

The demo is not a production multi-user authentication/rate-limit service.
Protect GPU access, uploads and logs with your deployment's authorization,
consent, retention and abuse controls. HTTPS alone is not authorization.


### Internal low-latency single-stream transport (26 October 2026)

`/api/turn` accepts optional `stream_transport: "pcm_f32"`, `live_only: true`,
and `first_chunk_frames: 2`. The normal `chunk_frames` controls later blocks.
`live_only` requires `generation: "single"`, streaming, and PCM. It preserves
configured alignment but skips ranking and full-take MP3 export after all live
chunks have been emitted; the final result has an empty candidates list and
`scoring_ms: 0`. PCM chunk events carry base64 little-endian float32 mono
`pcm_f32` and `sample_rate: 48000`. The default public transport and download
format remain mono 96-kbps MP3. This is used by Robby's server-to-server bridge;
its browser still receives the existing 24-kHz PCM WebSocket stream.

The optional first-block scheduling currently applies to the compiled fast
backend. It changes codec yield boundaries, not the sampling policy.
See [Robby's measured latency and caveats](../robby/QUICK_REACTIONS.md).
