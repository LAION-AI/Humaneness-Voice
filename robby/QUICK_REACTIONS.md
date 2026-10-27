# Local contextual quick reactions and conversation-first layout

**Current implementation:** [two-stage, memory-informed speech](TWO_STAGE_SPEECH.md).
The sections below preserve earlier experiments; their one-preface race, 140-ms
reserve and direct handoff were superseded on 2026-10-27. TTS now uses 0.8; the
local reaction text model remains at 0.35.

## Quality follow-up (2026-10-26)

See [the complete quality report](static/quick-quality-report.html) for 40 authored
situations, three seeds per eligible situation, rejected prompt variants, all
final outputs, four real TTS examples and a six-turn Qwen conversation. The local
model remains Bolt E4B at temperature 0.35; larger sampling temperatures made
phrasing worse. Tone-specific English/German prompt examples now separate play,
grief, support, affection, corrections and curiosity. They guide generation;
they are never selected directly as canned audio. Recent openings are excluded,
and exact repeated output is skipped without retrying. Play uses an amusement
TTS cue; serious speech retains soft delivery. Main-answer generation still starts
immediately and wins the race whenever it is ready first.

The blinded model reviewer preferred the change in 19 cases, the baseline in 9,
and tied 10; two malformed judgments remain in the receipts. Engagement rose
from 2.61 to 3.13/4 but naturalness fell from 3.58 to 3.45/4. These are advisory
model ratings, not human scores. Factual-topic prefaces remain too generic in
some cases. The six-turn test also exposed weak/repeated jokes in the **main
Qwen answer**, despite improved task routing; this is not certified as fixed.
Local generation median was 198 ms versus 197 ms previously (no new online
judge, retry or model). This quality study does not establish physical playback
below 900 ms.

Additional repairs: compound joke requests and critical “another one” follow-ups
preserve the task; “Please speak English” is honored; urgent help and trivial
arithmetic omit the preface; action promises are rejected; duplicate short
condolences are removed at the handoff without deleting substantive content.
The initial held-out run exposed the language issue, so the final rerun is a
regression check, not untouched hold-out evidence. Test code, prompts and every
rejected trial are published alongside the report.

Robby can ask its existing local **Nemotron 3.5 Bolt E4B Q4_K_M** for a
one-to-six-word contextual reaction while Qwen/Cerebras produces the complete
answer. No additional model download is required. This replaces the proposed
canned bridge bank in `BRIDGE_PHRASES_PLAN.md`; that PDF remains historical.

## Behavior and trade-off

The main request starts immediately. Its first complete sentence races the local
reaction. If the main sentence is ready first, or the local call fails, times out
(700 ms), abstains or violates the six-word contract, Robby proceeds directly to
the main answer. No fixed fallback phrase is substituted. Exact-output requests,
poems, explicit sentence/line counts, Stop and optional avatar events skip the
reaction. Local-only main answers skip the extra local call.

The reaction is a complete short utterance in the conversation language. Its
prompt sees the recent conversation and uncertain delivery hint, but forbids
invented facts, action promises, questions and spoken waiting announcements.
The main model starts with useful content rather than another acknowledgment.
Only a nearly identical short first acknowledgment is suppressed. The complete
answer, tools, verse structure and cancellation are preserved.

**Earlier sound is not an earlier substantive answer.** The shared TTS model
speaks the reaction and then the main answer, with a single playback sequence.
A short reaction takes time to say. Both first PCM and first main-answer PCM are
reported separately. A perfect acoustic splice is not guaranteed; this is two
separately synthesized utterances with the same reference voice.

The short reaction uses a 160-ms initial generated chunk followed by 400-ms chunks, 10-ms boundary fades and no
forced-alignment lookahead. Its requested duration is 2–2.4 seconds (depending
on word count). The main answer retains the existing 1600-ms chunks and
alignment/fades. Internal TTS transport uses direct PCM with `live_only: true`: no
per-chunk MP3 processes/downloads and no ranking/export of a single live take.
Normal Playground MP3 defaults remain intact. Older servers retain the MP3 fallback. The `seconds` parameter conditions speech duration; it is NOT
a hard maximum. An initial 6.4-second value stretched four words unnecessarily.

A tested additional three-sentence transition was rejected: it added roughly
0.4–0.5 s of model latency plus extra speech and could repeat or contradict the
main answer. The implemented route hands off directly.

## Controls and implementation

**Models & quick replies → Local quick reply** controls the feature.
It defaults on, persists as `robby.quickReplies`, and applies to subsequent
turns. FastAPI `/api/bolt/chat` accepts `quick_replies: false` to disable it.
`ROBO_QUICK_REPLIES=0` changes the server-side initial preference.

The avatar and recent conversation are now the first row, with component
latencies underneath. Model, camera, privacy, recognition, personality and
conversation controls are moved into the existing collapsible settings below.
Their IDs/listeners are retained. Camera remains off by default.

Core files: `quick_reaction.py`, `bolt_pipeline.py`, `bolt_director.py`,
`conversation_director.py`, `tts_bridge.py`, `server.py`, `bolt_api.py`,
`static/playground-layout.js`, `static/conversation-first.css`, `static/ui.js`,
`static/live-overview.js`. `bolt_text.py` now safely ignores malformed optional
acting cues rather than dropping the spoken answer.

## Reproduction and limitations

Run with the pinned local llama.cpp model and the normal shared TTS/ASR stack:

```bash
python run_unit_checks.py
python benchmark_quick_reactions.py
python benchmark_quick_reaction_audio.py
python browser_quick_reaction_check.py --url https://YOUR-DEMO
```

The first benchmark compares 16 local/API generation races and three longer
transition probes. The second generates three paired on/off trials through real
API and TTS. Outputs are stochastic and not length-matched. The HTTPS browser
check covers full answers, a four-line poem, disabling/persistence, PCM playback,
manual cancellation, mobile layout and preservation of existing controls.
Never run ASR-browser tests against an occupied single-slot conversation server.

The live host has two shared RTX 3090 GPUs; these timings do not establish an
idle-device benchmark or physical microphone/speaker round trip. Optional short
reactions can still be generic or repetitive. This experiment does not certify
all room-noise or acoustic Stop cases; the previously documented misses remain.
See `static/quick-reaction-report.html` and its receipts for actual results.

## Under-900-ms experiment

`benchmark_quick_onset.py` deliberately cancels after the first PCM block to
isolate onset. Eight actual local/API/TTS trials produced 1311 ms on the first
post-restart trial, then 707–827 ms across seven warm trials: **7/8 below 900 ms**,
median 745 ms. These are server HTTP timings, not a microphone-to-ear claim.
Browser/network/startup buffering is reported separately in the HTTPS receipt.

`benchmark_quick_chunking.py` compared identical text, reference, seed and
settings: uniform 160-ms chunks took 2862 ms to generate 2.48 s of audio; a
160-ms first / 400-ms subsequent schedule took 1975 ms. This changes yield
boundaries, not token sampling. The codec's block-size-dependent floating-point
output was not bit-identical (maximum sample difference 0.01953125 in this trial);
this is not proof of perceptual equivalence. Earlier uniform-small-block browser
runs had audible-scheduling gaps and are retained.

The reaction's browser scheduling reserve is 140 ms initially, with adaptive
increases after underruns. This does not alter TYGER's timestamp-based echo
reference. Real network jitter and acoustic-device latency can still exceed the
900-ms target; no unconditional upper latency guarantee is made.
