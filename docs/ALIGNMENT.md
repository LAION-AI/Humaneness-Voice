# End-trimming: fade in at the first word, cut after the last

## The problem, and why the obvious fix cannot work

Takes sometimes say the line and then keep going — a few more words, often not
real words. The tempting fix is to stop the audio at the duration the prompt
asked for.

That does not work, and the reason is the most useful fact in this document:
**the audio already matches the requested duration to within 0.02 s**, in every
condition ever measured, clean or degraded (`EXPERIMENTS.md` §8). The model does
not overrun. When it improvises it speaks the real line slightly faster and puts
the filler *inside* the same budget. There is no overhang to trim.

So the fix has to be content-based: something must notice that the words which
were asked for have been said.

## Why alignment rather than recognition

We already know which words were supposed to be spoken, so this is an alignment
problem, not a transcription problem. An aligner cannot invent a word that was
never in the script; a transcriber can, and then the trim point would be decided
by a hallucination.

## Two backends, and why the default changed

The first implementation used the MMS CTC aligner, which is **CC BY-NC 4.0** —
the only non-commercial licence in a stack that was otherwise chosen for
commercial usability on purpose. It has been replaced as the default by an
Apache-2.0 model, and both remain available:

| | licence | how | en | de |
|---|---|---|:-:|:-:|
| **`Qwen/Qwen3-ForcedAligner-0.6B-hf`** (default) | **apache-2.0** | one non-autoregressive forward pass; takes the transcript directly | ✅ | ✅ |
| `romara-labs/mms-300m-1130-forced-aligner-ONNX` | cc-by-nc-4.0 | Wav2Vec2 CTC emissions (31 tokens, 20 ms/frame) + `torchaudio.functional.forced_align` | ✅ | ✅ |

`MOSS_ALIGN_BACKEND=qwen,mms` is the order of preference; each falls through to
the next if it does not load.

Measured on the same clips:

| | Qwen | MMS |
|---|--:|--:|
| 10 s English | **38 ms** | 61 ms |
| 14 s German | **68 ms** | 135 ms |
| VRAM loaded | 1.84 GB | ~0.4 GB |
| word ends, mean disagreement (German) | — | 0.057 s |
| word ends, worst disagreement | — | 0.220 s |

Qwen is about twice as fast despite being three times the parameters, because it
is a single pass rather than emissions plus a Viterbi search. It is also far less
code to drive: it takes the transcript as text, so the romanisation, the
diacritic stripping and the ß mapping the CTC path needs all disappear. It costs
1.84 GB, which does not fit beside the voice model on a 24 GB card, so it goes on
the language model's card (`MOSS_ALIGN_QWEN_DEVICE`, default `cuda:1`).

**One thing is genuinely worse.** Qwen returns no per-word confidence. The CTC
path has a posterior per token, and the threshold on it is what refuses to edit
when the words were probably not found where the aligner thinks. For Qwen the
`score` is a **structural** check instead — right number of words, in order,
inside the audio, none implausibly long. That catches gross failures and will
not catch subtle ones, and the consequence is stated here rather than hidden in
the code.

**Qwen places word ends more generously**, including more of the decay. That
makes the tail rule stricter by construction: fewer takes have enough room after
the last word to be worth cutting (18 of 36 against 26 of 36). Whether that is
better or worse is a listening question, not a metric one.

The Viterbi pass on the CTC path is deliberately not hand-rolled. A trellis is
exactly the kind of code that is subtly wrong for a long time, and torchaudio
ships a tested one.

## The two edges

**Lead-in.** If the script does not open on a vocal burst, everything before the
first word is throat-clearing nobody asked for. Fade in so full level is reached
exactly at the first word's onset. When the script *does* open on a burst, the
opening is left completely alone — the burst is the performance.

**Tail.** After the last scripted word, keep a little air, ramp down over 150 ms,
and drop the rest.

**A scripted closing burst is protected.** A line ending on `(breathy giggle)`
had 1.03 s removed and read as a clean success on every metric — extra words
1 → 0, word error 0.10 → 0.00 — while actually deleting the giggle the director
wrote. Burst tags carry their own lengths, so the allowance is now exact rather
than guessed. This was found by reading the per-take rows, not the summary.

## Streaming

A stream cannot retract what it has played, so the guard keeps a lookahead: audio
is held `ALIGN_LOOKAHEAD_S` (0.5 s) behind generation, and the aligner runs on
everything generated so far every `ALIGN_EVERY_S` of new audio. Once the last
word is found to have ended, the tail is faded and the remainder dropped.

Two things had to be different from the offline path, and both were bugs first:

* **The lead-in needs a longer first look.** Measured offline, first-word onsets
  run to 1.3 s. Deciding at 0.5 s asked the aligner to fit the whole script into
  half a second, which fails — so the fade silently never fired in the stream
  while firing 28 times in 36 takes offline. The first decision now waits for
  `ALIGN_LEAD_SCAN_S` (1.3 s) and asks about the first three words only.
* **The tail must not be looked for too early.** Forced alignment always places
  every target somewhere, so against a prefix it reports the last word as
  finished while half the line is unspoken — and that cut removes real speech.
  The tail is only considered once `ALIGN_TAIL_AFTER` (60 %) of the requested
  duration exists, which the duration contract makes reliable.

Cost: the aligner runs on CUDA at **50 ms per 10 s of audio**, against 1000 ms on
CPU. That difference decides whether streaming is possible at all, and it depends
on load order — `onnxruntime` finds cuDNN and cuBLAS because torch has already
pulled them into the process, so the aligner is constructed *after* the TTS model.
Loaded first it falls back to CPU without failing.

Time to first audio grows by the lead-in scan, about 0.7 s, once per reply.

## Measured

36 takes: twelve scripts × three seeds, on the shipped stack. Each line was
generated once with trimming off and the trim then applied to that same audio, so
both halves are the identical take and the only difference is the edit.

| | |
|---|--:|
| takes edited | 32 / 36 (89 %) |
| lead-in fired | 28 / 36 |
| tail cut fired | 26 / 36 |
| removed, mean | 0.366 s |
| removed, max | 1.790 s |
| **invented words** | **0.22 → 0.03** |
| word error | 0.033 → 0.020 |

The invented-word column is the point: `extra_w` counts words transcribed after
the last one still matching the script, and it falls by 86 %. Word error falling
too says the trim is not eating real speech — if it were, that number would rise.

Both `opens_burst` scripts were left alone at the lead-in on every seed, which is
the rule working rather than a coincidence.

### The two backends are not a controlled comparison

Run with the Qwen backend, same twelve scripts, same three seeds, same stack:
31 of 36 edited, lead-in 29, tail cut 18, removed 0.225 s mean, invented words
**0.06 → 0.03**, word error 0.008 → 0.011.

Those numbers cannot be compared with the MMS table above, because the
*pre-trim* figures differ between the runs — 0.06 invented words against 0.22,
and word error 0.008 against 0.033. Same seeds, same adapter stack, same lines:
the generations themselves were not the same, so the two runs measure different
audio and the backend difference is buried under that.

What is established about it:

* **Within one server session, generation is bit-identical.** Four consecutive
  requests with the same seed produced the same sha256.
* **It is not merge drift.** A probe line was generated, 28 further takes were
  run to carry the merge counter past `RESYNC_EVERY = 25`, and the probe
  reproduced its hash exactly — twice. That also answers the open question in
  `TIPS.md`: no measurable drift over a session of that length.
* **What did cause it is not identified.** Something about the server state
  across a restart, and the honest position is that it is unexplained.

So: both backends work, they agree on word timings to within about 60 ms, and
the licence and the speed are the reasons to prefer Qwen. Which one trims
*better* is not answered by these data.

### A preliminary signal worth following up

An interrupted run with the burst+stop preference adapter loaded showed **0.09
invented words before any trimming**, against 0.22 without it — the adapter
appears to reduce the improvisation itself. That was 11 takes against 36 and the
run did not finish, so it is a signal and not a result. It agrees with what that
adapter was trained for, which is a reason to test it properly rather than a
reason to believe it.

## Switching it off

Checkbox in the chat page, on by default. `MOSS_ALIGN=0` for the server default.
With the aligner absent the guard is never constructed and the streaming path is
exactly what it was.

Every edit is reported per turn — which edges fired, where, how much went away,
and how many alignment passes it took — in the `align` block of the final event
and as one `[align]` line in the log.
