# SIDON restoration: it fixes the sound, not the performance

[`sarulab-speech/sidon-v0.1`](https://huggingface.co/sarulab-speech/sidon-v0.1)
is a speech restoration model — MIT licensed, `w2v-bert-2.0` as a feature
extractor into a DAC decoder, both shipped as TorchScript. It takes 16 kHz in
and returns 48 kHz, which matches this server's rate exactly.

It was tested because the arena run kept saying the same thing about our output:
of 1,140 clips, **`pleasant` returned a 5 exactly five times**. The ceiling was
the sound, so it was worth asking whether cleaning the sound raised the ceiling.

## How it was measured

Twenty-one clips drawn across the whole score range (totals 0 through 15), each
judged twice by `gemini-3.8-flash` with the identical rubric prompt: once as
generated, once after restoration. **Both sides were judged fresh.** Reusing the
stored score would have charged all of the judge's own run-to-run noise to the
enhancement, which is how a null becomes a result.

## What it does

| rubric | original | SIDON | delta | t | p |
|---|--:|--:|--:|--:|--:|
| pleasant | 2.71 | 3.14 | **+0.43** | 3.87 | **0.000** |
| fit | 2.19 | 2.38 | +0.19 | 0.85 | 0.397 |
| natural | 2.10 | 2.14 | +0.05 | 0.24 | 0.813 |
| total | 7.00 | 7.67 | +0.67 | 1.48 | 0.138 |

Better on 10 of 21, worse on 4.

**One dimension moves, and it is the one the model is for.** `pleasant` gains
0.43 of 5 at p < 0.001. `natural` does not move at all, and the justifications
say why in almost the same words as before:

> *original:* "It sounds like a speed-read text-to-speech model that completely
> breaks down into synthetic babble."
> *restored:* "The pacing is unnaturally rapid and metronomic before completely
> devolving into synthetic gibberish at the end."

Restoration cleans the signal. It does not change the timing, the phrasing or
the over-generation, so every complaint about the *performance* survives it
intact — including the trailing babble, which comes back sounding tidier.

## When to use it

**Offline, on a take worth keeping.** 7.3 s per clip on CPU, so it is far too
slow to stream, but it is a reasonable last step before exporting a chosen
take — the audible gain is real and it costs nothing but time.

**Not as a fix for anything the director controls.** If a take is rushed,
metronomic or runs past its script, restoration will hand back the same take
with a better noise floor.

Reproduce with `eval/sidon_test.py <n>`; the per-clip pairs land in
`runs/sidon_test.json`.

---

## It is the best lever we have measured for how the voice sounds

Across two arena runs and 2,949 rated clips, **nineteen prompt additions changed
the sound by nothing that survived a paired comparison.** The best of them came
in at +0.64 of 15 with p = 0.099 against a Bonferroni threshold of 0.0042, and
both of the promising ones from the first run changed sign when measured again.

SIDON moves `pleasant` by **+0.43 of 5 at p < 0.001** on the first attempt, with
no tuning and no search. On the one axis it touches it is not merely the best
lever in these studies — it is the only one that produced a significant effect
at all. It was shipped **on** for that reason and turned back **off** on
6 September after a listening report: on screams and loud vocal bursts it
distorts. The twenty-one clips behind the +0.43 were drawn across the score
range of an ordinary conversational run and contained no screaming, so the
number is right for ordinary speech and silent about the loud tail — which is
exactly where a restoration model trained on clean read speech has least
support. Turn it on by hand for a quiet take.

It is worth being exact about the scope: this is a claim about **how the voice
sounds**, not about how well it acts. `fit` and `natural` do not move, and the
arena run is clear that those two are where the system is actually weakest
(`natural` averaged 2.21 of 5 over 1,800 clips). Restoration raises the floor of
the sound and leaves the performance exactly where it was.

## Why it might work: a hypothesis, not a measurement

The demo stacks five or more LoRA adapters at once — the DPO adapter, the
speaker's own identity adapter, emotion, VoiceNet delivery, quality, sometimes a
vocal-burst adapter on top. **The suspicion is that each one costs a little
high-frequency detail, and that the stack ends up perceptibly duller than the
base model would be alone** — which is exactly the deficit a restoration model
trained on clean speech would put back.

Two things make this more than a guess, without making it a finding:

* The audio codec is **residual**, so the later codebooks carry the fine
  detail. Anything that perturbs the deeper channels shows up as a loss of air
  and sibilance rather than as an obvious artefact — quiet, cumulative, and
  invisible to every scorer in this stack, since all of them are models and none
  measures spectral content.
* The team that trained the adapters raised the same question independently,
  before SIDON was tested here, and it is still open in
  [`TIPS.md`](TIPS.md): *"If the demo has lost some high end, the first thing to
  test is whether it is the adapter stack rather than the prompt."*

**The experiment that would settle it** is cheap and has not been run: generate
one fixed line at several adapter counts — base alone, base + DPO, then adding
identity, emotion, VoiceNet, quality one at a time — and measure spectral
centroid and high-band energy on each, with and without restoration. If the
hypothesis holds, the high band should fall monotonically with the stack, and
SIDON's gain should grow with it. Until that is run, this is a plausible story
that fits the evidence, and it is written here as one.

---

## How it is wired in

Restoration needs the whole take, so **any turn that will be restored is
generated offline rather than streamed**, then sent down the same chunked
protocol. The UI says so on the toggle.

| where | what happens |
|---|---|
| ordinary chat turn | generated offline as a single take, restored, played; the take as generated is kept beside it as a second player that never plays by itself |
| best-of-N | all N candidates are restored **before** they are judged, so the reward is computed on the audio that will actually be heard; every candidate carries both versions |
| CFG sweep | each guidance value is restored, and both versions appear under it |

The `sidon` boolean on `/api/turn`, `/api/cfg_sweep` and the UI checkbox all
default to `SIDON_ON`, which ships **off** (see above). Turning it on costs
the streaming.

### It runs in its own process, and that is not an accident

Both halves are TorchScript with their constant tensors baked in at trace time,
and constants move with **neither `map_location` nor `.to()`** — so the CUDA
build runs on `cuda:0` and nowhere else, and the CPU build runs on the CPU and
nowhere else. Asking for `cuda:1` loads cleanly and then fails inside the first
layer norm.

Loading it inside the app would therefore put it on the TTS card, where it costs
**1.67 GiB and leaves 0.25 GiB of slack** — not enough for a guided best-of-N
batch, which has run out of memory on this box before. So `run.sh sidon` starts
it as a small service with `CUDA_VISIBLE_DEVICES` pointing at the language
model's card: its `cuda:0` is that card, and generation never sees the memory
disappear.

Measured on the service: **103 ms for one clip, about 1 s for eight**, against
7.3 s per clip on CPU. Every failure is soft — if the service is down, the
originals are returned and the only sign is a log line.
