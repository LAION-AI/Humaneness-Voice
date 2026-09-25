# Tips from the training side

Written 29 August 2026 by the team that trained the adapters this server loads,
after reading `LEARNINGS.md` and the code. It has two halves: what this PR
changed and why, and a handful of things that are easy to get wrong with this
particular model and are not obvious from the outside.

`LEARNINGS.md` is a good piece of debugging and the conclusions in it hold up.
Nothing below contradicts it.

---

## What this PR changed

### 1. Shared weights are now detected, not listed

`lora_bank.py` already refused to merge into `audio_lm_heads.*`, with a correct
comment explaining why: `tie_weights()` makes `audio_lm_heads.N.weight` and
`audio_embeddings.N.weight` **the same tensor**, so folding a head delta into the
weight silently rewrites the audio embedding table as well. That was the right
call and it was already in place.

The problem is that it is a **hand-maintained list of names**. The same model
makes a second tie the pattern does not cover:

```python
def tie_weights(self):                       # modeling_moss_tts.py:114
    self.text_lm_head.weight = self.transformer.embed_tokens.weight
    for embedding, head in zip(self.audio_embeddings, self.audio_lm_heads):
        head.weight = embedding.weight
```

Nothing in this stack adapts `text_lm_head` today, so that one is latent — but
latent is exactly what the audio-head tie was until an adapter set started
targeting it, and a regex cannot notice a new tie appearing in a future
checkpoint.

Asking the model is cheaper than remembering. Two parameters are the same tensor
exactly when they share storage, so `_tied_paths()` groups every module weight by
`data_ptr()` and treats any group with more than one owner as tied. The union of
that and the existing regex decides what gets hooked. There is also now an
assertion on the merge path, because the failure mode here is silent: the model
keeps generating, slightly wrong, and nothing in the logs says so.

### 2. The delivery adapters are no longer capped at 0.75

The comment in `config.py` said, correctly at the time:

> The sweep that produced 1.5 for the emotion adapters has NOT been repeated on
> this set, and the set has not been evaluated at all, so the demo stays well
> under the trained value of 1.0.

It has been evaluated now — a 5,740-cell dose-response sweep of 79 adapters at
six weights, scored against **each axis's own VoiceNet regression** rather than
against side effects alone, which is the measurement `sweep_voicenet.json` was
missing.

* 16 of the 17 delivery adapters have a usable weight. It is the best-behaved
  family in the stack: 12 monotone, 4 saturating, 1 below resolution.
* Median safe **and** strong weight is **1.5**.
* Going 0.75 → 1.5 buys **+0.375** on the target axis (t 5.18, better on 15 of
  17) for a word-error change of **+0.003** — t 0.55, statistically nothing.
  Individual gains reach +1.37 (`S_RANT_high`) against a noise floor of ~0.15.

`SFT3_VN_LEVELS` is therefore `(0.5, 0.75, 1.0, 1.25, 1.5)`. **If
`SFT3_VN_MAX` is ever raised above 1, bring this ladder back down with it** —
stacking is still untested, and the 0.75 ceiling was originally justified by
two axes running at once.

### 3. Aesthetics no longer silently fights ranting

Measured: pushing the aesthetic axis alone moves it +0.196…+0.317. Pushing
ranting alone moves `S_RANT` +0.464 (t 7.01, on **12 of 12** prompts). Both at
the same strength: **−0.012**, indistinguishable from zero. The two directions
are close to opposed in the model's representation, so asking for both produces
neither.

`config.QUALITY_CONFLICTS` records the pairs and `app.py` scales the quality
adapter down when the conflicting delivery axis is in play, with a log line
saying so. `S_DRAM_high` is halved rather than dropped: same family, not tested
pairwise, and halving is the conservative guess.

---

## Things that are easy to get wrong with this model

### Never merge an adapter into the weights without checking for ties

Covered above, but it generalises: **`merge_and_unload()` is unsafe on this
checkpoint**, and so is any hand-rolled equivalent. The published model cards
say so and they mean it literally. If you want the speed of merged weights,
merge everything *except* the shared modules and hook those.

### The duration budget is a contract, not a ceiling

`LEARNINGS.md` reached this conclusion the hard way and it is worth stating as a
rule: this model spends the time it is given. It does not overrun and it does not
stop early — it fills. Anything that loosens a budget produces filler, not
silence. The corollary for future work: a *tight* budget is a feature, and the
right lever for over-generation is always the budget, never the stop bias.

### Adapter families do not behave alike, and the differences are large

From the same 5,740-cell sweep:

| family | usable | median weight | shape |
|---|--:|--:|---|
| delivery (VoiceNet) | 16 / 17 | **1.5** | 12 monotone, 4 saturating |
| emotion | 11 / 40 | **1.0** | climbs to 1.0, then flat |
| quality | 1 / 3 | 1.5 | right direction, misses significance |
| vocal burst | 4 / 19 | 1.5 | mixed |

Two things this table does not say on its own. **Emotion adapters saturate at
1.0** — pooled they climb to +0.035 at w = 1.0 (t 4.82) and then stop
(1.5 − 1.0 = −0.0026, t −0.40). This server already runs them at 1.0, which is
right; going to 1.5 is not harmful, just past the point of return. And **"below
resolution" is not "no effect"**: the 24 quiet emotion adapters still move when
pooled (+0.024 at w = 1.0, t 2.70). Ten takes per cell is not enough to prove one
adapter at a time.

**No adapter in the sweep was harmful at any weight.** Every failure was a
failure to move the target, not a broken guardrail.

### Manner is easy to steer, feeling is not

Three completely different mechanisms — a weight patch, a hidden-state nudge and
a logit extrapolation — all move the **delivery** axes 18–20× further than the
**emotion** heads. That matches what a probe can read out of the same
activations (R² 0.85 for voice quality against 0.47 for emotion). If a scene
needs to land emotionally, reach for a delivery axis (`S_DRAM`, `S_WHIS`,
`S_RANT`) before reaching harder for an emotion adapter.

### Consider a different ASR for the harness

> **Note added when merging (demo side):** this server and its harness were
> already on `nvidia/parakeet-tdt-0.6b-v3` before this PR — `config.ASR_MODEL`
> and `eval_tail.ASR` both read it. There is no Whisper in this stack and no
> Whisper thresholds to keep. The advice below is right; it was already taken.

`LEARNINGS.md` records that ten samples at one seed is not a result, which is
true. Part of that variance is the ruler: on identical audio, Whisper
large-v3-turbo is **2.4× noisier** than `nvidia/parakeet-tdt-0.6b-v3`. Switching
buys resolution without generating more takes — `transformers` ≥ 5.14 ships
`ParakeetForTDT`, so no new dependency and no NeMo.

We switched expecting Whisper to *flatter* broken audio by hallucinating fluent
text. It does not: measured across 29,676 clips the offset does not grow with how
hard the model is being pushed (slope −0.0008 per unit of guidance, t −0.19), and
Whisper is if anything the harsher instrument on the worst clips. **Your existing
Whisper thresholds are fine.** The reason to switch is precision, not bias.

### Your `extra_w` metric is better than ours

Counting the words that appear *after* the last one still matching the reference
is a more direct measure of over-generation than anything in our harness. We are
adopting it.

---

## Two open questions we would take off your hands

**Does the audio sound duller than it should?** If the demo has lost some high
end, the first thing to test is whether it is the adapter stack rather than the
prompt: the audio codec is residual, so the later codebooks carry the fine
detail, and anything that perturbs the deeper channels shows up as a loss of air
and sibilance rather than as an obvious artefact. A one-adapter-at-a-time A/B on
a fixed line would settle it in an afternoon. We have not measured spectral
content at all — every score in both projects is a scoring *model*, and none of
them would notice a gentle low-pass.

**Does a long session drift?** `RESYNC_EVERY = 25` bounds the merge/unmerge
rounding, and with the tied modules now hooked the dangerous tensors are never
written at all. What is not measured is whether 25 turns of bf16 drift on the
remaining modules is audible. Generating the same line at turn 1 and turn 24 of a
session and comparing would answer it.

---

## Smoke test on merge (demo side, 29 August 2026)

Run before merging, as asked:

* **Tie detection works and finds more than the regex did.** `_tied_paths()`
  reports **26 modules in 13 groups**: the twelve `audio_lm_heads.N` /
  `audio_embeddings.N` pairs the regex already knew about, plus
  `transformer.embed_tokens` / `text_lm_head` — exactly the latent tie predicted
  above. Confirmed independently against a freshly loaded checkpoint.
* **Still latent.** None of `sft3_dpo:p2`, `sft3_quality:*`, `sft3_voicenet:*`,
  `sft3_emotion:*` or `sft3_voice:*` carries LoRA on `text_lm_head` or
  `embed_tokens` — all 268 of their target modules are elsewhere. So this is a
  guard against a future checkpoint, not a live bug that was being hit.
* **The interference rule fires correctly.** Forcing `S_RANT_high` at 1.5
  dropped `sft3_quality:esthetics_high` from 0.5 to 0 with the intended log
  line, and the adapter is absent from the applied set.
* **A full turn generates normally** under the merged code.

Everything cited here is in `LAION-AI/Voice-Acting-Pipeline-WIP`,
`research-log-2026-08/` — `lora-dose/` for the dose-response sweep,
`cfg-study/` for guidance, `combination-study/` for interactions, and
`layer-forensics/` for the probes and steering vectors.
