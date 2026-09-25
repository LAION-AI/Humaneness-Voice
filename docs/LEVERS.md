# Three levers, not one: generation modes

Until now this server had exactly one way to shape a performance — load adapters and write a
good prompt. Two more have since been measured on this checkpoint, and so has the way all
three combine. This document is the write-up of the modes that expose them, and of the
evidence each default rests on.

Everything numeric below comes from three studies in `LAION-AI/Voice-Acting-Pipeline-WIP`,
`research-log-2026-08/`: `lora-dose/` (5,740 cells, 79 adapters × 6 weights), `cfg-study/`
(guidance and its true cost), and `combination-study/` (the 2×2×2 factorial over all three
levers, 60 attributes). Where a number appears here it is the number in those files, not a
rounded one. **Nothing in this document has been run on this server's hardware** — see
*What has not been tested* at the end, and the smoke test asked for in the pull request.

---

## 1. The three levers

| lever | what it does | cost |
|---|---|--:|
| **adapter** | a rank-16 LoRA at a merge weight, as today | free |
| **steer** | `h ← h + α · (v/‖v‖) · ‖h‖` at the last position of each forward pass | 1–5 extra kernel launches per token |
| **cfg** | `logits = logits_uncond + g · (logits_cond − logits_uncond)` | **1.93×** |

`α` is **dimensionless**: it is the fraction of the current hidden state's own magnitude
added along the direction. That normalisation is not cosmetic — raw difference-vector norms
span three orders of magnitude across the 36 layers (‖h‖ runs 6.4 at the embeddings to 2237
at layer 35), so a fixed absolute α would be a no-op in one place and a catastrophe in
another.

`g = 1` cancels the unconditional term exactly and *is* ordinary sampling, so it is both the
"off" value and the control: with `g = 1` both branches still run, both caches still advance,
and the same draws happen in the same order.

---

## 2. The best single lever flips by family

Main effects on the target attribute, in SD units, from the factorial
(`combination-study/stats/analysis.json`, `t2.by_family`; n is attribute × prompt cells):

| family | adapter | t | steering | t | guidance | t | n |
|---|--:|--:|--:|--:|--:|--:|--:|
| emotion | +0.077 | 2.81 | **+0.384** | 9.41 | +0.050 | 1.84 | 399 |
| delivery | +0.377 | 7.10 | **+0.614** | 9.62 | +0.026 | 0.44 | 170 |
| quality | **+0.399** | 5.99 | +0.006 | 0.01 | +0.062 | 0.94 | 30 |

Three things follow, and they are the whole of the mode policy.

**On the quality axes the adapter is the only lever that does anything.** Steering moves them
by +0.006 at t 0.01 — not a small effect, an absent one. It is refused there, and the
steering study independently found that the quality directions break at k ≥ 2 layers
(genuineness −2.81 at k = 5). Genuineness, burst blend and aesthetics get `adapter`.

**On an emotion, steering is five times the adapter** and the two are **additive**:
interaction +0.038 at t 1.36, which is nothing. A combination whose interaction is null is a
combination you can reason about, so `adapter+steer` is the default for emotions.

**On a delivery axis the two levers do the same job.** adapter × steering is **−0.164
(t −3.7)** and adapter × guidance is **−0.125 (t −3.5)** — both significantly sub-additive.
Stacking them buys less than either alone would suggest. **Pick one.** The server picks the
adapter by default (`MOSS_DELIVERY_LEVER`), because the delivery adapters are the
best-behaved family in the stack and were swept at scale, which the steering vectors have not
been on this hardware.

---

## 3. Steering × guidance is the only real synergy, and it is a coupled package

On the emotion family the steering × guidance interaction is **+0.277 (t 7.5)** on the
target. The same term carries:

| carried by steering × guidance | value | t |
|---|--:|--:|
| target (SD) | +0.277 | 7.51 |
| word error | +0.078 | 11.80 |
| genuineness | −0.862 | −21.53 |
| burst realisation | −0.070 | −6.73 |

**Every damage term has a larger |t| than the gain.** This is not an argument for never
combining them; it is an argument for combining them the cheaper way. Steering **both** CFG
branches rather than only the conditioned one keeps 82 % of the effect and returns 0.209 of
word error and 0.75 of genuineness, so that is what `MOSS_CFG_STEER_BRANCH` does by default.

---

## 4. The modes

| mode | what runs | when it is right |
|---|---|---|
| `auto` | resolved per family, below | the default; right almost always |
| `adapter` | today's behaviour, unchanged | the quality axes; anywhere latency is tight |
| `adapter+steer` | adapter plus a steering vector | emotions — the two are cleanly additive there |
| `adapter+cfg` | adapter plus guidance | emotions where steering has not reached the band |
| `steer` | steering without the attribute's own adapter | delivery axes, where the two overlap |
| `cfg` | guidance without the attribute's own adapter | delivery axes, likewise |

`auto` resolves **emotion → `adapter+steer`**, **delivery → `adapter`**, **quality →
`adapter`**, and **never spends guidance**: 1.93× is not something to pay by default.

A lever-only mode (`steer`, `cfg`) drops **only the adapter carrying that attribute**. The
DPO adapter, the voice profile and the quality axes stay exactly where they were — which is
what `lora_w = 0` meant in the study, where `sft3_dpo:p2 @ 1.0` was always on.

Every mode is switchable off:

| switch | effect |
|---|---|
| `MOSS_GEN_MODE` | pins every turn to one mode; default `auto` |
| `MOSS_STEER=0` | steering off entirely |
| `MOSS_CFG=0` | guidance off entirely |
| `MOSS_AGENT_PICKS_MODE=0` | the director may not choose; `MOSS_GEN_MODE` decides |
| `MOSS_DELIVERY_LEVER` | `adapter` (default) or `steer` on a delivery axis |
| `MOSS_NUMBNESS` | `with_steer` (default) or `off` |

A request body may also carry `"gen_mode"`, which overrides both.

---

## 5. The settings, and where they come from

**α = 0.10 at the attribute's own top layer is the free setting.** Emotion percentile
0.4354 → 0.5840 with word error *falling*. Everything breaks above α = 0.3, and at α ≥ 0.5 a
random direction of matched norm does the same damage — which is how an earlier round
concluded, wrongly, that steering never works. The correction came from a control, not from a
better idea.

**k is per attribute and small.** Emotion is free only at k = 1; the quality axes break at
k ≥ 2; the delivery axes want 3–5. There is no global setting, and the layers differ per
attribute: Anger peaks at h21 h20 h19, genuineness at h12 h13 h21, blend at h25 h22 h20.
`taps: "top1"` means *that attribute's own best layer*.

**Two ceilings, and they are different numbers.** A single component may not exceed
`MOSS_STEER_ALPHA_CEILING` (0.15, half the break point). The **realised** magnitude at any
one layer may not exceed `MOSS_STEER_REALISED_CEILING` (0.25). Components that share a layer
sum there, and it is not a quadrature sum: `cos(emotion direction, quality axis)` runs −0.62
to −0.95 in this representation, so *subtracting* `Emotional_Numbness` adds almost entirely
**along** the emotion direction. Measured over the forty emotion recipes with the numbness
subtraction attached, the realised magnitude runs to **0.1926** (`emo:Interest` at h20;
Elation 0.1907, Amusement 0.1781). Those are compositions the study actually ran, so a
ceiling at 0.15 would have refused a measured recipe — which is precisely the check that
found this, and `setup/check_levers.py` now asserts it. Past the ceiling a composition is
**refused, not trimmed**: a trimmed composition is not the one that was measured.

**The numbness subtraction is free and automatic.** Subtracting `Emotional_Numbness` at
α = −0.10 returns **+0.60 of genuineness (t 9.64, on 67 of 80 prompts)** at no cost in
emotion when the adapter is carrying the emotion. It is attached to any emotion turn where
the steering machinery is already running. It is deliberately *not* attached to a bare
`adapter` turn: that mode is the fallback for everything, and it has to stay bit-for-bit
what it was.

**g = 3.0 for emotion, g = 2.5 for delivery**, at word error ≤ 0.20. Below g = 1 guidance
actively hurts (−0.0370 at g = 0.5, t −2.56), which is the directional control the arm needed.

---

## 6. Guidance does not stream, and that is the honest answer

The measured cost is **1.93×** at batch 1 — 1.89–1.94 over four cells, sd 0.053. The
intuition that only the semantic transformer doubles is wrong: the local transformer doubles
too, running twelve times per frame *per branch*, because the unconditional branch needs its
own local KV state to predict the next channel. The split is semantic 72–77 %, talker
18–19 %, heads 4–7 %, codec decode **1.6 %** — and the decode is the only genuinely shared
component, so sharing it saves nothing.

[`ADAPTERS.md` §1](ADAPTERS.md) records realtime factor **1.0** as the streaming budget and
**0.764** as the live merged baseline. 1.93 × 0.764 ≈ **1.47**, so a guided take generates
slower than it plays and the player would starve. This is the same argument that killed
forward hooks for the whole adapter stack at 1.75×; it applies here with more force.

So **a guided take is rendered whole and then played.** The chunk size is set larger than any
possible take, exactly one decode happens, and the response payload says `"streaming": false`.
Time-to-first-audio becomes the whole generation time. That is a real cost and the director is
told about it in its own prompt.

**What is left as a next step.** The two-branch loop itself
(`tts_engine._stream_frames_cfg`) yields on the same chunk boundaries as the single-branch
one, so nothing about it is structurally non-streaming — the arithmetic is. A streaming
guided path needs the cost down, not the loop changed: batching the two branches into one
forward pass instead of running them sequentially is the obvious candidate and is untried.
Until someone measures that, shipping a mode that claims to stream and then starves the
player would be worse than shipping one that says plainly what it does.

### Why the branches interleave at channel level

The twelve audio channels of a frame are sampled autoregressively: channel *c+1*'s logits
depend on the token sampled for channel *c*. A frame-level CFG — run branch A for a whole
frame, then branch B — would leave the two branches conditioned on **different channel
prefixes**, and their difference would then be mostly about that divergence rather than about
the condition. Both branches are therefore advanced with the **same** sampled token at every
channel step.

The continue/end decision is taken on the **conditional branch alone**. It is a structural
token: guiding it would change how long the clip is, and the duration error would then be
measuring guidance's effect on the stopping rule rather than on the performance.

### What "neutralised" means

The unconditional branch is the same prompt with the affect removed and nothing else:

* `GENERAL:` keeps the identity sentence, the base register, the continuity clause and the
  recording-quality clause, and loses the director's delivery line and the `reads as …`
  clause. It is built in `llm_agent._clean`, which is the only place that knows which clause
  was the director's and which are standing — a regex downstream would be guessing.
* `SCRIPT:` loses every round bracket **without** a number, which is the format's own
  definition of a delivery direction, and keeps every round bracket **with** one, which is a
  vocal burst. Square brackets — durations and pauses — are untouched.
* The words, the burst tags, the timings and the `Tokens` budget are **byte-identical**. A
  direction is zero seconds long, so removing it cannot change the arithmetic;
  `setup/check_levers.py` asserts that it does not.

The neutralised prompt is in distribution: 20 % of the CFG-DPO corpus had its instruction
words removed and 15–30 % of every supervised round rendered scripts without directions.

---

## 7. What the director may ask for

A second tool, `choose_generation_mode`, rides in the same constrained pass as
`select_reference_voice` — a separate tool-call turn would add a full prefill and decode to
time-to-first-audio for a decision the model can make in the same breath as the line.

```json
"perform": {"mode": "auto|adapter|adapter+steer|adapter+cfg|steer|cfg",
            "dimension": "<optional: an emotion, a delivery adapter or a quality axis>",
            "strength": "gentle|moderate|strong"}
```

`dimension` is drawn entirely from names the director has already been shown; there is no
second naming scheme. Left out, the lever pushes whatever `voice` or `style` already chose.

**`strength` is never a number.** A language model that is allowed to pick a dose picks it
wrong ([`ADAPTERS.md` §4.4](ADAPTERS.md)), so the three words map to a fixed table:
`moderate` is the attribute's measured **balanced** operating point, `strong` its
**high-effect** one, and `gentle` is the balanced point at half α. The server clamps
everything afterwards regardless.

**Code mode does not get this call.** The terse code language (`d`/`s`/`l`/`sp`) has no
`perform` key, so a code-mode turn always runs `auto`. Adding a fifth key there is a
straightforward follow-up; it was left out rather than guessed at, because the attribute a
code-mode turn is pushing has to be read out of the blend rather than named.

The director's prompt tells it, in this order: reach for a delivery axis before pushing a
feeling harder (which is the rule the training team added in `44ea226`, and it comes first);
use `adapter` alone for the quality axes because nothing else moves them; add steering for an
emotion; and spend guidance only when the first two have not reached the band.

---

## 8. Where the assets come from

Two files, neither of them in git history. Both are published as **release assets on this
repository**, which is public, so no token is needed:

```bash
B=https://github.com/LAION-AI/Humaneness-Voice-Demo-Server/releases/download/assets-2026-08-29

curl -fL -o /mnt/nvme/moss-15-v2-assets/wikiskills/coefficients.json --create-dirs \
  $B/coefficients.json
curl -fL -o /mnt/nvme/moss-15-v2-assets/steering/p3_vectors_server.npz --create-dirs \
  $B/p3_vectors_server.npz
```

Those are the paths `config.WIKI_COEFFICIENTS` and `config.STEER_PACK` already default to.
Keep the `-f`: without it curl writes an HTTP error page to the target path and the failure
surfaces later as a puzzling parse error. Check the hashes, then run
`python setup/check_levers.py`.

| file | bytes | sha256 | what it is |
|---|--:|---|---|
| `coefficients.json` | 324,511 | `7e029f0e…13e840` | the measured operating point for each of 60 attributes |
| `p3_vectors_server.npz` | 5,334,470 | `d81b6971…6de78f7` | 99 attributes × their own top 5 layers × 2560, float32 |

> The research log `LAION-AI/Voice-Acting-Pipeline-WIP` is **private**, so a `raw`
> `githubusercontent` link into it returns 404 for anyone outside it. That is why these are
> mirrored here rather than linked there.

The research library the vectors are distilled from is **112 MB** — three difference tables
of 99 × 38 × 2560 — and none of it belongs in git. `setup/build_steering_pack.py` reduces it
to the one table and the few layers the server can actually reach, on a machine that has the
research artefacts (they are not on the demo box):

```bash
python setup/build_steering_pack.py \
    --vectors  $SC/out/actforensics/vectors/p3_vectors_ext.npz \
    --tap-rank $SC/work_vb/tap_rank.json \
    --out      /mnt/nvme/moss-15-v2-assets/steering/p3_vectors_server.npz
```

float32 rather than float16 is deliberate: a steering direction is a *difference* of means,
one to two orders of magnitude smaller than the means themselves, and half precision would
quantise it to a few significant bits.

`coefficients.json` is generated from the measured JSON by
`wikiskills/code/build_wikiskills.py` in the research log, alongside one pattern page per
attribute. It is generated rather than written because a hand-typed table drifts.

`MOSS_STEER_PACK` and `MOSS_WIKI_COEFFICIENTS` point at them.

**Each lever is gated on what it actually uses, and only that.** The first version of this
gated both on the coefficient table, which meant the 0.4 MB download was holding up the lever
that needs no vectors at all — reported by the demo team on PR #3, and they were right.

| you have | `auto` | `adapter` | `adapter+cfg`, `cfg` | `adapter+steer`, `steer` |
|---|---|---|---|---|
| neither file | `adapter` | ✅ | ✅ family default | `adapter` |
| coefficients only | measured per attribute | ✅ | ✅ measured `g` | `adapter` |
| vector pack only | `adapter` | ✅ | ✅ family default | ✅ family default |
| both | measured per attribute | ✅ | ✅ | ✅ |

* **Guidance needs neither asset.** `g` has a family default measured in the CFG study — 3.0
  for emotion, 2.5 for delivery, at word error ≤ 0.20 — and the neutralised branch is built
  from the turn's own prompt. So a box with no assets at all can still test guidance, the
  neutralised branch and the director's tool use.
* **Steering needs the vector pack and nothing else.** The pack carries the per-dimension
  layer ranking embedded in it, so α = 0.10 at the attribute's own top-k layers — the
  measured free setting — is reachable without the coefficient table.
* **The coefficient table is what `auto` reads**, and what turns a family default into this
  attribute's own measured operating point.

**`auto` stays strict.** It is a claim about what was measured for this attribute, so with no
row to read it makes no claim and asks for no lever. An explicitly requested mode is an
operator or a director overriding that, and it runs on the documented family default with
`"operating_point": "family_default"` in the payload and a reason line saying so. **A default
from the study is not a guess**; what is refused is inventing a per-attribute number nobody
measured.

The refusals that are *findings* rather than missing recipes survive with no coefficient
table at all: steering is still refused on a quality axis and on a `_low` tail, and a
delivery adapter and a delivery steering vector still never both run.

A dial that reads a value while the thing it names is switched off is worse than no dial
([`LEARNINGS.md`](LEARNINGS.md)), so every degrade is reported in `/api/state` and in the
response payload, never silent.

---

## 9. What comes back

The `llm` event and the final `end` event both carry a `levers` block:

```json
{"mode": "adapter+steer", "mode_requested": "auto",
 "attribute": "emo/Anger", "family": "emo",
 "strength": "moderate", "operating_point": "balanced",
 "steer": [{"key": "emo:Anger", "alpha": 0.1, "taps": "top1", "layers": [21]},
           {"key": "emo:Emotional_Numbness", "alpha": -0.1, "taps": "top1", "layers": [20]}],
 "steer_branch": "cond", "realised_alpha": {"20": 0.1, "21": 0.1},
 "guidance": 1.0, "cost_factor": 1.0, "streaming": true,
 "dropped_adapter": null, "reasons": []}
```

`mode_requested` against `mode`, and `reasons`, are the point of the block: every downgrade
is recorded in the order it happened, so a bad take can be traced to its configuration rather
than guessed at. A lever refused *after* the plan was made moves the mode word with it — a
payload reading `adapter+steer` while nothing is being steered is exactly the dial that reads
a value while the thing it names is switched off. One extra word can appear there: `none`, for
the case where a lever-only mode dropped the attribute's adapter and the lever was then
refused inside the engine. It is rare, it is not offered to the director, and it is reported
rather than quietly relabelled `adapter`, because by then the adapter really is gone. The same is written to the log as one `[levers]` line per turn, including
the realised magnitude at each layer. `/api/state` reports what the box can actually do.

---

## 10. Sixty attributes, and seven of them have nothing

The coefficient table has a balanced operating point for **53 of 60** attributes and a
high-effect point for **56**. Seven — `emo/Bitterness`, `emo/Disappointment`, `emo/Disgust`,
`emo/Emotional_Numbness`, `emo/Relief`, `emo/Shame` and `vn/ARSH_low` — have **no
configuration that clears the balanced guardrails**.

For those the server runs the adapter and records why. It does not substitute a nearby
setting, and it does not interpolate. An absent recommendation is a finding, and the one
failure mode a coefficient table exists to prevent is filling a gap with something plausible.

The same applies to the low tails of the delivery axes. The vector table holds the
high-*minus*-low difference, and the two tails of an attribute are **orthogonal, not
opposite** (median cos −0.0004), so −α along it is not "the low tail". No balanced or
high-effect recipe for any `_low` axis uses steering, and the server refuses to invent one.

**And a floor, not a zero.** The matched random-direction control is null on its own
(−0.033 pooled) but **not** null at the combined operating point: **+0.106, t 2.78**. A
recommendation only counts if it beats that, which is what the `beats_random_floor` flag in
the coefficient table records.

---

## What has not been tested

* **`adapter`-mode bit-identity has not been byte-checked.** The offline check shows the
  injector is a genuine no-op object on that path and the loop shape is untouched, which is
  evidence and not proof. `setup/ab_codes.py` does the actual comparison — same seed, same
  prompt, same adapter set, on two checkouts, comparing the generated **code tensors** rather
  than the audio, since the decode is deterministic given identical codes.
* **None of this has run on the demo box.** `setup/check_levers.py` verifies the parts that
  do not need 4.55 B parameters — that the table and the vector pack agree, that the resolver
  refuses what the measurements say it should, that the neutralised prompt keeps the
  arithmetic identical, and that the injector's arithmetic matches its own definition and
  leaves no hooks behind. It cannot check what the audio sounds like or what any of it costs
  in wall-clock here.
* **The steering hooks' cost is arithmetic, not a measurement.** [`ADAPTERS.md` §1](ADAPTERS.md)
  records that forward hooks for the whole adapter stack took the realtime factor from 0.737
  to 1.29 — but that was **536 extra kernel launches per token**. This is one to five, each a
  norm and a fused multiply-add on a `[1, 1, 2560]` slice, which is the regime §2 already
  accepted for the twelve tied-module hooks ("twelve hooks is not 536"). Nobody has measured
  it here.
* **1.93× is measured on the study's hardware, not this one.** The ratio should carry — it is
  dominated by running the transformer twice — but the absolute realtime factor will not.
* **Every number in the studies is one model's judgement of another model's output.** No
  listening test has been run on any steering, guidance or combination result.
* **Ten prompts per cell.** The family-level rows above pool hundreds of cells and are solid;
  a single attribute's row is ten prompts and is not.
* **English and German only**, and almost everything measured so far is English.

## Guidance: what a listener said

The family defaults here (3.0 emotion, 2.5 delivery) were chosen by word error
rate. A person using the CFG sweep in the chat page reported that guidance stays
usable **up to about 4.0 and degenerates above it** — the first evidence anyone
has about how these values sound rather than how they score. The strength slider
allows 1–5 deliberately, so the limit can be heard rather than asserted. See
[`FIELD_NOTES.md`](FIELD_NOTES.md).

## What this server actually ships, and why it is not `auto`

**`GEN_MODE` defaults to `adapter`.** Clone this repository, install the assets, run it, and
you get adapters and a prompt — no steering, no guidance. Everything below still works and is
one environment variable away:

```
MOSS_GEN_MODE=auto|adapter|adapter+steer|adapter+cfg|steer|cfg
```

The reason is a listening report, and it is worth recording in full because it is the kind of
result the rest of this document cannot produce.

Running with `auto`, the resolver chose steering on nearly every turn — `α = +0.10` on the
target attribute and `−0.10` on `Emotional_Numbness`, and on some turns it dropped the emotion
adapter and steered instead. The measured attribute scores went the right way. A person
listening described the output as *more emotional, with strange artefacts, and the timbre off*,
and asked for it to be rolled back. With `GEN_MODE=adapter` the same person called it good
again.

Nothing in the numbers predicted that, and nothing in them contradicts it either. **Every
figure in the steering and guidance studies is one model's judgement of another model's
output**, which those studies say themselves — "no listening test has been run" is in their own
honest-limits section. A genuineness score and a human ear disagreeing is not a contradiction;
it is the blind spot of the instrument.

So: the levers are built, tested, documented and reachable. They are not the default until
somebody has listened to them on purpose. The obvious next experiments, in order of how cheap
they are:

1. **Halve α.** 0.10 is the study's balanced point; 0.05 was never listened to either.
2. **Delivery axes only.** The complaint arrived on emotion steering. Delivery is where the
   measured effect is largest anyway, and it is a different set of directions.
3. **Drop the `−Emotional_Numbness` component.** Subtracting a second direction doubles the
   perturbation at a shared layer, and the realised magnitude at `h20` is where the recipes
   come closest to the ceiling.
4. **A/B on fixed lines, listened to blind**, rather than scored. That is the measurement
   nobody has made.

One thing that is *not* suspected: the adapter path itself. `setup/ab_codes.py` compared the
generated RVQ codes before and after generation modes existed — same seed, same prompt —
and got the same sha256 over all 84 frames × 12 channels. Whatever changed the sound, it was
a lever being pulled, not the plumbing moving.
