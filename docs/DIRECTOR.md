# The director's manual

This is the authoritative page for the language model that directs a turn, and
for anyone changing what that model is told. Everything here is checked against
the code; where a rule exists because something was measured, the number is
given. If this page and the code disagree, the code is right and this page is a
bug.

Source files: `llm_agent.py` (what the director is told), `timed_script.py` (what
the server does with it), `skills.py` (which sounds exist), `config.py` (every
weight), `app.py` (how the two are assembled).

---

## 1. What the director's job is

The director is a language model that receives a user message and returns **one
JSON object** describing a single spoken turn. It does not produce audio. It
decides:

- **what to say** — the words,
- **how they are performed** — a delivery cue before every sentence, vocal
  bursts standing between them, where the silences fall,
- **which reference recording** the voice model conditions on,
- **which adapters** are merged, and at which of a fixed set of strengths,
- **which of the three levers** pushes the performance.

It does **not** decide timing. Every number in the final prompt is computed by
`timed_script.render()`. This is the single most misunderstood part of the
contract and section 3 is about nothing else.

Two prompts exist. `SYSTEM` (`llm_agent.py:90–291`) is the prose director. When
the request asks for `style="codes"` and a codebook is loaded, `CODE_SYSTEM`
(`llm_agent.py:440–479`) replaces it and the cues become codes instead of prose.
The rest of this page describes the prose director; the code director follows the
same bracket rules with a smaller vocabulary.

---

## 2. The output contract

### 2.1 The JSON object

Built by `build_schema()` (`llm_agent.py:294–384`). `voice`, `voice2`, `style`,
`perform`, `speed`, `language`, `delivery` and `script` are **all required** —
left optional, the model simply never emitted them (`llm_agent.py:380–381`).

| field | type | what it does |
|---|---|---|
| `voice` | object, `mode` required | picks the reference recording the take is conditioned on |
| `voice.mode` | `emotion \| voicenet \| character \| edge_case \| sports \| none` | which axis to pick along |
| `voice.emotion` | one of the catalog emotions | the emotion name |
| `voice.intensity` | `intense \| moderate` | how far it runs |
| `voice.containment` | `free \| contained` | let out, or held back and leaking at the edges |
| `voice.dimension` | a VoiceNet code, e.g. `S_WHIS` | for `mode: voicenet` |
| `voice.level` | `extremely_low \| moderately_low \| moderately_high \| very_high` | where on that axis |
| `voice.character` | a catalog character | for `mode: character` |
| `voice.edge_case` | a catalog edge case | scream / sob / laughter references |
| `voice2` | same eight keys, no `sports` | an optional **second** reference, concatenated after the first |
| `speed` | `much_slower \| slower \| normal \| faster \| much_faster` | swaps in a faster or slower take of the reference |
| `style` | array, at most `SFT3_VN_MAX` = **1** | the delivery adapter |
| `style[].adapter` | one of the 17 `SFT3_VN_ADAPTERS` | which talking-style axis |
| `style[].strength` | one of `0.5, 0.75, 1.0, 1.25, 1.5` | its merge weight |
| `perform.mode` | `auto \| adapter \| adapter+steer \| adapter+cfg \| steer \| cfg` | which levers run |
| `perform.dimension` | an emotion, a delivery axis, or a quality axis | what gets pushed |
| `perform.strength` | `gentle \| moderate \| strong` | a **word**, never a number |
| `language` | `English \| German` | told to the voice model |
| `delivery` | string | the standing GENERAL line for this turn |
| `script` | string | the performance, cues in position |

There is deliberately **no `reply` field**: the spoken words are the script with
its cues stripped, and generating them twice would cost about a third of the
output tokens for nothing (`llm_agent.py:298–302`).

`voice2` has an ordering rule: the state the line **starts** in goes in `voice`,
the state it **ends** in goes in `voice2` (`llm_agent.py:132–134`).

### 2.2 Brackets

Two bracket kinds reach the director, and a third meaning exists that the
director must never produce.

| written | means | example |
|---|---|---|
| `( … )` with no number | a **delivery direction** — how to speak the words after it | `(clearly amused, letting it show)` |
| `( … )` naming a burst | a **vocal burst** — an actual sound | `(chuckle)` |
| `[pause]`, `[long pause]` | a **beat** | `[pause]` |
| `( … , N.N seconds)` | a burst **with its duration** | written by the server only |
| `[N.N seconds duration]` | a speech segment's length | written by the server only |

`timed_script.py` resolves them in this order (`parse`, and the module
docstring): square bracket = seconds; round bracket **with** a number = a burst;
round bracket **without** a number = either a burst or a direction, decided by
whether it names a known label.

### 2.3 The hard rules, and what breaks when each is broken

**A vocal burst is always its own bracket.** Naming one inside a delivery cue
produces no sound at all, because the whole bracket is read as an instruction
about how to speak. `(clearly amused, with a small chuckle)` yields no chuckle;
`(clearly amused) … (chuckle) …` yields one, with its own slot in the timing.
Stated at `llm_agent.py:244–248`.

**Never a number inside any bracket.** A round bracket that contains a number
stops being a direction and becomes a vocal burst: `(quietly, 2 seconds)` is
performed as a sound, not as an instruction. Stated at `llm_agent.py:249–251` and
again at `473–475`. **Nothing in the code repairs this** — `_sanitise_script`
does not strip numbers — so the prompt rule is the only defence, and a violation
is silent.

**No capitals, anywhere, in a cue or in the spoken line.** This model spells
capitalised words out letter by letter: `AAAGH!` comes out as
"ay-ay-ay-gee-aitch". Write a scream as a cue — `(a raw, tearing scream)` — and
leave the words in ordinary lower case. `llm_agent.py:256–259`, and for the
`delivery` field at `217–218`. Partially repaired: `_sanitise_script`
(`llm_agent.py:779`) lower-cases runs of **three or more** capitals. Two-letter
all-caps words survive, and the comment above that line saying ">= 2" is wrong
about its own regex.

**No adapter names or underscore identifiers in a cue.** `(ga_pain_scream)` is
not a direction, it is a database key, and it will be read aloud.
`llm_agent.py:260–262`. Repaired at `llm_agent.py:776`, but only when the bracket
contains *nothing but* the identifier.

**Every line needs at least ten words.** Short lines get rushed and clipped.
`llm_agent.py:263–264`. **Nothing enforces this.** The `< 10` test at
`lora_bank.py:592` looks like enforcement and is not — it is choosing between the
inline and solo burst dose.

**Every sentence gets its own cue**, in round brackets, placed immediately
*before* the words it affects — not one cue for the whole reply
(`llm_agent.py:223–226`). And every cue names its strength with one of four
adverb bands, the same scale the model was trained against
(`llm_agent.py:236–243`):

| band | words |
|---|---|
| present but held down | barely / faintly / only slightly / just a little |
| plainly audible, controlled | clearly / plainly / noticeably / unmistakably |
| running hard | strongly / intensely / very / deeply |
| at the limit | overwhelmingly / extremely / utterly / completely |

**Never a `[pause]` directly after a burst**, and **never open or close the
performance with a burst** — words must follow it. `llm_agent.py:265–266`, and
three more times in the prompt. Partially repaired: `_sanitise_script:781` drops
a pause tag after *any* round bracket, and `:789` drops a trailing cue, which
enforces "never close". **Nothing enforces "never open".**

**Square brackets are only for beats.** Anything else in square brackets is
rewritten into round brackets (`_sanitise_script:784–786`) so that a burst
written as `[a soft sigh]` is not stripped as markup and lost entirely.

**No emoji and no markdown anywhere** — everything outside the brackets is spoken
aloud (`llm_agent.py:269`).

**Never describe the voice itself** — age, gender, timbre, accent, who it belongs
to. That is fixed and is added for the director. Describing it would recast the
part between turns (`llm_agent.py:198–201`).

---

## 3. The two layers, and their opposite rules about numbers

This is the part that is easy to get backwards, and getting it backwards
silently converts every delivery cue into a burst.

**The director layer** writes the performance and **never a number**.

**The timed-script layer** is `timed_script.render()`, computed by the server on
every turn. It adds **every** number, and in the SFT3 format those numbers are
mandatory:

- every speech segment is preceded by `[N.N seconds duration]`,
- every gap of `PAUSE_MIN` = 0.20 s or more by `[N.N seconds pause]`,
- every burst becomes `(label, N.N seconds)`,
- `Tokens:` is the sum of all of it at `FRAME_RATE` = **12.5** frames per second.

So the duration **exists and is mandatory in the format** — and the director must
**never write it**. The numbers are not left to the language model because
getting one to produce a set of decimals that sums to a given budget is a bad
bet, and length control is the one thing this checkpoint honours best
(`timed_script.py` docstring).

The constants, from `timed_script.py:26–33`:

| constant | value | why |
|---|--:|---|
| `BURST_DEFAULT` | 0.28 s | the median burst in the training data |
| `BURST_MIN` / `BURST_MAX` | 0.14 / 1.2 s | the 10th and 90th percentiles are 0.14 and 0.48; a burst asked for at 3 s is outside anything the model saw |
| `PAUSE_DEFAULT` | 0.30 s | |
| `PAUSE_MIN` | 0.20 s | below this, no gap is emitted |
| `LEAD_PAUSE` / `TAIL_PAUSE` | 0.30 / 0.30 s | a gap before the first word and after the last |
| `SEG_MAX` | 12.0 s | segments over 12 s were split again in training |
| `TIMED_FRAMES_PER_WORD` | 4.0 | words → seconds, divided by the frame rate |

A trailing pause is always popped before rendering, so a take never ends on
silence the model would have to fill (`timed_script.py`, `render`).

---

## 4. A worked example, both layers

This is real output from `timed_script.render()`, not an illustration.

**User message:** something the character did not expect to hear said out loud.

**What the director writes** — the `script` field. No numbers anywhere; one cue
per sentence; the burst standing in its own bracket between two sentences; lower
case throughout:

```
(clearly amused, letting it show) i genuinely did not think you were going to
say that out loud. (chuckle) but you did, and now we both have to live with it.
[pause] (quieter, warmer, taking his time) give me a second to think about it
properly.
```

**What the server computes** — `timed_script.render()` output, which goes into
both `SCRIPT:` and `Text:` byte-identically:

```
[0.3 seconds pause] (clearly amused, letting it show) [4.2 seconds duration] i
genuinely did not think you were going to say that out loud. (chuckle, 0.3
seconds) [3.8 seconds duration] but you did, and now we both have to live with
it. [0.3 seconds pause] (quieter, warmer, taking his time) [2.9 seconds
duration] give me a second to think about it properly.
```

`frames` = **148**, i.e. 11.84 s at 12.5 fps. `timed_script.check()` returns
`(11.8, 148, True)` — the printed numbers add up to the stated budget.

Note what happened to each item:

- `(clearly amused, letting it show)` stayed a **direction**: round bracket, no
  number, not a known label. It contributes **0.0 seconds**.
- `(chuckle)` became a **burst** and picked up `BURST_DEFAULT`, rendered as
  `(chuckle, 0.3 seconds)`.
- `[pause]` became `[0.3 seconds pause]` from `PAUSE_DEFAULT`.
- Each sentence got a `[N.N seconds duration]` from its word count.
- A `[0.3 seconds pause]` was inserted at the head; the tail pause was popped.

**The GENERAL line**, folded to the one-line shape the format's own example uses
(`timed_script.general_line`):

```
intensely amused, the laugh sitting just under the words; reads as amusement; 11.8s, EN.
```

**The full instruction** handed to the voice model
(`tts_engine.py:619–626`):

```
GENERAL: intensely amused, the laugh sitting just under the words; reads as amusement; 11.8s, EN.
SCRIPT:
[0.3 seconds pause] (clearly amused, letting it show) [4.2 seconds duration] i genuinely did not think you were going to say that out loud. (chuckle, 0.3 seconds) [3.8 seconds duration] but you did, and now we both have to live with it. [0.3 seconds pause] (quieter, warmer, taking his time) [2.9 seconds duration] give me a second to think about it properly.
```

**And the unconditional branch**, when guidance is running
(`timed_script.neutralise`) — every round bracket *without* a number is gone,
every one *with* a number stays, the arithmetic is untouched, so the two branches
differ in affect and nothing else:

```
[0.3 seconds pause] [4.2 seconds duration] i genuinely did not think you were going to say that out loud. (chuckle, 0.3 seconds) [3.8 seconds duration] but you did, and now we both have to live with it. [0.3 seconds pause] [2.9 seconds duration] give me a second to think about it properly.
```

Now the failure mode, in the same example. Had the director written
`(clearly amused, 2 seconds)` instead, `_BURST_RE` would have matched it and the
line would have been performed as a **two-second sound** named "clearly amused",
with no adapter, and the words would have lost their direction entirely. That is
the whole reason for the no-numbers rule.

---

## 5. The adapter taxonomy

From `config.LORA_ROOTS` (`config.py:133–172`). Every path and weight below is
the value in the code.

### 5.1 On by default — the quality and aesthetics stack

These are not per-moment acting choices; they are the floor the whole demo stands
on, and the director does not pick them.

| adapter | weight | source | why that weight |
|---|--:|---|---|
| `sft3_dpo:p2` | **1.0** | `sft3_dpo`, local | the general-quality adapter of the recommended stack, at its published weight. p2 supersedes the first DPO adapter: reward 0.4757 vs 0.4708, and the only preference-tuned model in this line whose word error (0.0977) beats the supervised baseline it is built on (0.0987), with the highest emotion percentile of any of them (0.3541). Rank 64, alpha 128. |
| `sft3_quality:genuineness_high` | **0.25** | `sft3_quality`, local | genuineness raises its own score only below 0.5 and collapses intelligibility above 1.0 (0.176 at 1.25) |
| `sft3_quality:blend_high` | **0.5** | `sft3_quality`, local | vocal-burst blend; safe at any weight measured |
| `sft3_quality:esthetics_high` | **0.5** | `sft3_quality`, local | aesthetics |
| `sft3_qdpo:quality_dpo` | **1.5** | `sft3_qdpo`, local | preference-tuned, step 376. On by default; never listened to |
| the voice adapter | **0.25** | `sft3_voice`, local | see 5.5 |

All three quality adapters at 1.0 scored word error 0.116 with an invented word
in 60 % of takes; at 0.25 / 0.5 / 0.5 the same three score **0.055**
(`config.py:495–499`). That is why they are not simply at their trained value.

One measured interference rule (`config.QUALITY_CONFLICTS`): pushing the
aesthetic axis alone moves it +0.196…+0.317 and pushing `S_RANT_high` alone moves
its axis +0.464 (t 7.01, on 12 of 12 prompts), but **both at the same strength
gives −0.012, indistinguishable from zero** — the two directions are close to
opposed in the model's representation. So when `S_RANT_high` is active the
aesthetics adapter is scaled to **0.0**, and for `S_DRAM_high` to **0.5**.

**Off by default:** `sft3_qdpo:burst_stop_dpo` (0.0) and
`sft3_qdpo:quality_dpo_step1504` (0.0). `AESTH_LORA` is empty and
`AESTH_LORA_LAM` is 0.0 — that dial pointed at the parked 57-dimension VoiceNet
set and could not load anything, while the real aesthetics adapter was running at
its own weight elsewhere, which made a UI reading "0.00" actively misleading.

### 5.2 Vocal bursts

`LORA_ROOTS["burst"] = /mnt/nvme/moss-15-v2-assets/loras/sft3_burst` — **71**
SFT3-native adapters on local disk. The Hugging Face line
`laion/vocal-burst-lora-adapters` is **commented out**, so nothing resolves it
today.

A burst adapter is pulled in automatically when the script contains that burst.
The director does not name the adapter; it writes the sound.

**Which weight a class actually gets**, precisely:

1. With skills **on** (`MOSS_SKILLS=1`, the default) and a measured recipe for
   that class in `wikiskills/VOCAL_BURSTS.md` — **the measured weight for that
   class**, via `skills.weight_for()`, applied in `app.py`. These run **0.25 to
   2.3**.
2. With skills on but **no** measured recipe for that class — the flat fallback:
   `BURST_LAM` = **0.25**, or `BURST_LAM_INTENSE` = **0.5** when the burst stands
   alone as its own beat.
3. With skills **off** — the flat fallback for every class, so the two settings
   are comparable rather than half-mixed.
4. In all cases the result is capped at `BURST_LAM_MAX`, default **2.3**, which

> **Note added on the demo side, 6 September 2026.** The weights this page
> assumes are not the ones this server runs. `BURST_LAM_MAX` is **1.25**
> here, under a **total** budget of 2.0 across a turn, because two burst
> adapters at 1.5 each destroyed the line in 5 of 5 seeds under the full
> shipped stack — it is the sum that breaks it, not the single weight.
> See [`BABBLE.md`](BABBLE.md).

   changes nothing today. See the note in `config.py`: the 2026-09-05 addendum
   argues for 1.5 and the table above it has not been rewritten to match.

The flat 0.25 was **not** chosen because a higher dose drags the line towards the
burst — that reasoning predates the measurement. It was the ceiling the
*genuineness* gate imposed, and that gate has been retired on purpose: a scream
is not supposed to sound like a composed natural address, so falling genuineness
is the expected price of a burst, not grounds for exclusion. The gate that
remains is word error: paired Parakeet WER no more than **+0.104** against the
class's own w = 0 cell, and for inline scripts no more than **0.25** absolute.

**Which sounds exist.** Of the 71 adapters on disk, `skills.Skills.offerable()`
offers **36** — the ones with a measured family hit rate at or above
`SKILLS_MIN_HIT` = **0.15** that are not on the never-realises list. **19 classes
never realise at any dose under either prompt form**, and 17 more sit below the
bar. Every mouth class and every whistle class in the bank is on one of those two
lists, none above 0.017. They are not weak adapters that need a larger weight;
the sound is absent from what the model can produce, and the fix is data.

The nine highest-scoring offered classes, with the weight each actually gets:

| class | measured weight | hit rate (family) | recipe from |
|---|--:|--:|---|
| `exasperated_sigh` | 1.0 | 0.833 | §64 |
| `relief_sigh` | 0.8 | 0.75 | §51/52 |
| `chuckle` | 2.0 | 0.73 | §51/52 |
| `contented_sigh` | 1.5 | 0.68 | §51/52 |
| `soft_hum` | 2.0 | 0.68 | §51/52 |
| `wistful_sigh` | 1.0 | 0.633 | §64 |
| `cackle` | 1.5 | 0.633 | §64 |
| `nervous_giggle` | 1.5 | 0.633 | §64 |
| `guffaw` | 1.0 | 0.633 | §64 |

**How to write a burst**, each measured on this model
(`skills.Skills.prompt_block`):

| rule | measured |
|---|---|
| put the burst **between** sentences, not inside one | worse on 15 of 15 classes: hit −0.07…−0.12, miss +0.31…+0.37, t 8–10. The one inversion is `clears_throat`, better mid-sentence (0.250 → 0.483) |
| name the **cause** of the sound in the GENERAL line | +0.026 hit, and it composes |
| a burst that matters gets a longer stated duration | +0.022, and +0.044 together with the cause |
| write the **sound**, never the action | `(chuckle)` works; `(he chuckles)` degrades to silence, −0.08…−0.11 hit, misses +0.12 |
| do not substitute a neighbour | measured as a harm, not a fallback |
| never open or close the line on a burst | words must follow |

And the standing instruction: **use them constantly.** Real speech is full of
them and a reply without a breath, a laugh or a sigh sounds read rather than
spoken. One in most replies, two or three when the moment is emotional.

Best-of-N is the more effective lever than any weight change. The wiki carries an
`N` column: at a hit rate of 0.38, five candidates give a 90 % chance that at
least one realises the sound.

### 5.3 Talking-style attributes (delivery axes)

`LORA_ROOTS["sft3_voicenet"]` — **17** adapters, each the top or bottom 1 % of a
3.1 M-utterance corpus along one axis, trained against SFT3 itself. The director
picks them through `style[]`.

The full set, with the gloss the director is shown (`config.SFT3_VN_ADAPTERS`):
`AROU_high`, `AROU_low`, `ARSH_high`, `ARSH_low`, `EMPH_high`, `EXPL_high`,
`S_ASMR_high`, `S_DRAM_high`, `S_RANT_high`, `TENS_high`, `VALN_high`,
`VALN_low`, `VALS_high`, `VALS_low`, `VFLX_high`, `VOLT_high`, `VULN_high`.

- **At most one** (`SFT3_VN_MAX` = 1). One delivery axis costs little; two took
  word error from 0.041 to 0.143 and put invented words in half the takes.
- **Strength is one of `0.5, 0.75, 1.0, 1.25, 1.5`**, and the director picks it.
  A 5,740-cell dose-response sweep of 79 adapters at six weights found 16 of
  these 17 have a usable weight — the best-behaved family in the stack, 12
  monotone and 4 saturating. The median safe *and* strong weight is **1.5, not
  0.75**: going 0.75 → 1.5 buys +0.375 on the target axis (t 5.18, better on 15
  of 17) for a word-error change of +0.003, t 0.55, i.e. none.
- **Default: none.** No delivery adapter is merged unless the director asks or a
  slider forces one.

The older 57-dimension VoiceNet set is **parked** — trained against the untuned
v2 weights, so off-distribution on SFT3. `BASE_STYLE_LORAS` is now empty for the
same reason, and the base register is carried by the prompt instead
(`config.BASE_REGISTER`).

### 5.4 Emotions

`LORA_ROOTS["sft3_emotion"]` — affect and no identity, trained against SFT3.
Chosen per turn by retrieval, not by the director naming an adapter.

**Weight: `SFT3_EMOTION_LAM` = 1.0** (`config.py:200`, applied at `app.py:812`).

The adapter card recommends **1.5**, and 1.5 is the published operating point
from a 31-adapter scale sweep: emotion 0.408 → 0.471, genuineness and burst blend
both rise, median word error still 0.000. It is nonetheless **not** the setting
here, because averaged over all forty adapters 1.5 costs far more word error than
1.0, and individual adapters at 1.5 do not degrade gently — they derail outright.
`Confusion` at 1.5 reached word error **1.285 with 19 invented words**. 1.0 keeps
most of the effect (genuineness 1.90 against 1.83) at word error 0.018 instead of
0.083.

The v3 emotion set (`emotion`) is commented out: trained against the untuned v2
weights, replaced by the SFT3 set.

### 5.5 Voice and character

| kind | source | weight | note |
|---|---|--:|---|
| `sft3_voice` | local | **0.25** | speaker identity, no affect. Stacks with emotion |
| `character` | HF `TTS-AGI/moss-character-loras-refined-public` | 0.75 | trained against the *untuned* v2 weights, so off-distribution on SFT3; offered as a switch, not a default |
| `speaker` | HF `TTS-AGI/moss-voice-lora-velvet-sage-baritone` | 1.0 | the anchor speaker as a trained adapter |
| `sports` | HF `laion/moss-sports-commentator-lora` | 0.75 | |
| `profile` | local | — | the v2-era profile set, off-distribution on SFT3 |

The voice weight is worth stating carefully, because `SFT3_VOICE_LAM = 1.0`
exists in `config.py:181` and is **never read by anything**. The weight actually
applied is `PROFILE_LORA_LAM` = **0.25** (`config.py:389`, used at `app.py:868`),
and the reasoning is measured: the reference recording in the prompt already
carries the identity — speaker similarity with no voice adapter at all is 0.513 —
and the adapter adds 0.068 at 0.25 but only 0.019 more by 1.0, while word error
goes the wrong way. Most of the identity, none of the cost.

Voice adapters carry identity and no affect; emotion adapters carry affect and no
identity; they stack. They are rank 64 where the others are rank 16, which is
exactly why PEFT's `add_weighted_adapter` refuses the combination — merging
deltas, as `lora_bank.py` does, has no such constraint.

---

## 6. The three levers

The director chooses one through `perform.mode`. Full write-up in
`docs/LEVERS.md` and `docs/ENSEMBLE.md`; the short version, from the "THE THREE
LEVERS" block above `GEN_MODE` in `config.py`:

| mode | what it adds | measured |
|---|---|---|
| `adapter` | merge weights only | the only lever that moves the quality axes at all (+0.399, t 6.0) and the cheapest everywhere |
| `adapter+steer` | plus a difference-of-means direction added to the hidden state | on the emotion heads the two are cleanly additive (interaction +0.038, t 1.36), and steering is five times the adapter's effect there: +0.384 t 9.4 against +0.077 t 2.8 |
| `adapter+cfg` | plus classifier-free guidance on the delivery condition | costs **1.93×** |
| `steer` / `cfg` | one lever *without* the attribute's own adapter | for the delivery axes, where adapter and steering are significantly **sub**-additive (−0.164, t −3.7) and the right move is to pick one rather than stack |
| `auto` | resolves per family from the measurements | emotion → `adapter+steer`, delivery → `adapter`, quality → `adapter`. Never spends guidance |

**The shipped default is `adapter`, not `auto`.** `auto` was rolled back after a
listening report: it picked steering on nearly every turn (alpha +0.10 on the
target attribute and −0.10 on `Emotional_Numbness`) and a human heard artefacts
and an off timbre even though the scoring models liked it. That is the documented
blind spot of the evidence — the steering study states plainly that no listening
test was run on any of its results, and every figure in it is one model judging
another model's output.

**How much of this choice actually survives**, precisely (`app.py`, the
`req_mode` block): an explicit `gen_mode` in the request wins. Otherwise, if
`AGENT_PICKS_MODE` is on — it is — the director's `perform.mode` is taken, **and
only if that is `auto` does the server substitute `GEN_MODE`**. So a director
that writes `adapter+steer` gets `adapter+steer`; a director that writes `auto`
gets `adapter`. Naming a lever explicitly is therefore the only way to reach
levers 2 and 3 from the director's seat, and `auto` is the one word that
guarantees they are not used.

`perform.strength` is a **word**, never a number: `gentle`, `moderate`, `strong`,
mapped to a measured dose table. A language model that is allowed to pick a dose
picks it wrong, and every dose here is measured (`llm_agent.py:60–62`).

Unknown words are not an error. `_clean` (`llm_agent.py:844–853`) falls back to
`auto` for an unrecognised mode and `moderate` for an unrecognised strength.

---

## 7. What is repaired for the director, and what is not

Worth knowing precisely, because the unrepaired rules are the ones that fail
silently.

| rule | repaired? | where |
|---|---|---|
| underscore identifier alone in a cue | yes | `_sanitise_script:776` |
| runs of 3+ capitals | yes | `:779` (two-letter all-caps survive) |
| `[pause]` right after a bracket | yes | `:781` |
| square brackets not containing "pause" | rewritten to round | `:784–786` |
| a cue at the very end | dropped | `:789` |
| cues leaking into the spoken text | yes | `_strip_tags:756–763` |
| bad `perform` mode or strength | falls back | `_clean:844–853` |
| bad `style` adapter or strength | dropped / clamped | `app.py` |
| **a number inside a round bracket** | **no** | becomes a burst, silently |
| **fewer than ten words in a line** | **no** | |
| **opening on a burst** | **no** | closing is repaired, opening is not |

---

## 8. Where the burst vocabulary comes from

Two lists used to be maintained by hand and drifted apart, which is worth
recording because it is the failure this section exists to prevent.

`skills.py` reads `wikiskills/` and decides which sounds the director is
**offered**. `timed_script.py` decides which round brackets are **read** as
sounds. When the second was a hard-coded list of 22 labels and the wiki carried
117 pages, **9 of the 36 classes the server actually offered were unrecognised**
— including `guffaw` at a measured 0.633 hit, the fourth-best recipe in the bank,
and `clears_throat` at 0.48. The director wrote them, the bracket was re-read as
a delivery direction, and no sound was produced. No error, no warning.

`timed_script.burst_vocabulary()` now sources the vocabulary from the same
directory, in this order:

1. `<SKILLS_DIR>/patterns/vb-<label>.md` — one page per label, **117** of them:
   the union of the classes callers ask for, the labels the detector can emit,
   and the members of the 23-group scheme.
2. `<SKILLS_DIR>/VOCAL_BURSTS.md` via `skills.py`, if the pages are absent.
3. `BURST_LABELS` — the hard-coded core of 22 — when the skills directory is not
   on disk at all.

Only the **exact-match** half saw the widening. The older fuzzy rule, which lets
"a soft chuckle" land, still matches against the 22-label core only: running it
across all 117 would start reading ordinary directions as sounds, because
`(spitting the words out)` contains `spitting` and `(panting after the stairs)`
contains `panting`.

`tests/test_burst_vocabulary.py` asserts that every label the wiki offers is
recognised, and that a list of real delivery cues is not, so the two cannot
diverge again silently.

---

## 9. Quick reference

**Always:**

- one cue per sentence, in round brackets, before the words, with a strength
  adverb
- bursts in their own brackets, often, between sentences, with words after
- `[pause]` and `[long pause]` for beats
- lower case everywhere
- at least ten words per line
- plain English prose in cues, describing what the voice *does*

**Never:**

- a number inside any bracket
- a burst named inside a delivery cue
- capitals, emoji, markdown
- an adapter name or an underscore identifier in a cue
- a `[pause]` straight after a burst
- opening or closing on a burst
- describing the voice itself
- a duration, a length, a token count, or any other arithmetic — the server does
  all of it
