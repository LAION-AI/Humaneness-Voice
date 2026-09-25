# Adapters

> **Release note:** This page retains implementation details and historical
> measurements from the source demo server. Some later doses and machine-local
> inventories are superseded. For the current Humaneness Voice installation,
> use [Models and adapters](MODELS_AND_ADAPTERS.md),
> [shipped defaults](DEFAULTS.md), the manifest and `config.py`. The `sft3_*`
> strings below are runtime compatibility keys.

How this server puts LoRA adapters into the speech model, which adapters exist,
which ones a turn actually gets, and what every switch in the UI does to that.

Everything below is read off the code in this repository:
[`lora_bank.py`](../lora_bank.py) (the mechanism),
[`app.py`](../app.py) (which adapters a turn gets),
[`config.py`](../config.py) (every set, root and weight) and
[`llm_agent.py`](../llm_agent.py) (what the director is allowed to ask for).
Where a number appears here it is the number in the code, not a rounded one.

---

## 1. The mechanism: deltas merged into the weights, not PEFT

An adapter here is a directory containing `adapter_model.safetensors` and
`adapter_config.json`. Nothing loads it through PEFT, and `PeftModel` is never
constructed. `LoraBank` reads the tensors itself:

1. **Parse.** For every key ending `.lora_A.weight` the matching `.lora_B.weight`
   is looked up. The module path is the part before `.lora_A.`, with the PEFT
   wrapper prefixes `base_model.model.` / `base_model.` stripped, and it is kept
   only if that path exists in `model.named_modules()`. Both matrices are cast to
   `float16` and held in host RAM.
2. **Scale.** The adapter's own `alpha/r` is read from `adapter_config.json` —
   `scale = float(alpha or r) / float(r)`. Every set in this stack has
   `alpha = 2r`, so `scale` is **2.0** everywhere (rank 64 / alpha 128 for the DPO
   adapter, rank 16 / alpha 32 for the SFT3 sets, rank 4 / alpha 8 for the old
   profile adapters).
3. **Merge.** For each requested `(name, lam)` the per-module gain is
   `scale * lam`. Shapes: `W` is `[out, in]`, `A` is `[r, in]`, `B` is `[out, r]`,
   so the delta is

   ```python
   def _delta(A, B, gain):
       return (B.float() @ A.float()).mul_(gain)      # [out, in]
   ```

   and it is added straight into the module's stored weight:
   `m.weight.add_(d.to(m.weight.dtype))`. When several adapters touch the same
   module their deltas are summed first and added once.
4. **Unmerge.** At the end of the turn the same products are recomputed and
   subtracted again.

`apply()` calls `unapply()` first, so a turn can never run under the previous
turn's adapters. Both happen inside the generation lock in
[`tts_engine.py`](../tts_engine.py), so two concurrent requests cannot interleave
a merge with someone else's decode.

Because the deltas are summed into one weight, **the order of the adapter list
has no effect on the arithmetic**. The order documented in section 4 is the order
the list is built and the order shown in the UI; it is not a precedence rule.
Precedence is expressed by *removing* entries from the list (a slider drops the
director's pick of the same adapter), never by ordering.

### Why not forward hooks

Hooks were the first design, and the module docstring at the top of `lora_bank.py`
still argues for them: a hook on each target `Linear` adding
`lam * (alpha/r) * (x @ Aᵀ) @ Bᵀ` to its output never writes to the base model,
makes adapter swapping a pointer change, and makes stacking a sum.

It did not survive measurement. The streaming loop runs **batch 1, one token at a
time**, so kernel-launch overhead dominates the rank-32/rank-16 arithmetic. 536
extra kernel launches per token took the measured realtime factor from **0.737 to
1.29** — slower than playback, which breaks streaming outright. Folding the delta
into the weights costs a fixed sum once per turn and nothing per token.

### Why a pristine host copy exists

Subtracting a recomputed delta is cheap and stays on the GPU, but it **rounds in
bf16**, and the error accumulates over a long session. So the first time a module
is touched, `_snapshot()` takes a byte copy of its weight into host RAM:

```python
self._pristine[mp] = m.weight.detach().to("cpu", copy=True)
```

That is roughly 9 GB of the machine's ~110 GB of free host RAM — deliberately not
VRAM, where the model already occupies 20 of 24 GB.

`unapply()` then has two modes, chosen by a counter:

```python
RESYNC_EVERY = 25
exact = (self._merges % self.RESYNC_EVERY) == 0
```

`_merges` is incremented on every successful merge. On 24 turns out of 25 the
delta is recomputed and subtracted on the GPU; on every 25th the pristine host
copy is written back with `m.weight.copy_(...)` instead, which restores the
original bits exactly and resets any accumulated rounding to zero.

### Caches

| cache | bound | set by |
|---|---|---|
| parsed adapters in host RAM (`fp16`) | 64 | `config.MAX_CPU_ADAPTERS` |
| adapters resident on the GPU | 8 | `LoraBank(..., max_gpu_adapters=8)` in `app.py` |

Both are LRU. The RAM cache used to be unbounded; with ~65 GB of adapters on disk
a long session that ranged over many emotions and bursts grew it until the kernel
killed the process. A host entry that is still resident on the GPU is not evicted.
`config.PRELOAD_LORA_KINDS` (`("emotion", "speaker")`) names the sets warmed at
startup.

---

## 2. The one exception: twelve tied modules are hooked, not merged

`MossTTSLocalModel.tie_weights()` makes `audio_lm_heads.N.weight` and
`audio_embeddings.N.weight` **the same tensor object**. Folding a head delta into
the weight therefore also rewrites the audio embedding table, and the model is
corrupted — not subtly: the DPO adapter's own model card records the measurement,
both tensors moving by exactly **6.103515625e-05** while the text embedding did
not move at all. The card's conclusion is that the adapter "ships unmerged and
should stay that way".

`LoraBank` detects those modules with

```python
TIED = re.compile(r"(?:^|\.)audio_(?:lm_heads|embeddings)\.\d+$")
```

and routes them to `_ensure_hook()` instead. The hook is registered once per
module path and reads a shared `self._active` dict, so it is a no-op when nothing
is active:

```python
d = F.linear(F.linear(x.to(A.dtype), A), B) * gain
return output + delta.to(output.dtype)
```

That is the same arithmetic as the merge — `h + scaling·B(A(x))` — with no stored
weight written to. `unapply()` clears `_active`, which switches the hooks off.

Every DPO, quality, VoiceNet, emotion, voice and burst adapter in this stack
carries LoRA on **268 modules**: the **12** audio LM heads (`audio_lm_heads.0`
through `audio_lm_heads.11`, one per codebook) plus **256** others. The 256 are
merged, the 12 are hooked. On startup the server says so once:

```
[lora] 12 tied modules run as hooks, not merged (shared weights: audio_lm_heads.0 …)
```

Twelve hooks is not 536, so the realtime-factor argument that killed hooks for
the whole model does not apply here — and correctness is not optional.

---

## 3. Which adapter sets exist

`config.LORA_ROOTS` maps a *kind* to a directory; every subdirectory holding an
`adapter_model.safetensors` becomes one adapter named `kind:subdir`. On this
machine that discovers **206 adapters across 11 sets**.

`config._snap(repo, local)` prefers a plain directory under
`$MOSS_ASSETS` (default `/mnt/nvme/moss-15-v2-assets/loras`) and falls back to the
Hugging Face cache snapshot for `repo`. The full published collection is 168 GB
and does not fit on this disk, so the sets below are the subsets that were kept.

### Trained against SFT3 — in distribution

These were trained against
`laion/moss-tts-local-transformer-4.55b-voice-acting-v2-sft3`, which is the
checkpoint this server runs.

| kind | count | rank / alpha | source | used by default |
|---|---|---|---|---|
| `sft3_dpo` | 2 (`dpo`, `p2`) | 64 / 128 | [`laion/moss-va-sft3-dpo-lora-p2`](https://huggingface.co/laion/moss-va-sft3-dpo-lora-p2), superseding [`…-dpo-lora`](https://huggingface.co/laion/moss-va-sft3-dpo-lora) | yes, `p2` at 1.0 |
| `sft3_voice` | 10 of 500 | 16 / 32 | [`laion/moss-va-sft3-voice-loras`](https://huggingface.co/laion/moss-va-sft3-voice-loras) | yes, the active profile at 1.0 |
| `sft3_emotion` | 40 | 16 / 32 | [`laion/moss-va-sft3-emotion-loras`](https://huggingface.co/laion/moss-va-sft3-emotion-loras) | yes, the retrieved one at 1.5 |
| `sft3_quality` | 3 | 16 / 32 | local set, `base_model` field points at the SFT3 checkpoint | yes, all three at 1.0 |
| `sft3_voicenet` | 17 | 16 / 32 | local set, `base_model` field points at the SFT3 checkpoint | only when the director asks |
| `burst` (the SFT3 burst set) | 71 | 16 / 32 | local set at `$MOSS_ASSETS/sft3_burst`, `base_model` field points at the SFT3 checkpoint | only when the script contains a burst |

`sft3_dpo:p2` is the one used. Its card: reward 0.4757 against 0.4708 for its
predecessor, the highest emotion percentile of any preference-tuned model in the
line (0.3541), and — uniquely — a word error rate (0.0977) slightly better than
the supervised baseline it is built on (0.0987).

The three `sft3_quality` axes are `genuineness_high`, `blend_high` (vocal-burst
blend) and `esthetics_high`. Each is trained on the top 1 % of a 3.14 M-utterance
corpus along one perceptual axis.

The 17 `sft3_voicenet` adapters are each the extreme tail of one delivery axis,
also drawn from the corpus; `config.SFT3_VN_ADAPTERS` maps every name to a gloss
of what its training clips *sound* like, and that gloss is what the director is
shown.

### Off-distribution leftovers

These were trained against the untuned v2 checkpoint,
[`laion/moss-tts-local-transformer-4.55b-voice-acting-v2`](https://huggingface.co/laion/moss-tts-local-transformer-4.55b-voice-acting-v2).
They are still discovered and can still be forced on, but nothing reaches for them
on a default SFT3 turn.

| kind | count | source | status |
|---|---|---|---|
| `emotion` | 40 | [`TTS-AGI/moss-emotion-loras-v3`](https://huggingface.co/TTS-AGI/moss-emotion-loras-v3) | superseded by `sft3_emotion`; still reachable through the legacy planner (see §4.4) |
| `character` | 12 | [`TTS-AGI/moss-character-loras-refined-public`](https://huggingface.co/TTS-AGI/moss-character-loras-refined-public) | only when the director picks `voice.mode == "character"` |
| `profile` | 10 | [`laion/moss-voice-profile-loras-500`](https://huggingface.co/laion/moss-voice-profile-loras-500), rank 4 | inactive: `PROFILE_LORA_KIND` is `sft3_voice` on this checkpoint |
| `speaker` | 1 (`velvet-sage-baritone`) | [`TTS-AGI/moss-voice-lora-velvet-sage-baritone`](https://huggingface.co/TTS-AGI/moss-voice-lora-velvet-sage-baritone) | only used when no profile adapter is active |
| `sports` | 0 here | [`laion/moss-sports-commentator-lora`](https://huggingface.co/laion/moss-sports-commentator-lora) | not installed on this machine; `voice.mode == "sports"` silently resolves to nothing |
| `voicenet` (57 dimensions) | — | [`laion/moss-voicenet-dimension-loras`](https://huggingface.co/laion/moss-voicenet-dimension-loras) | **commented out of `LORA_ROOTS`**. Off-distribution on SFT3 and replaced by the 17-adapter `sft3_voicenet` set |

Two consequences of parking the 57-dimension set are worth stating plainly:

* `config.BASE_STYLE_LORAS` is now `()`. The three base-style adapters that used
  to be merged into every turn (`vn_S_CONV__high` 0.25, `vn_S_CASU__high` 0.5,
  `vn_WARM__high` 0.25) are kept only as `_OLD_BASE_STYLE_LORAS` for reference.
  The conversational base register is carried by the prompt instead
  (`config.BASE_REGISTER`).
* `config.AESTH_LORA` still names `voicenet:vn_ESTH__high`, which no longer
  resolves. Its slider is therefore inert on this configuration, on top of
  already defaulting to 0.0.

### Honesty about evaluation

The model cards are explicit, and this document repeats them rather than
softening them:

* The **quality** trio, the **`sft3_voicenet`** delivery axes and the **SFT3
  burst** set are **unevaluated**. 1.0 (quality) and 0.25/0.5/0.75 (delivery) are
  where they were trained or a deliberately conservative fraction of it — not
  weights that were shown to be best. The 1.5 sweep that fixed the emotion weight
  has not been repeated on any of them.
* The **emotion** adapters *are* evaluated, and the result is mixed: over 17
  adapters, emotion rises +0.047 when asked for and +0.033 when *not* asked for —
  a selectivity ratio of only about 1.4 : 1 — at roughly half again the
  transcription error. Merging one permanently and expecting neutral speech to
  survive is not supported.

---

## 4. What a default turn gets, in order

The list is built in `app.py` around lines 505–670, from the director's JSON,
the retrieval result and the request body. This is the non-pure path with nothing
ticked or unticked.

### 4.1 The default stack

| # | adapter | weight | where the number comes from |
|---|---|---|---|
| 1 | `sft3_dpo:p2` | **1.0** | `config.SFT3_DPO_LAM`, the adapter's published weight |
| 2 | `sft3_voice:<profile>` | **1.0** | `config.PROFILE_LORA_LAM`. 1.0 is the trained value and has not been swept |
| 3 | `burst:<label>` | **0.25** | `config.BURST_LAM`. Only when the script contains a burst cue that has an adapter. **0.5** (`config.BURST_LAM_INTENSE`) when the whole line is shorter than 14 words, i.e. the burst carries the beat on its own |
| 4 | `sft3_quality:genuineness_high` | **1.0** | `config.QUALITY_LORAS` |
| 5 | `sft3_quality:blend_high` | **1.0** | `config.QUALITY_LORAS` |
| 6 | `sft3_quality:esthetics_high` | **1.0** | `config.QUALITY_LORAS` |
| 7–8 | `sft3_voicenet:<axis>` × up to 2 | **0.25 / 0.5 / 0.75** | the `strength` the director wrote, clamped to `max(config.SFT3_VN_LEVELS)` = 0.75; count capped at `config.SFT3_VN_MAX` = 2 |
| 9 | `sft3_emotion:<Emotion>` | **1.5** | `config.SFT3_EMOTION_LAM`. The emotion comes from retrieval, not from the director |

Rows 3 and 7–9 are conditional; rows 1, 2, 4, 5, 6 are on every turn.

**1.5 for the emotion adapter** is the published operating point from a
31-adapter scale sweep: emotion 0.408 → 0.471, genuineness and vocal-burst blend
both rising with it, median word error rate still 0.000 and mean at its lowest.
Intelligibility breaks only between 1.5 and 2.0, and as a tail of derailed clips
rather than as general decay.

**Not in the stack any more:** the three base-style VoiceNet adapters (the set is
parked), the aesthetics dial (`AESTH_LORA_LAM = 0.0`, and the adapter no longer
resolves), and the standalone `speaker:velvet-sage-baritone` adapter (suppressed
whenever a per-profile voice adapter is active — otherwise every profile drifted
towards Velvet Sage). `personas.loras_for()` exists and is applied, but no shipped
persona declares any adapters, so it contributes nothing.

### 4.2 A worked example

Turn: the German reply from the README's prompt example, spoken by the default
profile Mira (`emolia_c1699`). The director wrote

```
(klar amüsiert, leicht verschwörerisch) Also, ich finde es herrlich, dass mein
staubsauger immer genau dann dramatisch den dienst quittiert, wenn besuch kommt.
(chuckle) Dann stehe ich da, nicke würdevoll und tue so, als hätte ich gerade
eine sehr moderne wohnästhetik erfunden.
```

with `"style": [{"adapter": "S_DRAM_high", "strength": 0.5}]`, and retrieval
returned `Amusement`. The resulting list — exactly what the UI shows under
`loras` for that turn:

| adapter | weight |
|---|---|
| `sft3_dpo:p2` | 1.0 |
| `sft3_voice:emolia_c1699` | 1.0 |
| `burst:chuckle` | 0.25 |
| `sft3_quality:genuineness_high` | 1.0 |
| `sft3_quality:blend_high` | 1.0 |
| `sft3_quality:esthetics_high` | 1.0 |
| `sft3_voicenet:S_DRAM_high` | 0.5 |
| `sft3_emotion:Amusement` | 1.5 |

`burst:chuckle` gets 0.25 rather than 0.5 because the line is 40 words, well over
the 14-word threshold, so the chuckle sits *inside* speech rather than carrying a
beat.

Eight adapters × 268 modules each. They collapse into one summed delta per touched
module: 256 weights get `add_()` once, 12 heads get a hook. Eight is also exactly
the GPU cache bound, so a ninth adapter on the next turn evicts the least recently
used one.

### 4.3 Pure mode

`pure_mode: true` takes the *legacy planner* out of the loop, not everything else:

```python
specs = dpo_spec + q_spec + specs + vn_spec + emo_spec
```

where `specs` is at most `[(sft3_voice:<profile>, 0.5)]` — `config.PURE_PROFILE_LAM`,
half the normal identity dose, on the reasoning that with fewer expressive
adapters underneath a lighter dose keeps more of the base model's own acting.

So pure mode **still** gets the DPO adapter, the quality trio, the delivery axes
and the emotion adapter. What it drops is everything the legacy `lora_bank.plan()`
would have contributed (v2 emotion, character, burst, v2 VoiceNet, sports), the
base-style list, the speaker dial and the aesthetics dial. It is "no legacy
adapters and a lighter voice", not "no adapters".

### 4.4 The legacy planner, and the one way a v2 emotion adapter can still appear

On the non-pure path `app.py` first calls `lora_bank.plan()` (or `plan_blend()` in
code mode), which is the v2-era policy. Its doses are the manual's measured
values, hard-coded so a hallucinated number cannot get through:

| condition | adapter | dose |
|---|---|---|
| `voice.mode == "character"` | `character:<name>` | `LAM_CHARACTER` = 0.75 |
| `voice.mode == "emotion"` | `emotion:<name>` (v2 set) | `LAM_EMOTION` = 0.5, or `LAM_EMOTION_UNDER_BURST` = 0.5 then capped to half the burst dose |
| `voice.mode == "voicenet"` | `voicenet:vn_<dim>__<high\|low>` | `LAM_STYLE` = 0.5 — never fires, the set is parked |
| `voice.mode == "sports"` | `sports:r32_e2` | `LAM_SPORTS` = 0.75 — never fires here, not installed |
| a burst cue in the script | `burst:<label>` | 0.25 inline / 0.5 solo |

Of these, only the burst and (rarely) the character adapter survive to the final
list on a normal turn, because the last step is

```python
if emo_spec:
    specs = [s for s in specs if not s[0].startswith("emotion:")] + emo_spec
```

— the SFT3 emotion adapter replaces any v2 one. **But if `emo_spec` is empty** —
emotion nuances switched off, retrieval off, or retrieval returning an emotion
with no matching adapter — that filter never runs, and a v2 `emotion:<name>`
adapter picked by the legacy planner *does* reach the model at 0.5, on a
checkpoint it was not trained for. That is current behaviour, not a design
intention.

One related asymmetry: the "emotion at most half the burst dose" rule in
`plan()` rewrites entries whose name starts with `emotion:`. `sft3_emotion:` does
not match that prefix and is appended afterwards, so **the SFT3 emotion adapter is
never capped by a burst**. It stays at 1.5 alongside a burst at 0.25.

---

## 5. Every switch and slider

All of these are fields in the `POST /api/turn` body. The chat page (`/`) exposes
most of them; `/studio` exposes a subset.

### 5.1 Booleans

| field | default | on | off |
|---|---|---|---|
| `pure_mode` | `false` | see §4.3 | normal path |
| `char_lora` | `true` | in pure mode, the voice adapter is merged at `PURE_PROFILE_LAM` | in pure mode, no voice adapter; identity comes from the reference recordings alone. **Only read in pure mode** |
| `emotion_nuance` | `config.EMOTION_NUANCE_ON` = `true` | retrieval returns an emotion and `sft3_emotion:<E>` is merged at 1.5 | retrieval returns `""`, no SFT3 emotion adapter (and see the §4.4 caveat) |
| `retrieval` | `config.RETRIEVAL_ON` = `true` | reference clip *and* emotion adapter come from retrieval | neither; the turn runs on the base checkpoint and the profile anchor |
| `delivery_loras` | `true` | the director's `style` picks are merged | `sft3_voicenet` never merged |
| `dpo_lora` | `true` | `sft3_dpo:p2` at 1.0 | no DPO adapter |
| `base_style` | `true` | `config.BASE_STYLE_LORAS` merged — currently empty, so no effect | — |
| `speaker_lora` | `not have_profile` — and ANDed with `not have_profile` again, so it is off whenever a profile voice adapter is active, whatever the request says | `speaker:velvet-sage-baritone` at `SPEAKER_LORA_LAM` | — |
| `aesth_lora` | `true` | would merge `config.AESTH_LORA` at `aesth_lora_lam` | — |

### 5.2 Numbers

| field | default | range | meaning |
|---|---|---|---|
| `profile_lora_lam` | 1.0 (`PROFILE_LORA_LAM`), or 0.5 (`PURE_PROFILE_LAM`) in pure mode | ≥ 0 | merge weight of the speaker's own voice adapter. ≤ 0.001 ⇒ not merged at all |
| `speaker_lora_lam` | 1.0 (`SPEAKER_LORA_LAM`) | ≥ 0 | merge weight of the standalone velvet-sage adapter, which only applies when no profile adapter is active |
| `aesth_lora_lam` | **0.0** (`AESTH_LORA_LAM`) | ≥ 0 | merge weight of `voicenet:vn_ESTH__high`. 0.0 because it was trained against the untuned v2 weights and was the strongest off-distribution adapter in the old stack. On this configuration the adapter is not discovered at all, so the slider does nothing |
| `stop_bias` | 3.0 (`config.STOP_BIAS`) | any | **not an adapter.** A constant in nats subtracted from the end-of-take token's logit before sampling, so a line is not cut off a few words early. 1.0 makes stopping about *e* times less likely at any step. 0 restores the model's own sampler bit-for-bit |
| `quality_lams` | `config.QUALITY_LORAS`, i.e. 1.0 each | 0 – 2 in the UI | per-axis merge weight for the three quality adapters. Accepts either the full name (`sft3_quality:blend_high`) or the bare suffix (`blend_high`) |
| `adapter_overrides` | `{}` | 0 – 2 (bursts 0 – 1.5) | `{full_adapter_name: weight}`. Any entry above 0.001 whose name exists is appended to the list, and any same-named entry the director produced is removed first, so the slider's dose wins |

### 5.3 The 0 semantics, which are not uniform

This is the part that surprises people, and `/api/adapters` says it in its own
docstring:

> A default of 0 does not mean "off": it means nothing is forced and the director
> decides.

* **Delivery axes, emotions and bursts** in the overlay have a default of **0**.
  That 0 means *leave it to the director* — a delivery axis at 0 is not
  suppressed, it is simply not forced, and the director may still pick it. The
  overlay builds its payload with `if (v > 0.001) ov[k] = v`, so a slider at 0
  contributes no key at all, and `adapter_overrides` only ever adds and replaces,
  never removes.
* **The three quality axes** are the exception. They have a default of **1.0** and
  are sent as `quality_lams` on every request, whatever their value. Setting one
  to 0 means it is genuinely not merged: the code skips any entry with
  `lam <= 0.001`. Here 0 really is off.
* **The dials** (`profile_lora_lam`, `speaker_lora_lam`, `aesth_lora_lam`) behave
  like the quality axes: they are explicit weights, and 0 means not merged.

To *suppress* something the director would otherwise pick, use the booleans in
§5.1 — `delivery_loras: false`, `emotion_nuance: false`, `dpo_lora: false` — not
a slider at 0.

---

## 6. How the director chooses

The director is a language model constrained by a JSON schema
([`llm_agent.build_schema`](../llm_agent.py)). Two of its fields decide adapters,
and one important choice is taken away from it entirely.

### 6.1 `style` — the delivery axes

```json
"style": {
  "type": "array",
  "maxItems": 2,
  "items": {"type": "object",
            "properties": {
              "adapter":  {"type": "string", "enum": [ ...the 17 names... ]},
              "strength": {"type": "number", "enum": [0.25, 0.5, 0.75]}},
            "required": ["adapter", "strength"]}}
```

`maxItems` is `config.SFT3_VN_MAX` (2) and the strength enum is
`config.SFT3_VN_LEVELS`. `style` is in the schema's `required` list — together
with `voice2` — because when it was optional the model simply never emitted it.
An empty array is a valid answer and is the right one when the emotion alone
carries the line.

The catalogue is not left to the schema. `render_catalog()` prints all 17
adapters into the system prompt with their glosses, because the schema constrains
sampling but is never shown to the model:

```
DELIVERY ADAPTERS (17) — this is what "style" picks. Each one is the extreme tail of
one axis, and the gloss says what its training clips actually sound like, not what
the axis is called:
  AROU_high — highly aroused, very dominant, tense, elated, thin
  AROU_low — dialled down, not performed at full size; narrow pitch range, submissive, slow
  ...
  strength: 0.25 a touch, 0.5 clearly there, 0.75 strong. Leave "style" empty when the
  line needs no colouring beyond the emotion.
```

The prompt tells the director to pick on the *gloss* rather than the axis name,
and to keep strengths modest because the set is a pilot that has not been
evaluated. `app.py` enforces the rest: unknown names are dropped, duplicates are
dropped, the strength is clamped to `[0, 0.75]`, at most 2 survive, and an entry
whose adapter is not installed is skipped.

For the hosted backends a compact JSON skeleton is appended to the system prompt
as well. That exists because a truncated schema dump used to be sent instead, and
the `voice`/`voice2` enums alone overran the 1800-character cut — so every key
after them, including `style`, was invisible to the model and never emitted.

### 6.2 The emotion adapter is chosen by retrieval, not by the LLM

The director does name an emotion (in `voice.emotion`, from a fixed 40-name
vocabulary), but that is used to pick a *reference recording*. The **adapter** is
chosen by [`retrieval.py`](../retrieval.py):

1. The director's round-bracket cues are extracted from the script.
2. They are embedded with the VoiceCLAP text tower and matched against 40 emotion
   text anchors, each the mean of six caption templates.
3. The winner is `retr["emotion"]`, and `app.py` merges
   `sft3_emotion:<that name>` at 1.5 — if such an adapter exists and
   `emotion_nuance` is on.

Matching cues against the emotion text anchors scores **0.61 top-1** over 40
classes; matching the full direction against the audio centroids scores 0.28.
That is why the emotion axis leads and the audio axis only picks the level and
the take, with `config.RETRIEVAL_EMO_BONUS` = 0.5 as the bonus for conditions of
the winning emotion.

One guard rail: when the cues are German — or absent — the emotion is instead read
from the English label the director named, because German cues retrieve the wrong
emotion rather than merely a weaker one ("Stimme brüchig vor zurückgehaltener
Trauer" came back as `Teasing`).

### 6.3 The burst adapter is chosen by string matching

`lora_bank.detect_burst()` scans the round-bracket tags of the script, normalises
German umlauts, and matches against a synonym table covering both English and
German director wording (`_BURST_SYNONYMS`, `_BURST_SYNONYMS_DE`, plus `_EXTRA_*`
noun forms). Longer synonyms win, so "fearful gasp" beats "gasp". The director
never names an adapter; it writes `(sighs)` or `(kichert)` and the table resolves
it.

The reason to resolve a cue even when nobody asked for an adapter: a prompt tag
alone lands a burst **23.6 %** of the time; with the matching adapter merged it is
**71.9 %**.

`burst_catalog()` renders the installed bursts back into the system prompt as one
clean cue per adapter, so the director can only ask for bursts that exist.

---

## 7. Reproducing this

### 7.1 Setup scripts, in order

```bash
export HF_HOME=/path/with/space          # the corpus shards are large
python setup/fetch_profile_refs3.py      # best 3 takes per condition, ~3.5 GB kept
python setup/build_retrieval_index.py    # condition centroids + emotion text anchors
python setup/profile_traits.py           # measured gender/age/timbre per voice
```

`fetch_profile_refs3.py` must run first: it produces the clips and their metadata.
`build_retrieval_index.py` reads that output and writes the condition centroids
and emotion anchors into `config.RETRIEVAL_DIR` — without it retrieval is
unavailable and no emotion adapter is ever chosen. `profile_traits.py` only
affects the GENERAL text, not adapters, and can run at any point after the fetch.
(`setup/fetch_profile_refs.py` is the older v2 fetcher, kept for the `refs2`
anchors.)

The adapters themselves are not fetched by any script here. Download the
repositories listed in §3 with `huggingface_hub` into directories matching
`config.LORA_ROOTS` — the layout each root expects is one subdirectory per
adapter, each holding `adapter_model.safetensors` and `adapter_config.json`.
`config._snap()` looks under `$MOSS_ASSETS` first and falls back to the HF cache,
so either location works. Ten voice adapters plus forty emotion adapters is about
6.5 GB.

Discovery is logged at startup, and is the fastest way to check a layout:

```
[lora] discovered 206 adapters across 11 sets
[bursts] 71 burst adapters offered to the director
```

### 7.2 `GET /api/adapters`

Everything the overlay can dial, with its default weight and slider maximum:

```json
{
  "groups": [
    {"kind": "sft3_quality",  "title": "Quality axes",
     "note": "On by default at the trained value of 1.0.",
     "items": [{"name": "sft3_quality:blend_high", "label": "Burst blend",
                "default": 1.0, "max": 2.0}, ...]},
    {"kind": "sft3_voicenet", "title": "Delivery axes",
     "note": "The director picks up to two of these itself. A slider forces one on regardless.",
     "items": [{"name": "sft3_voicenet:AROU_high", "label": "AROU_high",
                "hint": "highly aroused, very dominant, tense, elated, thin",
                "default": 0.0, "max": 2.0}, ...]},
    {"kind": "sft3_emotion",  "title": "Emotions",
     "note": "The retrieval picks one per turn at 1.5. A slider adds or replaces it.",
     "items": [{"name": "sft3_emotion:Amusement", "label": "Amusement",
                "default": 0.0, "max": 2.0}, ...]},
    {"kind": "burst",         "title": "Vocal bursts",
     "note": "Added automatically when the script contains one. With skills on,
              at the weight measured for that class (0.25-2.3); a class with no
              measured recipe falls back to 0.25 (0.5 standing alone).",
     "items": [{"name": "burst:chuckle", "label": "chuckle",
                "default": 0.0, "max": 1.5}, ...]}
  ],
  "always": [
    {"name": "sft3_dpo:p2",          "label": "Quality (DPO p2)", "default": 1.0},
    {"name": "sft3_voice:<profile>", "label": "Voice identity",   "default": 1.0}
  ]
}
```

It returns `503 {"error": "adapters disabled"}` when `config.USE_LORA` is off.
The `always` entries are informational — they are not dialable from the overlay.
The endpoint lists only the four dialable kinds; `character`, `profile`,
`speaker`, `sports` and the legacy `emotion` set are discovered and usable through
`adapter_overrides` by full name, but are not offered in the UI.

Every turn's response also carries the resolved list, so what actually ran is
never a guess:

```json
"loras": [{"name": "sft3_dpo:p2", "lam": 1.0}, {"name": "sft3_voice:emolia_c1699", "lam": 1.0}, ...]
```

### 7.3 Runtime knobs worth knowing

| variable | default | effect |
|---|---|---|
| `MOSS_USE_LORA` | `1` | `0` disables the bank entirely; `/api/adapters` then 503s |
| `MOSS_MAX_CPU_ADAPTERS` | `64` | host RAM LRU bound |
| `MOSS_PRELOAD_LORA` | `emotion,speaker` | which kinds are parsed into RAM at startup |
| `MOSS_SFT3_DPO_LORA` | `sft3_dpo:p2` | which DPO adapter |
| `MOSS_SFT3_EMOTION_LAM` | `1.5` | emotion merge weight |
| `MOSS_SFT3_VN_MAX` | `2` | how many delivery axes the director may stack |
| `MOSS_LAM_GENUINE` / `MOSS_LAM_BLEND` / `MOSS_LAM_ESTH` | `1.0` | quality-axis defaults |
| `MOSS_BURST_LAM` / `MOSS_BURST_LAM_INTENSE` | `0.25` / `0.5` | burst doses |
| `MOSS_PROFILE_LORA_LAM` / `MOSS_PURE_PROFILE_LAM` | `1.0` / `0.5` | voice-identity weight |
| `MOSS_ASSETS` | `/mnt/nvme/moss-15-v2-assets/loras` | where the local adapter sets live |

The full generated list is in [`DEFAULTS.md`](DEFAULTS.md).

---

## 8. Known limits of this design

* Emotion retrieval is right about 61 % of the time over 40 classes, and a wrong
  pick merged at 1.5 is audible. Gating the adapter on a retrieval-score
  threshold is the obvious next step and is not implemented.
* The quality, delivery-axis and burst sets are unevaluated on this checkpoint
  (§3). Their weights are trained values or conservative fractions of them, not
  measured optima.
* A v2 emotion adapter can still reach an SFT3 turn when the SFT3 emotion adapter
  is absent (§4.4).
* The aesthetics dial is dead code on the current `LORA_ROOTS` (§3).
* `_snapshot` grows with the union of every module any adapter has ever touched.
  In practice that is a fixed 256 modules — every set here has the same target
  list — but a set with a different target list would enlarge it.
* The GPU adapter cache holds 8. A default turn uses 8, so any additional forced
  adapter causes an eviction and a PCIe copy on the next turn that needs it.
