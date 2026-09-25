# The burst term in the best-of-N reward

Added 2026-09-07. Study `vb_opt`, protocol §71. This page says what changed, what
is measured, and **what is deliberately left to the server that has the models**.

## Which detector to score with — read this before changing `BON_BURST_ENCODER`

**First, a correction to a note that was in `config.py` until 8 September.** It
said the 768-d `x2` head sits on an unpublished extractor, that its dimension
matching our commercial tower was a coincidence, and that feeding it commercial
embeddings would produce meaningless probabilities. Measured on the stored
feature vectors — 300 segments in both feature sets, same keys, same order —
the two embeddings agree to **max 1.57 × 10⁻⁷, cosine 1.000000**.
`FastScorer.emb.encode_waveform` **is** `laion/voiceclap-commercial`. The
reasoning behind the old note was sound; the fact was not, and the difference
only became measurable once both feature sets existed side by side.
`vocal-burst-detector-x2` now also ships a `commercial/` head trained directly
on commercial embeddings, so nothing needs to rest on that identity.

Two heads are available in `laion/vocal-burst-detector-x2` and **they fail on
different classes**. Measured per class, trained on both sources and tested on
real speech:

* The **3584-d `large-v2`** head wins 12 of 17 classes and is much better on the
  quiet, drawn-out sounds — Sharp Inhale +0.248, Humming +0.240, Deep Breath
  +0.216, Yawn +0.208 — and on knowing that nothing is there (0.912 against
  0.824).
* The **768-d `commercial`** head wins on the loud, short ones, and most sharply
  on **`Scream`: 0.768 against 0.576**.

That single exception is why `BON_BURST_ENCODER` defaults to `commercial`, and
the reason is not only that the tower is already loaded. **A detector decides
whether a hit counts as a hit**, so ranking candidates with the big head would
systematically undercount exactly the class this project has spent the most
effort on. The cheap option is also the better one for the case that matters
here — that is luck, not design, and it is worth writing down so nobody
"upgrades" the default without checking what it costs on screams.

Both heads answer "is a sound there at all" at 97-99 % and "which sound is it"
at 38 % (768-d) and 45 % (3584-d) over 17 classes, against a 5.9 % chance rate.
The errors are confusions inside a burst family, not deafness; family-level
scoring adds 16-21 points to both. Neither is reliable enough to gate anything —
which is why the term below is a soft re-ranking weight and not a filter.

## The problem

`bestofn.rank` scored four things — genuineness, burst blend, CLAP agreement and
a word-error gate — and **none of them asks whether the sound the script names is
actually in the audio.**

Measured over 1,940 candidate sets and 15,517 candidates under the full
production stack, of the four terms only `clap` tracks burst presence at all
(within-set r **+0.148**); `blend` points the **wrong way** (−0.084) and
`genuineness` does nothing (−0.009, CI straddles zero).

Pooled, the old reward still picks better than chance: the delivered take hits
0.1531 against 0.1183 for a random pick, a lift of **+0.0348 (t 5.47)**.

**But restricted to the configuration the server is always in it does not.** On
the six carrier voices that have an `sft3_voice` adapter — and one is always
loaded, because the server falls back to the speaker adapter otherwise — the
lift is **+0.0104 (t 1.51)**, not distinguishable from zero. On the four
carriers without one it is +0.0714 (t 5.97). The pooled number is carried
entirely by material that is off the shipped configuration.

## What changed in this repository

| file | change |
|---|---|
| `config.py` | `BON_BURST_WEIGHT` (2.0), `BON_BURST_DETECTOR`, `BON_BURST_ENCODER`, `BON_BURST_MODE`, `BON_BURST_WIN_S`, `BON_BURST_HOP_S` |
| `config.py` | `BURST_LAM_MAX_SOLO` (1.5) — the ceiling for a reply with exactly one burst adapter |
| `bestofn.py` | fifth summand in `rank()`, `n_burst` emitted per candidate |
| `app.py` | applies the solo ceiling when exactly one burst is tagged, best-of-N is on and the burst term is non-zero |
| `llm_agent.py` | director guidance: when in doubt write one burst, not two |
| `wikiskills/VOCAL_BURSTS.md` | the solo ceiling, and the `bestmem` entry rewritten with both stacks |

**`MOSS_BON` is unchanged and still `0`.** Best-of-N stays off in the chat path.
That is deliberate: everything below is measured **unguided**, the chat path
would run at guidance 3.0, and the thin guided sub-arm suggests guidance already
recovers much of the same gain (+0.0125, t 0.30, against +0.1125, t 2.39
unguided, on 80 sets). **Turn best-of-N on first and measure; judge the term
after.** Not both at once.

## What is NOT done here, and is the next commit

`rank()` reads `c["burst"]`. **Nothing populates it yet**, and `rank` forces the
term to exactly 0 when no candidate carries the key, so the ranking of every
turn is byte-identical to before this commit — verified on a fixture.

That guard is not cosmetic and is worth knowing about before anyone simplifies
it away: `_norm` of a constant vector returns **0.5, not 0**, and a constant 0.5
inside the sum is *not* harmless, because the sum is multiplied by a
per-candidate WER gate afterwards. Without the guard the inert term silently
re-weighted every candidate by its own word error rate, and on a three-candidate
fixture it changed the winner.

Leaving it inert is intentional: this side has
no GPU and none of the assets under `/mnt/nvme/moss-15-v2-assets`, so shipping an
untested model load would have been worse than shipping an inert term.

**To finish it**, `Judge.score` needs one more per-candidate field:

```
burst = mean over the bursts the turn requested of
        s_strict + 0.5 * (s_fam - s_strict) + 0.25 * s_pres + 0.5 * agree
```

* `s_strict` — the detector's probability for the requested class.
* `s_fam` — the same summed over that class's burst family. The failure is
  granularity, not deafness: family-relaxed scoring adds 16–21 points in every
  cross-source cell.
* `s_pres` — probability that any burst is present.
* `agree` — 1.0 when the detector's top-1 is in the requested family.
* `0.0` for a turn that requests no burst, so the term drops out.

Windows of `BON_BURST_WIN_S` at `BON_BURST_HOP_S`, restricted to
`[t − 1.0, t + 2.0]` around each bracket's expected onset — `timed_script.render`
already returns those onsets. Localised beats whole-clip, 0.2082 to 0.1907.

Encoder: `BON_BURST_ENCODER = "commercial"` is the drop-in, because that tower is
already loaded for retrieval and the marginal cost is one 768→256→18 MLP on an
embedding computed per candidate anyway. `large-v2` scores better (+0.0454
against +0.0314, both crossed) at the price of a second encoder on the same card.

## What it buys, and what it costs

Every figure crossed — ranked with one detector, scored by the other:

| | value | t | n |
|---|--:|--:|--:|
| strict hit rate of the delivered take, before | 0.1242 | — | 1940 |
| **Δ hit at weight 2.0** | **+0.0454** | **7.60** | 1940 |
| — relative | **+37 %** | | |
| Δ Parakeet WER (gate is +0.104) | +0.0040 | 2.49 | 1940 |
| Δ CLAP agreement | −0.0055 | −6.82 | 1940 |
| Δ genuineness | −0.067 | −7.24 | 1940 |

**This is a trade, not a free win, and the traded quantity is CLAP** — 3.6 % of
the 0.1507 within-set range a re-ranker can actually move, bought for a 37 %
relative increase in delivered bursts.

Why 2.0 and not 3.0: the exchange rate has a knee. In hit gained per unit of CLAP
lost, 1→2 buys **3.82**, 2→3 buys 2.09, 3→5 buys 1.16.

Rejected after measuring: **tiered ranking** (WER +0.046, t 3.55, against soft's
+0.004, for 0.002 of extra hit) and the **`agree` sweep** (moves the outcome by
0.002 — harmless, useless).

## No recipe changes

No weight in `VOCAL_BURSTS.md` moves. The change is to the ranker, not to any
recipe. What changes is which of the N candidates is delivered.

---

# Integration on the server that has the models, 7 September 2026

The commit above left the term inert and named the next step. Here is what
happened when the side with the GPU tried to take it, and why the term is
**still inert**.

## The encoder named in the config has no matching weights

`BON_BURST_ENCODER = "commercial"` was described as the drop-in because that
tower is already loaded for retrieval. Checked against the published weights,
`laion/vocal-burst-detector-x2` ships three heads and **none of them consumes a
VoiceCLAP-commercial embedding**:

| checkpoint | D | `embedder` / `encoder` field |
|---|--:|---|
| `vocal_burst_mlp_x2_s*.pt` | 768 | `FastScorer.emb.encode_waveform (frozen)` |
| `voiceclap/…_vclap_s*.pt` | 3584 | `FastScorer.emb.encode_waveform (frozen)` |
| `production/…_prod_s*.pt` | 3584 | `voiceclap-large-v2` |

The 768 of the first head and the 768 of the commercial tower are a
**coincidence**. That head sits on the old detector's own frozen extractor —
`laion/vocal-burst-detector-v2` publishes the 0.9 MB head and not the extractor.
Feeding commercial embeddings into it would run without error and return
meaningless probabilities, which is the worst failure available: the ranker
would look burst-aware and rank on noise.

The `production` head is the correct one and needs `voiceclap-large-v2` — an
**18.15 GB** Qwen2.5-Omni-7B encoder. With the speech model, the scorers and the
aligner loaded, this box has **1.0 GB free on one card and 2.5 GB on the other**.
It does not fit.

**So `Judge.score` still does not populate `burst`,** and the reason is a
missing artefact rather than missing work. What would unblock it, in order of
preference:

1. a head trained on `laion/voiceclap-commercial` embeddings — 768→256→17, the
   encoder is already resident and the marginal cost really would be one MLP;
2. the FastScorer extractor published, which makes the existing 768-d `x2` head
   usable;
3. someone with the VRAM for `voiceclap-large-v2` running the `production`
   ensemble.

## `BON_BURST_READY`

New flag, default **off**. While it is off the fifth summand is dead weight and
every gate that depends on a burst-aware ranker stays shut. Set it only when a
detector whose encoder is genuinely loadable is wired in. It exists so that
"inert" is a state the code knows about rather than a property of a comment.

Verified on this machine: with no `burst` key the rewards and ranks are
bit-identical to before the merge, and with values injected by hand the ranking
reorders as specified.

## The solo ceiling was gated on the wrong thing

It read `config.BON_ON`, which is the **server default** for best-of-N and is
`0` here. The UI and `/api/speak` turn best-of-N on per request, so the gate
could never open on a turn that actually ran best-of-N, and would have opened on
none at all. It now reads the request's own `best_of`, and additionally requires
`BON_BURST_READY` — the ceiling is only safe *because* a burst-aware ranker
rejects the takes it costs, and that ranker does not exist here yet.

**One bug of our own, worth recording because of how it failed.** The first
version of that gate referenced `bon_n`, which is computed further down the
handler. Referencing it threw, the whole burst-weight block fell into its
`except`, and **every burst adapter silently dropped to the flat 0.25 default** —
a scream that should merge at 1.25 merging at a fifth of that. The log line said
`[skills] burst weights unchanged: cannot access local variable 'bon_n'`, which
is true and reads like a note rather than a fault. Caught by checking the
applied weights on a turn rather than by reading the log.

## What does work now, tested

| scene | bursts written | adapter and weight | word error |
|---|--:|---|--:|
| VNET scream, English | 1 | `burst_abl:ablation_d2_matched__scream` @1.25 | 0.020 |
| German, missing someone | 1 | `burst:wistful_sigh` @1.0 | 0.240 |
| the funniest thing | 1 | `burst:chuckle` @1.25 | 0.091 |

Recipe weights are honoured and capped at 1.25, and all three replies wrote
**exactly one** burst — the director guidance added in the commit above is
landing.

---

# The detector is running, 7 September 2026

`burst_server.py` on port 8794, started by `./run.sh burst`, on the card the
language model would otherwise have. `Judge.score` populates `burst` and
`BON_BURST_READY` ships **on**.

## The encoder question, answered by measuring

`BON_BURST_ENCODER = "commercial"` was never usable — no published head takes
that embedding. The `production` ensemble needs `voiceclap-large-v2`, an 8.93 B
Qwen2.5-Omni thinker: 16.6 GB in bf16, which does not fit. Quantisation was the
only route, and the head was trained on bf16 embeddings, so it had to be checked
rather than assumed. Against a bf16 CPU reference over 36 windows:

| | cosine median | cosine min | top-1 agrees | VRAM | 36 windows |
|---|--:|--:|--:|--:|--:|
| bf16 (CPU reference) | — | — | — | 16.6 GB | 127 s |
| 8-bit | **0.9959** | 0.9810 | **35/36** | ~10 GB | 9 s |
| **4-bit (NF4)** | 0.9753 | 0.9517 | **35/36** | **~5.7 GB** | **4 s** |

**4-bit ships.** It blurs the embedding measurably more than 8-bit and its
*decisions* are exactly as good — the same 35 of 36, and the same single
borderline window, which was already undecided in bf16 (0.519 against 0.634).

The reason is headroom rather than fidelity. With 8-bit the encoder and the
app's own scorers came to 21.4 GB of 23.68, and a turn that allocated during
alignment died with `CUDA out of memory`. Fidelity nobody can measure in the
ranking is not worth an out-of-memory error in generation. At 4-bit the card
holds 6.2 GB spare with a turn in flight.

## What had to be built

| file | what it does |
|---|---|
| `burst_server.py` | encoder + five-head ensemble; raw int16 in, one score per candidate out |
| `burst_client.py` | soft in every direction — service down means `None` and `rank` forces the term to 0 |
| `timed_script.burst_onsets` | `(label, start_seconds)` from a **rendered** script; scoring is localised to `[t−1, t+2]`, which measured 0.2082 against 0.1907 whole-clip |
| `bestofn.Judge.score` | takes `tagged=` — the rendered script, the only one carrying the durations an onset is computed from |

Our burst labels are prose and the detector knows 17 names, so `ALIAS` maps the
ones that correspond exactly and `FAMILY_WORDS` places the rest in a family.
A label with no exact class is scored at the family level only, which is the
honest thing for a distinction the detector was never trained to make.

**Three things that had to be read out of the artefacts rather than assumed.**
The head is `BatchNorm`, not `LayerNorm` — the checkpoint carries `running_mean`,
and guessing would have measured through a wrongly normalised layer. The
processor needs `torchvision`, which was not installed. And `BatchNorm` in eval
mode still refuses a batch of one, so a single window is duplicated and halved.

**And one failure worth recording.** The term ranked correctly for two full test
rounds while appearing to do nothing: `bon["candidates"]` is built from a fixed
key list, and `burst` and `n_burst` were not in it. In the UI and in every log
that looks exactly like a term that is not working, and it invites fixing the
wrong thing.

## It reorders, as designed

```
"funniest thing", chuckle at 6.8 s
  rank0  reward 3.425  burst 0.211  n_burst 1.00  clap 0.463
  rank1  reward 3.000  burst 0.154  n_burst 0.00  clap 0.468
```

Rank 1 has the **better** CLAP score and loses — the trade the study describes.

## What it costs

**The local director, and the restoration service.** 5.7 GB of encoder plus
7.6 GB of `llama-server` plus the app's 11.4 GB does not fit on 23.68. Running
the detector means hosted directors only (`glm`, `luna`, `gemini-flash`).
`./run.sh stop` and starting without `burst` gives the local model back.

About **1.5 s per candidate**: 4 s for a best-of-3, roughly 15 s for a
best-of-10.

---

# The cheap path is the shipped path, 9 September 2026

The correction above changes the integration completely. `FastScorer` **is**
`laion/voiceclap-commercial`, the tower this server already loads for
retrieval — and `vocal-burst-detector-x2` now ships `commercial/` heads that
name that encoder explicitly (`D: 768`, `encoder: voiceclap-commercial`).

So the term is scored **in-process**. `Judge.burst_scores` embeds each window
with the same `encode_waveform` call `_clap_sim` already uses, and runs five
MLPs of about a megabyte. No second encoder, no service, no 18 GB download —
and **the local language model keeps its card**.

```
[burst] 5 commercial heads on cuda:0
Tell me the funniest thing that happened   wer=0.050  burst=[0.917, 0.811, 0.732]
Erzähl mir von jemandem, den du vermisst   wer=0.000  burst=[0.006, 0.002, 0.004]
```

The second row is the term doing its job: a `wistful sigh` was asked for and
essentially nothing was detected in any of the three candidates. The reward now
knows that, where before every term was blind to it.

`burst_server.py` is kept as the **optional** `voiceclap-large-v2` path. That
encoder wins 12 of 17 classes on real speech, mostly the quiet sustained ones,
and loses on Scream (0.576 against 0.768). It costs 5.7 GB in 4-bit and the
local director, so start it only when those classes matter.

**One bug from the switch, worth the line it takes.** A partial download leaves
a second snapshot directory, and `glob(...)[0]` found the older one without
`commercial/`. The heads silently did not load, the term returned `None` for
every candidate, and the only sign was `[burst] 0 commercial heads`. It now
picks the snapshot that actually has them and raises with the fetch command if
none does.
