# What changed on 5–6 September 2026, and how to check any of it

A working log. Every claim here has a script beside it, and every default that
moved names the measurement that moved it. If you want the short version, read
the table at the end.

---

## 1. The director could not place a pause, and nobody had noticed

The prompt had asked for silence *inside* sentences for a while and the replies
kept stopping only at full stops. The renderer turned out to support it
perfectly — a mid-clause `[0.6 seconds pause]` renders and folds into the token
sum — so this was a prompt problem, not a capability problem.

Making it work took three passes, and the pattern that emerged governs
everything below:

| what the prompt said | mid-clause pauses per reply |
|---|--:|
| the rule, buried among thirty others | ~0 |
| the rule plus a worked example | ~1 |
| the rule, an example per feeling, and `speed` explained | 1–3 |

**An exemplar moves this director where a rule does not.** That is the single
most useful thing learned this week, and it is why the prompt now carries
before/after pairs rather than more prose.

A deterministic floor backs it up: `benchmark.breathe()` counts the silences the
director wrote and tops the reply up to `BREATHE_WANT` (2) at commas, dashes and
the conjunctions that begin a new thought. It never overrides a choice.

*Reproduce:* send any turn and count `[N.N seconds pause]` tags whose sentence
already has spoken words before them.

## 2. Two bugs were deleting those pauses

Found only after the prompt work looked like it was failing.

**`_sanitise_script` deleted a pause after any round bracket.** The rule is real
— a pause directly after a *burst* truncates it — but the pattern matched
delivery directions too, and that is where a director puts the first silence of
a sentence. Every reply lost one silently.

**`_has_inner_pause` counted a pause at the start of a sentence as inside it**,
because `)` is not sentence punctuation. That switched the floor off on exactly
the replies that had no mid-clause silence.

*Reproduce:* `tests/test_burst_vocabulary.py`, and feed
`"(clearly hesitant) [0.5 seconds pause] I wanted to tell you something."`
through `LLMAgent._sanitise_script`.

## 3. A screamed line came back as babble — twice, for two different reasons

**In English: the sum of the burst adapters.** Two at 1.5 each destroy the line
whether or not a burst is asked for — 5 of 5 seeds, median word error 0.82 —
while one at 1.5 breaks none. The old budget of 3.0 came from a ladder measured
on a bare model; a turn here already carries a voice adapter, three quality
adapters, a preference adapter, an emotion adapter and a delivery axis before a
burst arrives.

| sum of burst weights | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|--:|--:|--:|--:|--:|
| median word error | 0.000 | 0.000 | 0.010 | 0.110 | **0.820** |

`BURST_LAM_BUDGET` 3.0 → **2.0**, `BURST_LAM_MAX` 1.5 → **1.25**, and over
budget the weights are **scaled rather than dropped**.

*Reproduce:* `eval/why_babble.py` (isolation), `eval/sweetspot.py` (the table).

**In German: the language of the brackets.** The corpus is captioned in English;
a German stage direction is off-distribution and takes the words with it.

| | median word error | unusable |
|---|--:|--:|
| German cues | 0.267 | 2 of 4 |
| English cues, same German words | **0.000** | 0 of 4 |

`cues.py` now rewrites them **before retrieval** — which also fixed a
long-standing oddity where a horror scene was conditioned on
`Jealousy_and_Envy`: the retriever falls back to the named emotion whenever the
cues are German.

*Reproduce:* `eval/de_cues.py`.

## 4. Guidance 3.0, measured twice

Four acting tasks, two English and two German, each rendered from **one** script
per guidance level so the director's choices are held constant.

| | pleasant | fit | natural | total of 15 |
|---|--:|--:|--:|--:|
| no guidance | 3.17 | 2.08 | 2.25 | 7.50 |
| **g = 3** | **3.50** | **2.92** | **3.00** | **9.42** |
| g = 4 | 2.92 | 2.83 | 2.42 | 8.17 |

g3 − g1 is +2.33 (p 0.006) in the first run and +1.92 in the second — it
replicates. **g4 − g3 is −1.25, worse on all three rubrics, winning 4 of 12
pairs.** German gains far more from guidance than English (+3.50 vs +0.33).

*Reproduce:* `eval/cfg_ab.py`.

## 5. Nineteen prompt additions changed nothing, and the search that said so

An evolutionary search over prompt additions, 2,949 rated clips across two runs.
The first run's design flattered itself: children were bred inside a single
(task, director) arm and compared against a control pooled over twenty, so the
comparison measured task difficulty. Corrected — every block on every arm,
fitness the mean rather than the maximum — **the control wins**, and the two
blocks that looked promising in run 1 changed sign in run 2.

The reason is worth keeping: take-to-take spread inside one cell is SD 1.57 of
15, as large as the spread between all the variants of an arm, so a
best-of-three fitness buys +2.78 points from luck alone.

*Reproduce:* `eval/harness2.py`, then `eval/analyse_fixed.py`.

## 6. SIDON: the one thing that moved the sound, and then came back off

Restoration raises "how pleasant does this sound" by **+0.43 of 5, p < 0.001**,
and moves fit and natural not at all. It shipped on for a day and was turned
**off** again after a listening report: it distorts on screams and loud bursts,
and the twenty-one clips behind the measurement contained no screaming.

*Reproduce:* `eval/sidon_test.py`.

## 7. `/api/speak`

Text in, MP3 out, the whole pipeline behind one request: best-of 10, guidance
3.0, every UI parameter optional. It drives the same `_turn()` the browser does
rather than a second copy. 27 checks cover every documented path.

*Reproduce:* the examples in [`SERVER.md`](SERVER.md).

## 8. Pacing follows the feeling

The director now sets `speed` for the scene and writes `[N.N seconds duration]`
per sentence. That needed the renderer: `parse()` stripped every duration tag
and recomputed it from the word count, so the instruction went into a field the
server threw away. Durations are honoured and clamped to **0.6–1.5×** natural,
and the ceiling is measured:

| duration | 1.0× | 1.5× | 2.0× | 2.5× | 3.0× |
|---|--:|--:|--:|--:|--:|
| median word error | 0.00 | 0.00 | 0.06 | 0.50 | 0.75 |

Past 1.5× the budget stops being silence and becomes invented words, so extra
slowness is sent to the pauses instead.

On the benchmark scene this came from, with no chat prompting: `speed` came back
`slower`, pauses **0.7 / 0.6 / 0.8** against 0.3 / 0.4 / 0.3, and 27.7 seconds
against 18.6.

---

## Everything that moved

| setting | was | now | why |
|---|--:|--:|---|
| `BURST_LAM_BUDGET` | 3.0 | **2.0** | §3 |
| `BURST_LAM_MAX` | 1.5 | **1.25** | §3 |
| burst over budget | dropped | **scaled** | §3 |
| `BON_GUIDANCE` | 4.0 | **3.0** | §4 |
| CFG in the UI | off, 2.0 | **on, 3.0** | §4 |
| `SIDON_ON` | on | **off** | §6 |
| `ENGLISH_CUES` | — | **on** | §3 |
| `BREATHE_WANT` | fired at 0 | **tops up to 2** | §1 |
| director-set durations | discarded | **honoured, 0.6–1.5×** | §8 |
| `SPEAK_BEST_OF` / `SPEAK_GUIDANCE` | — | **10 / 3.0** | §7 |

## Checking it all at once

```bash
python setup/check_docs.py    # defaults, generated pages, links, arithmetic
python -m pytest tests/ -q    # 15 tests
python setup/check_levers.py  # 48 lever assertions
python setup/fetch_all.py --check
```

## What is still open

* **Hesitation sounds** appear in roughly one reply in four or five. They are
  deliberately not floored the way pauses are: adding silence is reversible,
  adding words is not.
* **The VoiceNet track** scores 5.93 of 15 against `emotion`'s 8.88 and no
  prompt addition in either arena run touched it. That points at the adapters.
* **Whether the pause emphasis itself helps.** The one replicated negative in
  the arena runs was `breath`, a block asking for *more* pauses than the
  standing prompt. It measures the marginal push, not the default. The block
  that *removes* the emphasis has not been run.
* **Why SIDON works at all.** The hypothesis — that stacking five or more LoRA
  adapters costs high-frequency detail, which a restoration model puts back —
  fits the evidence and has not been tested. The experiment is written down in
  [`SIDON.md`](SIDON.md).
