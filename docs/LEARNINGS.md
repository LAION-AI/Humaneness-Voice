# What we learned making this thing stop talking

This is written for someone who has not seen the project before. It describes a
real bug, how it was chased, and what turned out to be true — including the
guesses that were wrong, because those cost the most time.

## The symptom

The demo would say its line correctly and then keep going: a few more words,
often not real words, tacked onto the end of every second or third reply. It
sounded like the voice trailing off into mumbling.

## What the system does, in one paragraph

A language model writes the reply and also directs it: a description of the
voice, and an instruction in round brackets before each sentence saying how to
perform it. A second step turns that into the format the speech model was
trained on, which is unusual in one important way — **it states how long each
sentence should take**, in seconds, and states the total as a token count. The
speech model then generates audio frame by frame until it decides to stop.

## The four things we suspected

1. The time budget is too generous, so the model has spare time to fill.
2. The stop-token brake — a dial that makes stopping more or less likely — is not
   working.
3. Too many LoRA adapters are loaded at once, pushing the model off-distribution.
4. Rounding error accumulating as adapters are merged and unmerged each turn.

Three of these were partly or wholly wrong. Here is how each was tested.

## How we measured it

Guessing was not going to settle this, so we built a measurement.

Ten fixed sentences, always the same, always in the same order. For each
configuration we generate all ten, transcribe them with a speech-recognition
model, and compare the transcript with the sentence we asked for. Two numbers
matter:

* **word error rate** — how much of the transcript differs from the intended line;
* **extra words** — how many words appear *after* the last word that still
  matches. This is the direct measurement of "it keeps talking".

We also score each clip for naturalness, for how well non-speech sounds blend in,
and for whether it still sounds like the intended speaker.

Everything is fixed except the one thing being tested, including the random seed.
Later runs use three different seeds, because ten sentences at one seed is not
enough to trust a single number — a lesson learned the hard way, see below.

## What turned out to be true

### The adapters are expensive, and that part of the suspicion was right

Loading nothing but the base model and its quality adapter: word error 0.013, and
not one invented word in ten takes. Adding the full set the demo shipped with:
word error 0.258, with roughly two invented words per take. The damage grows
both with the *number* of adapters and with how strongly each is mixed in.

Turning the same adapters down rather than removing them recovered most of it.
The demo now runs each at a measured weight instead of the value it was trained
at, and that alone took word error from 0.273 to 0.018.

There is a real trade here, not a free win: the heavy stack scored *highest* on
the naturalness measure. It made the voice sound less like reading aloud, and it
did that by loosening the same control that keeps it on script.

### The stop-token brake does almost nothing

The dial subtracts a constant from the "stop now" option before the model
chooses. Turning it up makes stopping less likely — it was built for the
opposite problem, lines ending too early.

Swept across its range, three settings produced **bit-identical output**. Not
similar: identical, through two different code paths. That is only possible if
the stop-or-continue decision is never close. Something else was deciding when
the take ended.

### It was the clock

Across every configuration measured — clean, degraded, any adapter set — the
audio came out matching the requested duration to within 0.02 seconds.

That is the whole answer. The model was not overrunning its budget. It was
honouring it exactly. Whatever time it did not need to say the words, it filled,
and what it filled it with was invented speech.

Two things were setting the budget too high:

* the **rate** the script was rendered at — 4.5 audio frames per word, which is
  the rate in the format's own worked example, but more time than this voice
  needs;
* a **closing pause tag** the renderer appended after the last word, because the
  format's description asks for a pause after the final word. A stated pause at
  the end is time the model still owes.

Neither change alone was enough. At the old rate, removing the closing pause
still left invented words in 20–40 % of takes. At the tighter rate, keeping the
closing pause still left 10–20 %. **Together** they were the only configuration
that held at 0–10 % across three different random seeds — and in the final
measurement, 0 %.

### The obvious fix would have made it worse

The intuitive move is to remove the timing tags altogether and let the model
decide. We measured that too. It is the **worst** configuration in the study:
0.9 invented words per take and nearly four times the trailing audio. The timed
format is not the problem. An over-generous budget inside it was.

### Rounding drift was never shown to matter

Suspicion 4 was never confirmed. It may still be real over a long session, but
nothing in these measurements needed it as an explanation, so it stays an open
question rather than a finding.

## The mistakes worth repeating out loud

**A measurement bug that looked like a model failure.** The first version of the
harness rebuilt transcripts one token at a time. That drops the marks that
separate words, so `The kettle boiled` came out as `Thekettleboiled` — one long
word matching nothing. Word error read 1.088 on audio that was in fact perfect.
Always check that a shocking number is not your own ruler.

**Ten samples at one seed is not a result.** An early sweep showed a clean
downward trend across six settings. Repeated with a different seed, the trend did
not replicate. Only conclusions that survived three seeds were kept.

**Comparing against the wrong reference.** The harness conditioned the model on
one speaker's recording while testing another speaker's adapter — they fought
each other, and speaker similarity came out near zero. The adapter was fine; the
experiment was not.

**A control that could not control anything.** The interface had an "Aesthetics"
slider pointing at an adapter that had been removed from the configuration. It
showed 0.00 and could load nothing, while the real aesthetics adapter ran at a
different weight somewhere else entirely. A dial that reads zero while the thing
it names is switched on is worse than no dial.

## Where it landed

| setting | before | after |
|---|--:|--:|
| frames per word | 4.5 | 4.0 |
| closing pause tag | 0.3 s | none |
| stop-token brake | 3.0 | 2.0 (1–3 measured identical) |
| genuineness adapter | 1.0 | 0.25 |
| burst-blend adapter | 1.0 | 0.5 |
| aesthetics adapter | 1.0 | 0.5 |
| voice adapter | 1.0 | 0.25 |
| emotion adapter | 1.5 | 1.0 |
| delivery adapters at once | 2 | 1 |

Invented words: **none in thirty takes** at the final settings, against roughly
two per take at the settings this started from.

The full method, every prompt, every hyper-parameter and the raw per-take data
are in [`EXPERIMENTS.md`](EXPERIMENTS.md), and the adapter machinery is described
in [`ADAPTERS.md`](ADAPTERS.md).
