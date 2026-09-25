# WikiSkills — the state before the 2026-09-04 vocal-burst revision

A verbatim snapshot of `wikiskills/` as it stood at 11:36 on 4 September 2026, immediately
before the vocal-burst recipes were rewritten against the re-measured evaluation. It is kept
because the revision changed recommendations that had been in use, and a reader comparing a
result produced last week against the current table needs to see the table that produced it.

**Nothing outside the vocal bursts differs.** Exactly 33 of the 101 files changed:
`VOCAL_BURSTS.md`, all 31 `patterns/vb-*.md`, and the burst paragraph in `index.md`. The
emotion patterns, VoiceNet dimension patterns, `interactions.md` and `coefficients.json` are
byte-identical in both trees, so anything that reads a coefficient or a non-burst pattern can
use either directory and get the same answer.

## What changed in those 33 files

The revision was a merge, not a replacement. Of the recipes that existed before:

- **14 were kept** — the old prompt form measured better and stayed;
- **12 were replaced** by a re-measured recipe;
- **5 were not re-measured** and stand unchanged with that stated;
- **19 are new** classes that had no recipe before.

Every row's wording also changed from "the best setting" to "a good setting". That was not
cosmetic: re-scoring the same audio on a second detector moves the argmax cell for 14 of 16
checkable classes, but the cost of keeping either instrument's choice is symmetric (median
0.083 against a seed-noise standard deviation of 0.068). Neither instrument has the better
recipe, so a superlative was not supportable. Best-of-N candidate count is the larger lever
and is named as such in each file.

## Do not use this directory as an input

It exists to be read, not consumed. Anything that globs `wikiskills/patterns/*.md` should point
at `../wikiskills/`; this tree is deliberately a sibling rather than a subdirectory so that such
a glob cannot pick it up by accident.
