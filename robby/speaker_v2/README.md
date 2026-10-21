# TYGER V2: lightweight speaker-embedding pilot

This optional network predicts a normalized 192-dimensional ECAPA speaker cue
from the **frozen Nemotron ASR 3.5 layer-15 representation** already computed by
TYGER: 1024 dimensions per 80 ms, four new frames every 320 ms. It does not replace
or retrain the original TYGER GRU, endpointing, interruption, or backchannel rules.
It is not authentication. Synthetic voices and a small held-out pilot cannot
establish accuracy for real users, microphones, accents, or unseen acoustic rooms.

## Measured outcome

Fifteen actual CPU training runs were completed: eleven initial architecture,
width, context, and loss ablations; one centered-target control; and three
class-balanced runs on a second, broader-emotion corpus. The final experimental
integration checkpoint is `artifacts/best.pt`, with `artifacts/selection.json`.
It is a centered MLP with 470,147 trainable parameters and 640 ms context.
Among five predefined candidates it wins on the broader corpus's **validation**
EER (18.0%), not on test metrics. Held-out test EER is still **25.3%**, and the
conservative validation-calibrated threshold rejects roughly 90% of genuine
short probes. Its measured warmed CPU inference is about 0.32 ms per update.
The long-clean ECAPA teacher control achieves 2.5% EER on that corpus but sees
far more clean audio. It is not an equal-context comparison.

**Keep the student identity gate off by default.** It is useful as an optional
research signal, not reliable multi-user filtering. The 100,000-conversation /
500,000-window expansion was not started because this pilot failed reliability
checks. Read `report.html` for the combined results and `DATASET_CARD.md` for
the data/license/coverage limitations. V2 covers anger, amusement, elation and
sadness, but is German-heavy; this does not establish broad English robustness.

## Reproduce the architecture comparison

From `robo-integration`, using the existing Python environment:

```bash
python -m unittest speaker_v2.test_predictor -v
python -m speaker_v2.train --data speaker_v2/pilot --out speaker_v2/results \
  --epochs 30 --threads 6 --device cpu
```

Run `prepare_pilot.py --help` for the separate data preparation workflow. It uses
selective downloads, disjoint clean enrollment/probe source clips, a deterministic
global speaker split, frozen encoders, and compact cached features. Inspect its
manifest and source-license records before distributing any derived dataset.
No microphone recordings or enrolled user embeddings belong in this dataset.

The first concrete pilot contains 32 identities (24 training, 4 validation,
4 testing), 64 procedural 24-second conversations, and 4,800 decision hops.
It has 192 source clips (98 German, 94 English), but bounded prefix extraction
reached only `Fatigue_Exhaustion` and `VoiceNetBKGN_moderately_low` conditions.
This is **not broad emotional coverage**; a later, independently sampled pilot
must test additional emotions before claiming robustness or starting a large
publication-quality corpus. Two long source clips per identity are enrollment
teacher targets, while four different source clips supply conversation probes.
The additional complete-probe ECAPA evaluation quantifies identity/domain noise.

The initial ablation compares MLP, GRU, and causal temporal CNN predictors at
widths 96 and 192 (148k–470k parameters). The validation winner is then compared
at 4, 8, 12, and 25 frames (320, 640, 960, and 2000 ms), and with cosine, MSE, and
mixed regression objectives. Because outputs and targets are normalized,
`192 * MSE = 2 * (1 - cosine)`: this particular loss ablation changes the relative
weight of regression versus the auxiliary activity head, not the embedding
geometry. A constant training-teacher mean and frozen-feature ridge regression
are included as controls. The ridge baseline uses oracle activity labels only
to isolate embedding quality and is not a deployable silence detector.

Architecture selection uses validation identities only. Verification thresholds
are calibrated from validation impostor scores (95th percentile, approximately
5% false accepts), never from test labels. Evaluation reports held-out EER,
teacher cosine, closed-set identification, known acceptance, unknown rejection,
and speech/silence/ambiguous-overlap classification. Trials from adjacent windows
are correlated: their count must not be mistaken for independent people.
Temporal accumulation uses oracle same-speaker boundaries and is explicitly an
optimistic diagnostic rather than a deployed diarization result.

Outputs are `results/results.json`, `results/report.html`, individual reproducible
checkpoints, and `results/best.pt`. The report records actual CPU throughput and
extrapolates only the small predictor's training cost, excluding download,
augmentation, ASR/teacher feature extraction, and evaluation. A pilot should be
assessed before authorizing or beginning 500,000-window training.

The broader corpus uses `--data speaker_v2/pilot_v2`. Controlled alternatives
use `--balanced-activity`, `--centered-target`, and `--only-config` (see help).
`python -m speaker_v2.select` reproduces the validation-only integration
selection once all five predefined candidate reports exist; it never promotes
the student to default-on. `python -m speaker_v2.compile_report` rebuilds the
combined English report. Frozen cross-corpus evaluation uses the immutable
`results/best.pt` and `artifacts_centered/best.pt` checkpoints, **not** the mutable
integration destination `artifacts/best.pt`.

## Incremental integration

```python
from speaker_v2.inference import Predictor

predictor = Predictor("speaker_v2/results/best.pt", device="cpu")
result = predictor.update(new_layer15_frames, energy=microphone_rms)
# result: embedding (normalized numpy[192] or None), kind,
#         confidence, context_ms, optional activity_probabilities
predictor.reset()  # new session / explicit identity-memory reset
```

Call `update` with **only newly produced frames**, not the full rolling ASR window.
Use actual microphone RMS if available. Silence and ambiguous overlap abstain;
short warm-up context reduces confidence. Accumulation resets on silence or an
obvious embedding change, but this heuristic is not guaranteed diarization.
Confidence is an activity/context indicator, **not** a calibrated probability
that a person matches an enrolled identity. The identity manager must separately
check similarity, enrollment status, margin, activity, timeouts, and uncertainty.

Do not automatically enable a predictor that fails held-out controls. Keep the
teacher-embedding fallback or disable identity filtering instead. Store session
voiceprints only with informed consent; do not publish them.
