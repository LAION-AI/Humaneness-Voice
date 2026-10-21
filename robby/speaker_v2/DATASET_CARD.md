# TYGER V2 bounded pilot dataset

This is a small, reproducible research pilot, **not** the requested final
500,000-window production dataset. It contains only public synthetic speech and
procedurally generated augmentation. No webcam frames, private microphone
recordings, uploaded references, or enrolled-user voiceprints are included.

## Sources, permissions, and attribution

The source is [LAION's annotated voice profiles](https://huggingface.co/datasets/laion/laion-voice-profiles-annotated),
revision `88ea857d6a6f20839674dfde30936d8fd6b23c62`, attributed to **Christoph
Schuhmann and LAION**, under the dataset's **CC BY 4.0** terms. Its card explicitly
describes all released speech as model output, not original human recordings.
The source card also lists upstream provenance and cautions about synthetic
identity variation. Retain that provenance when redistributing selected audio.

The frozen teacher is `speechbrain/spkrec-ecapa-voxceleb`, revision
`0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`. The Nemotron/TYGER feature extractor is
the separately licensed `nineninesix/tyger` bundle; model files and its OpenMDW
terms are **not** relicensed as CC BY merely because the input audio has that
license. Before public release, include all applicable model notices and review
the terms for derived feature data and learned predictors.

## Reproduction

Run from `robo-integration` using the existing Python environment:

```sh
python speaker_v2/prepare_pilot.py --stage sources
python speaker_v2/prepare_pilot.py --stage teacher
python speaker_v2/prepare_pilot.py --stage features
python speaker_v2/audit_pilot.py --save
```

The last stage requires the local Robo service on port 8793. Its loopback-only
feature endpoint reserves the **same existing EARS encoder**, never loads another
GPU encoder, and returns HTTP 423 while an interactive conversation owns it.
Requests contain mono float32 little-endian 16-kHz public synthetic audio, with
at most 30 seconds per request. The dataset script uses 24-second examples.
ECAPA inference runs on CPU, with six threads, using existing pinned weights.

The source cache is `/home/c4r33u19/.cache/robo-pilot`. Source retrieval reads
bounded 8-MiB prefixes from two approximately 2-GB tar shards per voice, retaining
only selected MP3 clips. A server that ignores HTTP Range is rejected before its
body is downloaded. No full tar or entire dataset is downloaded. The script
checks for at least 5 GiB free on both filesystems and stops if its source cache
exceeds 1 GiB. It never deletes unrelated user data.
Override the cache location with `ROBO_PILOT_CACHE`. Set `ROBO_ECAPA_ROOT` to a
preexisting pinned teacher snapshot on another machine; otherwise the script
fetches only that small teacher's necessary files, not the source speech corpus.

## Splits and supervision

All 500 voice names are sorted by
`sha256("tyger-v2-split-v1:" + voice)`: first 10 validation, next 10 test,
remaining 480 training. The initial pilot takes 24 training, four validation,
and four test identities from those fixed pools. A future expansion therefore
does not silently move pilot test voices into training.

For each identity, two complete clean utterances form a normalized average
192-dimensional ECAPA enrollment target. Four **different** utterances supply
student examples. Teacher and prediction recording IDs are disjoint, including
for held-out speakers. Pairwise teacher cosine is recorded, not assumed perfect;
the synthetic source may drift in timbre across acting conditions.
`probe_teacher_embeddings.npz` provides complete-utterance ECAPA embeddings for
prediction-source clips, allowing a separate teacher-vs-enrollment verification
ceiling rather than presenting regression loss as identity accuracy.

Each procedural conversation has one to five identities from **one split only**,
alternating turns, partial interruptions, pauses, and non-320-ms-aligned speaker
boundaries. Five scenarios cycle deterministically: clean, generated colored
noise, generated sparse room impulse responses, telephone bandpass/quantization,
and delayed synthetic playback leakage. No third-party music or environmental
audio is used. The pilot echo source shares one scenario identity; dedicated
distinct-Robo self-voice evaluation is still required.

Word-level source alignments determine an activity mask. Each 80-ms frame is
speech-labeled only when one speaker occupies at least half of it; simultaneous
speaker activity is explicitly ambiguous and excluded from identity regression.
These model-derived alignments are imperfect labels, not manually annotated
ground truth. Energy-based uncertainty and live identity gating must therefore
remain separate safeguards.

## Files and incremental feature format

`pilot/manifest.json` is the entry point. `pilot/global_split.json` stores the
complete 500-speaker split. `pilot/speakers.json` and each conversation JSON retain
source IDs, byte hashes, conditions, seeds, and precise procedural boundaries.

Each conversation NPZ contains:

| Field | Meaning |
|---|---|
| `features [T,1024]` float16 | Frozen Nemotron layer-15 states, new frames only |
| `frame_end_s [T]` | Frame endpoint on the synthetic input timeline |
| `speaker_idx [T]` | Global pilot ID, `-1` silence/uncertain, `-2` overlap |
| `valid_mask [T]` | Unambiguous speech eligible for identity regression |
| `speech_fraction [T]` | Dominant speaker's aligned occupancy in the frame |
| `ambiguity [T]` | Multi-speaker activity flag |

The feature cadence is 80 ms; four new frames support each 320-ms prediction.
Train causal temporal windows only and do not pool future frames into a past
prediction. Evaluation must remain speaker-disjoint. Store final chosen
architecture, normalization, thresholds, and all pilot limitations with its
checkpoint. Approximate identity embeddings are **not authentication**.

## Limits before scale-up

The initial source selection spans two acting-condition shard prefixes per
identity, not the full 842-condition matrix. It has modest language/emotion
coverage and does not establish natural-recording or unseen-language robustness.
Only expand to the requested 480/10/10 speakers, 50 snippets per identity,
100,000 conversations, and 500,000 windows after held-out pilot evidence supports
it. Estimate cost from measured extraction/training throughput, not nominal GPU
FLOPS. Keep any larger expansion under the separately authorized 40-GB budget.

## Emotion-diverse second pilot

```sh
python speaker_v2/prepare_emotion_pilot.py
python speaker_v2/audit_pilot.py --data speaker_v2/pilot_v2 --save
```

V1 remains unchanged in `pilot/`. `pilot_v2/` keeps exactly the same 32 identities,
same global split, and same teacher-only enrollment utterances. Every student
prediction utterance is new relative to V1. Each voice adds two source takes
under each named **Anger, Amusement, Elation, and Sadness** condition, plus an
ordinary read under a background-condition prompt. The actual source condition
IDs—not claims that the model perfectly realized each emotion—are recorded in
the source audit. The background condition is `BKGN/very_high`, so those clips are
not clean neutral controls and should not be described as such. Whispering is
not covered by this small second pilot.
The V2 source selection is **German-heavy: 338 German and 45 English clips,
including the unchanged enrollment references**. The emotion-range midpoint
usually lands in the German `C` condition. Thus its emotion test is not evidence
of balanced English emotional generalization. V1 is approximately balanced
(98 German / 94 English source clips), but covers fewer acting conditions.

To find those examples without transferring multi-gigabyte archives, the script
uses HTTP Range to read only the Parquet `uid` and `audio_bytes` columns (roughly
100 KB per index). Cumulative audio sizes estimate each emotion's archive
position. A bounded 3-MiB response is resynchronized using checksum-validated
512-byte TAR headers, and only complete MP3/JSON pairs are retained. A 1-MiB
background-condition range supplements each voice. Wrong-voice records and
missing word alignments are rejected. The tested safety guards reject a server
that ignores byte ranges before reading its body.

The second source acquisition has a **persistent 500,000,000-byte download
reservation cap**, independent of the first pilot. The completed source pass
reserved 463,134,685 bytes, including a conservative 200-MB allowance for an
earlier interrupted pass and exploratory range probes; this is an upper-budget
account, not a precise measurement of network bytes delivered. Selected source
clips, provenance, feature arrays, and evaluation results remain far below the
40-GB working budget. The script never erases unrelated caches to make space.

Evaluate the V1-selected frozen predictor on V2 first as an out-of-condition
check, then report any V2-trained model separately. Do not use test results to
choose an architecture or threshold. Even this broader pilot contains only
four validation and four test identities and cannot establish real microphone
deployment performance.
