# Robby voice-assistant test protocol

Version 1 · 2026-10-26 · English · implementation and development evidence, not a room-accuracy certification.

This is the repeatable test procedure for Robby's microphone → ASR → person memory → turn/interrupt decision → LLM → TTS → browser playback pipeline. Its primary question is: **can the assistant continue a useful conversation with the current person while a television or other people keep speaking?** Tests must also prove that an invited person can join, and that the current person can stop an unwanted spoken reply.

Start with the executable regression checks below. Use the configuration-search procedure only after those checks work. **Implemented**, **executed**, and **proposed** are different statuses throughout this document. A passing unit test is not a successful microphone conversation. A lower cosine threshold is not automatically an improvement.

## 1. Current evidence and its limits

The initial October 26 development evidence is in these machine-readable receipts, relative to this directory:

| Receipt | Executed scope | Result recorded before this protocol was introduced |
|---|---|---|
| `logs/unit-checks.json` | Offline contracts, including clocks, stale results, prompt/task handling, speaker permissions and bounded actions | 432 tests passed after the continuity and real-bridge memory contracts were added (initial baseline: 428); no microphone or cloud API |
| `static/rebalance-decision-check.json` | Six real requests to the pinned CPU Laya worker, through the production decision client | Six correct actions, including negated/quoted-topic controls; approximately 275–296 ms/request in the initial run |
| `static/rebalance-stop-browser.json` | CPU Chromium, real HTTPS WebSocket, paced synthetic audio, real ASR/Qwen/TTS/playback | Four short Stop/Wait/Halt/Pause cases cancelled long TTS; subsequent spoken request received audio |
| `static/rebalance-permissions-browser.json` | Spoken Alice invitation, original speaker return, background invitation denied, spoken only-me revocation | Passed, including the actual admitted participant roster |
| `static/conversation-stress-check.json` | Eight mixtures for one synthetic main voice; actual ECAPA/V4/Nemotron plus foreground cropping | Both foreground voices were subsequently tested (16 rows): speaker/crop gates passed. Christoph WER 0 in seven / 0.429 in hard overlap; Selma WER 1.0 and 1.167 under the two strongest overlap conditions. Passing the speaker gate does not imply clean ASR |
| `static/rebalance-browser.json` | Combined actions, handover, banana game, Stop and guest flow | Earlier attempts first exposed short Stop, then a combined-history unnamed-guest naming failure after seven successful checks; see `static/rebalance-before-group-fix.json`. The isolated group test had passed. Consult the current receipt/run archive for the post-fix combined result; do not combine partial passes into a fictional full pass. |

The four initial browser Stop capture-start-to-cancel measurements were **1291.0, 2089.0, 3211.3 and 3351.8 ms**. Their input files last approximately 1.07, 1.39, 2.75 and 2.59 seconds. These values include the utterance and padding; **they do not measure reaction time after the last word**. The earlier banana run admitted three interjections during a 16.43-second input, with its first decision around 4.49 seconds from input-file start. The exact current numbers live in the receipts.

Failures that motivated these regressions included: a short speaker window rewinding the main speech clock; silence/TV padding contaminating the final transcript; a one-word Stop missing VAD despite a correct ASR preview; an unreliable short student score vetoing that Stop; the sentence-stream branch skipping the existing live decision controller; a manual handover admitting a stale queued turn; model prose promising a banana rule without installing one; delayed action metadata losing a requested spin; and the test harness shadowing its `audio()` helper. Each is a distinct failure class. The helper failure was a harness error, not an assistant accuracy failure.

These are development examples reused during repair, with correlated mixtures and synthetic voices. They are not held-out human accuracy, a false-accept rate, or a guarantee of performance in a real room. Strong simultaneous speech still inserts background words. No physical camera, microphone, loudspeaker, acoustic echo path or human listening study was exercised by these particular browser tests.

## 2. Freeze the environment before comparing results

Run from the **same deployed `robo-integration` / exported `robby` directory**, with Python 3.12 and the dependencies in `requirements.txt`. Complete the service/model setup in [README.md](README.md) first. Do not install the entire upstream TYGER/vLLM training stack just to run these tests.

Working-host interpreter: `/home/c4r33u19/moss15v2/.venv/bin/python`. Replace it with your own environment's Python on another machine. Browser tests additionally need Playwright and Chrome/Chromium. On the working host:

```bash
cd /mnt/nvme/moss-15-v2/robo-integration
PY=/home/c4r33u19/moss15v2/.venv/bin/python
$PY -m pip show numpy scipy torch httpx
curl --fail http://127.0.0.1:8793/api/state
curl --fail http://127.0.0.1:8797/health
curl --fail http://127.0.0.1:8792/api/state
```

Required running components:

| Component | Working-host endpoint | What to pin / verify |
|---|---|---|
| Robby UI, native ASR, TYGER and session memory | `127.0.0.1:8793` | `ready=true`; deployed `source_sha256`; sentence streaming enabled |
| Public FastAPI facade | `127.0.0.1:8798`, when used | Proxies the same service; HTTPS tunnel changes do not define a model version |
| Laya decision worker | `127.0.0.1:8797` | `/health` model, revision, policy hash and actual request counter |
| Humaneness TTS | `127.0.0.1:8792` | Selected checkpoint/revision/hash, precision, reference profile, temperature, CFG, chunk frames and conversion state |
| Local dialogue fallback | `127.0.0.1:8799` | Exact GGUF/hash, llama.cpp commit, sampler and context |
| Selected alternative final ASR, if used | `127.0.0.1:8801` | Model ID/revision and actual selection; native ASR still supplies live features |
| Cloud dialogue model, if selected | OpenRouter | Model and actual provider/endpoint, prompt hash, sampler, response IDs and date; credentials stay private |

Current regression baseline: current-partner cosine **0.28**, normal speech pause **700 ms**, native live-decision hop **320 ms**, camera **10 FPS** with face-based speech gating off, and existing 1600-ms TTS chunks. Open speaker labeling has a separate threshold, normally **0.38**; do not confuse it with conversation retention. ECAPA uses the strongest bounded context, up to **3.04 seconds** of an utterance. Longer context does not mean always waiting another three seconds: work can overlap ASR. A short Stop uses a separate policy.

Source pins: TYGER `97d011dcd4c8d2a117e54f00bf5b2d457cefa160`, TYGER bundle `a5d4d73bb74fa4a7082dcd4387e44a1e5ac151b6` (`paths.py`); ECAPA `speechbrain/spkrec-ecapa-voxceleb` revision `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` (`speaker_teacher.py`); Laya source `ffc142012af63d015fa19f05f29f9c5ff329a2cc`, multilingual weights revision `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`. Student selection and its exact checkpoint come from `speaker_runtime.registry()` and the completed V4/V5 validation records, not the current user's name. Save the checkpoint file SHA256 when changing a student.

The tested TTS checkpoint is `laion/Humaneness-Voice-Small-1.1-Alpha-RL-r128-v1`, revision `182bc700cac29b2d69295a33398b13e1ca7fba69`, base file `checkpoints/step-00006109/model.safetensors`, SHA256 `6fe4a757c505732d376a1116a3e5dc130fa5af845e9a9dc8474c2cb4f73b67c6`; merged RL adapter SHA256 `b441554a730000f96617958d73e74474ad83ffde486763750b4a1fd5b52da379`. The live service reports the selected compute precision; do not label merged FP32, BF16 and reference FP32 runs equivalent.

Use a disposable browser profile: camera off, idle play/petting invitations off, requested avatar actions on, no persistent biometric profiles. Default regression tests deliberately seed the old 0.32 setting to check its one-time migration to 0.28. Explicit search settings mark that migration complete, so a requested 0.32 candidate stays 0.32. Use fresh session memory for independent scenarios; keep memory only for explicitly stateful group-conversation sequences.

**Serialize live audio tests.** The native encoder has one shared session slot. Research endpoints return HTTP 423 when a browser owns it. This is a busy test environment, not an accuracy failure: record the attempt as blocked/environmental, let the real user finish, then create a new run. Never evict someone, restart production to obtain a pass, or run two audio suites concurrently. A local runner lock serializes cooperating runners; it cannot reserve the server against a real user.

## 3. Restore exact test inputs

The fixture pack contains only thirteen explicitly named public synthetic TTS clips, approximately **0.68 MB compressed**, no microphone recordings and no enrolled voiceprints. `static/voice-test-fixtures.json` records text, voice-profile ID, seed when known, file SHA256, decoded float32 SHA256, duration and FFmpeg version. Source profiles: [LAION voice profiles](https://huggingface.co/datasets/laion/laion-voice-profiles-annotated), CC BY 4.0 with attribution. Texts are authored test requests.

The archive is a separate [GitHub release asset](https://github.com/LAION-AI/Humaneness-Voice/releases/download/robby-test-fixtures-v1/robby-synthetic-fixtures.zip), so the audited source tree stays text-only. Locally the build is `releases/robby-synthetic-fixtures.zip`. Download it as `test-fixtures.zip`; its SHA256 is `ad3c6036e22d7032e1751089a6f616084a4ab79db0c3b934a85177dc5f907695`. Restore the pack, rather than resynthesizing until a good take appears:

```bash
curl --fail --location --output test-fixtures.zip \
  https://github.com/LAION-AI/Humaneness-Voice/releases/download/robby-test-fixtures-v1/robby-synthetic-fixtures.zip
mkdir -p /path/to/robby-test-fixtures
unzip test-fixtures.zip -d /path/to/robby-test-fixtures
export ROBBY_TEST_FIXTURES=/path/to/robby-test-fixtures
```

`speaker_live.live_test.audio(name)` decodes MP3 through FFmpeg to mono 16-kHz float32LE, then scales each whole source clip to RMS **0.055**. Clip padding is retained. No denoising or speaker-dependent selection is hidden in the loader. The runner rejects MP3 hashes that differ from the pinned corpus. FFmpeg/library changes may change decoded bytes slightly; archive versions and decoded hashes as well as the MP3 hashes.

To generate a **new version** of the inputs on a configured TTS service:

```bash
$PY -m speaker_live.fixtures
$PY generate_speaker_permission_fixtures.py
$PY snapshot_test_fixtures.py --pack
```

Both generators cache existing files. Do not delete the baseline to make it regenerate. Use a new `ROBBY_TEST_FIXTURES` directory. Some older takes used a service-default seed; their exact output cannot be reconstructed from a seed alone. The archived audio and hashes provide repeatable inputs; regeneration is explicitly a new corpus. GPU sampling/cloud responses are not promised bit-exact even with a seed. Never substitute “the best of several takes” without listing every attempt.

The permission generator uses seed **20261026**, temperature **0.7**, CFG **2.0**, no VC, alignment enabled, `small_rl`, and the same public profile IDs. Full requests and other special-case seeds are in the two fixture generators. The names Christoph/Alice in this suite are scripted labels for synthetic voices, not recorded people.

## 4. Exact commands and archived outcomes

The serial runner creates a fresh `logs/voice-assistant-runs/<UTC>-<random-id>/` folder per invocation, with a manifest, per-suite stdout/stderr and copied JSON receipts. It hashes source and inputs, records package versions and the deployed server signature, and rejects a stale success receipt. The current runner also records exact student/teacher file hashes and TTS checkpoint/precision metadata before and after live suites. The successful initial run preceded that inventory instrumentation; its report labels the later read-only model snapshot explicitly. There is **one attempt per job, no hidden retry**. An optional `--keep-going` continues independent suites after preserving a failure. It never changes production defaults or restarts a service.

Start with cheap offline checks:

```bash
$PY run_voice_assistant_checks.py --suites unit
$PY run_voice_assistant_checks.py --suites decision
$PY run_voice_assistant_checks.py --suites stress --speakers christoph,selma --seeds 42
```

The stress suite's loopback research calls currently target `127.0.0.1:8793`; it must run on the ASR host. `--url` selects the browser/readiness endpoint, **not a remote research endpoint**. The real-model continuity helper also uses the host-local FastAPI facade at port 8798. Do not expose loopback research APIs through the public tunnel.

Run the actual browser integrations, in sequence:

```bash
/home/c4r33u19/.local/bin/uv run --no-project --with playwright \
  --python "$PY" python run_voice_assistant_checks.py \
  --suites stops,permissions,browser \
  --url https://YOUR-CURRENT-TUNNEL.trycloudflare.com
```

The default browser executable is `/usr/bin/google-chrome`; set `ROBBY_TEST_CHROME` to another installed Chromium path if necessary. Use the **current** tunnel, not an expired URL copied from an old report. Localhost is allowed for quick tests but is not an HTTPS deployment check. Calls to the selected real cloud LLM incur its usual usage. The assistant's existing private server credential is used normally; never put a key in the command, manifest or fixture archive.

Equivalent direct diagnostic commands, useful for isolating a failing layer:

```bash
$PY validate_live_decision_route.py --base http://127.0.0.1:8797
$PY conversation_stress.py --asr --speaker christoph --seed 42 --threshold .28 --pause-ms 700
ROBBY_TEST_URL=https://YOUR-CURRENT-TUNNEL.trycloudflare.com \
  ROBBY_TEST_QUICK_STOP=1 "$PY" browser_rebalance_check.py
ROBBY_TEST_URL=https://YOUR-CURRENT-TUNNEL.trycloudflare.com \
  ROBBY_TEST_PERMISSIONS=1 "$PY" browser_rebalance_check.py
```

Direct browser commands assume Playwright is installed in the chosen interpreter; otherwise use the `uv --with playwright` wrapper shown above. Direct diagnostics overwrite their named current receipt. Use the runner for comparisons that must retain every attempt. Receipt `passed` and process exit code must both succeed. A timeout, browser error, missing service or occupied session is never a pass. Record environmental failures separately from genuine model/policy failures without discarding them.

## 5. What the implemented tests actually do

### Offline contracts

`run_unit_checks.py` lists all test modules explicitly. `test_sentence_reliability.py` covers full-transcript preservation, distinct identity/audio clocks, foreground bounds, short-Stop freshness, stale reply/admission IDs, VAD-missed literal controls, decision routing and malformed JSON repair without replaying an action. `test_speaker_management.py` covers invite/revoke/ignore parsing, existing-name reuse, membership and authority. Existing audio, playback, ASR selection, secret boundary, multimodal settings and avatar contracts remain in the suite. Mocks are intentional here; they cannot establish acoustic accuracy or actual gesture rendering.

### Real decision-service routing

`validate_live_decision_route.py` sends six **actual** Laya requests with `ROBO_DIRECTOR_MODE=bolt`. This catches accidental replacement of the Stop endpoint by the dialogue model. “Stopp”, “Hör bitte auf” and “Warte bitte…” must stop; “hör nicht auf”, a question about the word Stop and a Stop Motion topic must listen. The health counter must increase by at least six. This does not test who said the words, playback cancellation or general semantic-rule accuracy.

### Component mixture matrix

`conversation_stress.py --asr` runs the real CPU VAD/cropping policy, ECAPA reference comparison, existing V4 predictions from native encoder features, and native final ASR. It constructs a **known target** at 0.64 seconds, followed by 1.5 seconds of continued background. The target is one ordinary question. The second background source is gated at 0.42 Hz, with genuine overlap. Music is an authored four-tone chord with amplitude modulation; foley is authored decaying noise/tone impacts. This is reproducible music-like interference, not a representative music benchmark.

| Condition | Background A gain | Background B gain | Music gain | Foley gain | Main delivery |
|---|---:|---:|---:|---:|---|
| Clean | 0 | 0 | 0 | 0 | Original |
| Quiet TV | .06 | 0 | 0 | 0 | Original |
| TV | .12 | 0 | 0 | 0 | Original |
| Two quiet people | .08 | .08 | 0 | 0 | Original |
| Two people + soft music + foley | .10 | .10 | .04 | .03 | Original |
| Changed delivery + two people | .10 | .08 | .04 | .03 | 1.04× speed/pitch proxy |
| Louder overlapping people | .20 | .16 | .05 | .04 | Original |
| Hard overlap + music | .35 | .25 | .10 | .06 | Original |

Gains are linear amplitudes on RMS-normalized sources, **not measured in-room SNR**. The target source and its background vary in time. Mixtures are clipped at ±0.95. Offsets and foley use a fixed seed; current seed 42 preserves the initial offsets, other seeds shift them deterministically. Every waveform has a SHA256 in the receipt. Do not call a 4% resampling change a full emotional robustness test.

For each row report main acceptance, background-only acceptance, main cosine, supportive fast cosine, crop start/end, source-activity retention, ASR transcript/WER, work time and RAM. “Retained” means the crop contains known energy support (5% of source peak after 20-ms smoothing), with at most 80-ms start / 120-ms end tolerances. It is **not** word alignment. WER is a separate check. The suite's exit gate covers the first five ordinary conditions; `all_stress_cases_pass` exposes the remaining three rather than hiding their failures. Current WER is diagnostic and is not part of that binary speaker/crop gate. A pass therefore does not imply clean transcription.

The old `previous_032_cosine_accepts_main` column is only a teacher-cosine counterfactual. It is **not** a replay of the entire previous application. The component setup pre-enrolls a known reference and mocks conversation transport. It cannot prove that a real introduction, speaker roster, interruption or subsequent reply worked.

### CPU Chromium, paced digital microphone and real services

`browser_rebalance_check.py` opens a fresh CPU-rendered Chrome page, sends 20-ms chunks at real-time pace through the actual client `robo.mic` method, and listens to genuine server events and scheduled browser audio. It does **not** use a fake ASR, stub LLM or canned TTS. Because injection starts at the client microphone method, this test bypasses microphone permissions, capture drivers and the physical acoustic path; retain the existing fake-device lifecycle tests and the real-room checklist for those.

The script checks a full introduction over TV, the same identity with delivery change and overlapping voices/music/foley, manual handover during a spoken story, a rendered full 360° spin with concurrent audio, rendered anger/grimace/nod sequence, a live banana rule during a long uninterrupted monologue, short spoken Stop and subsequent conversational recovery, and the following group sequence:

1. Christoph says “Das ist meine Freundin Alice. Rede bitte auch mit ihr.”
2. Alice speaks without a wake phrase; both Alice and Christoph remain admitted.
3. Christoph returns and keeps his identity.
4. A quiet, unadmitted third voice tries to invite Bob; no endpoint or permission grant is allowed.
5. Christoph says “Ignoriere die Stimmen im Hintergrund. Rede bitte nur mit mir.”
6. Alice's later speech must not produce a conversational endpoint.

The quick Stop mode tests **Stopp / Warte bitte / Halt, bitte hör auf / Pause, bitte**, each during a fresh long real story, then confirms that another ordinary voice request still receives audio. It checks cancellation events and resumed audio, not a hardware DAC measurement. The permissions mode isolates the group sequence. The combined mode checks interactions between features; isolated passes do not replace it.

## 6. Metrics and clock definitions

| Metric | Definition and denominator | Guard against misleading results |
|---|---|---|
| Main-speaker turn recall | Correctly submitted eligible main turns / all eligible main turns | A fragment or wrong identity is not a successful whole turn |
| Background false replies | Replies caused by unadmitted/background speech / background-only hours; also per incident | Report speaking exposure separately from total wall time |
| Unintended TTS aborts | Non-command/background-induced cancellations / bot-speaking hours | Separate legitimate Stop, voluntary user barge-in and manual Stop |
| Speaker continuity | Correct person assignments / eligible accepted turns; identity switches per conversation | Track original anchor and explicit user-corrected aliases separately |
| Permission correctness | Authorized grants/revokes applied, unauthorized grants rejected | Do not score a verbal promise as a state update |
| Explicit Stop recall | Current-person commands that cancel the correct active reply / eligible commands | Include negative controls, stale decisions and next-turn recovery |
| Banana-rule success | Exact interjection admitted and PCM produced before probe ends | Record false positives for apples/no-rule/negated rule/background trigger |
| WER | `(substitutions + deletions + insertions) / reference words` | Lowercase Unicode `\w+` tokenization in current script; no punctuation score; can exceed 1 |
| Background word leakage | Words attributed to the target that came only from a distractor | Empty target references need insertion counts/rates, not a fabricated WER |
| End-of-speech → submit | Endpoint time minus annotated last target speech sample | Do not use end of file, final ASR result or room silence as the source endpoint |
| Cue → cancellation | Cancellation receipt minus annotated last command word (and separately command onset) | Current browser receipt measures file-capture onset only; word-end metric needs added alignment |
| Trigger → first audio | First matching partial, verified decision, PCM receive/schedule and audible output as separate times | PCM arrival is not audible playback; stale/unrelated chunks must match reply ID |
| Stage latency | Final ASR, remaining identity wait, first LLM sentence, first TTS PCM, browser lead | Use monotonic clocks; overlapping stages must not be summed |
| Stability / resources | Crashes, lost/duplicate turns, queue depth, CPU/GPU/RAM and disk growth | Report warm-up separately and include failed/time-out runs |

The existing `asr.observed_quiet_ms` field measures the raw neural speech clock, which can be kept active by TV. It can therefore be small even when the separate foreground/word-stability policy permits submission. Do not use it as ground truth for the main person's speech offset. A future instrumentation test should expose raw, foreground and word-stability quiet durations separately and assert the exact commit basis.

Use `performance.now()` for browser-local events and `time.monotonic()` / `perf_counter()` for server-local durations. Cross-clock comparisons need synchronization and uncertainty bounds. Input source offsets are known in digital tests. For physical tests, record an audible loopback marker and estimate device/acoustic delay rather than subtracting a guessed constant. Do not derive on-device throughput from HTTP wall time.

Report counts plus p50/p95/max and the number of independent conversations/voices. Eight mixtures of one utterance are not eight independent people. Use paired seeds/clips for A/B comparisons and bootstrap by speaker or conversation, not by adjacent 320-ms frame. For binary outcomes provide confidence intervals. Zero false replies in a short test does not prove a low hourly false-reply rate: even the simple Poisson zero-event 95% upper bound is approximately `3 / exposure_hours`.

## 7. Search for a better configuration without overfitting

`voice_assistant_test_matrix.json` separates implemented suites/search axes from proposed future tests. Create the twelve-candidate first-stage plan without running it:

```bash
$PY plan_voice_assistant_search.py --url https://YOUR-CURRENT-TUNNEL.trycloudflare.com
```

It writes `logs/voice-assistant-search-plan.json`: thresholds **.24/.28/.32/.36** × pauses **500/700/1000 ms**, with deterministic configuration IDs and copyable screen/finalist commands. Candidates alter only test-session settings, not production defaults. Example:

```bash
$PY run_voice_assistant_checks.py --suites unit,stress \
  --threshold .28 --pause-ms 700 --speakers christoph,selma --seeds 42,43,44
```

Recommended selection procedure:

1. **Development:** freeze source/model/fixture hashes and use the same ordered inputs/seeds for every candidate. Run offline contracts, then component screening. Preserve every failure and a full configuration table, not only the winner.
2. **Validation:** replay the best few candidates through real browser conversations, including Stop, TV exposure, handover and group permissions. Vary execution order to reduce warm-cache/API-load bias. Component cropping alone cannot rank actual endpoint delay; the current component script evaluates a completed mixture, not a live commit deadline.
3. **Select with hard constraints first:** reject lost/duplicate turns, unexpected identity learning, unauthorized permission grants, ordinary-condition failures or major task-quality regression. Then inspect the Pareto tradeoff between main recall, background false replies/aborts and p95 latency. Do not optimize a single mean cosine or mean latency.
4. **Held-out test:** freeze the chosen configuration and evaluate different identities, scripts and rooms that were never used to choose thresholds. Report all results once. If those results guide another fix, they become development data; obtain a new held-out set.

The JSON includes **proposed**, not achieved, release targets: ≥95% main-turn and explicit-Stop recall, ≤0.1 background false replies and unintended aborts per exposure hour, ≤1 s p95 from the last Stop word to cancellation, ≤2 s p95 from target speech end to submission, and zero unintended permission grants. The current tiny corpus cannot establish these rates. Choose product-specific targets and enough exposure before treating them as release gates.

Next search axes should be separate controlled ablations: longer 2/3-second identity context, prototype-update confidence, known-rival margin, foreground relative-level gate, conversational-owner timeout, hysteresis, alternative final ASR, diarization ensemble and echo handling. Change one family first, then test selected interactions. Different models/context/pause policies require fresh real inference; cached embeddings cannot simulate a changed model, changed crop or live latency.

## 8. More demanding tests to add next

The following are **proposed extensions**, not already executed evidence:

| Priority | Scenario | Necessary assertion / reason |
|---|---|---|
| P0 | Shift short commands by 0/80/160/240 ms against the encoder hop; 300/700/1500-ms words | Stop cannot depend on favorable chunk phase or require a long vowel |
| P0 | Main silent for 700–1000 ms while TV keeps speaking; main resumes just before/after boundary | Submit the complete main turn once; invalidate late ASR when it genuinely resumes |
| P0 | TV says Stop/bananas/name/introduction while the owner remains quiet | No reply/abort/rule firing/identity or permission learning from background |
| P0 | Known main changes emotion, pitch, language, speed, distance and microphone angle | Preserve identity without relaxing the gate enough to admit a rival |
| P0 | Actual two-to-four-person room, similar voices and repeated invited/uninvited returns | Keep a stable admitted roster and current thread; score confusion and false switches |
| P0 | Main says “Das ist Bob”, pauses, then “Rede auch mit ihm”; “Nein, ignorier ihn”; revokes while work pending | Named invitation, pronoun resolution and stale-result rejection remain correct |
| P0 | Bot echo + current-person Stop + a simultaneous unknown speaker | Correctly synchronized Second Ear reference, no echo-induced aborts, Stop still works |
| P1 | Recorded/procedural reverb, varying SNR, telephone bandpass, clipping, AGC and device noise | Report actual mixture power/SNR, overlap ratio and room parameters, not just gain labels |
| P1 | Controlled 0/80/160/320-ms playback delay, 20–150-ms jitter and short packet loss | No stale authorization, repeated speech or accumulating drift; record buffering |
| P1 | TV with acoustically similar voice, same synthetic generator voice, or addressed-looking dialogue | Expose the limit of voice identity as a proxy for “addressed to Robby” |
| P1 | Simultaneous two-person disagreement and quoted/negated Stop instructions | Abstain or ask briefly when uncertain; no hidden forced identity assignment |
| P1 | User correction “Ich bin auch Christoph”; wrong-name collision; forgetting and reconnect | Preserve old trusted anchors; do not merge named people silently; session privacy |
| P1 | Thirty-minute soak with TV, main absence/return, guests, long poems/news and multiple Stops | Measure false events per exposure hour, memory/queue growth and complete tasks |
| P2 | 48-kHz real capture, Bluetooth device switch, suspended tab, CPU throttling, mobile | Exercise capture/resampling/device clocks beyond digital `robo.mic` injection |
| P2 | Camera off, lost track, static open mouth, several faces, visual gating off/on | Audio must still work with gating off; gaze must not imply identity certainty |

For richer soundscapes define a JSONL timeline **before inference**: scenario ID, split, seed, person/source ID, clip hash, source sample range, source onset/offset, gain, room/noise parameters, overlap intervals, admitted roster, expected current speaker, intended addressee, commands and expected state changes. Keep a distinction between *audible speaker identity*, *authorized participant* and *speech addressed to Robby*. A loud foreground voice is not proof of conversational intent.

Generate mixtures procedurally/on demand, store compact manifests and events, and cache only reusable public clips/features. Put a size limit on each run and total cache, retain failures plus a small fixed listening sample, and delete only verified redundant generated intermediates. Do not fill disks with every waveform variant. Do not synthesize reference labels from the same ASR output being evaluated. Word-end latency needs independently checked alignments; diarization evaluation needs source activity labels/RTTM.

If testing Nemotron diarization, pin its checkpoint and NeMo commit and report overlap inclusion, collar, postprocessing and streaming context/cache settings. DER alone cannot measure whether Robby answered the intended person. Keep conversation metrics and diarization metrics side by side. NVIDIA's [evaluation instructions](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/diarization_evaluation.md) describe these required scoring choices. [NIST SCTK](https://github.com/usnistgov/SCTK/blob/master/doc/sclite.htm) is the reference for word-error scoring conventions; the small local script uses the normalization explicitly stated above.

## 9. Physical-room acceptance checklist

Use consenting testers and keep any private recordings/profile embeddings local. Record the scenario/measurements without publishing faces, microphones or voiceprints. Add at least one quiet room and one reflective/noisy room; two microphone distances; a real TV; two nearby people; and a real loudspeaker echo path.

For each room: establish the main person without a wake phrase, ask a multi-sentence task, let TV continue during a one-second pause, change delivery, say a short Stop during long output, resume a task, invite Bob, let Bob and the main person alternate, reject an uninvited third person, revoke Bob, then return to the main thread. Repeat with randomized order and held-out scripts. A human should independently label intended addressee, source-word boundaries, speaker identity and whether the answer/action fulfilled the request. Preserve refusals/abstentions as outcomes, not missing data.

The current foreground-level heuristic can fail if the main person moves away, a TV is louder, AGC equalizes levels or two voices overlap. Very short literal Stop is deliberately more permissive than ordinary speech: a fresh, clear foreground imperative can cancel output despite uncertain short speaker identity; known strong rivals and bot-echo guards still apply, and no voice anchor is learned from that cue. A loud unknown voice might still cancel. This is a user-control tradeoff, **not secure speaker authentication**. Include it explicitly in false-abort tests.

## 10. Required result record and release checklist

Each comparison must retain: source commit/hash and server signature; model/weight/prompt hashes; explicit settings, precision and actual provider; fixture/mixture hashes and seeds; machine/library/browser versions; warm/cold state and concurrency; annotated expectations; every attempt with pass/fail/blocked status; transcripts, identity/permission/decision/reply IDs; stage times and units; aggregate counts/p50/p95 and uncertainty; resource usage; and human-audited listening samples where authorized.

Before publication: run offline checks, component and browser suites on frozen source; inspect every failure; separate current and historical receipts; scan staged code/reports/archives for credentials; publish no private recordings or enrolled voiceprints; link the exact protocol, pinned input manifest and tested source. Keep the old attempts. Update this document and the machine-readable matrix whenever adding a new model, policy, fixture version, metric definition or known limitation.


## 11. Multi-turn task completion and appropriate answer length

A task test must inspect what Robby actually says, not just whether it produced one sentence or some audio. A joke may be short, but a promise to tell one is not a joke; an expanded joke needs its payoff. “Noch einen” must refer to the most recent kind of work and subject, request a new example and avoid repeating the old one. “Den ausführlicher” asks for a retelling, not an unrelated story. A new explicit task/topic overrides an older one.

`validate_conversation_continuity.py` runs three independent **six-turn** conversations through the real selected LLM SSE API. It sends each actual generated answer back in the next request's history. The sequence is: complete robot joke → “Noch einen” → expand the last joke with dialogue in about six sentences → another robot joke at a station → a cat joke without a robot → “Und noch einen”. There is one generation per turn, no hidden retry. Full synthetic transcripts and provider/latency metrics are retained. Run:

```bash
$PY validate_conversation_continuity.py --repeats 3
$PY run_voice_assistant_checks.py --suites unit,continuity
# Compare the local model separately, with the same dialogue procedure:
$PY validate_conversation_continuity.py --model local --repeats 3 --out logs/continuity-local.json
```

The first real cloud baseline passed 17 of 18 automated checks but **repeated one entire cat joke verbatim**. The retained receipt is `static/conversation-continuity-before.json`. Reading those outputs also revealed weak/missing punchlines and robot details leaking into a cat story. The repair adds a bounded joke/follow-up task hint with complete payoff and different setup/punchline, without generating jokes in code. An interrupted reply now retains its actual generated words under an explicit “only a prefix may have been heard” warning; replacing all of those words with a generic interruption note had discarded useful context. This is not a claim that all generated words were played.

Automated checks measure minimum requested content, current topic cues, exact-repeat rejection, non-offer response and stream completion. They **do not establish that a joke is funny or coherent**. Review the full text for premise, punchline, grammar, character continuity and fulfillment. Record a human 0/1 completion label plus a short reason; blind A/B order if comparing models. Add paraphrase/plot duplication checks, not only exact string equality. Do not inflate success by counting every long answer as adequate. One-sentence jokes can be valid when explicitly requested; generic greetings should remain concise.

The SSE test establishes model-context behavior but supplies history explicitly. `browser_conversation_continuity.py` now verifies six actual WebSocket/session-owned turns, with no caller-supplied history and real streamed TTS. Run it via the runner with `--suites dialogue` and Playwright. The plain first request is “Erzähl mir einen Witz über einen kleinen Roboter.” All six structure/topic/audio checks passed in the successful combined run. The interrupted-history unit additionally exercises the real MouthBridge wire; a physical spoken Stop-and-continuation conversation remains a separate room test. For longer work include poems, explanations and stories with counted requirements and a follow-up, plus a return to an earlier topic after a deliberate topic change. Never use synthetic one-sentence latency benchmarks as a response-length policy for production.


## 12. Executed run ledger and maintained regression entry points

The frozen-runtime acceptance run **`20261026T123934Z-e4d72e0f` passed all eight stages**: 432 offline tests; six real Laya decisions; two eight-row component matrices; 18 real-model task turns; 11 combined browser checks; four short Stops plus subsequent recovery; and six actual session-owned/TTS dialogue turns. Its archive is `logs/voice-assistant-runs/20261026T123934Z-e4d72e0f/`; the publishable summary is `static/voice-assistant-test-results.json`. Source and deployed runtime signatures remained unchanged during that run. It took about 8.5 minutes on the working shared host; stage durations describe test execution, not response latency.

Earlier attempts stay separate: `20261026T123219Z-d5f11f3e` failed preflight while ASR was still starting, with zero completed suites. `20261026T123231Z-5c744a3b` ran all six then-selected suites but failed the stateful unnamed-guest naming check; Stop already passed. The corrected full run above was a new attempt, not a silently retried case. The first 18-turn joke baseline had one exact-repeat failure; the revised 18-turn model test and six-turn session test passed automated checks. Punchline/humor quality is still variable, so structural passes are not a quality guarantee.

The working host has **two NVIDIA RTX 3090 cards (24 GB each), driver 550.54.15**, shared with unrelated workloads. These conversational timings are not isolated GPU benchmarks. The protocol does not reuse historical 4090 numbers as if they were measured on this host.

Existing maintained entry points remain available for focused regressions:

| Area | Entry points / evidence | Scope reminder |
|---|---|---|
| Prior partner-focus diagnosis | `validate_conversation_focus.py`, `browser_conversation_focus.py`, `static/conversation-focus-update.html` | Historical operating point; old defaults/failed acoustic controls are retained |
| Complete poems/stories/detail | `validate_long_replies.py`, `static/long-replies-update.html` | Count requirements and read actual words, not only first PCM |
| OpenRouter endpoint/key boundary | `validate_openrouter.py`, `browser_openrouter_check.py`, `static/openrouter-update.html` | Live provider catalogue may change; archive actual selected endpoint and date |
| Alternative German ASR | `static/asr-model-update.html`, `asr_catalog.py` | Different recognizers need real re-inference; no accuracy ranking from a few synthetic clips |
| Visual/action behavior | `static/avatar-command-update.html`, `static/avatar-checks.html`, offline avatar modules | Camera/renderer behavior and conversation correctness are different layers |
| Older Stop/echo diagnostics | `static/stop-fix-update.html`, `speaker_live/` | Historical strict-path results do not automatically cover the current open-sentence policy |

Before reusing an older script, inspect its model selection, assumptions, service URLs and fixture version. Some historical scripts deliberately select local mode or earlier speaker gates. They are useful diagnostics, not automatically valid comparisons against today's default route.

For the separate user-requested local 4B hardware investigation, see the [published local E4B report](https://projects.laion.ai/Humaneness-Voice/robby/gpu-unlock-4b.html) and its pinned upstream/benchmark scripts. A compile failure or required GPU reset is not a measured inference speedup. Keep hardware experiments and conversational-quality comparisons separate.
