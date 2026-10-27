# Robby voice-assistant test protocol

Version 1 · 2026-10-26 · English · implementation and development evidence, not a room-accuracy certification.

This is the repeatable test procedure for Robby's microphone → ASR → person memory → turn/interrupt decision → LLM → TTS → browser playback pipeline. Its primary question is: **can the assistant continue a useful conversation with the current person while a television or other people keep speaking?** Tests must also prove that an invited person can join, and that the current person can stop an unwanted spoken reply.

Start with the executable regression checks below. Use the configuration-search procedure only after those checks work. **Implemented**, **executed**, and **proposed** are different statuses throughout this document. A passing unit test is not a successful microphone conversation. A lower cosine threshold is not automatically an improvement.

## 1. Current evidence and its limits

### October 27: role confusion, person memory and curiosity regression

See [SOCIAL_IDENTITY.md](SOCIAL_IDENTITY.md) and the
[English report](static/social-identity-report.html). Run the browser audio suites
sequentially against the deployed source hash, using authored synthetic fixtures;
keep private microphone recordings and voiceprint arrays out of published results.

| Test | Required assertions | Executable / receipt |
| --- | --- | --- |
| User/robot role | “I am Christoph” names the user; a correction is acknowledged; another name in history does not become the current person | `check_social_dialogue.py`; `static/social-dialogue-*.json` |
| Current-person name lookup | A narrow identity question uses the named current speaker; weak continuity asks for confirmation; general knowledge/relationship questions still use the LLM | `test_social_profiles.py`; actual browser name turns |
| Voice variation | Calm and playful questions keep the correct person; weak continuity never creates a new acoustic anchor | `test_social_profiles.py`, `browser_social_profiles.py` |
| Speaker change and return | Christoph → Selma → Christoph must yield both the correct endpoint identity AND the correct spoken name | `browser_social_profiles.py`; the earlier failed receipt is retained |
| Optional profile persistence | Default OFF; explicit consent; reconnect without another introduction; disable removes local storage; Forget clears face/voice state | `browser_social_profiles.py`; multimodal consent checks |
| Traits are uncertain | GPU impressions toggle changes returned dimensions; trait changes do not determine identity; disabling use hides current summaries | `static/social-vocal-traits.json`, `test_social_profiles.py` |
| Repeated audible background | Two rejected background observations may produce one tentative, audible question, without enrollment; relevant question content is required, not just playback; known soft voice, invalid floor and ignore instruction must not | `browser_social_curiosity.py`, `check_social_curiosity_prompt.py`, `test_social_profiles.py` |
| Camera still optional | Camera, Face ID and face VAD start OFF; position tracking independent; real CPU two-face detection and descriptors; unambiguous speaking face can bind without gating speech | `browser_multimodal_checks.py`, `test_social_profiles.py` |

The browser profile suite uses 20-ms digital microphone packets, actual Parakeet,
ECAPA, Ears, Qwen or the explicit name-lookup controller, real TTS and PCM playback.
The camera suite uses fake device permission, synthetic landmark trajectories and
public imagery. Do not describe these as physical-device accuracy measurements.
The profile suite has six turns on two synthetic voices, not a representative
biometric benchmark. Ten role probes include six deterministic name lookups and
four real LLM generations; never report all ten as LLM accuracy successes.

Future held-out expansion: at least 20 consenting human speakers, quiet and
television rooms, two microphones/distances, multiple emotions/languages per
speaker, explicitly corrected identity variants, twins/similar voices, replayed
voices, stale camera frames and overlap. Measure name confusion, duplicate-profile
rate, unnecessary naming questions per ten minutes, missed actual speaker changes,
false proactive questions, turn completion and interruption latency. Sweep
threshold, rival margin and continuity timeout jointly; do not optimize recall
alone by accepting every voice. Include private test fixtures only with suitable
permission; publish anonymized aggregate results rather than voiceprints.

### Historical initial protocol evidence

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


## 13. Version 2: adversarial, stateful conversation protocol

**Initial complete adversarial baseline (26 October 2026): 12/34 steps passed,
22 failed, none blocked.** This stronger test therefore does not support a
general reliability claim. See `static/adversarial-before-fix.json` and the
[English report](static/adversarial-report.html) for the full failed cases.
The earlier seven-scenario harness attempt blocked all 34 steps because its
setup assertion incorrectly equated one audio file with one conversation turn;
that attempt is retained separately as `static/adversarial-harness-baseline.json`.
One transient speech-service error also occurred in that initial harness run.

The baseline exposed a genuine cached-speaker-window problem: prefetch selected
the last 3.2 seconds even when they mostly contained TV after the actual request.
The correction selects the most audible 3.04-second window within the current,
bounded utterance, using the same energy selector as final enrollment. Thresholds
and known-rival rejection are retained. A one-word literal Stop uses a 640-ms
voice window, while longer commands retain 1.28 seconds; fresh phonemes can
extend a partial cue without refreshing its expiration. A separate bounded LLM
continuation handles an acknowledgement in place of a requested story/joke.
These are implementation changes to verify in new frozen runs, not assumed wins.

The input name is spoken: `Fips` and the acoustically equivalent `Fipps` spelling
are accepted by the name assertion. This alias is explicit in the versioned
plan; it does not excuse a missing name, changed character or lost setting.

The next complete run (`20261026T134528Z-41df665b`) passed 16/34 automated steps
versus 12/34 before the changes. Stop/Wait improved from 3/8 to 6/8, with successful
commands cancelling 424–894 ms after the isolated source's estimated energy end.
Both known negative interruption controls passed. These eight correlated inputs
do not establish a population accuracy or a robust p95 latency. There were no
observed false events in 47.46 seconds of declared negative windows; that short
exposure gives a weak Poisson upper bound of roughly 227/hour, not certification.

The subsequent frozen run (`20261026T142143Z-be95cdec`) passed **18/34** steps,
with no blocked steps. Source, deployed server and model inventories remained
unchanged throughout its outer eight-suite run (`20261026T141603Z-f6e5eac5`).
Stop/Wait remained **6/8**, with successful cancellations 325–853 ms after source
energy end; Stop at 240 ms and Wait at 0 ms still failed. The separate clean
18-turn conversational continuity regression passed 18/18. The older combined
browser's eleven behavioral assertions passed, but its overall run correctly
failed on a planning-service ValueError. These are different tests and must not
be merged into a claim that noisy conversation or all browser behavior passed.

| Complete development run | Passed steps | Stop/Wait | Interpretation |
|---|---:|---:|---|
| `20261026T132456Z-b5b8be08` | 12/34 | 3/8 | Initial complete baseline |
| `20261026T134528Z-41df665b` | 16/34 | 6/8 | Foreground reference window and shorter literal-stop window |
| `20261026T142143Z-be95cdec` | 18/34 | 6/8 | Matching volume-gate context and creative-task completion guard |

This is development iteration on reused inputs, not a held-out causal A/B study.
API generation varies, and the later plan corrects phonetic name aliases as
documented below. Preserve the exact plan hashes and inspect failed assertions,
not just the totals. In the last run, the guest invitation itself failed; later
successful rejection of that guest cannot prove successful admission/revocation.

Tool controls independently exposed unrelated Wikipedia results, spoken decimals
split into integers, and a news answer without a real lookup. The original
domain-only Wikipedia assertion passed but **source-title inspection rejected it**.
The updated assertion also requires a relevant article title. Runtime fixes
strip inflected delivery instructions from the entity query, require real calls
for explicit current-source requests, remind the model to preserve decimal
numbers, and repair malformed final JSON/schema once with tools disabled. A
repair never re-executes successful actions. Targeted post-fix results are
separate receipts; they do not overwrite the three complete noisy runs above.

The separate older combined browser regression then exposed another real issue:
a correct voice match was rejected as quiet because a long TV pre-roll diluted
the full-buffer level percentile. Its correction measures the level on the same
most-audible utterance context used for the voice reference. The follow-up run
must verify that correction; the earlier failure remains archived. The
18-turn model-only regression passed 17/18 in that run, with one incomplete/weak
joke. Transcript inspection also found `Phipps` retained across a topic change: the
raw name-regex check failed on spelling, while the more serious incomplete first
joke and ASR background insertions remain real failures. Do not treat every raw
assertion failure as an independent model error.

The original regression is retained. The additional `adversarial` suite runs real browser conversations with prerecorded **public synthetic** voices through the current microphone transport, Nemotron ASR, speaker gate, selected language model, live lookup tools, TTS and browser playback. It is not a mock conversation and it does not inject caller-authored conversation history. The application owns its memory across turns. Every scenario starts with a fresh session; turns inside that scenario deliberately share memory.

### Freeze expectations before inference

`adversarial_scenarios.py` declares seven scenarios and 34 steps. Each step specifies the input clip, expected speaker, noise condition, task/content requirements and expected permission or interruption behavior. `validate_plan()` rejects duplicate identifiers and unsupported cue phases. Each run saves the exact plan and its hash before opening the conversation.

| Scenario | Steps | What must work |
|---|---:|---|
| Conversation thread | 4 | A complete joke about Fips at a railway station; another joke with the same character/place; a real Wikipedia lookup; recall Fips and the location after the topic change |
| Long tasks and live tools | 3 | An eight-line poem rather than a one-line offer; actual Berlin weather lookup; two current Tagesschau news items |
| Noisy group permissions | 6 | The current speaker invites Alice, Alice speaks, the owner returns, Alice is revoked, then the revoked guest and an unauthorized background invitation are ignored |
| Interruptions | 11 | TV Stop/name/invitation spoof does not cancel; “Bitte hör nicht auf” does not cancel; Stop and Wait each tested at four input-hop offsets; conversation resumes afterward |
| Background-only exposure | 2 | A television-like voice saying Stop, names, bananas or invitations must not gain permission or start a conversational reply |
| Acoustic stress, Christoph | 4 | Complete target request and sensible answer with two competing voices, instrumental music, colored noise and transient knocks at 20/12/6/0 dB SNR |
| Acoustic stress, Selma | 4 | Same interference severity with a different foreground identity and script |

The 20/12-dB acoustic conditions are required regression cases. The 6/0-dB conditions are predeclared diagnostic boundaries; they still appear as failures when unsuccessful. A report exposes both `passed` (all required scenarios) and `all_cases_passed` (including diagnostic boundaries). Never describe a required-only pass as “all stress conditions passed.” Speaker admission alone cannot make an acoustic case pass: transcript, task, identity, completion and audio checks are separate.

The 6-dB cases also use a 4% speed/pitch change and a deterministic sparse synthetic reverberation impulse. These are combined stress cases, **not causal ablations** of speed, reverberation or SNR. To isolate a cause, rerun the same utterance with exactly one changed factor in a separately versioned plan.

### Fixture provenance, storage and ground truth

`generate_adversarial_fixtures.py` creates eleven new clips with a fixed seed and copies fifteen existing synthetic clips without modifying their bytes. Each new clip is generated once; a bad take is retained, not silently replaced. The generator records the full request, model/adapter hashes and precision from the TTS service. Its 26 MP3 inputs total approximately 1.7 MB. `static/adversarial-fixtures.json` pins file hashes, decoded normalized float32 hashes, authored text, voice-profile identifiers, FFmpeg version and generation metadata.

The input root is `ROBBY_ADVERSARIAL_FIXTURES`, defaulting locally to `/mnt/nvme/robo-speaker-studies/robby-adversarial-v2`. Existing inputs come from `ROBBY_TEST_FIXTURES`. The generator refuses to overwrite a mismatching cached baseline or a half-written new clip. Restore published exact inputs for comparisons; resynthesis on a different precision/backend is a new corpus, even with the same seed.

Authored scripts are independent of the ASR under evaluation, but they are **not an independently listened transcription of every synthesized clip**. A listening reviewer must flag TTS omissions/pronunciation errors separately before attributing every word error to ASR. Keep that distinction in the result ledger. No personal microphone recordings, actual enrolled identities, API credentials or arbitrary cache contents belong in the public pack.

Restore the exact v2 pack without resynthesizing:

```bash
curl -fL -o robby-adversarial-fixtures-v2.zip \
  https://github.com/LAION-AI/Humaneness-Voice/releases/download/robby-test-fixtures-v2/robby-adversarial-fixtures-v2.zip
echo '1319d3ddf25c3c8bf213fb5857d67093d11812c2bfd56fec5235503d01e62002  robby-adversarial-fixtures-v2.zip' | sha256sum -c -
mkdir -p fixtures-v2
unzip robby-adversarial-fixtures-v2.zip -d fixtures-v2
export ROBBY_ADVERSARIAL_FIXTURES="$PWD/fixtures-v2"
```

The archive contains 26 MP3s, their generation metadata, manifest and provenance
notice (54 entries, 1,458,538 compressed bytes). Keep the v1 pack as well for the
older independent regression suites. Source attribution is in the pack; no
private microphone recordings or human voiceprints are included.

Only reusable MP3s and compact manifests/events are retained. Mixtures are generated in RAM and limited to 120 seconds per allocation. Soaks must stream bounded blocks. Do not save every float32 mixture. A replay can rebuild each input from the fixture hashes, seed, offset, speed, synthetic room impulse and SNR in its receipt.

### Measured soundscapes

`adversarial_audio.py` mixes independently offset synthetic speech sources with authored instrumental chords, colored noise and short impulsive sounds. It measures target/interferer RMS on the isolated target's active samples, then scales interference to the requested SNR. A common final gain keeps the mixed peak below 0.94 without changing SNR. The record contains requested/measured SNR, seed, target/mixture hashes, source duration, target activity support, tail duration and peak scaling. Background continues after the target ends and while the assistant answers.

Target activity uses a 20-ms RMS envelope with a fixed 5%-of-peak floor. This is an **energy-support annotation, not a word alignment or perfect speech VAD**. Do not rename its boundaries “last word.” Finite looping clips and sparse synthetic reflections are controlled approximations, not recorded television, a measured room impulse response, real AGC or source separation.

### Run the suites

```bash
cd /path/to/Humaneness-Voice/robby
export ROBBY_TEST_FIXTURES=/path/to/v1-fixtures
export ROBBY_ADVERSARIAL_FIXTURES=/path/to/v2-fixtures

# First-time generation on a host with the documented TTS service; normally restore the pinned pack instead.
python generate_adversarial_fixtures.py

# Offline harness contracts: SNR, clipping, determinism, WER insertion penalties,
# confidence bounds and completeness/uniqueness of the declared test matrix.
python -m unittest test_adversarial_protocol

# All seven real-browser scenarios, serially. Playwright/Chromium required.
python run_voice_assistant_checks.py --suites unit,adversarial \
  --url http://127.0.0.1:8793 --seeds 42 --keep-going

# Focused repair cycle, then rerun the combined suite to detect state interactions.
python run_voice_assistant_checks.py --suites adversarial \
  --adversarial-scenarios interruptions,background_only --seeds 42

# A new augmentation seed does not create a new independent human speaker.
python run_voice_assistant_checks.py --suites adversarial \
  --adversarial-scenarios acoustic_christoph,acoustic_selma --seeds 43,44 --keep-going
```

When Playwright is not installed in the model environment, use the same `uv run --no-project --with playwright --python "$PY" python ...` wrapper as the earlier browser suites. Use the current HTTPS URL for a deployment transport check. A localhost pass is not an HTTPS test. Neither mode exercises actual microphone permissions, capture hardware or loudspeaker acoustics.

The parent runner holds its existing lock; a standalone `browser_adversarial_check.py` also acquires that lock. They never restart a service or evict a user's session. Live audio suites remain serial because the native encoder has one shared slot. The wrapper timeout for the complete adversarial suite is one hour; that is a test execution limit, not an acceptable conversational response time.

### Assertions and timing

- **ASR:** Unicode lowercase word edit distance divided by authored reference word count. Inserted TV words count; WER can exceed 1. A target speech case needs WER ≤ 0.35. Report the exact hypothesis, not just the number.
- **Identity:** accepted endpoints must belong to the expected current participant. An invitation must create the expected admitted roster; revocation must remove the guest. Do not equate “some speaker accepted” with correct identity.
- **Turn segmentation:** one-sentence requests normally allow one submission. Explicitly marked two-sentence inputs allow at most two distinct submissions because their real source pause may exceed the user's threshold. Duplicate transcripts and extra TV turns fail. The two-sentence enrollment fixture is not forcibly treated as one utterance.
- **Task completion:** minimum substantial content, explicit remembered names/place, topic retention and non-identical follow-up, exact requested poem structure, source metadata and completed audio playback. These checks reject common regressions; they do not prove humor quality, story coherence or factual accuracy.
- **Real tools:** trusted `reply.sources` must contain the expected provider domain, and the real tool director must have executed. A URL merely mentioned in generated prose is insufficient. Review the returned facts against the source snapshot/date; live news and weather cannot have one immutable expected answer.
- **Stop:** a real long spoken reply must be active before the cue. The test requires a genuine server cancellation, cleared browser playback and a bounded reaction after the isolated command's energy support ends. Cleanup by the UI Stop button occurs only **after** evidence is captured and is never counted as voice interruption success.
- **Phase:** Stop and Wait begin at sample offsets equivalent to 0/80/160/240 ms in the 320-ms input stream hop. This is the **fixture-start** phase; clip-leading silence can shift the first spoken phoneme. Log both activity support and input-sample count. These are deterministic input phases, not claims about a physical device's clock.
- **Negative controls:** background speech and negated Stop must cause no conversational endpoint, cancellation or applied permission change within their declared exposure window. Record how many seconds were exposed, not just how many “tests passed.”
- **Latency:** browser `performance.now()` records injection start/end, endpoint, actual PCM arrival and cancellation. Report source-energy-end → endpoint/PCM and cue-energy-end → cancellation, alongside capture-start values. Packet scheduling lag is recorded separately. Do not sum overlapping server component times or subtract unrelated server/browser clocks.

### Failure taxonomy and uncertainty

Keep failures distinct: fixture intelligibility, ASR contamination/omission, premature or missing turn submission, wrong/rejected speaker, unauthorized permission, lost conversational context, incomplete task, missing/incorrect tool output, missed/false interruption, TTS/service failure, playback failure and environmental contention. Several may co-occur. A failed enrollment blocks dependent steps; blocked is neither success nor a model-quality measurement. No automatic retries convert a failure to a pass. A rerun is a new timestamped attempt with its own source and configuration.

The runner keeps full synthetic event traces in `logs/adversarial-runs/<run-id>/result.json`, plus the predeclared plan. Its current public receipt is `static/adversarial-check.json`. The outer protocol runner also archives that receipt, subprocess logs, source/model inventories, package versions and server signatures. A changed source/runtime/model during the run invalidates the combined pass.

Report numerator/denominator and exposure, not unsupported percentages. Wilson intervals can summarize counts descriptively, but these reused synthetic scenes are correlated and do not represent independent real-human trials. With zero false events in T seconds, the one-sided 95% Poisson upper bound is `-ln(0.05) * 3600 / T` events/hour. Zero events in one minute gives an upper bound near 180/hour, **not proof of zero false activations**. To support a bound of 0.1/hour even under a simple homogeneous Poisson assumption takes about 30 event-free hours; real deployment additionally needs diverse conditions and representative participants.

### Required human review and next extensions

A reviewer should score each complete transcript for (1) answering the requested task rather than acknowledging it, (2) keeping character/place/facts across follow-ups, (3) respecting exact length/structure constraints, (4) a coherent joke setup and payoff or a coherent story, (5) source-supported tool facts, and (6) natural recovery after an interruption. Use pass/fail/uncertain plus a short reason; automated keyword counts are screening checks. Keep model-generated self-grading separate from human acceptance.

The next unclaimed tiers are synchronized actual playback echo with several measured delays; genuine room/TV recordings; 30-minute and multi-hour soak runs; packet jitter/loss and suspended-tab recovery; mobile/Bluetooth/camera operation; same/near-identical background voices; and held-out consenting human speakers. The optional experimental model/configuration search must reserve a held-out split, keep all failed candidates and compare paired scenes before selecting a setting. Do not retune on the final held-out evaluation.


### Clean-input and isolated tool controls

Run `python run_voice_assistant_checks.py --suites input_audit,tool_evidence --keep-going`
serially with the other audio tests. `audit_adversarial_inputs.py` transcribes
all 26 exact clips without interference and compares their ECAPA embeddings
against the corresponding clean introduction. `passed` means the audit completed;
`all_inputs_unflagged` is a separate field. A low clean cosine can indicate
synthetic voice inconsistency, an imperfect reference or a teacher limitation.
It is not automatically an application failure or a reason to discard a take.
Neither this audit nor author-provided scripts replace independent listening.

`browser_tool_evidence_check.py` supplies three typed controls through the actual
browser, using real Wikipedia, weather and news tools and spoken output. This
isolates tool/LLM/TTS failures from acoustic admission. Trusted lookup receipts now
survive the sentence-streaming bridge and include tool name, timestamp, result
hash and success/error status. Weather receipts additionally retain bounded
structured numeric facts, allowing the predicted daily min/max temperatures to
be compared with German number mentions in the spoken answer (0.6 C rounding
tolerance). This is a numerical guard, not verification of every weather claim
or its grammatical association. News and Wikipedia still need human factual
review against retrieved sources. A URL in generated prose alone cannot pass.
No raw articles, private microphone recordings or API credentials are added to
these public synthetic receipts.

The clean audit flagged 8 of 26 takes, without excluding or regenerating any.
Two flags are the spelling difference Stop/Stopp. Others include low ECAPA
similarity for the same synthetic source profile and corrupted clean ASR. In
particular, the negated-stop clip was transcribed as unrelated words; a passing
no-cancellation assertion on that input is **not evidence that negation was
understood**. The release keeps the exact original take for independent review.
The current input audit's `passed=true` means the audit ran to completion;
`all_inputs_unflagged=false` records its actual finding.

A bad preceding turn can invalidate the premise of a follow-up: for example,
if the first joke was rejected, failure to remember its character does not by
itself establish a memory-system bug. Inspect the whole scenario rather than
adding correlated downstream assertions as independent failures. Similarly,
background words can trigger an unintended news lookup inside a joke request;
inspect both the transcript and source metadata, not only whether a tool ran.


For the final recipe, the spoken-name aliases explicitly include `Fips`, `Fipps`,
`Phips` and `Phipps`, which are acoustically equivalent in this context. The
predeclared plan hash records this scoring correction; older raw receipts are
not rewritten. Creative-task completion additionally requests one bounded
continuation when an ordinary joke stops at a single setup sentence or an
ordinary story stops before three sentences. Explicit one-liner requests are
exempt. This structural safeguard does not judge whether a punchline is funny
or a story makes sense; the original user request and actual preceding text
remain the model context.


### Targeted tool/schema validation after the complete stress run

Run `20261026T144116Z-50ba538a` passed all three requested suites: **453 offline
checks**, the **three actual typed tool controls**, and the **11-case combined
browser regression**. Source/runtime/model inventories stayed fixed during that
run. It did not rerun the 34-step acoustic/phase matrix; its successes do not
replace the 18/34 noisy result or the two remaining phase-specific Stop misses.
The targeted manifest is `static/adversarial-targeted-run.json`; the earlier
complete manifest is `static/adversarial-full-run.json`.

The repaired tool run produced a relevant Pinguine article, actual Tagesschau
receipts, and accurate spoken decimal min/max temperatures. It still added an
unrequested joke about penguins. Its automated pass is therefore evidence for
lookup routing and the stated numeric/structural checks, not certification of
all factual prose or excellent conversational taste. Genuine human listening
and source-fact review remain outstanding.


## 14. Interrupted replies: private playback metadata and broader Stop phrases (2026-10-26)

The legacy `[My preceding reply was interrupted…]` note was stored inside an
assistant history message. It could be imitated by the LLM and split into
visible/spoken fragments by sentence streaming. History now stores generated
words separately from `playback_interrupted`. Both directors put playback
uncertainty in an internal system instruction, never in assistant speech
examples. Legacy history is migrated on read. The sentence framer buffers and
drops the known legacy note even across single-character token boundaries;
compact validation removes it from the TTS script, and the UI has a compatibility
guard for old-server reply payloads. Legitimate pause cues remain intact.

While Robby speaks, the live controller uses configured local multilingual Laya
for semantic Stop decisions at 320-ms opportunities, plus a fast path for clear
commands. Before the fix, `Danke, das reicht` never entered the semantic check
and was not a direct command. `Halt die Klappe` was classified as Stop by Laya
but missed the fast literal-command path. Anchored closing/blunt requests now
include these phrases, `Das reicht mir jetzt`, and the bounded ASR variant
`Danke, cas reicht`. `Danke` alone, `Das reicht nicht`, ordinary mentions and
questions do not become direct Stops. Existing acoustic speaker, rival, quiet
background and echo checks still apply; these text rules do not bypass them.

Verification at the initial note/wording fix: 457 offline checks passed, including interrupted-history retention,
legacy migration, arbitrary streamed token boundaries, TTS prompt filtering and
negative Stop controls. Six actual Laya service requests distinguish the three
new positive phrases from continuation/negation/quoted-meaning controls. Two
real Qwen/Cerebras follow-ups retain clean spoken text. An injected legacy
reply in real HTTPS Chromium is absent from both the latest-reply card and chat.

The new actual-ASR/identity/TTS test is `browser_completion_stop_check.py`: it
generates three fixed synthetic inputs once, tests the two positive commands
at 0/160-ms phases with 12-dB interference, a negated control and a subsequent
spin reply. Its first attempted run was **blocked before enrollment by an
occupied ASR session**; no user's connection was evicted. Do not describe that
blocked run as a passing acoustic test. The later completed attempt missed all four positive noisy cancellation cases; the negated control passed, and the spin follow-up timed out. These failures are retained in `static/completion-stop-browser.json`. A clean Parakeet audit also found poor transcription of the synthetic blunt-Stop take, so input fidelity needs review. The preceding 6/8 Stop phase result
remains a documented limit, not a result proven fixed by text-only checks.

```bash
python check_completion_decisions.py
# Requires the pinned v2 inputs, Playwright/Chrome and a free native ASR slot.
python browser_completion_stop_check.py --prepare --url http://127.0.0.1:8793
```

Use `ROBBY_COMPLETION_FIXTURES` for the three additional MP3s and metadata; their
one-take generation records and hashes are retained. Every browser attempt has
a timestamped `logs/completion-stop-*.json` receipt; blocked and failed attempts
are never silently retried into a pass. [Current explanation and evidence](interruption-update.html).


## 15. Default ASR, camera privacy and GPU delivery captions

Application defaults now select `parakeet_v3` for final recognition and do not
request camera access automatically. Live previews/interruptions remain Nemotron.
Historical stress runs are pinned to Nemotron: `run_voice_assistant_checks.py`
and `browser_adversarial_check.py` accept `--asr-model parakeet_v3` for explicit
comparisons; their default remains `nemotron` so an app default change does not
silently change the benchmark. Each new runner manifest records this selection.

The new `browser_voice_caption_check.py` explicitly selects Parakeet and checks
GPU captions on/off with real synthetic microphone transport, final transcripts,
identity assignment, LLM responses and actual TTS playback. It separately checks
fresh settings, migration of old defaults and persistence of a later explicit
Nemotron choice. The real camera is never opened in this test. It passed over
HTTPS; available captions added zero awaited delay after final ASR/identity.

`check_voice_captions.py` runs 36 measured HTTP probes: four audio lengths, three
modes (ASR alone, captions alone, concurrent), three repeats. These repeated
waveforms are timing probes, not emotion-quality evaluation. A proposed future
quality test must use independent recordings, diverse emotions/languages, noise
and overlapping speakers, plus blinded human judgments and caption abstention.
`check_voice_caption_prompt.py` records an actual response to a conflicting
acoustic cue; it requires manual semantic review, not just a substring pass.

See [VOICE_CAPTIONS.md](https://github.com/LAION-AI/Humaneness-Voice/blob/main/robby/VOICE_CAPTIONS.md)
for model revisions, normalization, device selection, exact commands and known
limitations. All 465 current offline tests pass. Caption tests cover disabled,
short, silent, invalid, timed-out and stale inputs; a late future must not block
an answer. The text-only model test does not establish real emotion accuracy.

The first completion-Stop browser attempt was blocked and is retained separately
as `completion-stop-blocked.json`. The subsequent noisy attempt is a failure:
0/4 positive cancellation cases, one recognized negative control preserved, and
a timeout before the final spin assertion. Do not replace it with a unit-test
pass or silently regenerate a better-sounding input. The original three
one-take MP3/metadata pairs are the `robby-completion-stop-fixtures-v1.zip`
release asset, alongside the pinned v2 base corpus. Their manifest and SHA-256
values are in `completion-stop-fixtures.json`. Restore them under
`ROBBY_COMPLETION_FIXTURES` to rerun the same evidence.

```bash
python check_voice_captions.py
python check_voice_caption_prompt.py
uv run --no-project --with playwright --python /path/to/python \
  python browser_voice_caption_check.py --url https://YOUR-TUNNEL
python run_voice_assistant_checks.py --help  # --asr-model pins the comparison
```


## Local quick-reaction quality regression

Use `quick_reaction_cases.py` for the fixed 40 authored cases, including serious
news, jokes, elliptical follow-ups, language switches and deliberate omissions.
`benchmark_quick_quality.py --label LABEL --split all` runs three fixed seeds
per eligible case and records full prompt, raw output, validated output and
local latency. `judge_quick_quality.py` performs randomized, blinded A/B ratings
with Qwen; invalid judge replies remain explicit. Model preferences do not replace
human listening or semantic review. Preserve failed prompt variants and original
held-out failures; a rerun after a repair is a regression test, not an untouched
holdout. Never publish real user conversations as benchmark fixtures.

Run `benchmark_quick_dialogue.py` for a six-turn actual-model thread,
`benchmark_quick_quality_audio.py` for four real TTS handoffs, and
`browser_quick_reaction_check.py --url HTTPS_URL` for the complete browser
contract. These scripts refuse to overwrite existing quality receipts; archive
old results before an intentional rerun. Main answers are tested separately from
optional first reactions: a good preface does not imply a good joke or a shorter
time to substantive content. See `static/quick-quality-report.html` for all
results and limitations.


## Two-stage speech and memory-grounded first reactions (2026-10-27)

See [TWO_STAGE_SPEECH.md](TWO_STAGE_SPEECH.md) for the exact scripts, settings,
receipt files and scope. Preserve all failed attempts; use fresh output filenames.
The GPU cache-isolation pilot is deterministic at temperature zero and deliberately
truncated, while conversational TTS uses 0.8. Do not compare those as quality runs.
Measure first sound separately from the time the substantive answer is heard.
Test the first reaction with known people, unknown people, memory disabled, empty
memory, current-session-only facts, explicit name corrections, English, and hostile
instructions embedded in reference data. Reconnect with a zero history budget to
prove that recall comes from wiki retrieval. Each browser test uses and resets its
own isolated memory capability; never inspect/reset a real visitor's store.
Test cancelling at the reaction and at the informed transition, not only in the
legacy exact-length story route. Retain browser scheduling gaps and server PCM
timings without treating them as physical loudspeaker measurements.


## Proactive environment regression suite — 2026-10-27

See [PROACTIVE_ENVIRONMENT.md](PROACTIVE_ENVIRONMENT.md) for model pins, launch
commands and detection thresholds. The new authored fixtures cover steady noise,
impacts, beeps, speech onset, echo/playback suppression, cooldowns and cancellation.
`check_sound_caption.py` compares actual serial and parallel GPU services.
`browser_environment_check.py` checks HTTPS fake-device capture, automatic welcome,
real autoplay blocking, a simulated permission denial with typed fallback, sound
reaction, visible timing, mobile layout and persistent shutdown.
`browser_environment_voices.py` uses the existing synthetic speaker corpus with
real ASR/ECAPA/Ears/LLM/TTS: quiet TV must not elicit a reply; a clearly different
foreground voice may elicit a question without enrollment or focus transfer.
The ordinary memory and two-stage browser suites are rerun with automatic start
and environmental sensing explicitly disabled to isolate those contracts.
The adversarial harness disables page autostart but leaves environment defaults
intact; its quiet-background expectations must still hold. Run browsers serially.

Save every attempt with source hash, exact fixture seed and event trace, including
failures. Do not count an interrupted question as a fully heard reply. Speaker
identity and physical echo immunity need separate room/device trials. Future
coverage should add held-out licensed real room events, level/SNR sweeps, unknown
voice variants, long TV programs, far-field microphones and measured acoustic
loopback; do not tune solely to the small procedural test set.


Music follow-up: `python browser_environment_music.py --url https://YOUR-DEMO`
uses authored melody/bass/drums through Chrome's microphone path. Quiet music
must not trigger, a prominent onset must yield an actual spoken music response,
and a second onset after the detector cooldown must be suppressed by the
three-minute music-category cooldown. Receipt: `environment-music-*.json`.
The ordinary speech VAD remains enabled; this is an instrumental-music test.

## Environmental sounds, music and after-question answers (2026-10-27)

This extension tests three distinct outcomes separately: **capture/routing**, **sound
interpretation**, and **delivered speech**. A visible reply or a nonempty caption
alone is never evidence that the user heard it or that its label was correct.
All recordings in this automated suite are authored synthetic speech/music or
identified public test clips; never use or publish private microphone recordings.

### Fixed sources and reproducible commands

Run in the Robby directory using the pinned Python environment. Start the existing
ASR/ECAPA/Ears/LLM/TTS services plus `sound_caption_worker.py` (GPU, port 8803)
and `sound_event_worker.py` (CPU, port 8804). Check both `/state` routes. The
`launch_bolt_stack.py --with-sound-captions` option starts both sidecars. Install
matched PyTorch/torchaudio/torchvision builds; tested versions are 2.6.0+cu124,
2.6.0+cu124 and 0.21.0+cu124. Event model bootstrap downloads only inference
code and a 4.07-MB checkpoint. The model source revision and weight hash are
recorded in `bootstrap_sound_events.py` and the measurement receipt.

```bash
python run_unit_checks.py
python bootstrap_sound_events.py
python generate_environment_question.py
python check_environment_diversity.py
python check_sound_events.py
# Run browsers SEQUENTIALLY; the public demo currently has one ASR session slot.
python browser_environment_check.py --url https://YOUR-DEMO
python browser_environment_music.py --url https://YOUR-DEMO
python browser_environment_expanded.py --url https://YOUR-DEMO --group all
python browser_two_stage_check.py --url https://YOUR-DEMO
python browser_two_stage_memory.py --url https://YOUR-DEMO
```

`environment_fixtures.py` supplies deterministic CC0 melody, bass/drums, impacts,
beeps and low-level noise. `check_environment_diversity.py` chooses the **first
two metadata rows**, without success-based selection, for dog, cat, breaking
glass, car horn, engine and rain in ESC-50 commit
`33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`. It downloads twelve five-second files,
not the dataset. Per-file SHA256, source URL and original credit/license line
are in `environment-diversity*.json` and `environment-event-tagger.json`.
[ESC-50 licensing](https://github.com/karolpiczak/ESC-50/blob/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6/LICENSE)
is retained; downloaded audio stays in the excluded local `logs/environment-fixtures`
cache and is not redistributed as CC0. Existing synthetic speaker fixtures use
the adversarial manifest described above. The spoken sound question has one fixed
TTS take, with script, generation settings and hash in
`environment-question-fixture.json`; never silently regenerate a failed take.

### Required scenarios and assertions

| ID | Input and procedure | Required outcome / measurement |
| --- | --- | --- |
| ENV-01 | Quiet authored music at gain 0.005, then silence | No unsolicited caption/reply; record actual digital level. |
| ENV-02 | Prominent melody/bass/drums after two seconds of room floor | Music is routed, tentative reply has real TTS PCM and completes. |
| ENV-03 | Music present from microphone start, at least 16 seconds | Sustained probe detects it despite calibration; no permanent floor absorption. |
| ENV-04 | Repeat music after onset cooldown but before 180-second music cooldown | A new caption may occur; no repeated unsolicited commentary. Explicit questions remain answerable. |
| ENV-05 | Each pinned animal, traffic, weather, impact clip | Record captured window, both model outputs, expected category, latency and errors. Routing and semantic correctness are separate columns. |
| ENV-06 | Repeat detector replay with VAD forced positive | Capture is not discarded merely because music activates speech VAD. This synthetic control does not certify singing detection. |
| ENV-07 | Type “Was ist das hier?”, then begin music 200 ms later | Microphone ingress continues; requested after-question caption enters the actual answer. Test bridges OFF and ON. |
| ENV-08 | Play the authored spoken question, pause 350 ms, then music | Real Parakeet endpoint, sound context, main answer and TTS; retain recognized text, including ASR spelling errors. |
| ENV-09 | Explicit sound question followed by silence | No hallucinated sound. Bounded listening timeout and an uncertain/no-clear-sound answer. |
| ENV-10 | Disable sensing or issue Stop during capture/caption | Old results cannot start a reply; queued audio is cancelled appropriately. |
| ENV-11 | During an environmental spoken reply, inject a high RMS update plus foreground VAD activity | Browser must not send `avatar:cancel` merely for energy/VAD. Genuine explicit Stop still uses the server verifier. |
| ENV-12 | Caption/event disagreement, e.g. rain vs pouring or meow vs siren | Both uncertain sources reach the LLM; no unsupported certainty, danger or spatial direction. Human review of response wording required. |
| ENV-13 | CPU tagger absent, busy or delayed beyond 180 ms | Caption/ASR/TTS continue without its optional hint; no stale queue. |
| ENV-14 | Alternate caption-only and parallel caption+events on identical audio, six repetitions | Report individual and median/p95 wall times; do not sum overlapping stages or claim speedup from timing noise. |
| ENV-15 | Existing robot playback and delayed echo, low-level room noise | No self-triggered environmental turn. Explicit interruption remains a separate path. |

The dedicated music test uses Chrome's fake media device through the actual
capture worklet. Expanded tests inject float32 PCM through the real browser
client's 20-ms microphone sender and playback-position tags. This exercises
HTTPS, ASR, speaker assignment, GPU/CPU models, hosted/local LLMs and actual TTS,
but **does not reproduce a physical speaker–room–microphone acoustic path**.
Camera remains OFF. Browser settings must be recorded/reset deliberately.
The stronger foreground-focus tests set focus ON explicitly; the open-speaker
suite sets it OFF. Never mix these policies in a single claimed comparison.

### Addressed speech versus background: controlled matrix

Current heuristic: very quiet relative level (<0.28 of session reference), no
confident known-voice match (<0.62), and no explicit address/self-introduction
must all agree before withholding a turn. Neither RMS nor estimated SNR alone
proves intent. No anchor means no automatic background rejection. Open foreground
speakers remain welcome. Turn this filter OFF to compare with the raw open policy.

Required positives: known speaker at normal and low levels; a different foreground
speaker with no wake phrase; an explicit quiet “Robby”; self-introduction;
changed intonation; a name correction adding another prototype. Required
negatives: a quieter unrelated weather/news statement; the same signal during a
bot reply; delayed echo; music/impacts incorrectly activating VAD. Report false
accept/reject counts separately. The expanded speaker browser test verifies
Christoph → unknown foreground guest → “Ich bin Selma” → later Selma, plus a
background clip at gain 0.1. It also catches a prior bug where a name answer at
cosine 0.315 created a duplicate person instead of labeling the recent guest.

For systematic extension, cross three independent voices with relative levels
{0.1, 0.2, 0.35, 0.6, 1.0}, speech-to-background ratios {−5, 0, 5, 10, 20 dB},
clean/telephone/reverberant audio, and direct/unaddressed wording. Hold out voices
and text when selecting thresholds. Include a loud television and a quiet real
user: these adversarial cases reveal why volume alone is inadequate. Measure
missed genuine turns, unsolicited TV replies, false barge-ins, name-fragmentation,
question completion and first audible PCM. Bootstrap confidence intervals over
independent scenes/speakers, not individual overlapping frames. Do not call the
small current replay a statistically validated intent classifier.

Additional future checks: real singing/rap, human speech over music, different
microphone devices/AGC implementations, off-axis speech, diffuse room music,
long recordings already present at startup, real loudspeaker echo, and a
counterbalanced blind listening review. Spatial localization needs synchronized
multi-microphone hardware; 1.28-second event windows supply only coarse time.

### Evidence, privacy and publication

Every browser attempt writes a unique `environment-expanded-<group>-<UTC>.json`,
including setup failures and traces. Record source SHA256, active models,
settings, synthetic input hashes, accepted/ignored endpoints, captions/tags,
reply text, PCM arrival counts and cancellation events. A busy public slot is a
**blocked setup**, not an accuracy failure or a passed test. Keep its receipt.
A reply generated but cancelled before audio must be reported as not spoken.
The UI now distinguishes text awaiting audio, scheduled PCM and cancellation.

The live report is `/static/environment-report.html`; the publication copy is
[the English environment report](https://projects.laion.ai/Humaneness-Voice/robby/environment-report.html).
Raw microphone recordings, private wiki entries, voiceprints and credentials
must remain excluded from GitHub and Pages. Session-only voice identity and
opt-in persistent factual memory are distinct systems.
