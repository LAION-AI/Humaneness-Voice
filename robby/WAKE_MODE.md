# Hey Robby mode and on-demand sound questions

## Automatic startup and eye contact (October 28)

For a new browser, the page automatically starts the microphone and waits for
the wake phrase as soon as the server is ready and browser permissions and
autoplay policy allow it. Camera startup and position-based gaze following are
also on by default. A one-time migration applies these defaults to existing
browser settings; later explicit choices are remembered. Face ID and face-VAD
gating remain separately **off** by default. Camera face landmarks and gaze
run in browser CPU workers; enabling the camera does not enroll a face identity.

Browser permission prompts and autoplay restrictions cannot be bypassed. If
the microphone or playback is blocked, click **Enable microphone & sound** and
grant permission. If camera permission is denied, voice conversation continues
and the camera controls offer a retry. **Turn Robby off** stops microphone,
camera and playback, and remembers auto-start OFF; enable the controls again
when desired. Closing the page also stops its media streams.

The wake detector accepts “Hey”, “Hi”, “Hallo” and “Hello” with Robby/Robbie/
Robbi/Robi, including commas and case variations. This operates on live ASR
text, so recognition errors can still prevent activation. The same voice may
continue during the answer. The ten-second follow-up period is renewed only
after the browser reports that playback has completed (or the existing TYGER
playback-timeout fallback ends a stalled playback). It does not run out just
because Robby speaks for 20–30 seconds.

The default microphone mode requires an utterance to begin with “Hey Robby”
(also recognizes common Robby/Robbie/Robi spellings and greetings such as
“Hallo”). After this wake turn, the same voice can begin another turn within
ten seconds without repeating the wake phrase. Each admitted follow-up renews
the window. The same voice may also speak while Robby is answering; after the
answer finishes playing, a fresh ten-second window prevents the answer itself
from consuming all the user's response time. Other voices still need a wake
phrase. A different or uncertain voice needs its own wake phrase. Disable
**Hey Robby mode** next to the avatar to return to open conversation.

An acoustic ECAPA embedding from the wake utterance anchors the current voice.
Follow-ups must match the anchor/known prototypes at the configured 0.32 cosine
threshold and beat a known competing speaker by at least 0.06. The temporal
check uses the estimated speech start, so ASR/GPU queuing after the window does
not reject a sentence that began in time. If the wake audio is too short or
embedding inference fails, the wake utterance can be answered but cannot open
an unverifiable follow-up window; repeat the wake phrase with more speech.
If a longer segment fails the voice check during the active window, up to three
short tail windows are re-embedded to reduce contamination by preceding sound.
The rival margin still applies, and a recovered match never trains a new voice
prototype. This is not source separation and does not guarantee a match.
For a modest ECAPA dip (at least 0.25), a provisional match may also be
accepted if the independently estimated fast voice cue scores at least 0.85
and the ECAPA score leads known rivals by at least 0.10. It cannot update the
reference profile. These thresholds need calibration on held-out speakers.
The model cannot reliably separate all overlapping speakers, spoofed audio or
similar voices. A voice embedding is an approximate cue, not authentication.
Age/gender impressions are not used as proof of identity.

The speech-pause default is **300 ms**, with the existing slider for adjustment.
This is only the speech-pause criterion: VAD scheduling, ASR, speaker inference,
LLM, TTS and playback add latency. A 300-ms pause can split sentences while a
person is still thinking; raise the slider if that happens. A bare “Hey Robby”
waits up to 1.2 seconds for following words, to avoid responding to the wake
phrase before the actual request. Longer statements
can cross the ten-second deadline as long as they begin inside it. Playback
interruptions use the wake speaker's acoustic cue, including after expiry for
the currently playing response. Very short Stop words and overlapping speech
remain harder to identify; the manual Stop button remains available.

Automatic proactive sound reactions are now **OFF by default**, independently
switchable near the avatar. A one-time migration also switches off the previous
default in existing browsers; later user choices persist. When enabled, salient
non-speech sound may prompt an observation only inside an active wake window;
unfamiliar background voice observations do not start a conversation in wake
mode. The user can still ask what a sound was even with automatic reactions
off. A rolling microphone buffer of at most about 15 seconds stays in RAM for explicit questions,
and the existing bounded post-question listening window permits sounds after
the question. Earlier and later windows are captioned separately, so the question itself
is less likely to drown out an earlier sound. Neither path writes a recording to disk. The question can add
latency, and its sound caption is uncertain and can confuse speech with sound.
Browser or server stop/disconnect clears the session buffer.

In wake mode, unsolicited welcome and spoken idle avatar actions are suppressed
outside the active window. A user can still type, use the manual Stop button,
or disable the wake mode. Live-interruption decisions are gated by current
speaker evidence; a very short or unseen speaker may need to repeat the wake
phrase rather than being guessed. Existing focus, visual, wiki-memory,
interruption and optional reaction features remain available.

In an open-mode browser test, the returning synthetic speaker was admitted but
given a new unnamed label. Previously the LLM then guessed the previous
person's name. A direct “Who am I?” from an unnamed enrolled speaker now asks
for confirmation instead. This avoids asserting an unsupported name; it does
not make open-mode voice matching infallible.

## Reproduction

Run `python run_unit_checks.py`. The dedicated `test_wake_mode.py` covers wake
prefixes, timeout, same-speaker follow-ups, a second speaker, uncertain voice,
open mode, synthetic sentence-queue integration and explicit sound requests
with automatic reactions disabled. Existing open-conversation tests explicitly
exercise the opt-out mode. Run `browser_wake_check.py --url https://YOUR-HOST`
with Playwright, authored synthetic fixtures, live ASR, ECAPA, Qwen and TTS.
See the test protocol and `static/wake-browser-*.json` for actual results and
failures. The browser replay uses the single live ASR slot; do not run it while
a user is conversing. Synthetic digital audio is not a room/microphone test.
`generate_wake_fixture.py` creates the one-take synthetic wake fixture when
absent. `browser_sound_on_demand.py` separately tests music before and after a
spoken question with automatic reactions off. Both browser scripts accept
`--local` to select the local Nemotron LLM explicitly for that test only.
Provider rate limits and each test's actual selected model are retained in
the receipts; testing a local route does not prove Qwen availability.

`generate_auto_wake_fixture.py`, `browser_auto_wake.py` and
`browser_auto_wake_policy.py` exercise default migration, fake microphone and
camera, an authored spoken “Hi Robbie” through real ASR/LLM/TTS, playback-end
window renewal, camera/microphone denial and Chromium's autoplay recovery.
Run sequentially because the live ASR worker has a single session slot. The
browser tests use fake devices and local synthetic audio, not a real webcam,
microphone, or a perceptual eye-contact accuracy study. Exact outcomes and
runtime fingerprints are in `static/auto-wake-*.json`.
