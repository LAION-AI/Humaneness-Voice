"""Central configuration for the MOSS voice-acting demo."""
import os, glob

# ---------------------------------------------------------------- models
# Full-parameter SFT + DPO tuning of the v2 base — same architecture, tokenizer
# and prompt format, so it swaps in directly.  Note the character adapters were
# trained against the *untuned* v2 weights, so stacking one on this checkpoint is
# off-distribution; it is offered as a switch rather than a default.
# SFT round 3: the round that trained the inline delivery directions back in.
# Word error rate on direction-carrying prompts 0.447 -> 0.099, vocal-burst hit
# rate 0.516 -> 0.666, every clip within 0.5 s of the requested length.  Since
# this demo writes a direction into every single sentence, that is the whole
# ballgame.  Same architecture and prompt format, so it swaps in directly.
TTS_REPO = os.environ.get(
    "MOSS_TTS_REPO",
    "laion/moss-tts-local-transformer-4.55b-voice-acting-v2-sft3")
TTS_REPO_DPO = "laion/moss-tts-local-transformer-4.55b-voice-acting-v2-sft-dpo"
TTS_REPO_BASE = "laion/moss-tts-local-transformer-4.55b-voice-acting-v2"
CODEC_REPO = os.environ.get("MOSS_CODEC_REPO", "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2")

# GPU split: language model on one card, voice-acting model on the other.
# The app is launched with CUDA_VISIBLE_DEVICES=1,0 so cuda:0 is the voice card
# and cuda:1 is the one the language model sits on, which still has room for the
# small speech-recognition and voice-conversion models.
TTS_GPU = os.environ.get("MOSS_TTS_GPU", "1")
LLM_GPU = os.environ.get("MOSS_LLM_GPU", "0")

ASR_MODEL = os.environ.get("MOSS_ASR_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
ASR_DEVICE = os.environ.get("MOSS_ASR_DEVICE", "cuda:1")

# optional voice conversion, off unless the user ticks the box
MEANVC2_ROOT = os.environ.get("MOSS_MEANVC2_ROOT",
                              "/mnt/nvme/moss-15-v2-assets/MeanVC2")
VC_DEVICE = os.environ.get("MOSS_VC_DEVICE", "cuda:1")
VC_CONTEXT_S = float(os.environ.get("MOSS_VC_CONTEXT_S", "1.5"))
# short equal-power join between converted chunks, hides residual drift
VC_CROSSFADE_S = float(os.environ.get("MOSS_VC_CROSSFADE_S", "0.04"))

# ---------------------------------------------------------------- reference bank
def _find_refs():
    p = os.environ.get("MOSS_REF_DIR")
    if p:
        return p
    hits = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--TTS-AGI--moss-voice-profile-references"
        "/snapshots/*/")))
    return hits[-1] if hits else ""

REF_DIR = os.environ.get("MOSS_REF_DIR_OVERRIDE") or "/mnt/nvme/moss-15-v2-assets/refs2"
# decoded-to-wav cache so the codec never has to touch mp3
# on /mnt/nvme, not the root fs: five tempo variants of 832 clips is ~3.4 GB and
# / is down to single-digit gigabytes
WAV_CACHE = os.environ.get("MOSS_WAV_CACHE",
                           "/mnt/nvme/moss-15-v2-assets/ref_wav_cache")
# only these languages are pre-tokenized into RAM (the demo speaks en/de)
PRELOAD_LANGS = tuple(os.environ.get("MOSS_PRELOAD_LANGS", "en,de").split(","))

# "voice_converted" pushes every condition onto the same target speaker
# (similarity to the anchor 0.726 vs 0.507 for "original", improved in 100% of
# groups) at the cost of ~0.057 emotional cosine.  Identity across turns is what
# this demo needs, so it is the default.
REF_VARIANT = os.environ.get("MOSS_REF_VARIANT", "vc_sidon")

# ---------------------------------------------------------------- identity
# The manual's first rule for consistency across clips: keep the GENERAL voice
# description BYTE-IDENTICAL and vary only the delivery.  A model that rewrites
# its own voice every turn recasts the part every turn, which is exactly the
# drift we are avoiding.  The language model writes only the delivery half.
# This must describe the ACTUAL anchor speaker of the reference corpus
# (index.json -> _speaker: "Velvet Sage Baritone", male, late 40s-60s).  A
# description that contradicts the reference codes fights them instead of
# reinforcing them, and speaker similarity collapses.
# Fallback only: with a voice profile selected (the normal case) the identity
# sentence comes from that profile's own speaker card instead.
SPEAKER_IDENTITY = os.environ.get("MOSS_SPEAKER_IDENTITY", (
    "the voice of one man in his early fifties. a warm, unhurried baritone that "
    "sits low and forward in the chest, like aged oak and morning mist. the "
    "timbre is dark and resonant with a soft gravel at the bottom of the "
    "register, the vowels are round and slightly drawn out, and consonants land "
    "softly rather than sharply. he speaks in an easy, measured, conversational "
    "manner, commanding without ever raising his voice for effect, and deeply "
    "contemplative between phrases — he thinks before he speaks and it is "
    "audible. this is always the same speaker, the same person, the same age and "
    "the same throat, recorded on the same microphone in the same room, no cut, "
    "no new narrator, no change of casting, no matter what he is feeling."
))
# appended to every instruction so the model is told the take continues the voice
CONTINUITY = ("the same speaker continues without interruption: identical voice, "
              "identical person, same microphone and same room.")

# The baseline register, appended to every GENERAL block.  Without it the model
# performs at the volume of a stage actor by default; this asks for the register
# of someone talking to one person in a room.  It is a floor, not a ceiling —
# the director's own delivery line sits before it and an explicit "roaring" or
# "screaming" there still wins, because it is the more specific instruction.
BASE_REGISTER = os.environ.get("MOSS_BASE_REGISTER", (
    "spoken softly and naturally at close conversational volume, relaxed and "
    "unforced, the way someone actually talks to one person in a quiet room "
    "rather than performing to a room full of them."))

# Stack the corpus anchor clip in front of the delivery clip as a second
# reference, so identity and performance come from separate recordings.
USE_ANCHOR = os.environ.get("MOSS_USE_ANCHOR", "1") not in ("0", "false", "")

# ---------------------------------------------------------------- tempo
# Every reference is also kept at four other tempi (pitch preserved, via
# audiostretchy), so "say that faster" swaps in a genuinely faster take of the
# same performance instead of just asking the model to hurry.
SPEEDS = (0.5, 0.75, 1.0, 1.25, 1.5)
SPEED_WORDS = {"much_slower": 0.5, "slower": 0.75, "normal": 1.0,
               "faster": 1.25, "much_faster": 1.5}

# ---------------------------------------------------------------- adapters
ASSETS = os.environ.get("MOSS_ASSETS", "/mnt/nvme/moss-15-v2-assets/loras")


# Every adapter set this box keeps as a plain directory also exists on the hub.
# Without the mapping a fresh clone finds an empty string, loads no adapters and
# gives no reason, which is the least useful way for this to fail.
HUB_FOR_ROOT = {
    "sft3_dpo":      "laion/moss-va-sft3-dpo-lora-p2",
    "sft3_voice":    "laion/moss-va-sft3-voice-loras",
    "sft3_emotion":  "laion/moss-va-sft3-emotion-loras",
    "sft3_voicenet": "laion/moss-va-sft3-voicenet-lora-adapters",
    "sft3_quality":  "laion/moss-va-sft3-quality-lora-adapters",
    "sft3_qdpo":     "laion/moss-va-sft3-quality-dpo-lora",
    "burst":         "laion/moss-va-sft3-vocal-burst-lora-adapters",
    "burst_v2":      "laion/moss-va-sft3-vocal-burst-lora-adapters-v2",
    "burst_v2_top1": "laion/moss-va-sft3-vocal-burst-lora-adapters-v2",
    "burst_grp":     "laion/moss-va-sft3-vocal-burst-lora-adapters-v2",
    "burst_grp25":   "laion/moss-va-sft3-vocal-burst-lora-adapters-v2",
    "burst_abl":     "laion/moss-va-sft3-vocal-burst-lora-adapters-v2",
    "burst_dose":    "laion/moss-va-sft3-vocal-burst-lora-adapters-v2",
    "profile":       "laion/moss-voice-profile-loras-500",
    "character":     "TTS-AGI/moss-character-loras-refined-public",
    "speaker":       "TTS-AGI/moss-voice-lora-velvet-sage-baritone",
    "sports":        "laion/moss-sports-commentator-lora",
}


def _hub_dir(repo, sub=None):
    """The snapshot for `repo` if it is already downloaded, else "".

    Looks in HF_HOME as well as the default cache, because this box keeps its
    models on a second disk and a fresh machine will not.
    """
    roots = [os.environ.get("HF_HOME"), os.path.expanduser("~/.cache/huggingface")]
    for r in roots:
        if not r:
            continue
        hub = os.path.join(r, "hub") if not r.rstrip("/").endswith("hub") else r
        hits = sorted(glob.glob(os.path.join(
            hub, f"models--{repo.replace('/', '--')}", "snapshots", "*")))
        if hits:
            d = hits[-1]
            return os.path.join(d, sub) if sub and os.path.isdir(
                os.path.join(d, sub)) else d
    return ""


def _root(kind, local, sub=None):
    """A plain directory if this box has one, otherwise the hub copy.

    Prints one line naming the repo and the fetcher when neither exists, so a
    fresh clone says what is missing instead of quietly running without it.
    """
    if local and os.path.isdir(local):
        return local
    repo = HUB_FOR_ROOT.get(kind)
    if repo:
        d = _hub_dir(repo, sub)
        if d:
            return d
        print(f"[config] adapter set '{kind}' not found. It is "
              f"{repo} on the hub — run: python setup/fetch_all.py",
              flush=True)
    return ""


def _snap(repo, local=None):
    """Prefer a plain directory under ASSETS, fall back to the HF cache.

    The full adapter collection does not fit on the root filesystem, so the big
    sets live on /mnt/nvme instead.
    """
    if local:
        d = os.path.join(ASSETS, local)
        if os.path.isdir(d):
            return d
    hits = sorted(glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*/")))
    return hits[-1] if hits else ""

# Only what fits: the full 168 GB of published adapters does not go on this disk.
# See PLAN_LORA.md for what was left out and why.
LORA_ROOTS = {
    # Parked with the rest of the v2-era sets: trained against the untuned
    # weights, and reachable by the legacy planner even when the SFT3 emotion
    # adapter was switched off.  The SFT3 set replaces it.
    # "emotion": _snap("TTS-AGI/moss-emotion-loras-v3"),
    "character": _snap("TTS-AGI/moss-character-loras-refined-public"),
    # SFT3-native burst adapters (71), replacing the v2-era set.
    # "burst":   _snap("laion/vocal-burst-lora-adapters", "bursts"),
    "burst":     _root("burst", "/mnt/nvme/moss-15-v2-assets/loras/sft3_burst"),
    # The 57-dimension VoiceNet set is trained against the *untuned* v2 weights
    # and is off-distribution on SFT3.  Parked, not deleted — restore this line
    # and BASE_STYLE_LORAS below to bring it back.
    # "voicenet":  _snap("laion/moss-voicenet-dimension-loras", "voicenet"),
    # 16 tails of the axes that actually vary with delivery, trained against
    # SFT3 itself.  Each is the top (or bottom) 1 % of a 3.1 M-utterance corpus
    # along one axis.
    "sft3_voicenet": _root("sft3_voicenet", "/mnt/nvme/moss-15-v2-assets/loras/sft3_voicenet"),
    # genuineness / vocal-burst blend / aesthetics, one per perceptual axis
    "sft3_quality":  _root("sft3_quality", "/mnt/nvme/moss-15-v2-assets/loras/sft3_quality"),
    # Two preference-tuned adapters, both rank 16.  Each targets audio_lm_heads
    # 0-11 AND text_lm_head -- 12 of its 23 modules are weight-tied, so all
    # twelve are hooked rather than merged (see docs/ADAPTERS.md).
    "sft3_qdpo":     _root("sft3_qdpo", "/mnt/nvme/moss-15-v2-assets/loras/sft3_qdpo"),
    # The v2 burst release: 105 adapters in six arms.  Registered so every one is
    # addressable by name from the overlay and from `adapter_overrides`; which
    # set the AUTOMATIC burst resolution uses is a separate switch, below.
    "burst_v2":      _root("burst_v2", "/mnt/nvme/moss-15-v2-assets/loras/_v2raw/per_class", "per_class"),
    "burst_v2_top1": _root("burst_v2_top1", "/mnt/nvme/moss-15-v2-assets/loras/_v2raw/per_class_top1", "per_class_top1"),
    "burst_grp":     _root("burst_grp", "/mnt/nvme/moss-15-v2-assets/loras/_v2raw/groups_full", "groups_full"),
    "burst_grp25":   _root("burst_grp25", "/mnt/nvme/moss-15-v2-assets/loras/_v2raw/groups_dose25", "groups_dose25"),
    "burst_abl":     _root("burst_abl", "/mnt/nvme/moss-15-v2-assets/loras/_v2raw/ablation", "ablation"),
    "burst_dose":    _root("burst_dose", "/mnt/nvme/moss-15-v2-assets/loras/_v2raw/dose", "dose"),
    "sports":    _snap("laion/moss-sports-commentator-lora"),
    # the anchor speaker as a trained adapter — the direct route to a consistent
    # voice, as opposed to converting after the fact
    "speaker":   _snap("TTS-AGI/moss-voice-lora-velvet-sage-baritone", "speaker"),
    "profile":   os.environ.get("MOSS_PROFILE_LORAS",
                                "/mnt/nvme/moss-15-v2-assets/loras/profiles"),
    # Trained against sft3 itself, so these are the in-distribution pair for the
    # checkpoint above.  Voice adapters carry identity and no affect; emotion
    # adapters carry affect and no identity; they stack.
    # rank 64 where the other two are rank 16 — which is exactly why PEFT's
    # add_weighted_adapter refuses the combination and the three are activated
    # and scaled separately.  Merging deltas, as this bank does, has no such
    # constraint.
    "sft3_dpo":     _root("sft3_dpo", "/mnt/nvme/moss-15-v2-assets/loras/sft3_dpo"),
    "sft3_voice":   _root("sft3_voice", "/mnt/nvme/moss-15-v2-assets/loras/sft3_voice"),
    "sft3_emotion": _root("sft3_emotion", "/mnt/nvme/moss-15-v2-assets/loras/sft3_emotion"),
}

# Which adapter set the speaker comes from.  The sft3 voice adapters were trained
# against the sft3 weights; the older profile adapters were not, so they are
# off-distribution on this checkpoint.
SFT3 = "sft3" in TTS_REPO
PROFILE_LORA_KIND = os.environ.get("MOSS_PROFILE_LORA_KIND",
                                   "sft3_voice" if SFT3 else "profile")
# 1.0 is the trained value for the voice adapters and has not been swept.
SFT3_VOICE_LAM = float(os.environ.get("MOSS_SFT3_VOICE_LAM", "1.0"))
# The general-quality adapter of the recommended three-way stack, at its
# published weight.
# p2 supersedes the first DPO adapter: reward 0.4757 vs 0.4708, and the only
# preference-tuned model in this line whose word error rate (0.0977) beats the
# supervised baseline it is built on (0.0987), with the highest emotion
# percentile of any of them (0.3541).  Same rank 64, alpha 128.
SFT3_DPO_LORA = os.environ.get("MOSS_SFT3_DPO_LORA", "sft3_dpo:p2")
SFT3_DPO_LAM = float(os.environ.get("MOSS_SFT3_DPO_LAM", "1.0"))
# 1.5 is the published operating point from a 31-adapter scale sweep: emotion
# 0.408 -> 0.471, genuineness and burst blend both rise with it, median word
# error rate still 0.000 and mean at its lowest.  Intelligibility only breaks
# between 1.5 and 2.0, and as a tail of derailed clips rather than general decay.
# 1.5 is the weight the adapter card recommends, and it is the single most
# expensive item in this stack: averaged over all forty adapters it costs far
# more word error than 1.0, and individual adapters at 1.5 do not degrade gently
# but derail outright (Confusion @1.5 reached word error 1.285 with 19 invented
# words).  1.0 keeps most of the effect — genuineness 1.90 against 1.83 — at
# word error 0.018 instead of 0.083.
SFT3_EMOTION_LAM = float(os.environ.get("MOSS_SFT3_EMOTION_LAM", "1.0"))
USE_LORA = os.environ.get("MOSS_USE_LORA", "1") not in ("0", "false", "")
# emotion adapters are the ones the agent reaches for most, so they sit in RAM
PRELOAD_LORA_KINDS = tuple(
    os.environ.get("MOSS_PRELOAD_LORA", "emotion,speaker").split(","))

# Speaker adapter: merged into every turn unless the demo switches it off.
# Its own config carries alpha/r = 2.0; this is the extra dose on top.
SPEAKER_LORA = os.environ.get("MOSS_SPEAKER_LORA", "speaker:velvet-sage-baritone")
SPEAKER_LORA_LAM = float(os.environ.get("MOSS_SPEAKER_LORA_LAM", "1.0"))

# Aesthetics adapter, off by default: it is a quality/polish dimension rather
# than a per-moment acting choice, so it gets its own dial instead of being
# something the director picks.
# Retired.  It pointed at the parked 57-dimension VoiceNet set, so the slider
# bound to it could not load anything — while the real aesthetics adapter
# (sft3_quality:esthetics_high) was running at its own weight elsewhere, which
# made a UI reading "0.00" actively misleading.  The slider now drives the
# quality adapter directly; set this to an adapter name to bring the dial back.
AESTH_LORA = os.environ.get("MOSS_AESTH_LORA", "")
# Off by default on SFT3: it was trained against the untuned v2 weights, so on
# this checkpoint it is off-distribution, and at 1.1 it was the strongest
# such adapter in the stack.  The slider still turns it back on.
AESTH_LORA_LAM = float(os.environ.get("MOSS_AESTH_LORA_LAM", "0.0"))

# Merged into every turn unless the request overrides them: without these the
# delivery leans formal and read-aloud, and "looser" is what a chat should sound
# like.  A persona that wants a different register just names other codes, which
# stack on top rather than fighting these.
# Parked with the 57-dimension set: these names no longer resolve.  The base
# register is carried by the prompt instead (see BASE_REGISTER).
BASE_STYLE_LORAS = ()
_OLD_BASE_STYLE_LORAS = (("voicenet:vn_S_CONV__high", 0.25),
                    ("voicenet:vn_S_CASU__high", 0.5),
                    ("voicenet:vn_WARM__high", 0.25))

# ---------------------------------------------------------------- services
LLM_BASE = os.environ.get("MOSS_LLM_BASE", "http://127.0.0.1:8790")
LLM_MODEL = os.environ.get("MOSS_LLM_MODEL", "gemma-4-12b-it-qat")
APP_PORT = int(os.environ.get("MOSS_APP_PORT", "8792"))

# hosted alternative brain.  The key is read from a 0600 file outside the repo and
# never leaves the server — the browser only ever sends the string "luna".
LUNA_BASE = os.environ.get("MOSS_LUNA_BASE", "https://api.hyprlab.io")
LUNA_MODEL = os.environ.get("MOSS_LUNA_MODEL", "gpt-5.6-luna")
LUNA_KEY_FILE = os.environ.get("MOSS_LUNA_KEY_FILE",
                               "/home/c4r33u19/moss15v2/.hyprlab_key")


def luna_key():
    try:
        with open(LUNA_KEY_FILE) as f:
            return f.read().strip()
    except Exception:
        return os.environ.get("HYPRLAB_API_KEY", "")

# ---------------------------------------------------------------- generation
# 12.5 Hz codec frames.  chunk_frames sets how often a partial decode is emitted.
DEFAULTS = dict(
    language="English",
    audio_temperature=1.0,
    audio_top_p=0.95,
    audio_top_k=25,
    audio_repetition_penalty=1.1,
    text_temperature=1.0,
    text_top_p=1.0,
    text_top_k=50,
    max_new_tokens=0,     # derived from the duration budget, see TOKEN_HEADROOM
    chunk_frames=12,      # ~1 s of audio per streamed chunk
    stop_bias=None,       # None -> config.STOP_BIAS
    seed=0,
)

# streaming decode window (see tts_engine): 80 ms hold-back, 30 ms crossfade,
# 48 frames (~4 s) of left context keeps timbre stable and seams click-free.
# duration control.  The model card: "pass tokens ~= words * 6 (12.5 Hz codec
# frames) via build_user_message to avoid rushed pacing".  Without it the model
# chooses its own length and often stops mid-line.
TOKENS_PER_WORD = float(os.environ.get("MOSS_TOKENS_PER_WORD", "6"))
# hard generation cap as a multiple of the requested duration, so cues, pauses
# and bursts have room without letting a runaway take go forever
TOKEN_HEADROOM = float(os.environ.get("MOSS_TOKEN_HEADROOM", "2.5"))
# below this many spoken words a reply is generated in one take: truncation was
# only ever observed on longer, multi-sentence replies, and splitting costs a
# full re-prefill per part
# Never split a reply into separate generations by default: separate takes are
# sampled independently and the voice and emotion visibly jump at the seam.
# Truncation is handled inside the single take instead (MIN_FRAME_FRACTION).
SPLIT_MIN_WORDS = int(os.environ.get("MOSS_SPLIT_MIN_WORDS", "100000"))
# refuse the end token below this fraction of the requested duration
MIN_FRAME_FRACTION = float(os.environ.get("MOSS_MIN_FRAME_FRACTION", "0.55"))
# Logit bias on the token that ends a take, in nats, subtracted before sampling.
# The floor above is all-or-nothing — below it the end token cannot win at all,
# above it nothing is changed — so a take that runs past the floor and then stops
# a few words early is untouched by it.  A gentle constant bias leans against
# stopping everywhere instead, without ever forbidding it.
# Measured on the current prompt shape, three seeds of ten utterances each:
# 1.0, 2.0 and 3.0 give bit-identical output (word error 0.030, no invented
# words in any take), 0.0 is marginally worse and 4.0 brings invented words back
# in 7% of takes.  2.0 is the middle of that plateau.  The dial matters far less
# than the duration budget does — see docs/EXPERIMENTS.md, experiments 7 and 8.
STOP_BIAS = float(os.environ.get("MOSS_STOP_BIAS", "2.0"))

FRAME_RATE = 12.5          # RVQ frames per second
HOLDBACK_S = 0.08
CROSSFADE_S = 0.06         # wider seam: 30 ms left an audible edge on long takes
# Left context for each partial decode.  At 48 frames (~3.8 s) the window starts
# sliding partway through a normal reply, and from that point the decoded audio
# diverges from a full decode — measured max deviation 0.91 late in a 6.8 s take.
# 160 frames covers ~13 s, so most replies never slide at all.
CTX_FRAMES = int(os.environ.get("MOSS_CTX_FRAMES", "160"))

# how much conversation the director keeps in view (messages, not turns)
HISTORY_TURNS_LUNA = int(os.environ.get("MOSS_HISTORY_LUNA", "40"))
HISTORY_TURNS_LOCAL = int(os.environ.get("MOSS_HISTORY_LOCAL", "8"))

# Sidon speech restoration, used once to turn the quiet 16 kHz corpus anchor into
# a full-bandwidth conversion target
SIDON_SRC = os.environ.get("MOSS_SIDON_SRC", "/mnt/nvme/moss-15-v2-assets/sidon/src")
SIDON_CKPTS = os.environ.get("MOSS_SIDON_CKPTS",
                             "/mnt/nvme/moss-15-v2-assets/sidon-ckpts")
SIDON_OUT_SR = int(os.environ.get("MOSS_SIDON_OUT_SR", "48000"))
SIDON_GPU = os.environ.get("MOSS_SIDON_GPU", "0")

# ---------------------------------------------------------------- scoring
# points at the copy that already existed on this box; a second one was
# downloaded by mistake and removed again (16 GB, byte-identical)
EIV_DIR = os.environ.get("MOSS_EIV_DIR", "/mnt/nvme/empathic-insights-voice-small")
WHISPER_DIR = os.environ.get("MOSS_WHISPER_DIR",
                             "/mnt/nvme/moss-15-v2-assets/bude-whisper")
VN_DIR = os.environ.get("MOSS_VN_DIR", "/mnt/nvme/moss-15-v2-assets/voicenet-pred")
SCORE_DEVICE = os.environ.get("MOSS_SCORE_DEVICE", "cuda:1")
# the EIV suite mixes 40 emotions with attributes (Age, Arousal, ...); only the
# emotions are ranked as "what it heard"
EMOTION_NAMES = {
    "Affection", "Amusement", "Anger", "Astonishment_Surprise", "Awe",
    "Bitterness", "Concentration", "Confusion", "Contemplation", "Contempt",
    "Contentment", "Disappointment", "Disgust", "Distress", "Doubt", "Elation",
    "Embarrassment", "Emotional_Numbness", "Fatigue_Exhaustion", "Fear",
    "Helplessness", "Hope_Enthusiasm_Optimism", "Impatience_and_Irritability",
    "Infatuation", "Interest", "Intoxication_Altered_States_of_Consciousness",
    "Jealousy_and_Envy", "Longing", "Malevolence_Malice", "Pain",
    "Pleasure_Ecstasy", "Pride", "Relief", "Sadness", "Sexual_Lust", "Shame",
    "Sourness", "Teasing", "Thankfulness_Gratitude", "Triumph",
}
# a few attributes are worth showing next to the emotions
SCORE_ATTRS = ("Arousal", "Valence", "Authenticity", "Confident_vs._Hesitant")
VN_BASELINE = os.environ.get("MOSS_VN_BASELINE",
                             "/mnt/nvme/moss-15-v2-assets/vn_baseline.json")

# hosted brains, all through the same OpenAI-compatible endpoint
HOSTED_MODELS = {
    "luna": "gpt-5.6-luna",
    "gemini-flash": "gemini-3-flash",
    "gemini-flash-lite": "gemini-3.5-flash-lite",
    "glm": "glm-5.3",
}
# Models whose endpoint takes `max_tokens` rather than `max_completion_tokens`.
# glm-5.3 documents exactly five parameters and `max_completion_tokens` is not
# among them; sending it there is a silently ignored cap, which is how a reply
# comes back truncated mid-JSON with nothing in the log.
HOSTED_PLAIN_MAX_TOKENS = {"glm-5.3"}
# no deliberation needed for a character decision, and it halves the latency
HOSTED_REASONING = os.environ.get("MOSS_HOSTED_REASONING", "none")
HOSTED_MAX_TOKENS = int(os.environ.get("MOSS_HOSTED_MAX_TOKENS", "3000"))
# gemini-3.5-flash-lite with reasoning off: measured 1.5 s against luna's 2.4 s
# and gemini-3-flash's 3.2 s, and this turn needs a character decision, not
# deliberation
DEFAULT_BRAIN = os.environ.get("MOSS_DEFAULT_BRAIN", "luna")

# ---------------------------------------------------------------- continuity
# Carry a short tail of the previous replies into the next generation, so the
# voice continues rather than restarts.  Tails only: the manual measured whole-
# clip chaining collapsing speaker similarity from 0.777 to 0.280 by clip four.
USE_TAIL_CONTEXT = os.environ.get("MOSS_TAIL_CONTEXT", "1") not in ("0", "false", "")
TAIL_FRAMES = int(os.environ.get("MOSS_TAIL_FRAMES", "50"))   # 50 / 12.5 Hz = 4 s
TAIL_TURNS = int(os.environ.get("MOSS_TAIL_TURNS", "2"))
MAX_SESSIONS = int(os.environ.get("MOSS_MAX_SESSIONS", "200"))

# ---------------------------------------------------------------- profiles
# Ten speaker profiles: one adapter and one anchor recording each.  The adapter
# collection is small (0.38 GB for all ten at the shipped rank) because only the
# top level of each voice folder is used — the ranks/ tree is an audit trail,
# not a menu.
PROFILE_LORAS = os.environ.get("MOSS_PROFILE_LORAS",
                               "/mnt/nvme/moss-15-v2-assets/loras/profiles")
PROFILE_REFS = os.environ.get("MOSS_PROFILE_REFS",
                              "/mnt/nvme/moss-15-v2-assets/refs2")
DEFAULT_PROFILE = os.environ.get("MOSS_DEFAULT_PROFILE", "emolia_c1699")
# The profile adapter is the speaker; at 0.25 the identity barely came through.
# It also replaces the standalone velvet-sage dial rather than stacking with it.
# Identity.  Swept with the script held fixed and conditioned on the profile's
# OWN reference recording, three scripts, two seeds:
#
#     w        0.00   0.25   0.50   0.75   1.00   1.25
#     spk_sim  0.512  0.513  0.464  0.545  0.556  0.549
#     wer      0.197  0.015  0.027  0.012  0.012  0.062
#
# 1.0 is best on both axes at once, and 1.25 gives up intelligibility for
# nothing.  This dial was 0.25 for a long time on the strength of an earlier
# sweep that conditioned on the CORPUS anchor -- a different speaker -- so the
# adapter was measured while fighting the reference clip and looked useless.
# Against the right anchor it is worth 0.044 of similarity over no adapter, and
# it costs no word error until 1.25.
PROFILE_LORA_LAM = float(os.environ.get("MOSS_PROFILE_LORA_LAM", "1.0"))

# Host-RAM cache bound for adapters.  65 GB exist on disk; without a cap a long
# session loads its way through them until the kernel intervenes.
MAX_CPU_ADAPTERS = int(os.environ.get("MOSS_MAX_CPU_ADAPTERS", "64"))

# Pure mode: base model + retrieved reference clips + the voice's own character
# adapter, nothing else.  The 500-voice release measured its adapters at scale
# 1.0 only, but with no expressive adapters underneath, a lighter dose keeps more
# of the base model's own acting; 0.5 is the starting point.
PURE_PROFILE_LAM = float(os.environ.get("MOSS_PURE_PROFILE_LAM", "0.5"))

# ---------------------------------------------------------------- retrieval
# The director's prose, matched against the corpus rather than decoded into
# codes.  See retrieval.py for the measurements that set the shape of this.
RETRIEVAL_DIR = os.environ.get("MOSS_RETRIEVAL_DIR",
                               "/mnt/nvme/moss-15-v2-assets/retrieval")
VOICECLAP_REPO = os.environ.get("MOSS_VOICECLAP", "laion/voiceclap-commercial")
REF3_DIR = os.environ.get("MOSS_REF3_DIR", "/mnt/nvme/moss-15-v2-assets/refs3")
# on by default: retrieval picks the reference clip and the emotion adapter.
# Turned off, the turn runs on the base checkpoint and the retrieved clip alone.
RETRIEVAL_ON = os.environ.get("MOSS_RETRIEVAL", "1") not in ("0", "false", "")
EMOTION_NUANCE_ON = os.environ.get("MOSS_EMOTION_NUANCE", "1") not in ("0", "false", "")
# The two axes are not equally good, so they are not equally weighted.  Matching
# the cues against the emotion text anchors scores 0.61 top-1; matching the full
# direction against the audio centroids scores 0.28.  A bonus for conditions of
# the winning emotion lets the clip choice inherit the stronger axis while audio
# similarity still picks the level and the take within it.  Swept on the same
# held-out set: 0.0 -> 0.17, 0.15 -> 0.39, 0.3 -> 0.61, and flat above that.
# 0.3 was the knee while the raw GENERAL went into the query.  With the static
# identity clauses filtered out the audio side is cleaner but weaker on its own,
# and the knee moves to 0.5: 0.1 -> 0.33, 0.3 -> 0.50, 0.5 -> 0.61, flat above.
RETRIEVAL_EMO_BONUS = float(os.environ.get("MOSS_RETRIEVAL_EMO_BONUS", "0.5"))
RETRIEVAL_LEVEL_PENALTY = float(os.environ.get("MOSS_RETRIEVAL_LEVEL_PEN", "0.05"))

# Tokenised reference codes, kept on disk so a restart does not re-encode the
# corpus and an un-preloaded clip costs a file read rather than a GPU pass.
CODE_CACHE = os.environ.get("MOSS_CODE_CACHE",
                            "/mnt/nvme/moss-15-v2-assets/code_cache")

# Speech rate for the timed script format, in frames per word.  TOKENS_PER_WORD
# (6) is the model card's budget for an *untimed* prompt, where overshooting
# only costs headroom.  Here the number is an instruction the checkpoint obeys,
# and 6 frames a word drags: the format's own worked example runs 13 words in
# 4.7 s, which is 4.5 frames a word.
# 4.5 is the rate of the format's own worked example, and it leaves this voice
# slack: the model hits the requested duration to within 0.02 s whatever else
# changes, so any budget it does not need to speak the line, it fills.  4.0 with
# no closing pause invented words in 0-10% of takes across three seeds, against
# 20-40% at 4.5 with one.
TIMED_FRAMES_PER_WORD = float(os.environ.get("MOSS_TIMED_FPW", "4.0"))
# The SFT3 timed script: durations, pauses and burst lengths that add up to the
# Tokens budget, with the script repeated byte-identically into the Text field.
TIMED_SCRIPT = os.environ.get("MOSS_TIMED_SCRIPT", "1") not in ("0", "false", "")

# ---------------------------------------------------------------- delivery tails
# The 16 SFT3 VoiceNet adapters, with the gloss from the model card: not a
# definition of the axis, but what the clips it was trained on are described as.
# Shown to the director so it can pick on sound rather than on a code it half
# remembers.
SFT3_VN_ADAPTERS = {
    "AROU_high":   "highly aroused, very dominant, tense, elated, thin",
    "AROU_low":    "dialled down, not performed at full size; narrow pitch range, submissive, slow",
    "ARSH_high":   "energised, slightly dominant, brisk, wide pitch range",
    "ARSH_low":    "involuntary rather than performed; weeping, tearful, searing",
    "EMPH_high":   "tense, highly aroused, very dominant, guarded, elated",
    "EXPL_high":   "mildly negative, normal-paced, conversational, quiet background",
    "S_ASMR_high": "dialled down, pouring out, uncertain, hedging, full of doubt",
    "S_DRAM_high": "dramatic; highly aroused, very dominant, very wide pitch range",
    "S_RANT_high": "ranting, very rough, guarded, very dominant, highly aroused",
    "TENS_high":   "tense, guarded, volatile, highly aroused, very dark",
    "VALN_low":    "involuntary rather than performed; blood-curdling, horrified, at the edge of a scream",
    "VALN_high":   "elated, positive, wide pitch range — the top of the valence axis",
    "VALS_high":   "positive affect, light breath, wide pitch range",
    "VALS_low":    "involuntary rather than performed; blood-curdling, weeping, tearful",
    "VFLX_high":   "slightly bright, fairly smooth, energised, positive, wide pitch range",
    "VOLT_high":   "volatile, heavy breath, slurred, cool timbre, dark",
    "VULN_high":   "dialled down, involuntary rather than performed, in every breath, impossible to hide",
}
# The director picks one of these merge weights.  The set HAS now been evaluated:
# a 5,740-cell dose-response sweep of 79 adapters at six weights, scored against
# each axis's own VoiceNet regression rather than against side effects alone.
#
#   * 16 of these 17 adapters have a usable weight -- the best-behaved family in
#     the whole stack, 12 monotone and 4 saturating.
#   * The median safe AND strong weight is 1.5, not 0.75.
#   * Going 0.75 -> 1.5 buys +0.375 on the target axis (t 5.18, better on 15 of
#     17 adapters) for a word-error change of +0.003 -- t 0.55, i.e. none.
#     Individual gains reach +1.37 (S_RANT_high) against a noise floor of ~0.15.
#
# The 0.75 ceiling was set when two delivery axes could run at once; SFT3_VN_MAX
# is 1, so that reason no longer applies.  Stacking is still untested, so if
# SFT3_VN_MAX is ever raised above 1 this ladder should come back down with it.
# Source: research-log-2026-08/lora-dose/ in LAION-AI/Voice-Acting-Pipeline-WIP.
SFT3_VN_LEVELS = (0.5, 0.75, 1.0, 1.25, 1.5)
# One delivery axis costs little; two took word error from 0.041 to 0.143 and put
# invented words in half the takes.  The director gets one.
SFT3_VN_MAX = int(os.environ.get("MOSS_SFT3_VN_MAX", "1"))

# ---------------------------------------------------------------- quality axes
# Three adapters, each the top 1 % of a 3.14 M-utterance corpus along one
# perceptual axis.  On by default at the trained value; the sliders in the UI
# override them per turn.  Unevaluated, like the rest of this family — 1.0 is
# where they were trained, not where they were shown to be best.
# Doses measured, not assumed.  All three at 1.0 scored word error 0.116 with an
# invented word in 60% of takes; at 0.25 / 0.5 / 0.5 the same three score 0.055.
# Genuineness is the one that has to stay low — it raises its own score only
# below 0.5 and collapses intelligibility above 1.0 (0.176 at 1.25).  Blend is
# safe at any weight measured.  See docs/EXPERIMENTS.md.
# ---------------------------------------------------------------- interference
# Pairs of adapters measured to work against each other.  These are not style
# preferences; they are measurements, and composing them produces neither effect.
#
#   ESTH x S_RANT: pushing the aesthetic axis alone moves it +0.196..+0.317;
#   pushing ranting alone moves S_RANT +0.464 (t 7.01, on 12 of 12 prompts).
#   Both at the same strength: -0.012, indistinguishable from zero.  The two
#   directions are close to opposed in the model's representation.
#
# When a delivery axis in the key of this table is active, the paired quality
# adapter is scaled by the factor given rather than silently fighting it.
# Source: research-log-2026-08/layer-forensics/w3/ (arm G) and combination-study/.
QUALITY_CONFLICTS = {
    "sft3_quality:esthetics_high": {
        "S_RANT_high": 0.0,     # measured to cancel outright
        "S_DRAM_high": 0.5,     # same family, untested pairwise -- halve, do not drop
    },
}

QUALITY_LORAS = {
    "sft3_quality:genuineness_high": float(os.environ.get("MOSS_LAM_GENUINE", "0.25")),
    "sft3_quality:blend_high":       float(os.environ.get("MOSS_LAM_BLEND", "0.5")),
    "sft3_quality:esthetics_high":   float(os.environ.get("MOSS_LAM_ESTH", "0.5")),
}
QUALITY_LABELS = {
    "sft3_quality:genuineness_high": "Genuineness",
    "sft3_quality:blend_high":       "Burst blend",
    "sft3_quality:esthetics_high":   "Aesthetics",
}
# Burst adapter dose.  THESE TWO ARE THE FALLBACK, NOT THE USUAL CASE: with
# skills on (the default) every class that has a measured weight in
# `wikiskills/VOCAL_BURSTS.md` gets that weight instead, per class, applied in
# app.py.  These flat numbers are what a class with no measured recipe gets, and
# what every class gets when MOSS_SKILLS=0.
#
# The old comment here said 0.25 was chosen because "higher starts to drag the
# whole line towards the burst".  That reasoning predates the measurement and is
# retired: 0.25 was in fact the ceiling the *genuineness* gate imposed, and that
# gate has since been dropped on purpose — a scream is not supposed to sound like
# a composed, natural address, so falling genuineness is the expected price of a
# burst and not grounds for exclusion.  The gate that remains is word error:
# paired Parakeet WER no more than +0.104 against the class's own w = 0 cell,
# and for inline scripts no more than 0.25 absolute.
#
# Measured per-class optima now run 0.25 to 2.3 (VOCAL_BURSTS.md §51/52 + §64),
# i.e. the shipped flat dose sat far under the optimum for most classes:
# `chuckle` alone goes 0.25 -> 2.0.
BURST_LAM = float(os.environ.get("MOSS_BURST_LAM", "0.25"))
BURST_LAM_INTENSE = float(os.environ.get("MOSS_BURST_LAM_INTENSE", "0.5"))
# Ceiling applied to a per-class weight after it is looked up.  2.3 is the
# largest weight any recipe names, so the default changes nothing.
#
# It is a knob rather than a fixed clamp because the evidence is not settled.
# The 2026-09-05 addendum (study `vb_grp`, 28 classes, group-level detector)
# measured all four adapter arms breaking the WER gate at w = 2.0 — retrained
# +0.167, borrowed +0.168, group-full +0.123, group-25% +0.109 against the
# +0.104 bound — and states that no recipe should name w = 2.0.  The §51/52
# table it sits above has not been rewritten to match and still carries ten
# recipes at 1.8-2.3, which is what skills.py parses and serves.  Those cells
# have their own passing absolute WER (sharp_inhale at 2.3 measured 0.091), so
# the two are not flatly contradictory; they are a different harness measuring a
# different delta.  Set MOSS_BURST_LAM_MAX=1.5 to enforce the addendum's ceiling.
# The recipe weights run to 2.3 and they do not survive this stack.  Swept with
# the SCRIPT HELD FIXED -- three burst-carrying scripts, two seeds, only the
# weight varying, because comparing across /api/turn calls measures the director
# writing a different line each time and not the adapter:
#
#     w   0.00  0.25  0.50  1.00  1.50 | 2.00  2.30
#     wer 0.285 0.319 0.304 0.322 0.341| 0.859 1.337
#     inv 0.67  0.67  0.67  0.67  0.67 | 1.83  3.17
#
# It is a cliff, not a slope: flat to 1.5, then intelligibility collapses.  1.5
# is the last safe rung, so the ceiling goes there.  Nine recipes ask for 2.0 or
# 2.3 -- chuckle, sharp_inhale, soft_hum among them -- and every one of those is
# on the far side of it here.
#
# Not a contradiction of the study: its ladder measured ONE adapter, for several
# classes with no production stack under it, while a turn here merges it on top
# of a voice adapter, three quality adapters, a preference adapter, an emotion
# adapter and often a delivery axis.  The study saw the same edge from its own
# side -- nine of 400 ladder cells produced no decodable audio, all at w >= 2.3.
# Raise it with MOSS_BURST_LAM_MAX or the slider to hear the recipe weights.
# Measured 6 September 2026 on a fixed scream script under the full shipped
# stack, six seeds per cell, word error against the intended line:
#
#   sum 1.0 (2 x 0.5)   median 0.000   0/6 broken
#   sum 1.5 (3 x 0.5)   median 0.000   0/6
#   sum 2.0 (2 x 1.0)   median 0.010   0/6
#   sum 2.0 (3 x 0.67)  median 0.000   0/6
#   sum 2.5 (2 x 1.25)  median 0.110   0/6, mean 0.103 -- degrading
#   sum 3.0 (2 x 1.5)   median 0.820   5/5 broken, unusable
#
# So it is the SUM that breaks a line, not the single weight: one adapter at
# 1.5 was clean in 5 of 5, two at 1.5 failed in 5 of 5.  That is the opposite
# of what the single-adapter ladder concluded on a bare model, and the
# difference is the rest of the stack -- a delivery axis at 1.5 and a
# preference adapter at 1.5 are already merged before a burst arrives.
BURST_LAM_MAX = float(os.environ.get("MOSS_BURST_LAM_MAX", "1.25"))

# The SOLO ceiling: what a single burst adapter may reach when it is the only
# one in the stack.  Two separate measurements agree that one adapter at 1.5 is
# safe and two are not -- the merge planner's own note records two adapters at
# 1.5 destroying the line in 5 of 5 seeds (median word error 0.82) while one at
# 1.5 came back clean every time -- so the ceiling that protects a two-burst
# line is stricter than a one-burst line needs.
# Study `vb_opt` measured raising the ceiling to 1.5 across the board: +0.0275
# on the candidate MEAN (t 2.31) and +0.0050 on the DELIVERED take (t 0.15).
# The extra burst was generated and then not selected.  That is exactly the
# condition this becomes worth doing under, and it is now met: with
# BON_BURST_WEIGHT the ranker can see the burst it is choosing.
# So this applies only when BOTH hold -- exactly one burst adapter, and a
# burst-aware ranker -- and falls back to BURST_LAM_MAX otherwise.
BURST_LAM_MAX_SOLO = float(os.environ.get("MOSS_BURST_LAM_MAX_SOLO", "1.5"))

# ---------------------------------------------------------------- generation modes
# THE THREE LEVERS.  Until now this server had exactly one way to shape a performance:
# load adapters and write a good prompt.  Two more have since been measured on this
# checkpoint, together with how all three combine, and they are offered here as switchable
# modes.  docs/LEVERS.md is the write-up; the short version:
#
#   adapter        today's behaviour.  Merge weights.  The only lever that moves the quality
#                  axes at all (+0.399, t 6.0) and the cheapest everywhere.
#   adapter+steer  plus a difference-of-means direction added to the hidden state.  On the
#                  emotion heads the two are cleanly additive (interaction +0.038, t 1.36),
#                  and steering is five times the adapter's effect there (+0.384 t 9.4
#                  against +0.077 t 2.8).
#   adapter+cfg    plus classifier-free guidance on the delivery condition.  Costs 1.93x.
#   steer / cfg    one lever without the attribute's own adapter.  For the delivery axes,
#                  where adapter and steering are significantly SUB-additive (-0.164,
#                  t -3.7) and the right move is to pick one rather than stack them.
#
# Default is `auto`, which resolves per family from the measurements: emotion ->
# adapter+steer, delivery -> adapter, quality -> adapter.  `auto` never spends guidance.
# Source: research-log-2026-08/combination-study/ in LAION-AI/Voice-Acting-Pipeline-WIP.
# Rolled back to `adapter` after a listening report: with `auto` the demo picked
# steering on nearly every turn (alpha +0.10 on the target attribute and -0.10 on
# Emotional_Numbness), and a human heard it as artefacts and an off timbre even
# though the scoring models liked it.  That is the documented blind spot of this
# evidence — the steering study states plainly that no listening test was run on
# any of its results, and every figure in it is one model judging another model's
# output.  `auto` and the rest stay one environment variable away:
#     MOSS_GEN_MODE=auto|adapter+steer|adapter+cfg|steer|cfg
GEN_MODE = os.environ.get("MOSS_GEN_MODE", "adapter")
# Both levers are individually killable, and with either off the modes that need it degrade
# to `adapter` and say so in the response payload rather than reporting a mode that is not
# running (docs/LEARNINGS.md: a dial that reads a value while the thing it names is off is
# worse than no dial).
STEER_ENABLED = os.environ.get("MOSS_STEER", "1") not in ("0", "false", "")
CFG_ENABLED = os.environ.get("MOSS_CFG", "1") not in ("0", "false", "")
# Whether the director may choose the mode at all.  Off, every turn uses GEN_MODE.
AGENT_PICKS_MODE = os.environ.get("MOSS_AGENT_PICKS_MODE", "1") not in ("0", "false", "")

# ---- steering ----------------------------------------------------------------
# The vectors.  ~5 MB distilled from the 112 MB research file by
# setup/build_steering_pack.py; NOT committed to this repository.  With no file here every
# steering mode degrades to `adapter`.  See docs/LEVERS.md.
STEER_PACK = os.environ.get("MOSS_STEER_PACK",
                            "/mnt/nvme/moss-15-v2-assets/steering/p3_vectors_server.npz")
# Only needed when STEER_PACK points at the raw research .npz instead of the distilled pack.
STEER_TAP_RANK = os.environ.get("MOSS_STEER_TAP_RANK",
                                "/mnt/nvme/moss-15-v2-assets/steering/tap_rank.json")
# How many of each dimension's own top layers the distilled pack carries.  5 covers every
# shipped k; the delivery axes are the only family that wants more than 1.
STEER_PACK_K = int(os.environ.get("MOSS_STEER_PACK_K", "5"))
# alpha is DIMENSIONLESS: the fraction of the hidden state's own magnitude added along the
# direction.  0.10 at the attribute's own top layer is the free setting -- emotion percentile
# 0.4354 -> 0.5840 with word error FALLING.  Everything breaks above 0.3, and at 0.5 a random
# direction of matched norm does the same damage, which is how an earlier round concluded
# wrongly that steering never works.
STEER_ALPHA = float(os.environ.get("MOSS_STEER_ALPHA", "0.10"))
# Hard ceiling on any SINGLE component.  0.15 is half the measured break point, and no
# shipped recipe exceeds it.
STEER_ALPHA_CEILING = float(os.environ.get("MOSS_STEER_ALPHA_CEILING", "0.15"))
# Hard ceiling on the REALISED magnitude at any one layer, which is a different number,
# because two components that share a layer sum there and the sum is not the larger alpha.
# It is not even the quadrature sum: cos(emotion direction, quality axis) runs -0.62 to
# -0.95 in this representation, so SUBTRACTING Emotional_Numbness adds almost entirely
# ALONG the emotion direction rather than orthogonally to it.  Measured over the 40 emotion
# recipes with the numbness subtraction attached, the realised magnitude at the shared layer
# runs to 0.1926 (emo:Interest at h20; Elation 0.1907, Amusement 0.1781) -- and those are the
# compositions the study actually ran and scored, so a ceiling at 0.15 would refuse a
# measured recipe.  0.25 sits above every one of them and well below the 0.30 at which
# steering collapses.  Above this the composition is REFUSED rather than trimmed: a trimmed
# composition is not the one that was measured.
STEER_REALISED_CEILING = float(os.environ.get("MOSS_STEER_REALISED_CEILING", "0.25"))
# k is per attribute and small.  Emotion is free only at k = 1; the quality axes break at
# k >= 2 (genuineness -2.81 at k = 5); the delivery axes want 3-5.  A recipe from the wiki
# overrides this; it is the fallback when the wiki's own point does not use steering.
STEER_K = {"emo": 1, "vn": 3, "qual": 0}
# The director's three strength words scale the recipe's alpha.  A language model that is
# allowed to pick a dose picks it wrong, and these are measured -- so the words map to a
# fixed table here and never reach the model as numbers.
STRENGTH_ALPHA_SCALE = {"gentle": 0.5, "moderate": 1.0, "strong": 1.0}
# Subtracting Emotional_Numbness at -0.10 returns +0.60 of genuineness (t 9.64, on 67 of 80
# prompts) at NO cost in emotion when the adapter is carrying the emotion.  It rides along
# automatically on any emotion turn where the steering machinery is already running.
#   with_steer  attach it whenever an emotion is steered            (default)
#   off         never
# It is deliberately not "always": attaching it to a bare `adapter` turn would mean `adapter`
# is no longer bit-for-bit today's behaviour, and that mode is the fallback for everything.
NUMBNESS_SUBTRACTION = os.environ.get("MOSS_NUMBNESS", "with_steer")

# ---- guidance ----------------------------------------------------------------
# logits = logits_uncond + g * (logits_cond - logits_uncond).  g = 1 cancels the
# unconditional term exactly and IS ordinary sampling, so it is the control and the "off"
# value.  Below 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56).
CFG_G = {"emo": 3.0, "vn": 2.5, "qual": 2.5}
CFG_G_MIN = float(os.environ.get("MOSS_CFG_G_MIN", "1.5"))
CFG_G_MAX = float(os.environ.get("MOSS_CFG_G_MAX", "3.0"))
# Measured at batch 1: 1.89-1.94x over four cells, sd 0.053.  The intuition that only the
# semantic transformer doubles is wrong -- the local transformer doubles too, running twelve
# times per frame per branch, and the codec decode, the only genuinely shared component, is
# 1.6 % of the total.  ADAPTERS.md §1 records realtime factor 1.0 as the streaming budget and
# the live merged baseline as 0.764, so 1.93x lands at ~1.47 and the stream would underrun.
# CFG therefore renders the take whole and then plays it; see docs/LEVERS.md.
CFG_COST_FACTOR = float(os.environ.get("MOSS_CFG_COST", "1.93"))
# When steering and guidance are both on, steer BOTH branches rather than only the
# conditioned one: it keeps 82 % of the effect and returns 0.209 of word error and 0.75 of
# genuineness.  The combined interaction is the only real synergy in the study and every
# damage term it carries has a larger |t| than the gain, so the cheaper wiring is the default.
CFG_STEER_BRANCH = os.environ.get("MOSS_CFG_STEER_BRANCH", "both")

# ---- what the levers push ----------------------------------------------------
# The three perceptual axes, by the name the director already uses for them.
QUALITY_AXES = ("genuineness_high", "blend_high", "esthetics_high")
# On a delivery axis the adapter and the steering vector do the same job and are
# significantly sub-additive, so exactly one of them runs.  "adapter" keeps the incumbent --
# the delivery adapters are the best-behaved family in the stack and were evaluated at scale
# on this hardware, which the steering vectors have not been.  "steer" picks the other.
DELIVERY_LEVER = os.environ.get("MOSS_DELIVERY_LEVER", "adapter")

# The generated coefficient table: one entry per attribute, with the balanced and high-effect
# operating points and the measured cost of each on all five guardrails.  Produced by
# wikiskills/code/build_wikiskills.py in the research log.  Absent, steering and guidance are
# refused for want of a measured setting -- the server never invents one.
WIKI_COEFFICIENTS = os.environ.get(
    "MOSS_WIKI_COEFFICIENTS",
    "/mnt/nvme/moss-15-v2-assets/wikiskills/coefficients.json")

# ---------------------------------------------------------------- end-trimming
# Forced alignment against the script, used to fade in at the first word and to
# cut after the last one.  This is the only fix that can work: the model spends
# the duration it is given rather than overrunning it, so the filler sits INSIDE
# the requested length and no clock-based cut can find it (EXPERIMENTS.md §8).
#
# LICENCE: the aligner is CC BY-NC 4.0, unlike the rest of this stack.  It is
# fetched at runtime, not redistributed, but a commercial deployment has to turn
# this off or swap the model.
ALIGN_DIR = os.environ.get("MOSS_ALIGN_DIR", "/mnt/nvme/moss-15-v2-assets/aligner")
ALIGN_DEVICE = os.environ.get("MOSS_ALIGN_DEVICE", "cuda")
ALIGN_ON = os.environ.get("MOSS_ALIGN", "1") not in ("0", "false", "")
# Below this alignment confidence the words were probably not found where the
# aligner thinks they were, and nothing is cut.  Refusing to edit is always the
# safe failure here: a wrong cut removes speech the director asked for.
ALIGN_MIN_SCORE = float(os.environ.get("MOSS_ALIGN_MIN_SCORE", "0.35"))
# Lead-in: only bother when there is more than this much before the first word.
ALIGN_LEAD_MIN_S = float(os.environ.get("MOSS_ALIGN_LEAD_MIN", "0.12"))
ALIGN_LEAD_RAMP_S = float(os.environ.get("MOSS_ALIGN_LEAD_RAMP", "0.10"))
# Tail: keep a little air after the last word, then ramp down over the rest.
ALIGN_TAIL_PAD_S = float(os.environ.get("MOSS_ALIGN_TAIL_PAD", "0.12"))
ALIGN_TAIL_RAMP_S = float(os.environ.get("MOSS_ALIGN_TAIL_RAMP", "0.15"))
# Do not cut unless there is at least this much audio after the last word, or
# every take loses its natural decay for nothing.
ALIGN_TAIL_MIN_S = float(os.environ.get("MOSS_ALIGN_TAIL_MIN", "0.25"))
# Streaming: how far behind generation the player runs, so there is room to cut
# before the filler is audible, and how much new audio between alignment passes.
ALIGN_LOOKAHEAD_S = float(os.environ.get("MOSS_ALIGN_LOOKAHEAD", "0.5"))
ALIGN_EVERY_S = float(os.environ.get("MOSS_ALIGN_EVERY", "0.5"))

# ------------------------------------------------------- preference adapters
# Both repositories publish their recommended checkpoint at the root, not their
# final one, and both argue against the final in their own cards:
#
#   quality_dpo      root == step376.  The preference task is solved there
#                    (accuracy 1.0000); every later step only inflates the
#                    reward margin -- about 1 nat per token of drift away from
#                    the reference model for no measured gain.  step1504 is
#                    fetched alongside as `quality_dpo_step1504` so the two can
#                    be compared by ear, which nobody has done.
#   burst_stop_dpo   root == step896, and the card says "use 896, not final":
#                    step902 is a 6-step tail and the only late reversal.
#
# burst_stop is the one whose evidence points at this server's open complaints:
# on 317 held-out pairs pooled accuracy 0.707 -> 0.943, and the checkpoint it
# started from scored 0.258 on two-sentence stops -- below chance, i.e. it
# actively preferred the take that keeps talking.  Off by default all the same,
# because nobody has heard it.
QDPO_LORAS = {
    "sft3_qdpo:quality_dpo":    float(os.environ.get("MOSS_LAM_QDPO", "1.5")),
    "sft3_qdpo:burst_stop_dpo": float(os.environ.get("MOSS_LAM_BSDPO", "0.0")),
    "sft3_qdpo:quality_dpo_step1504": 0.0,
}
QDPO_LABELS = {
    "sft3_qdpo:quality_dpo": "Quality DPO (step376)",
    "sft3_qdpo:burst_stop_dpo": "Burst + stop DPO (step896)",
    "sft3_qdpo:quality_dpo_step1504": "Quality DPO (step1504, not recommended)",
}
# A scripted burst after the last word gets its stated length times this before
# the cut point, because the model rarely realises a burst at exactly the length
# asked for and cutting one off is worse than keeping a little filler.
ALIGN_BURST_SLACK = float(os.environ.get("MOSS_ALIGN_BURST_SLACK", "2.0"))
# Streaming lead-in: how much audio to hold before deciding where the first word
# starts, and how many opening words to ask the aligner about.  Costs this much
# once, at the start of a reply, and only when trimming is on.
ALIGN_LEAD_SCAN_S = float(os.environ.get("MOSS_ALIGN_LEAD_SCAN", "1.3"))
ALIGN_LEAD_WORDS = int(os.environ.get("MOSS_ALIGN_LEAD_WORDS", "3"))
# Fraction of the requested duration that must exist before the tail is looked
# for at all.  Below this, an alignment against the prefix reports the last word
# as finished while half the line is unspoken.
ALIGN_TAIL_AFTER = float(os.environ.get("MOSS_ALIGN_TAIL_AFTER", "0.6"))
# Which aligner, in order of preference.  `qwen` is Apache-2.0 and is the
# default because it removes the only non-commercial licence in this stack; the
# MMS CTC model stays available and is the lighter of the two.
ALIGN_BACKEND = os.environ.get("MOSS_ALIGN_BACKEND", "qwen,mms")
ALIGN_QWEN_REPO = os.environ.get("MOSS_ALIGN_QWEN", "Qwen/Qwen3-ForcedAligner-0.6B-hf")
# 1.84 GB loaded, which does not fit beside the voice model on a 24 GB card, so
# it goes on the language model's card by default.
ALIGN_QWEN_DEVICE = os.environ.get("MOSS_ALIGN_QWEN_DEVICE", "cuda:1")
# Structural sanity bound: no single word lasts this long in ordinary speech, so
# a span longer than it means the alignment collapsed rather than fitted.
ALIGN_QWEN_MAX_WORD_S = float(os.environ.get("MOSS_ALIGN_QWEN_MAX_WORD", "3.0"))

# ---------------------------------------------------------------- wikiskills
# The generated knowledge layer.  `coefficients.json` in here is what levers.py
# reads; `VOCAL_BURSTS.md` is what the director reads, via skills.py.
def _skills_dir():
    """Where the wiki lives: the assets copy, else the one in this repository.

    `wikiskills/` is committed here, so a fresh clone has it without downloading
    anything.  A deployment that keeps a newer copy alongside its other assets
    wins, because that is the one the training side updates.
    """
    p = os.environ.get("MOSS_SKILLS_DIR")
    if p:
        return p
    for cand in ("/mnt/nvme/moss-15-v2-assets/wikiskills",
                 os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "wikiskills")):
        if os.path.isdir(cand):
            return cand
    return "/mnt/nvme/moss-15-v2-assets/wikiskills"


SKILLS_DIR = _skills_dir()
SKILLS_ON = os.environ.get("MOSS_SKILLS", "1") not in ("0", "false", "")
# Do not offer a burst class whose measured family hit rate is below this.  The
# study's own shipping bar; under it a request is more likely to produce nothing
# than the sound asked for.
SKILLS_MIN_HIT = float(os.environ.get("MOSS_SKILLS_MIN_HIT", "0.15"))

# ------------------------------------------------------- which burst adapters
# `recipe` is the default and the only setting that follows the measurements:
# each class's own page names the arm that won for it, and the merge rule moved
# 12 of them while the rest stayed on the shipped adapter.  Serving one set for
# everything would override that per-class judgement in one direction or the
# other.  The fixed sets remain selectable for comparison.
BURST_SET = os.environ.get("MOSS_BURST_SET", "recipe")
BURST_SET_ROOT = {"shipped": "burst", "v2": "burst_v2",
                  "v2_top1": "burst_v2_top1", "group": "burst_grp"}   # "recipe" -> per class
# How many burst adapters one turn may merge.  The planner used to return a
# single best match over the whole reply, so a line with two bursts got one
# adapter and the choice fell to whichever name was the longer string.  Reading
# the tags instead means a reply can want several; this bounds the merge cost.
BURST_MAX_ADAPTERS = int(os.environ.get("MOSS_BURST_MAX_ADAPTERS", "3"))
# The recipe weights were each measured with ONE adapter merged, and for several
# classes with no production stack under it at all ("nur der Adapter").  Serving
# three of them at once summed to 4.8 on top of eight other adapters and produced
# babble -- far outside anything that was measured.  So the total is budgeted:
# bursts are kept in script order until the budget is spent, which leaves every
# adapter that IS merged at exactly its measured weight rather than rescaling it
# to a value nobody tested.
# The total across a turn is budgeted too, but generously: measured on the same
# fixed scripts, two adapters cost nothing over one (1.0 x2 -> 0.285 against
# 0.319 for x1; 1.5 x2 -> 0.320 against 0.322).  It is the weight of a single
# adapter that breaks a line, not the sum, so this only bounds the merge cost.
# An earlier budget of 1.0 here was set from turn-to-turn comparisons that were
# dominated by the director writing different scripts, and was wrong.
# The cliff sits between 2.5 and 3.0; 2.0 is the last value clean on every
# arrangement tried.  Over budget the weights are SCALED, never dropped: a
# scream that is quietly not merged is a worse outcome than a scream at two
# thirds weight, and dropping made the set depend on tag order.
BURST_LAM_BUDGET = float(os.environ.get("MOSS_BURST_LAM_BUDGET", "2.0"))

# ---------------------------------------------------------------- best-of-N
# Generate the turn several times and keep the best.  The burst recipes quote an
# N per class -- "at a hit rate of 0.27, 8 candidates for 90 % confidence" -- and
# without this that column is advice nobody can act on.  All N are one batched
# forward pass, so the set costs little more than one take.
BON_ON = os.environ.get("MOSS_BON", "0") not in ("0", "false", "")
BON_N = int(os.environ.get("MOSS_BON_N", "8"))
# Guidance for the candidates.  1.0 is off.  4.0 is the ceiling the crossfade
# study measured -- anchor separation rises monotonically to it with no reversal
# anywhere in the grid -- and it is what a listener reported as the top of the
# usable range.  Its cost over 3.0 is not separable on the data we have.
BON_GUIDANCE = float(os.environ.get("MOSS_BON_GUIDANCE", "3.0"))
# How much the "is this the performance that was asked for" term counts.  Double,
# because the other two terms measure whether a take is good at all rather than
# whether it is the right one.
BON_CLAP_WEIGHT = float(os.environ.get("MOSS_BON_CLAP_W", "2.0"))
# The burst term, added 2026-09-07 from study `vb_opt` (protocol section 71).
# The four terms above score whether a take is good and whether it is the
# performance that was asked for.  NONE of them asks whether the sound the
# script names is actually present.  Measured over 1,940 candidate sets and
# 15,517 candidates under the full production stack: of the four, only `clap`
# tracks burst presence at all (within-set r +0.148), `blend` points the WRONG
# WAY (-0.084) and `genuineness` does nothing (-0.009, CI straddles zero).
# Restricted to the six carrier voices that have an `sft3_voice` adapter -- the
# case the server is ALWAYS in, because it falls back to the speaker adapter
# otherwise -- the shipped reward's burst lift is +0.0104 (t 1.51), i.e. not
# distinguishable from zero.  The pooled +0.0348 is carried entirely by material
# that is off the shipped configuration.
# With this term at 2.0: strict hit rate of the DELIVERED take +0.0454
# (t 7.60, n 1940, +37 % relative), scored by an independent detector rather
# than the one doing the ranking.  Price: WER +0.0040 (t 2.49, gate is +0.104)
# and CLAP -0.0055 (t -6.82) = 3.6 % of the within-set range a re-ranker can
# actually move.  That is a TRADE, not a free win, and it is stated as one.
# Why 2.0 and not 3.0: the exchange rate has a knee.  lambda 1->2 buys 3.82 hit
# per unit of CLAP, 2->3 buys 2.09, 3->5 buys 1.16.
# CAVEAT ON THE SIZE: every figure above is measured UNGUIDED.  The chat path
# would run this at guidance 3.0, where the thin guided sub-arm suggests
# guidance already recovers much of the same gain (+0.0125, t 0.30, against
# +0.1125, t 2.39 unguided, on 80 sets).  Turn best-of-N on first and measure,
# then judge this term -- not both at once.
BON_BURST_WEIGHT = float(os.environ.get("MOSS_BON_BURST_W", "2.0"))
# Which detector names the sound.  `x2` is ours and is the only one graded
# across generators: trained on one source and tested on the other it scores
# 34.2 % / 34.3 % over 17 classes against a 5.88 % chance rate, and beats the
# old detector on the same test sets (26.6 % / 25.5 %).
BON_BURST_DETECTOR = os.environ.get("MOSS_BON_BURST_DET",
                                    "laion/vocal-burst-detector-x2")
# CORRECTED 8 September 2026.  The note that stood here said the 768 of the
# `x2` head and the 768 of our commercial tower were a COINCIDENCE, that the
# head's extractor was unpublished, and that feeding it commercial embeddings
# would produce meaningless probabilities.  That was a sound reading of what was
# published at the time and it is wrong.  Measured since, on the stored feature
# vectors of 300 segments present in both feature sets, same keys, same order:
#
#   max |difference|   1.57e-07        cosine min/median/max   1.000000
#   mean |difference|  6.75e-09        bit-identical           no
#
# `FastScorer.emb.encode_waveform` IS `laion/voiceclap-commercial`.  The two
# differ only by float non-determinism and batch composition.  Feeding the x2
# head commercial embeddings is therefore correct, not meaningless.
#
# Independently: `laion/vocal-burst-detector-x2` now also ships `commercial/`,
# five seeds trained directly on commercial embeddings, so a head that names
# its encoder explicitly exists and nothing has to rest on the identity above.
#
# The class split is what should decide this, not the mean.  Tested on real
# speech, `large-v2` wins 12 of 17 classes -- the quiet sustained sounds, Sharp
# Inhale +0.248, Humming +0.240, Deep Breath +0.216, Yawn +0.208 -- and LOSES
# ON `Scream`, 0.768 against 0.576.  A detector decides whether a hit counts as
# a hit, so ranking with the big head would systematically undercount the class
# this project has spent the most effort on.  Do not "upgrade" this default
# without checking what it costs on screams.  docs/BURST_REWARD.md.
#
# The 18 GB `production` head remains out of reach on this box (1.0 GB free on
# one card, 2.5 GB on the other with the speech model loaded); the 110 M
# commercial tower is already resident for retrieval.
BON_BURST_ENCODER = os.environ.get("MOSS_BON_BURST_ENC", "commercial")
# Whether anything can actually populate `c["burst"]`.  While this is false the
# fifth summand is dead weight and every gate that depends on a burst-aware
# ranker — the solo ceiling below — stays shut.  Set it when a detector whose
# encoder is genuinely loadable here is wired into Judge.score.
BON_BURST_READY = os.environ.get("MOSS_BON_BURST_READY", "1") not in ("0", "false")
# Soft, not tiered.  Tiering was proposed as the sharper instrument; measured it
# costs WER +0.046 (t 3.55) against soft's +0.004 -- an order of magnitude --
# and buys 0.002 of extra hit.  Ranking on a hard tier first discards the
# intelligibility factor's ordering inside the tier.
BON_BURST_MODE = os.environ.get("MOSS_BON_BURST_MODE", "soft")
# 1.0 s windows at a 0.25 s hop, restricted to [t-1.0, t+2.0] around the
# bracket's expected onset.  Localised beats whole-clip, 0.2082 to 0.1907.
BON_BURST_WIN_S = float(os.environ.get("MOSS_BON_BURST_WIN", "1.0"))
BON_BURST_HOP_S = float(os.environ.get("MOSS_BON_BURST_HOP", "0.25"))
# The intelligibility factor is the raw inverse word error rate.  It used to be
# flattened to 1.0 above 0.85, which covered most of a candidate set -- six of
# eight in one run -- so the factor stopped separating precisely where the
# candidates were closest.  0 disables the flattening; set it to 0.85 to get the
# old behaviour back.
BON_WER_KNEE = float(os.environ.get("MOSS_BON_WER_KNEE", "0"))
BON_DEVICE = os.environ.get("MOSS_BON_DEVICE", "cuda:0")
# How many candidates go through the model at once.  Guidance doubles the batch
# -- two branches per candidate -- so it gets its own, smaller figure; eight
# candidates guided is sixteen sequences and that does not fit beside the model
# on a 24 GB card.  Exceeding it is not fatal: the batch halves and retries.
BON_BATCH = int(os.environ.get("MOSS_BON_BATCH", "8"))
BON_BATCH_CFG = int(os.environ.get("MOSS_BON_BATCH_CFG", "4"))

# ---------------------------------------------------------------- breath ----
# The prompt asks for silence *inside* a sentence and the model still returns
# replies that only stop at full stops.  When a reply comes back with no pause
# inside any sentence, put one at up to BREATHE_MAX natural breathing points —
# a comma, a dash, the conjunction that starts a new thought.  A pause the model
# chose itself is never touched.
BREATHE_ON = os.environ.get("MOSS_BREATHE", "1") not in ("0", "false", "no")
# How many the floor may ADD to one reply.  It has to be at least BREATHE_WANT,
# or the target is unreachable from a reply that arrived with none: the floor
# adds min(BREATHE_MAX, want - have), so a max of 2 silently capped a want of 3
# at 2.
BREATHE_MAX = int(os.environ.get("MOSS_BREATHE_MAX", "3"))
# How many silences a reply should end up with INSIDE its sentences.  The floor
# used to fire only at zero, which left every reply that managed one sitting at
# one; it now tops up to this.  Two bugs were deleting or hiding these before
# the number mattered: the sanitiser stripped any pause following a round
# bracket, including a delivery direction, and the inner-pause test counted a
# pause at the start of a sentence as being inside it.
BREATHE_WANT = int(os.environ.get("MOSS_BREATHE_WANT", "3"))

# ----------------------------------------------------------------- sidon ----
# Speech restoration as the last step before a clip is heard.  Measured +0.43 of
# 5 on "how pleasant does this sound" (p < 0.001, 21 paired clips) with no
# movement on how natural or how fitting the performance is: it cleans the
# signal, not the delivery.  See docs/SIDON.md.
#
# DEFAULT OFF since 6 September 2026.  It was shipped on and turned back off
# after a listening report: on screams and loud vocal bursts it distorts.  The
# measurement that put it on used twenty-one clips drawn across the score range
# of an ordinary conversational run, and that sample contained no screaming --
# so the +0.43 is real for ordinary speech and says nothing about the loud tail,
# which is exactly where a restoration model trained on clean read speech has
# least support.  Still worth turning on by hand for a quiet take.
# This is the TorchScript pair served from the hub snapshot, which is a
# different path from SIDON_SRC above — that one is the research checkout used
# once, offline, to restore the corpus anchor.
# The app sees the TTS card as cuda:0 and the language model's as cuda:1;
# restoration goes on the second so it never competes with generation.
SIDON_ON = os.environ.get("MOSS_SIDON", "0") not in ("0", "false", "no")
SIDON_DEVICE = os.environ.get("MOSS_SIDON_DEVICE", "cuda:0")
SIDON_SNAPSHOT = os.environ.get(
    "MOSS_SIDON_SNAPSHOT",
    "/mnt/nvme/hf_cache/hub/models--sarulab-speech--sidon-v0.1/snapshots/*")
SIDON_BASE = os.environ.get("MOSS_SIDON_BASE", "http://127.0.0.1:8793")
SIDON_TIMEOUT = float(os.environ.get("MOSS_SIDON_TIMEOUT", "120"))

# Every bracket in English, whatever language is being spoken.  Measured on one
# German line, four seeds a cell: German cues median word error 0.267 (2 of 4
# unusable) against 0.000 with the same German words and English cues.  The
# prompt has asked for this all along and the director does not comply on German
# turns, so the server rewrites them before the reference is retrieved.
ENGLISH_CUES = os.environ.get("MOSS_ENGLISH_CUES", "1") not in ("0", "false")


# ----------------------------------------------------------------- speak ----
# `/api/speak` is the whole demo behind one request: text in, MP3 out.  Its
# defaults are deliberately the UI's, so a bare curl gives what the browser
# gives.  Best-of is higher here than the UI's 8 because a headless caller is
# not sitting and waiting the way a person in a chat window is.
SPEAK_BEST_OF = int(os.environ.get("MOSS_SPEAK_BEST_OF", "10"))
SPEAK_GUIDANCE = float(os.environ.get("MOSS_SPEAK_GUIDANCE", "3.0"))

# ----------------------------------------------------------------- burst ----
# The burst detector as a service.  Encoder `laion/voiceclap-large-v2` in 8-bit
# (~9 GB) plus the five-head production ensemble; it needs a card to itself, so
# it runs where the language model would otherwise be.  8-bit was checked
# against a bf16 CPU reference: cosine median 0.9959 and the same top-1 class on
# 35 of 36 windows.  See docs/BURST_REWARD.md.
BURST_BASE = os.environ.get("MOSS_BURST_BASE", "http://127.0.0.1:8794")
BURST_TIMEOUT = float(os.environ.get("MOSS_BURST_TIMEOUT", "600"))
BURST_ENCODER_PATH = os.environ.get(
    "MOSS_BURST_ENCODER_PATH",
    "/mnt/nvme/hf_cache/hub/models--laion--voiceclap-large-v2/snapshots/*")
BURST_DETECTOR_PATH = os.environ.get(
    "MOSS_BURST_DETECTOR_PATH",
    "/mnt/nvme/hf_cache/hub/models--laion--vocal-burst-detector-x2/snapshots/*")
# 4-bit ships: it agrees with bf16 on 35 of 36 top-1 decisions exactly as 8-bit
# does, at 5.7 GB instead of 10.  The 8-bit build left 2 GB of headroom on this
# card and a turn allocating during alignment ran out of memory.
BURST_QUANT = os.environ.get("MOSS_BURST_QUANT", "4bit")
