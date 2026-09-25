"""Match the director's prose against the reference corpus.

The director writes how the line should sound — a standing GENERAL description
plus a bracketed cue on every sentence.  That prose, concatenated, is embedded
with the VoiceCLAP-commercial text tower and matched against the index built by
build_retrieval_index.py, which gives back three things: the reference clip to
condition on, the emotion whose adapter to merge, and the measured scores of
what was picked.

Two measured facts shape how the matching is done.

Anisotropy: raw cosines in this space sit in a narrow band around 0.9 and a few
conditions are near-neighbours of everything, so both sides are mean-centred
first.  On a 40-way held-out set of director prose that moves top-1 from 0.35 to
0.61 against the text anchors and from 0.22 to 0.44 against the audio centroids.

Single clips are noisy: classifying one clip's audio embedding against 40
emotion captions scores 0.071, which is precisely the emonet top-1 on this
model's own card.  Averaging a condition's takes into a centroid is what makes
the audio side usable at all, so nothing here matches against a lone clip.
"""
import json, os, re, threading

import numpy as np

import config

_LEVEL_WORDS = {
    "A": ("intense", "extreme", "full", "unrestrained"),
    "B": ("contained", "restrained", "quiet", "held back", "subtle"),
}


# German markers that are NOT also English words.  The first version of this
# list contained `was` and `die`, which are ordinary English; two of either
# marked an English line as German, and the whole GENERAL block then declared
# itself DE on an English take.  Umlauts and eszett count double because no
# English word carries them.
_DE_UMLAUT = re.compile(r"[äöüßÄÖÜ]")
_DE_MARK = re.compile(
    r"\b(?:und|nicht|noch|sehr|ganz|dich|dir|ist|das|der|ein|eine|einen|auf|"
    r"für|über|wie|dass|weil|ich|du|wir|sie|aber|schon|auch|mich|mir|kann|"
    r"haben|hatte|sind|wird|nur|doch|jetzt|immer|nichts|etwas|mehr|schön|"
    r"leise|stimme|atem|trauer|traurig|wut|zärtlich|vielleicht|wirklich)\b",
    re.I)


def looks_german(text):
    """Cheap language check, used where guessing wrong is expensive.

    The text tower is all-MiniLM-L6-v2, which is English; German cues do not
    merely score lower, they score wrongly — "Stimme brüchig vor
    zurückgehaltener Trauer" came back as Teasing.  And the director's own
    `language` field cannot be trusted for this: it reported English on a turn
    it had written entirely in German.
    """
    t = str(text or "")
    return len(_DE_MARK.findall(t)) + 2 * len(_DE_UMLAUT.findall(t)) >= 2


# Clauses of a GENERAL line that describe *who* is speaking rather than how the
# moment sounds.  These are constant across every turn of a voice, so they add
# the same vector to every query and push the whole index the same way.
_STATIC = re.compile(
    r"\b(?:aged?|age|years?\s+old|\d0s\b|late\s+\d0s|mid[- ]\d0s|early\s+\d0s"
    r"|masculine|feminine|male|female|man'?s|woman'?s|boy|girl"
    r"|accent|american|british|australian|irish|scottish|german|romanian|french"
    r"|standard\s+\w+|native|speaker of|recording|studio|microphone|mono|khz"
    r"|timbre|register\s+of|vocal\s+tract|voice\s+of\s+an?)\b", re.I)
# ...and the parts that do describe the moment.  "reads as" is where the corpus
# puts its emotion names, so it is always kept.
_EMOTIVE = re.compile(
    r"\b(?:reads as|sounds?|delivery|tempo|arousal|warmth|brightness|clarity"
    r"|breath|tension|pace|energy|affect|pitch|emotion|intensity|strain"
    r"|whisper|shout|soft|loud|bright|dark|rough|smooth|warm|cold|tight|relaxed"
    r"|urgent|calm|slow|fast|amused|angry|sad|tender|hostile|playful|weary)\b",
    re.I)
# the corpus's own 40 emotion names, so a bare "contempt" left over from
# "reads as bitterness, contempt" is not thrown away as unrecognised
_EMO_WORDS = re.compile(
    r"\b(?:affection|amusement|anger|astonishment|surprise|awe|bitterness|"
    r"concentration|confusion|contemplation|contempt|contentment|disappointment|"
    r"disgust|distress|doubt|elation|embarrassment|numbness|fatigue|exhaustion|"
    r"fear|helplessness|hope|enthusiasm|optimism|impatience|irritability|"
    r"infatuation|interest|intoxication|jealousy|envy|longing|malevolence|malice|"
    r"pain|pleasure|ecstasy|pride|relief|sadness|lust|shame|sourness|teasing|"
    r"thankfulness|gratitude|triumph|vulnerability|grief|joy)\b", re.I)


def emotive_general(general):
    """Keep only the parts of GENERAL that describe this moment's sound.

    A GENERAL line mixes two things: the standing identity of the speaker — age,
    gender, accent, timbre — and how the clip is performed.  The identity half
    never changes between turns, so it contributes the same offset to every
    query and drowns the half that varies.  Dropping it is what makes GENERAL
    worth including at all.
    """
    out = []
    for clause in re.split(r"[;.]|,(?=\s)", str(general or "")):
        c = clause.strip(" .,;")
        if not c:
            continue
        if _STATIC.search(c) and not re.search(r"\breads as\b", c, re.I):
            continue
        # No fallback: a clause that names neither a felt state nor a sound
        # property is describing the speaker, not the moment, and is dropped
        # even if that leaves GENERAL empty.  The cues carry the turn on their
        # own, and an empty half is better than a constant one.
        if _EMOTIVE.search(c) or _EMO_WORDS.search(c):
            out.append(c)
    return ", ".join(out)[:400]


def clean_cues(script):
    """Round-bracket text only: directions and bursts, with the numbers gone.

    Durations are arithmetic, not description — "(chuckle, 0.3 seconds)" should
    retrieve the same clip as "(chuckle)".
    """
    cues = re.findall(r"\(([^)]{2,160})\)", str(script or ""))
    out = []
    for c in cues:
        c = re.sub(r",?\s*[0-9]*\.?[0-9]+\s*(?:s|sec|seconds?)\b", "", c, flags=re.I)
        c = c.strip(" ,;")
        if c:
            out.append(c)
    return " ".join(out)[:600]


def split_direction(general, script, text=""):
    """The standing description and this moment's cues, kept apart.

    GENERAL says who is speaking and barely changes between turns; the round
    brackets say what to do right now.  Concatenated, the identity boilerplate
    is several times longer than the cues and dominates the embedding — a
    delighted line whose cues read "sharp inhale, delighted laugh, voice
    bursting upward" came back as Jealousy_and_Envy because the GENERAL text
    swamped it.  So the clip is matched on both, and the emotion on the cues.

    Square brackets are excluded on purpose: in this prompt format they carry
    [pause], which is timing rather than delivery.
    """
    gen = emotive_general(general)
    cue_txt = clean_cues(script)
    if not gen and not cue_txt and text:
        gen = str(text).strip()
    return gen[:600], cue_txt


def parse_direction(general, script, text=""):
    """Both halves concatenated — what the clip is matched against."""
    gen, cues = split_direction(general, script, text)
    return " ".join(b for b in (gen, cues) if b)[:800]


class Retriever:
    """Nearest-neighbour lookup over the acting conditions of one corpus."""

    def __init__(self, root=None, device=None):
        self.root = root or config.RETRIEVAL_DIR
        self.ok = False
        self.lock = threading.Lock()
        self.model = self.tok = None
        self.device = device or "cuda"
        p = os.path.join(self.root, "index.npz")
        if not os.path.exists(p):
            print(f"[retrieval] no index at {self.root}", flush=True)
            return
        z = np.load(p)
        meta = json.load(open(os.path.join(self.root, "index.json"), encoding="utf-8"))
        self.cond = meta["conditions"]
        self.emotions = meta["emotions"]
        self.templates = meta["templates"]
        self.C = self._center(z["cond_emb"], z["cond_mean"])
        self.A = self._center(z["emo_anchor"], z["emo_anchor_mean"])
        # A single query cannot be centred against a batch of its own, so it is
        # centred against the text anchors' mean — the one hub direction that is
        # known at inference.  This is the arrangement the 0.61 top-1 was
        # measured under; centring the query on the audio mean instead was not.
        self.qmu = z["emo_anchor_mean"]
        self.by_voice = {}
        for i, c in enumerate(self.cond):
            self.by_voice.setdefault(c["voice"], []).append(i)
        self.ok = True
        print(f"[retrieval] {len(self.cond)} conditions, {len(self.emotions)} emotions, "
              f"{len(self.by_voice)} voices", flush=True)

    @staticmethod
    def _center(M, mu):
        X = M - mu[None, :]
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(n, 1e-8)

    # ------------------------------------------------------------------ text
    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        repo = config.VOICECLAP_REPO
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo, trust_remote_code=True,
                                               dtype=torch.float32).to(self.device).eval()

    def embed(self, text):
        import torch
        self._load()
        b = self.tok([text], return_tensors="pt", padding=True, truncation=True,
                     max_length=128).to(self.device)
        with torch.no_grad():
            v = self.model.encode_text(b["input_ids"], b["attention_mask"])[0]
        return v.float().cpu().numpy()

    # ------------------------------------------------------------- retrieval
    def query(self, direction, voice=None, lang=None, level=None, k=3,
              emotion_nuance=True, cues=None):
        """Return the reference conditions, and the emotion to merge.

        `level` is the intensity band the director asked for: "A" for intense,
        "B" for contained.  It is a filter on the clip, not on the emotion.
        """
        if not self.ok or not direction:
            return None
        q = self.embed(direction)
        qa = q - self.qmu
        qc = qa / max(np.linalg.norm(qa), 1e-8)
        # the emotion is read off the cues alone when there are any
        if cues:
            qe = self.embed(cues) - self.qmu
            qe = qe / max(np.linalg.norm(qe), 1e-8)
        else:
            qe = qc

        emo_rank = []
        if len(self.emotions):
            se = self.A @ qe
            order = np.argsort(-se)
            emo_rank = [(self.emotions[i], float(se[i])) for i in order[:5]]

        idx = self.by_voice.get(voice) if voice else None
        idx = list(idx if idx else range(len(self.cond)))
        # Language is a hard filter, not a preference.  Conditioning a German
        # turn on an English clip put an audible English accent on the output
        # earlier in this project, and a soft penalty was not enough to stop the
        # occasional cross-language clip from winning on similarity alone.
        if lang:
            same = [i for i in idx if self.cond[i]["lang"] == lang]
            if same:
                idx = same
        idx = np.array(idx)
        sc = self.C[idx] @ qc

        top_emo = emo_rank[0][0] if emo_rank else ""
        # the fusion: the audio centroid says which clip sounds like the ask, the
        # text anchor says which emotion it is.  Agreeing conditions win.
        bonus = np.array([config.RETRIEVAL_EMO_BONUS
                          if (top_emo and self.cond[i]["emotion"] == top_emo) else 0.0
                          for i in idx])
        lvl_pen = np.array([0.0 if (not level or not self.cond[i]["level"]
                                    or self.cond[i]["level"] == level)
                            else -config.RETRIEVAL_LEVEL_PENALTY for i in idx])
        total = sc + bonus + lvl_pen
        order = np.argsort(-total)[:k]
        hits = []
        for j in order:
            c = dict(self.cond[int(idx[j])])
            c["score"] = float(total[j])
            c["audio_sim"] = float(sc[j])
            hits.append(c)
        return {
            "direction": direction[:300],
            "emotions": emo_rank,
            "emotion": top_emo if emotion_nuance else "",
            "hits": hits,
        }
