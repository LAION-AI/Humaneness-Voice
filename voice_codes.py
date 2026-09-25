"""A short code language for the director, expanded to prose before the TTS sees it.

The voice model was trained on prose captions, so it needs prose.  The language
model does not need to *write* prose: emitting a 60-word GENERAL block plus a
hand-written cue before every sentence is ~90 output tokens per turn, and output
tokens are the whole of time-to-first-audio.

So the model emits codes — `ANG3,TENS4,VOLT4` — and this module turns them back
into the captions the TTS expects.  Emotion codes are derived from the corpus
names deterministically (no hand-typed table to drift), VoiceNet codes are the
corpus codes as they already are.

    ANG3            emotion Anger at strength 3
    TENS4           voice quality TENS at level 4
    (SAD2)          the same, inline in a SCRIPT, becomes a prose cue
"""
import re

# strength 1-4 -> what it means, and which reference condition it maps to
STRENGTH = {
    1: ("faintly, mostly held in and only leaking at the edges of phrases",
        "moderate", "contained"),
    2: ("clearly present but kept under control", "moderate", "free"),
    3: ("strongly, pushing through and hard to contain", "intense", "contained"),
    4: ("at full force, completely unleashed, nothing held back", "intense", "free"),
}
LEVELS = {1: "extremely_low", 2: "moderately_low",
          3: "moderately_high", 4: "very_high"}
LEVEL_WORDS = {1: "extremely low", 2: "somewhat low",
               3: "noticeably high", 4: "very high"}


def build_emotion_codes(emotions):
    """Deterministic 3-letter codes, collisions resolved by walking the name."""
    codes, used = {}, set()
    for e in sorted(emotions):
        head = re.sub(r"[^A-Za-z]", "", e.split("_")[0]).upper()
        cand = head[:3] or "XXX"
        if cand in used:                       # try later letters of the word
            for i in range(1, max(1, len(head) - 2)):
                alt = (head[:2] + head[2 + i:3 + i]).upper()
                if len(alt) == 3 and alt not in used:
                    cand = alt
                    break
            else:                               # fall back to a numeric suffix
                n = 2
                while f"{cand[:2]}{n}" in used:
                    n += 1
                cand = f"{cand[:2]}{n}"
        used.add(cand)
        codes[cand] = e
    return codes


class CodeBook:
    def __init__(self, catalog, descriptions=None):
        self.emotions = build_emotion_codes(catalog.get("emotion", []))
        self.rev_emotion = {v: k for k, v in self.emotions.items()}
        self.dims = {d: d for d in catalog.get("voicenet_dimension", [])}
        self.gloss = (descriptions or {}).get("voicenet_dimension", {})

    # ------------------------------------------------------------- parsing
    _TOK = re.compile(r"([A-Z][A-Z0-9_]{1,7})\s*([1-4])")

    def parse(self, s):
        """'ANG3, TENS4' -> [('emotion','Anger',3), ('voicenet','TENS',4)]"""
        out = []
        for code, lvl in self._TOK.findall(str(s or "").upper()):
            lvl = int(lvl)
            if code in self.emotions:
                out.append(("emotion", self.emotions[code], lvl))
            elif code in self.dims:
                out.append(("voicenet", code, lvl))
        return out

    # ------------------------------------------------------------ expansion
    def expand_delivery(self, s):
        """Codes -> the prose GENERAL fragment the voice model was trained on."""
        parts = self.parse(s)
        if not parts:
            return ""
        bits = []
        for kind, name, lvl in parts:
            if kind == "emotion":
                how, _, _ = STRENGTH[lvl]
                bits.append(f"{name.replace('_', ' ').lower()} {how}")
            else:
                g = self.gloss.get(name, name)
                bits.append(f"{LEVEL_WORDS[lvl]} in {g}")
        return "gripped by " + "; ".join(bits) + "."

    # Short adverbs for an inline cue.  The manual's own examples are terse —
    # "(amused)", "(quietly, with lethal control)" — and it warns that cues
    # should be short and sit right before the words they affect.  Pasting the
    # full prose definition of a code inline produced 90-character cues that
    # buried the vocal burst sitting next to them.
    CUE_STRENGTH = {1: "faintly", 2: "", 3: "very", 4: "utterly"}
    # the emotion names are nouns; a cue reads as direction only as an adjective
    CUE_ADJ = {
        "Anger": "furious", "Sadness": "heartbroken", "Amusement": "amused",
        "Affection": "tender", "Fear": "frightened", "Disgust": "disgusted",
        "Contempt": "contemptuous", "Pride": "proud", "Shame": "ashamed",
        "Embarrassment": "embarrassed", "Relief": "relieved", "Awe": "awestruck",
        "Bitterness": "bitter", "Confusion": "confused", "Doubt": "doubtful",
        "Distress": "distressed", "Elation": "elated", "Longing": "longing",
        "Teasing": "teasing", "Triumph": "triumphant", "Interest": "curious",
        "Contentment": "content", "Disappointment": "disappointed",
        "Helplessness": "helpless", "Concentration": "focused",
        "Contemplation": "thoughtful", "Pain": "in pain",
        "Fatigue_Exhaustion": "exhausted", "Emotional_Numbness": "numb",
        "Impatience_and_Irritability": "impatient", "Jealousy_and_Envy": "jealous",
        "Malevolence_Malice": "malicious", "Sexual_Lust": "hungry",
        "Pleasure_Ecstasy": "blissful", "Sourness": "sour", "Infatuation": "smitten",
        "Thankfulness_Gratitude": "grateful", "Astonishment_Surprise": "astonished",
        "Hope_Enthusiasm_Optimism": "hopeful",
        "Intoxication_Altered_States_of_Consciousness": "woozy",
    }

    def expand_cue(self, s):
        """A single inline cue: 'ANG4' -> '(completely furious)'."""
        parts = self.parse(s)
        if not parts:
            return None
        bits = []
        for kind, name, lvl in parts[:2]:      # two feelings is already a lot inline
            if kind == "emotion":
                word = self.CUE_ADJ.get(name) or \
                    name.replace("_", " ").lower().split()[0]
                pre = self.CUE_STRENGTH.get(lvl, "")
                bits.append(f"{pre} {word}".strip())
            else:
                g = self.gloss.get(name, name)
                g = g.split("—")[0].strip()
                bits.append(f"{LEVEL_WORDS[lvl]} {g}")
        return "(" + ", ".join(b for b in bits if b) + ")"

    def expand_script(self, script):
        """Replace coded cues in a SCRIPT with prose ones, leave prose alone."""
        def sub(m):
            inner = m.group(1).strip()
            if not re.fullmatch(r"[A-Z0-9_ ,]+", inner):
                return m.group(0)          # already prose
            return self.expand_cue(inner) or m.group(0)
        return re.sub(r"\(([^)]*)\)", sub, script or "")

    def reference_for(self, s):
        """First emotion code in the string -> a reference-bank selection."""
        for kind, name, lvl in self.parse(s):
            if kind == "emotion":
                _, inten, cont = STRENGTH[lvl]
                return {"mode": "emotion", "emotion": name,
                        "intensity": inten, "containment": cont}
            if kind == "voicenet":
                return {"mode": "voicenet", "dimension": name,
                        "level": LEVELS[lvl]}
        return None

    def style_for(self, s):
        """VoiceNet codes -> style adapters (level 3/4 high, 1/2 low)."""
        out = []
        for kind, name, lvl in self.parse(s):
            if kind == "voicenet":
                out.append({"dimension": name,
                            "direction": "high" if lvl >= 3 else "low"})
        return out[:4]

    def blend(self, s):
        """The full mix, for merging several adapters at once."""
        return self.parse(s)

    # ---------------------------------------------------------------- prompt
    # A few dimensions have no usable gloss in the corpus prompts; and several
    # of the glosses are too terse to act on ("velocity flux" says nothing about
    # when to reach for it), so the ones that matter get a practical note.
    EXTRA_GLOSS = {
        "EXPL": "explicitness, crude or profane wording",
        "S_RANT": "ranting style",
        "S_TECH": "technical, explanatory style",
    }
    HINT = {
        "RANG": "pitch range — HIGH is wide, swooping, song-like; LOW is flat and monotone",
        "VFLX": "pitch/velocity movement — HIGH keeps the melody of the line moving",
        "HARM": "tonal purity — HIGH is clear and sung; LOW is breathy or noisy",
        "TEMP": "tempo — HIGH is fast, LOW is slow and deliberate",
        "VOLT": "loudness — HIGH shouts, LOW is barely voiced",
        "TENS": "tension in the throat — HIGH is tight and strained",
        "WARM": "warmth — HIGH is round and kind, LOW is cold",
        "BRGT": "brightness — HIGH is forward and crisp, LOW is dark and muffled",
        "ROUG": "roughness/rasp — HIGH is gravel, growl, damage",
        "SMTH": "smoothness — HIGH glides, LOW is jagged",
        "ATCK": "how hard each word starts — HIGH is punchy and percussive",
        "DFLU": "disfluency — HIGH stumbles, restarts, fills",
        "CLRT": "articulation — HIGH over-enunciates, LOW slurs",
        "AGEV": "apparent age — LOW is a child, HIGH is elderly",
        "GEND": "perceived gender of the voice",
        "REGS": "register — HIGH is head voice/falsetto, LOW is chest",
        "RESP": "audible breathing — HIGH is breathy and panting",
        "VULN": "vulnerability — HIGH sounds exposed and fragile",
        "DARC": "dynamic arc across the line — HIGH builds or collapses",
        "CHNK": "phrasing in chunks — HIGH breaks the line into deliberate pieces",
        "METL": "metallic, ringing edge to the timbre",
        "FULL": "body/fullness of tone",
        "S_WHIS": "whispering", "S_RANT": "ranting", "S_AUTH": "commanding authority",
        "S_STRY": "storytelling", "S_NARR": "narrating", "S_DRAM": "dramatic",
        "S_ASMR": "close, intimate ASMR", "S_CART": "cartoonish, exaggerated",
        "S_NEWS": "newsreader", "S_PLAY": "playful", "S_CASU": "casual",
        "S_CONV": "conversational", "S_FORM": "formal", "S_TECH": "explanatory",
    }
    RECIPES = (
        "  sing / melodic / sing-song   RANG4,VFLX4,HARM4  (+S_PLAY3 for light, "
        "S_DRAM3 for grand). There is no dedicated singing dimension — melody is "
        "wide pitch range plus tonal purity plus movement.\n"
        "  monotone / robotic / dead    RANG1,VFLX1,VALN2,SMTH3\n"
        "  whisper / secret             S_WHIS4,VOLT1,RESP3\n"
        "  shout / furious              VOLT4,TENS4,ATCK4,ROUG3\n"
        "  rant                         S_RANT4,TEMP4,VOLT3\n"
        "  bedtime story / calm         S_STRY4,WARM4,TEMP1,SMTH3\n"
        "  authoritative / commanding   S_AUTH4,R_CHST3,ATCK3\n"
        "  child                        AGEV1,BRGT4,RANG3\n"
        "  old and frail                AGEV4,ROUG3,VULN3\n"
        "  drunk / slurred              CLRT1,DFLU4,SMTH1\n"
        "  breathless / panicked        RESP4,TEMP4,TENS4\n"
        "  intimate / close             S_ASMR4,VOLT1,WARM3\n"
    )

    def legend(self, max_dims=None):
        dims = sorted(self.dims)
        def g(d):
            return self.HINT.get(d) or self.gloss.get(d) or self.EXTRA_GLOSS.get(d) or d
        return (
            "CODE LANGUAGE — write codes, not prose. A code is a tag plus a "
            "strength 1-4, several separated by commas.\n"
            "  strength: 1 barely there · 2 controlled · 3 pushing through · "
            "4 fully unleashed\n\n"
            "EMOTIONS (40) — the feeling behind the line:\n  "
            + ", ".join(f"{c}={n}" for c, n in sorted(self.emotions.items()))
            + "\n\nVOICE QUALITIES (" + str(len(dims)) + ") — HOW it is said, on the "
              "same 1-4 scale where 1 is the low end of the dimension and 4 the high "
              "end. Each of these is a separate trained adapter, so naming one really "
              "does change the voice:\n"
            + "".join(f"  {d:8s} {g(d)}\n" for d in dims)
            + "\nRECIPES — reach for these when the user asks for the thing on the left:\n"
            + self.RECIPES
            + "\nExample — \"d\": \"ANG3,TENS4,VOLT4\"  and in the script a cue is "
              "written the same way: \"(ANG4) you never listen to a single word I say\"."
        )
