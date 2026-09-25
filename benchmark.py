"""Benchmark items as acting briefs.

A benchmark item arrives as a JSON object: an emotion target, a situation, a
performance direction and — the part that matters — a script that must be
spoken *exactly* as written.  Handed that raw, a 12B model has to parse JSON,
work out which of nine fields is the line, and follow two hundred lines of
format rules at the same time.  It reliably fails the first step and answers
*about* the elevator instead of performing the sentence about it.

So the JSON never reaches the model.  `detect` recognises an item, `brief`
rewrites it as a short imperative task, and `verbatim_ok` checks afterwards that
the words came back unchanged.  `annotate` is the floor: a correct, plainly
directed rendering of the item, used when the model drifts off the script.
"""
import json
import re

import timed_script

# the adverb scale the voice model was trained against, keyed by the
# intensity word benchmark items use
_ADVERB = {
    "subtle": "barely", "slight": "barely", "low": "barely", "mild": "barely",
    "moderate": "clearly", "medium": "clearly",
    "strong": "intensely", "high": "intensely",
    "extreme": "overwhelmingly", "intense": "overwhelmingly",
    "overwhelming": "overwhelmingly",
}

# benchmark items name the emotion as a noun; a direction needs the adjective,
# because "(clearly amusement)" is not something a director would ever write
_ADJ = {
    "amusement": "amused", "anger": "angry", "fear": "afraid", "joy": "joyful",
    "sadness": "sad", "disgust": "disgusted", "surprise": "surprised",
    "contempt": "contemptuous", "pride": "proud", "shame": "ashamed",
    "guilt": "guilty", "relief": "relieved", "gratitude": "grateful",
    "affection": "affectionate", "love": "loving", "excitement": "excited",
    "boredom": "bored", "confusion": "confused", "curiosity": "curious",
    "embarrassment": "embarrassed", "envy": "envious", "hope": "hopeful",
    "nostalgia": "nostalgic", "awe": "awed", "anxiety": "anxious",
    "frustration": "frustrated", "disappointment": "disappointed",
    "admiration": "admiring", "sympathy": "sympathetic", "regret": "regretful",
    "determination": "determined", "irritation": "irritated",
    "satisfaction": "satisfied", "worry": "worried", "grief": "grieving",
    "jealousy": "jealous", "panic": "panicked", "delight": "delighted",
    "concern": "concerned", "sarcasm": "sarcastic", "triumph": "triumphant",
    "loneliness": "lonely", "desire": "wanting", "tenderness": "tender",
    "resentment": "resentful", "distress": "distressed", "calm": "calm",
    "contentment": "contented", "apprehension": "apprehensive",
}


def _adj(label):
    """Adjective for an emotion label, however it was written."""
    w = (label or "").strip().lower()
    if not w:
        return "engaged"
    if w in _ADJ:
        return _ADJ[w]
    for suf, rep in (("ment", "ed"), ("ness", ""), ("tion", "ted"),
                     ("ity", ""), ("ance", "ed"), ("ence", "ed")):
        if w.endswith(suf) and w[:-len(suf)] in _ADJ:
            return _ADJ[w[:-len(suf)]]
    # already an adjective, or a word we do not know: say it plainly
    return w if not w.endswith(("ment", "ness", "tion", "ity")) \
        else f"feeling {w}"


def _norm(s):
    """Spoken words only, for comparing what was asked against what came back."""
    s = re.sub(r"\[[^\]]*\]", " ", s or "")     # pauses and durations
    s = re.sub(r"\([^)]*\)", " ", s)            # directions and bursts
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("—", " ").replace("–", " ")
    s = re.sub(r"[^\w\s']", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def detect(message):
    """Return the item dict if this message is a benchmark item, else None.

    Accepts the object on its own, wrapped in a fenced block, or embedded in a
    sentence, and tolerates the trailing comma that comes with pasting one
    element out of a JSON array.
    """
    if not message or "{" not in message:
        return None
    txt = message.strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt).strip()
    start = txt.find("{")
    if start < 0:
        return None
    depth, end, instr, esc = 0, -1, False, False
    for i, ch in enumerate(txt[start:], start):
        if esc:
            esc = False
            continue
        if ch == "\\" and instr:
            esc = True
        elif ch == '"':
            instr = not instr
        elif not instr:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
    if end < 0:
        return None
    try:
        item = json.loads(txt[start:end])
    except Exception:
        return None
    if not isinstance(item, dict):
        return None
    # an item is recognised by carrying a script to perform, however nested
    if isinstance(item.get("script"), dict) and item["script"].get("text"):
        return item
    if isinstance(item.get("script"), str) and item.get("instruction"):
        return item
    return None


def script_text(item):
    sc = item.get("script")
    if isinstance(sc, dict):
        return (sc.get("text") or "").strip()
    return (sc or "").strip()


def is_emotion_item(item):
    """Does this item name a feeling, or a scene / a voice axis?

    Only the emotion track carries a label that can be spoken as a direction.
    `acting_challenge` labels are categories ("Conflict", "Emergency") and
    `voicenet` labels are axis codes ("VALS_high"), and forcing either into the
    adverb-plus-adjective form produces "(clearly conflict)" — which is not a
    direction, and was measured producing flat, unfitting takes.
    """
    tgt = item.get("target") or {}
    track = str(item.get("track") or "").lower()
    if track:
        return track == "emotion"
    return (str(tgt.get("intensity") or "").lower() in _ADVERB
            and (tgt.get("label") or "").strip().lower() in _ADJ)


def _burst_line(vb):
    """The item's burst request, however it is written."""
    if not vb:
        return None
    if isinstance(vb, str):
        return vb
    bits = [str(vb.get("type") or "").strip()]
    place = str(vb.get("placement") or "").replace("_", " ").strip()
    if place:
        bits.append(f"placed {place}")
    d = str(vb.get("description") or "").strip()
    line = ", ".join(b for b in bits if b)
    return f"{line}. {d}" if d else line


def brief(item):
    """The item as a task a small model can follow without parsing anything."""
    text = script_text(item)
    ins = item.get("instruction") or {}
    tgt = item.get("target") or {}
    desc = ", ".join(tgt.get("descriptors") or [])

    L = ["THIS TURN IS A PERFORMANCE, NOT A CONVERSATION.",
         "",
         "Speak these words, and only these words:",
         "",
         f"    {text}",
         ""]
    if is_emotion_item(item):
        label = (tgt.get("label") or "").strip()
        adv = _ADVERB.get(str(tgt.get("intensity") or "moderate").lower().strip(),
                          "clearly")
        if label:
            line = (f"The feeling is {label.lower()}, running {adv} — write it as "
                    f'"{adv} {_adj(label)}", that is the adverb the voice model '
                    f"was trained against.")
            if desc:
                line += f" It shows as {desc}."
            L.append(line)
    else:
        # scene- or axis-defined: the words for the feeling are the director's
        # to choose, from the direction and the descriptors
        if desc:
            L.append(f"The voice should read as: {desc}.")
        L.append("No single emotion is named for you. Decide what the person is "
                 "feeling from the situation and the direction below, name it in "
                 "plain words in your first bracket, and give it the adverb that "
                 "matches how hard it is running.")
    if tgt.get("valence") or tgt.get("arousal"):
        L.append(f"Valence {tgt.get('valence') or 'n/a'}, arousal "
                 f"{tgt.get('arousal') or 'n/a'}.")
    if str(tgt.get("mode") or "").strip().lower() == "arc":
        L.append("THIS IS AN ARC: the feeling must not stay where it starts. "
                 "Let the later brackets turn it somewhere the first one did not "
                 "promise.")
    if ins.get("context"):
        L.append(f"The situation: {ins['context']}")
    if ins.get("performance_direction"):
        L.append(f"How to play it: {ins['performance_direction']}")
    vb = _burst_line(ins.get("vocal_burst"))
    if vb:
        L.append(f"This item calls for a vocal burst: {vb} Give it its own "
                 f"bracket with a length in seconds.")
    else:
        L.append("This item does not require a vocal burst. Add one only if the "
                 "moment genuinely wants it.")
    L += ["",
          "Return those exact words in the `script` field with your cues added in "
          "position: a delivery direction before every sentence, pauses written "
          "where the voice would actually stop — including inside a sentence — and "
          "bursts where a person would make one.",
          "Do not add, remove, reorder or reword a single spoken word. Do not "
          "reply to the situation, do not comment on it, do not introduce it. "
          "The words above are the whole of what is said."]
    return "\n".join(L)


def verbatim_ok(script, item):
    """Did the model come back with the words it was given?"""
    return _norm(script) == _norm(script_text(item))


def annotate(item):
    """A correct rendering of the item, without the model.

    Used when the model returns something other than the script it was handed.
    Plain rather than inspired: full direction on the first sentence, short
    reminders after it, and a pause at the commas a speaker would breathe at.
    """
    text = script_text(item)
    tgt = item.get("target") or {}
    ins = item.get("instruction") or {}
    if is_emotion_item(item):
        label = _adj(tgt.get("label"))
        adv = _ADVERB.get(str(tgt.get("intensity") or "moderate").lower().strip(),
                          "clearly")
    else:
        # no usable emotion word: fall back to the first descriptor
        ds = [d for d in (tgt.get("descriptors") or []) if d]
        label = str(ds[0]).lower() if ds else "engaged"
        adv = "clearly"
    held = "letting it out, not hiding it"
    d = (ins.get("performance_direction") or "").lower()
    if any(w in d for w in ("tucked", "held", "contain", "not to", "restrain",
                            "under", "hide", "suppress", "careful")):
        held = "held in and only leaking at the edges"
    sents = [s for s in timed_script._split_sentences(text) if s.strip()]
    out = []
    for i, s in enumerate(sents):
        out.append(f"({adv} {label}, {held})" if i == 0
                   else f"(still {adv} {label})")
        out.append(add_breath_pauses(s, force=True))
    return " ".join(out)


# ---------------------------------------------------------------- pauses ----

# where a speaker actually breathes: after a comma, at a dash, before a
# conjunction that starts a new thought
_BREATH = re.compile(
    r"(?<=,)(?=\s)|(?<=\s)(?=[—–-]\s)|(?<=\s)(?=(?:and then|but|until|"
    r"because|so that|although|though|except|while)\s)", re.I)


def _has_inner_pause(script):
    """A pause with spoken words before it inside the same sentence.

    This used to accept any pause whose preceding character was not sentence
    punctuation — which includes the ")" of an opening delivery direction.  A
    pause written there sits at the START of a sentence, not inside one, and
    counting it as inner switched `breathe()` off on exactly the replies that
    had no mid-clause silence at all.
    """
    for m in re.finditer(r"\[[^\]]*pause[^\]]*\]", script or "", re.I):
        before = script[:m.start()]
        # back to the start of this sentence
        cut = max(before.rfind("."), before.rfind("!"), before.rfind("?"),
                  before.rfind("…"))
        head = before[cut + 1:] if cut >= 0 else before
        # spoken words only: brackets are cues, not speech
        head = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", head)
        if head.strip():
            return True
    return False


def add_breath_pauses(text, force=False, limit=2, length=0.3):
    """Insert a short silence at up to `limit` natural breathing points."""
    n, out, cuts = 0, text, []
    for m in _BREATH.finditer(text):
        pos = m.start()
        head = text[:pos].split()
        tail = text[pos:].split()
        # never strand a word or two on either side of the silence
        if len(head) < 4 or len(tail) < 3:
            continue
        if cuts and pos - cuts[-1] < 25:
            continue
        cuts.append(pos)
        n += 1
        if n >= limit:
            break
    if not cuts and not force:
        return text
    for pos in reversed(cuts):
        out = out[:pos] + f" [{length} seconds pause] " + out[pos:]
    return re.sub(r"\s+", " ", out).strip()


def _count_inner(script):
    """How many pauses have spoken words before them in their own sentence."""
    n = 0
    for m in re.finditer(r"\[[^\]]*pause[^\]]*\]", script or "", re.I):
        if _has_inner_pause(script[:m.end()]):
            n += 1
    return n


def breathe(script, limit=2, want=1):
    """Top a reply up to `want` silences inside its sentences.

    It used to fire only when there were none at all, which left every reply
    that managed one sitting at one.  It now counts what the model wrote and
    adds the difference, so an explicit choice is still never overwritten — the
    additions go at commas, dashes and the conjunctions that begin a new
    thought, and only where four words precede and three follow.
    """
    if not script:
        return script, 0
    have = _count_inner(script)
    limit = min(limit, max(0, want - have))
    if limit <= 0:
        return script, 0
    parts = re.split(r"(\([^)]*\)|\[[^\]]*\])", script)
    n = 0
    for i, p in enumerate(parts):
        if p.startswith(("(", "[")) or not p.strip() or n >= limit:
            continue
        new = add_breath_pauses(p, limit=limit - n)
        if new != p:
            n += new.count("seconds pause")
            parts[i] = (" " if p[:1].isspace() else "") + new + \
                       (" " if p[-1:].isspace() else "")
    return "".join(parts), n
