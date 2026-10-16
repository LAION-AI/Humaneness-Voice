"""Turn the director's script into the timed script SFT3 was trained on.

SFT3's format carries its own arithmetic: every speech segment is preceded by
`[N.N seconds duration]`, every gap of 0.2 s or more by `[N.N seconds pause]`,
and every vocal burst is `(label, N.N seconds)`.  The Tokens field is the sum of
all of it at 12.5 frames per second, and when the numbers disagree with the
budget the model has to choose which to honour.

So the numbers are not left to the language model.  It writes the performance —
directions, bursts, where the silences fall — and this module does the
arithmetic, then makes the total match the budget exactly.  Getting a language
model to produce a set of decimals that sums to a given figure is a bad bet, and
length control is the one thing this checkpoint honours best.

Brackets are what tell the three tag kinds apart, and the rule is unforgiving:
square bracket = seconds; round bracket *with* a number = a burst; round bracket
*without* a number = a delivery direction.  A direction that picks up a number
becomes a burst, so directions here never carry one.

A round bracket without a number is then read one of two ways depending on
whether it NAMES a burst, and getting that wrong is silent: an unrecognised
label becomes a direction, the round bracket turns into an instruction about how
to speak, and no sound is produced.  The vocabulary is therefore taken from the
same wikiskills directory `skills.py` uses to decide what to offer the director
— see `burst_vocabulary()` — rather than from a second list that would drift.
"""
import re

import config

# Median burst in the training data is 0.28 s; the 10th and 90th percentiles are
# 0.14 and 0.48.  A burst asked for at 3 s is outside anything the model saw.
BURST_DEFAULT = 0.28
BURST_MIN, BURST_MAX = 0.14, 1.2
# The director may mark a burst short or long instead of writing a number, which
# would turn its bracket into something else entirely.  The values are the
# corpus's own 10th and 90th percentiles, so "long" is half a second and not two:
# the longest burst ever observed is 2.46 s and the median is 0.28.
BURST_SHORT, BURST_LONG = 0.14, 0.48
_LEN_WORD = re.compile(r"^\s*(short|brief|quick|tiny|long|drawn[- ]out|extended)\s+",
                       re.I)
PAUSE_DEFAULT = 0.30
PAUSE_MIN = 0.20
LEAD_PAUSE = 0.30
TAIL_PAUSE = 0.30
# Segments over 12 s were split again in training, so nothing longer is emitted.
SEG_MAX = 12.0

# The core labels, hard-coded: the ones that actually occur in the training data
# and the fallback vocabulary when the skills directory is not on disk.  This is
# NOT the whole vocabulary — see `burst_vocabulary()` — but it is the only list
# the fuzzy fragment rule in `_is_burst_label` is allowed to match against.
BURST_LABELS = [
    "low mumble", "ahem", "contented sigh", "surprised gasp", "chuckle",
    "breathy giggle", "childlike giggle", "wistful sigh", "exhausted groan",
    "sharp inhale", "resonant hum", "scream", "yawn", "deep breath", "soft hum",
    "exasperated sigh", "cackle", "shriek", "coughing", "mournful wail",
    "growl", "purr",
]
_BURST_RE = re.compile(r"\(([^)]*?),\s*([0-9]*\.?[0-9]+)\s*(?:s|sec|seconds?)\)",
                       re.I)
_PAUSE_RE = re.compile(r"\[\s*(?:([0-9]*\.?[0-9]+)\s*(?:s|sec|seconds?)?\s*)?"
                       r"(?:pause|beat)\s*\]", re.I)
_DUR_RE = re.compile(r"\[\s*([0-9]*\.?[0-9]+)\s*(?:s|sec|seconds?)?\s*duration\s*\]", re.I)
_CUE_RE = re.compile(r"\(([^)]*)\)")


_VOCAB_CACHE = {}


def burst_vocabulary(root=None):
    """Every label that counts as a vocal burst rather than a delivery direction.

    Resolution order, first source that yields anything wins for the *wiki* half;
    the hard-coded core is always unioned in on top:

      1. `<SKILLS_DIR>/patterns/vb-<label>.md` — one page per label.  This is the
         authoritative space: the union of the classes callers ask for, the
         labels the detector can emit, and the members of the 23-group scheme.
      2. `<SKILLS_DIR>/VOCAL_BURSTS.md` — the recipe/never/weak tables, via
         `skills.py`, for the case where the pages are absent but the file is not.
      3. `BURST_LABELS` alone, when the skills directory is not on disk at all.

    Sourcing it rather than transcribing it is the point: `skills.py` already
    reads this directory to decide which sounds to OFFER the director, and a
    second hand-maintained list would drift away from the first.  A label the
    director is offered and then writes must not be silently re-read as an
    instruction about how to speak, which is what happens to any label this
    function does not return (see `parse`).

    Labels are returned in both spellings — `sharp_inhale` and `sharp inhale` —
    because the wiki keys on underscores and the director writes spaces.
    """
    key = root or getattr(config, "SKILLS_DIR", "")
    if key in _VOCAB_CACHE:
        return _VOCAB_CACHE[key]
    labels = set()
    try:
        import os
        pat = os.path.join(key, "patterns")
        if os.path.isdir(pat):
            for fn in os.listdir(pat):
                if fn.startswith("vb-") and fn.endswith(".md"):
                    labels.add(fn[3:-3].strip().lower())
        if not labels:
            import skills as _sk
            s = _sk.load()
            if s is not None and s.ok:
                labels |= set(s.recipes) | set(s.never) | set(s.weak)
    except Exception as e:                       # never break a turn over this
        print(f"[timed_script] burst vocabulary fell back to the core list: {e}",
              flush=True)
    vocab = set(BURST_LABELS)
    for lab in labels:
        vocab.add(lab)
        vocab.add(lab.replace("_", " "))
    _VOCAB_CACHE[key] = vocab
    return vocab


def burst_length(label):
    """(cleaned label, seconds) for a label that may carry short/long."""
    m = _LEN_WORD.match(str(label or ""))
    if not m:
        return str(label or "").strip(), BURST_DEFAULT
    word = m.group(1).lower()
    secs = BURST_SHORT if word in ("short", "brief", "quick", "tiny") else BURST_LONG
    return _LEN_WORD.sub("", str(label), count=1).strip(), secs


def _is_burst_label(txt):
    """Is this round bracket a vocal burst, or a direction about how to speak?

    Two tests, and only the first one saw the widened vocabulary.  An exact match
    against `burst_vocabulary()` is a burst.  The older fuzzy rule — a fragment of
    at most four words containing the last word of a known label, so that "a soft
    chuckle" lands — still matches against the 22-label core ONLY.  Running it
    across all 117 wiki labels would start reading ordinary directions as sounds:
    "(spitting the words out)" contains `spitting`, "(panting after the stairs)"
    contains `panting`.  Widening the exact half is free; widening the fuzzy half
    is not.
    """
    t = txt.strip().lower().rstrip(".")
    if t in burst_vocabulary():
        return True
    # a short fragment naming a known burst word counts too ("a soft chuckle")
    return len(t.split()) <= 4 and any(b.split()[-1] in t for b in BURST_LABELS)


def _seconds_for(words, speed=1.0):
    """How long this many words should take, in the model's own units."""
    per = config.TIMED_FRAMES_PER_WORD / config.FRAME_RATE   # seconds per word
    return max(0.4, words * per / max(speed, 0.1))


def parse(script):
    """Split the director's script into ordered items.

    Returns a list of ('pause', secs) | ('burst', label, secs) |
    ('direction', text) | ('speech', text).
    """
    # A duration the DIRECTOR wrote is kept, not stripped.  It used to be
    # deleted here and recomputed from the word count, so a director asked to
    # slow a melancholy line down was writing into a field the server threw
    # away.  `render` clamps it; see `_seconds_for`.
    s = str(script or "")
    # A duration written in ROUND brackets is a direction by the format's own
    # rule, so it was read as prose and then dropped, and the director's pacing
    # went with it.  Measured on the first three turns after the pacing rules
    # were added: every duration came back as "(4.8 seconds duration)".  Move
    # it rather than lose it.
    s = re.sub(r"\(\s*([0-9]*\.?[0-9]+)\s*(?:s|sec|seconds?)?\s+duration\s*\)",
               r"[\1 seconds duration]", s, flags=re.I)
    items, pos = [], 0
    marks = []
    for m in _DUR_RE.finditer(s):
        try:
            marks.append((m.start(), m.end(),
                          ("duration", float(m.group(1)))))
        except (TypeError, ValueError):
            marks.append((m.start(), m.end(), ("direction", "")))
    for m in _PAUSE_RE.finditer(s):
        marks.append((m.start(), m.end(), ("pause", float(m.group(1) or PAUSE_DEFAULT))))
    for m in _BURST_RE.finditer(s):
        secs = min(max(float(m.group(2)), BURST_MIN), BURST_MAX)
        marks.append((m.start(), m.end(), ("burst", m.group(1).strip(), secs)))
    taken = [(a, b) for a, b, _ in marks]
    for m in _CUE_RE.finditer(s):
        if any(a <= m.start() < b for a, b in taken):
            continue
        body = m.group(1).strip()
        if _is_burst_label(body):
            lbl, secs = burst_length(body)
            kind = ("burst", lbl, secs)
        else:
            kind = ("direction", body)
        marks.append((m.start(), m.end(), kind))
    marks.sort()
    for a, b, kind in marks:
        chunk = s[pos:a]
        if chunk.strip():
            items.append(("speech", chunk.strip()))
        items.append(kind)
        pos = b
    if s[pos:].strip():
        items.append(("speech", s[pos:].strip()))
    return items


def _split_sentences(text):
    out = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", text) if p.strip()]
    return out or ([text.strip()] if text.strip() else [])


def render(script, speed=1.0, budget_frames=None):
    """Return (tagged_script, frames, plain_text).

    `tagged_script` goes into both SCRIPT: and Text:, byte-identical, and
    `frames` into Tokens: — the sum of every number in the script.
    """
    items = parse(script)
    seq = []          # (kind, payload, seconds)
    pending_dir = None
    want_secs = None          # a duration the director asked for
    for it in items:
        if it[0] == "duration":
            want_secs = it[1]
            continue
        if it[0] == "pause":
            seq.append(("pause", None, max(it[1], PAUSE_MIN)))
        elif it[0] == "burst":
            if pending_dir:
                seq.append(("direction", pending_dir, 0.0))
                pending_dir = None
            seq.append(("burst", it[1], it[2]))
        elif it[0] == "direction":
            pending_dir = it[1]
        else:
            for sent in _split_sentences(it[1]):
                words = len([w for w in sent.split() if w.strip()])
                if not words:
                    continue
                secs = _seconds_for(words, speed)
                if want_secs is not None:
                    # Honour it, but only as far as the model can carry it.
                    # Measured on a fixed eight-word line, four seeds a cell,
                    # word error against the intended text:
                    #   1.0x 0.00   1.5x 0.00   2.0x 0.06   2.5x 0.50   3.0x 0.75
                    # This model spends whatever time it is given, so past about
                    # 1.5x the budget stops being silence and becomes invented
                    # words.  Extra slowness belongs in pauses, where silence
                    # stays silence.
                    lo, hi = secs * 0.6, secs * 1.5
                    got = min(max(want_secs, lo), hi)
                    if abs(got - want_secs) > 0.05:
                        print(f"[timed] asked for {want_secs:.1f}s on "
                              f"{words} words, using {got:.1f}s "
                              f"(natural {secs:.1f}s)", flush=True)
                    secs = round(got, 1)
                    want_secs = None
                if pending_dir:
                    seq.append(("direction", pending_dir, 0.0))
                    pending_dir = None
                # anything past 12 s was split again in training
                while secs > SEG_MAX:
                    seq.append(("speech", sent, SEG_MAX))
                    secs -= SEG_MAX
                    sent = ""
                seq.append(("speech", sent, round(secs, 1)))
    if pending_dir:                       # a direction with nothing after it
        seq.append(("direction", pending_dir, 0.0))
    if not any(k == "speech" for k, _, _ in seq):
        return "", 0, ""

    # a gap before the first word and after the last, as the format expects
    if seq and seq[0][0] != "pause":
        seq.insert(0, ("pause", None, LEAD_PAUSE))
    if TAIL_PAUSE > 0 and seq and seq[-1][0] != "pause":
        seq.append(("pause", None, TAIL_PAUSE))
    while seq and seq[-1][0] == "pause":
        seq.pop()                      # never end on silence the model must fill

    total = round(sum(s for _, _, s in seq), 1)
    frames = int(round(total * config.FRAME_RATE))
    if budget_frames:
        # scale the speech to the requested budget, leaving pauses and bursts
        # alone — they are performance, not padding
        fixed = sum(s for k, _, s in seq if k != "speech")
        spoken = total - fixed
        want = budget_frames / config.FRAME_RATE - fixed
        if spoken > 0.1 and want > 0.1:
            f = want / spoken
            seq = [(k, p, round(s * f, 1) if k == "speech" else s) for k, p, s in seq]
            total = round(sum(s for _, _, s in seq), 1)
            frames = int(round(total * config.FRAME_RATE))

    out, plain = [], []
    for kind, payload, secs in seq:
        if kind == "pause":
            out.append(f"[{secs:.1f} seconds pause]")
        elif kind == "burst":
            out.append(f"({payload}, {secs:.1f} seconds)")
        elif kind == "direction":
            out.append(f"({payload})")
        else:
            out.append(f"[{secs:.1f} seconds duration] {payload}")
            plain.append(payload)
    return " ".join(out), frames, " ".join(plain).strip()


def check(tagged, frames):
    """Do the printed numbers add up to the stated budget?"""
    tot = 0.0
    for m in re.finditer(r"\[([0-9]*\.?[0-9]+) seconds (?:duration|pause)\]", tagged):
        tot += float(m.group(1))
    for m in _BURST_RE.finditer(tagged):
        tot += float(m.group(2))
    return round(tot, 2), frames, abs(tot * config.FRAME_RATE - frames) <= 1.0


# Carried verbatim at the end of every GENERAL, on every turn and every path.
# These are the words the training corpus was captioned with; a paraphrase is a
# different conditioning signal, so this block is fixed rather than written.
STANDING = ("the same speaker continues without interruption: identical voice, "
            "identical person, same microphone and same room. genuine and "
            "spontaneous, like a real person in a real moment, not acted. "
            "pristine high-quality studio recording, no background noise.")


def general_line(general, seconds, lang_code, reads_as=None):
    """Fold GENERAL into the one-line shape the format's own example uses.

    The guide's GENERAL is a single compact line that ends with the emotions and
    then the clip's length and language — "…; reads as bitterness, contempt;
    9.5s, EN."  Ours had grown into several standing sentences about register,
    continuity and recording quality, which is prose the model has to wade
    through before it reaches the part that changes between turns.
    """
    g = " ".join(str(general or "").split())
    # The standing sentences — continuity, genuineness, room — are carried in
    # FULL and identically on every turn.  They were compressed to short
    # equivalents ("same speaker throughout", "genuine, not acted") to keep the
    # line close to the guide's one-line example, and that compression is now
    # reverted deliberately: these exact words are what the corpus was captioned
    # with, and a paraphrase of a caption is a different conditioning signal
    # from the caption.  Whatever the director wrote in their place is dropped
    # and replaced, so the block cannot drift turn to turn.
    keep_short = []
    for sent in re.split(r"(?<=[.;])\s+", g):
        low = sent.lower()
        if ("the way someone actually talks" in low
                or "close conversational volume" in low):
            keep_short.append("close conversational volume, unforced")
        elif ("the same speaker continues" in low
                or "genuine and spontaneous" in low
                or "studio recording" in low
                or "background noise" in low):
            continue                      # replaced verbatim below
        elif sent.strip(" .;"):
            keep_short.append(sent.strip(" .;"))
    seen, g2 = set(), []
    for k in keep_short:
        if k.lower() not in seen:
            seen.add(k.lower())
            g2.append(k)
    g = "; ".join(g2).rstrip(" .;")
    if reads_as and "reads as" not in g.lower():
        names = reads_as if isinstance(reads_as, str) else ", ".join(reads_as)
        names = names.replace("_", " ").lower()
        g += f"; reads as {names}"
    g = g.strip(" ;.,")            # an empty GENERAL left a leading "; "
    return (f"{g}. {STANDING} {seconds:.1f}s, {lang_code}." if g
            else f"{STANDING} {seconds:.1f}s, {lang_code}.")


def neutralise(tagged):
    """The same timed script with its delivery directions removed, and nothing else changed.

    This is the "unconditional" half of a classifier-free-guidance pair.  Only the affect
    leaves: every round bracket WITHOUT a number is a direction and goes; every round bracket
    WITH one is a vocal burst and stays; the square brackets carrying durations and pauses
    stay; the words are untouched.  The arithmetic is therefore identical -- a direction is
    zero seconds long -- so both branches carry the same `Tokens` and the guidance difference
    is about affect and nothing else.

    The format's own rule is what makes this safe: square bracket = seconds, round bracket
    *with* a number = a burst, round bracket *without* = a direction (see this module's
    header).  A cue that picked up a number would already have been read as a burst by
    `parse`, so nothing that survives here was ever a direction.

    The neutralised prompt is in distribution: 20 % of the CFG-DPO corpus had its instruction
    words removed and 15-30 % of every supervised round rendered scripts without directions.
    """
    s = str(tagged or "")
    keep = [(m.start(), m.end()) for m in _BURST_RE.finditer(s)]
    cut = []
    for m in _CUE_RE.finditer(s):
        if any(a <= m.start() < b for a, b in keep):
            continue
        cut.append((m.start(), m.end()))
    if not cut:
        return " ".join(s.split())
    out, pos = [], 0
    for a, b in cut:
        out.append(s[pos:a])
        pos = b
    out.append(s[pos:])
    return " ".join("".join(out).split())


def burst_onsets(tagged):
    """[(label, start_seconds), ...] for every burst in a RENDERED script.

    The burst reward has to look at the audio where the sound was asked for,
    not at the whole clip — localised scoring measured 0.2082 against 0.1907.
    Walking the rendered tags in order is exact: every number in the script is
    a duration the model was told to fill, so the clock is the sum of what
    came before.
    """
    t, out = 0.0, []
    for m in re.finditer(
            r"\[\s*([0-9]*\.?[0-9]+)\s*seconds?\s+(pause|duration)\s*\]"
            r"|\(([^),]+),\s*([0-9]*\.?[0-9]+)\s*seconds?\s*\)",
            str(tagged or ""), re.I):
        if m.group(1):
            t += float(m.group(1))
        else:
            out.append((m.group(3).strip(), round(t, 3)))
            t += float(m.group(4))
    return out
