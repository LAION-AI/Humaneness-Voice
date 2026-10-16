"""Keep every bracket in English, whatever language is being spoken.

The prompt has told the director to write its cues in English since the corpus
was first described, and on German turns it writes them in German anyway.  That
is not cosmetic.  Measured on one German line, four seeds a cell, word error
against the intended text:

    German cues  + German GENERAL    median 0.267   2 of 4 unusable
    German cues  + English GENERAL   median 0.389   3 of 4 unusable
    English cues + German GENERAL    median 0.033   0 of 4
    English cues + English GENERAL   median 0.000   0 of 4

The brackets dominate: translating only the cues, with the German words
untouched, took the line from babble to clean.  The corpus is captioned in
English -- its German lines read "Das zerreisst einen einfach, weisst du?
(relief sigh)" -- so a German cue is outside the distribution the voice model
learned, and it takes the words down with it.

So the server repairs what the prompt cannot enforce.  The lexicon covers the
vocabulary the prompt prescribes (the adverb scale, the emotion adjectives, the
handful of delivery verbs), which is what the director actually writes; anything
left over goes to the language model in one batched call.
"""
import re

import retrieval

# The prescribed adverb scale, then the words the directions are built from.
# Longest first at match time, so "völlig ungehemmt" wins over "völlig".
LEX = {
    # intensity
    "kaum": "barely", "leicht": "faintly", "ein wenig": "just a little",
    "deutlich": "clearly", "klar": "clearly", "hörbar": "noticeably",
    "unverkennbar": "unmistakably", "stark": "strongly", "intensiv": "intensely",
    "sehr": "very", "tief": "deeply", "überwältigend": "overwhelmingly",
    "extrem": "extremely", "völlig": "utterly", "vollkommen": "completely",
    "weiterhin": "still", "immer noch": "still", "wieder": "again",
    # feeling
    "verängstigt": "terrified", "ängstlich": "afraid", "panisch": "panicked",
    "entsetzt": "horrified", "erschrocken": "startled", "wütend": "angry",
    "zornig": "furious", "traurig": "sad", "amüsiert": "amused",
    "belustigt": "amused", "erleichtert": "relieved", "besorgt": "concerned",
    "verzweifelt": "desperate", "empört": "outraged", "angeekelt": "disgusted",
    "überrascht": "surprised", "stolz": "proud", "beschämt": "ashamed",
    "zärtlich": "tender", "liebevoll": "affectionate", "kalt": "cold",
    "bitter": "bitter", "spöttisch": "mocking", "trocken": "dry",
    "nachdenklich": "thoughtful", "ruhig": "calm", "beschützend": "protective",
    "dringlich": "urgent", "erschöpft": "exhausted", "hoffnungsvoll": "hopeful",
    # containment, the fork that matters
    "völlig ungehemmt": "letting it out, not hiding it",
    "ungehemmt": "letting it out, not hiding it",
    "zurückgehalten": "held in and only leaking at the edges",
    "unterdrückt": "fought down rather than shown",
    "beherrscht": "contained", "kontrolliert": "controlled",
    # manner
    "die stimme bricht": "the voice breaking",
    "am rand eines weiteren schreis": "at the edge of another scream",
    "atemlos": "breathless", "rau": "ragged", "heiser": "hoarse",
    "flüsternd": "whispering", "leiser": "quieter", "lauter": "louder",
    "schneller": "faster", "langsamer": "slower", "zitternd": "trembling",
    "brüchig": "breaking", "gepresst": "strained", "tonlos": "flat",
    "durch die panik gezwungen": "forced through panic",
    "die dringlichkeit übernimmt": "urgency taking over",
    "die worte": "the words", "die stimme": "the voice",
    "und": "and", "aber": "but", "mit": "with", "ohne": "without",
    "noch": "still", "nicht": "not", "sehr leise": "very quietly",
}
_ORDER = sorted(LEX, key=len, reverse=True)

_BRACKET = re.compile(r"\(([^)]*)\)")


def _is_direction(body):
    """A round bracket WITHOUT a number is an instruction; with one it is a
    burst, and burst labels are already English by construction."""
    return not re.search(r"[0-9]", body)


# Words that only appear in an English cue.  A bracket built from these is
# already right; anything else on a German turn is sent to be rewritten.
_EN = re.compile(
    r"\b(barely|faintly|slightly|clearly|plainly|noticeably|unmistakably|"
    r"strongly|intensely|very|deeply|overwhelmingly|extremely|utterly|"
    r"completely|still|letting|held|hiding|fought|shown|edges|voice|words|"
    r"breath|quieter|louder|the|and|with|not|out|in|at|of|a)\b", re.I)


def _looks_english(body):
    """Cheap and deliberately strict: a cue is English when it is built out of
    English words and carries no German marker.  `retrieval.looks_german` is
    tuned for whole sentences and returns False on a short phrase like
    "weiterhin intensiv entsetzt" — which is exactly the case that has to be
    caught here, so it cannot be the only test."""
    if re.search(r"[äöüßÄÖÜ]", body):
        return False
    words = re.findall(r"[\w']+", body)
    if not words:
        return True
    hits = len(_EN.findall(body))
    return hits >= max(2, len(words) // 3)


def german_cues(script, force=False):
    """The direction brackets that are not written in English."""
    out = []
    for m in _BRACKET.finditer(str(script or "")):
        b = m.group(1)
        if not _is_direction(b):
            continue
        if retrieval.looks_german(b) or (force and not _looks_english(b)):
            out.append(b)
    return out


def _lex(body):
    """Translate with the lexicon.  Returns (text, fully_covered)."""
    s = body
    for de in _ORDER:
        s = re.sub(r"(?<![\w])" + re.escape(de) + r"(?![\w])", LEX[de], s,
                   flags=re.I)
    # anything left with a German marker is not covered
    return s, not retrieval.looks_german(s)


def englishise(script, translate=None, force=False):
    """Rewrite German direction brackets into English.

    `translate` is an optional callable taking a list of strings and returning
    the same number of English strings — used only for the brackets the lexicon
    cannot fully cover.  Without it, the lexicon's partial result is used, which
    is still closer to the corpus than the German original.
    """
    script = str(script or "")
    spans, bodies = [], []
    for m in _BRACKET.finditer(script):
        b = m.group(1)
        if not _is_direction(b):
            continue
        if retrieval.looks_german(b) or (force and not _looks_english(b)):
            spans.append(m.span(1))
            bodies.append(b)
    if not spans:
        return script, 0, []
    # The language model is the primary path, not a fallback: the lexicon on
    # free prose produced half-translated cues ("intensely von überwältigender
    # Angst erfasst"), which is worse than either language on its own.  The
    # lexicon is only what happens when no model is reachable.
    if translate:
        done, hard = list(bodies), list(range(len(bodies)))
    else:
        done, hard = [], []
        for b in bodies:
            done.append(_lex(b)[0])
    if hard and translate:
        try:
            got = translate([bodies[i] for i in hard])
            for j, i in enumerate(hard):
                if j < len(got) and got[j]:
                    done[i] = str(got[j]).strip().strip("()")
        except Exception as e:
            print(f"[cues] translation failed, keeping the lexicon pass: "
                  f"{type(e).__name__}: {e}", flush=True)
    out, at = [], 0
    for (a, b), new in zip(spans, done):
        out.append(script[at:a])
        out.append(new)
        at = b
    out.append(script[at:])
    return "".join(out), len(spans), bodies


def englishise_general(general, translate=None, force=False):
    """The same for the delivery clause of GENERAL.

    Worth less than the brackets — English cues with a German GENERAL still
    measured 0.033 — but it is the same defect and the same fix.
    """
    g = str(general or "")
    parts = re.split(r"(?<=[.;])\s+", g)
    hit = [i for i, p in enumerate(parts)
           if retrieval.looks_german(p) or (force and p.strip()
                                            and not _looks_english(p))]
    if not hit:
        return g, 0
    hard = list(hit)
    if not translate:
        for i in hit:
            parts[i] = _lex(parts[i])[0]
        hard = []
    if hard and translate:
        try:
            got = translate([parts[i] for i in hard])
            for j, i in enumerate(hard):
                if j < len(got) and got[j]:
                    parts[i] = str(got[j]).strip()
        except Exception as e:
            print(f"[cues] general translation failed: {e}", flush=True)
    return " ".join(parts), len(hit)


async def englishise_async(script, translate, force=False):
    """`englishise` with an awaitable translator."""
    script = str(script or "")
    spans, bodies = [], []
    for m in _BRACKET.finditer(script):
        b = m.group(1)
        if not _is_direction(b):
            continue
        if retrieval.looks_german(b) or (force and not _looks_english(b)):
            spans.append(m.span(1))
            bodies.append(b)
    if not spans:
        return script, 0, []
    try:
        done = await translate(bodies)
    except Exception as e:
        print(f"[cues] translation failed, using the lexicon: {e}", flush=True)
        done = [_lex(b)[0] for b in bodies]
    if len(done) != len(bodies):
        print(f"[cues] translator returned {len(done)} for {len(bodies)}; "
              f"keeping the originals", flush=True)
        return script, 0, bodies
    out, at = [], 0
    for (a, b), new in zip(spans, done):
        out.append(script[at:a])
        out.append(str(new).strip().strip("()"))
        at = b
    out.append(script[at:])
    return "".join(out), len(spans), bodies


async def englishise_general_async(general, translate, force=False):
    g = str(general or "")
    parts = re.split(r"(?<=[.;])\s+", g)
    hit = [i for i, p in enumerate(parts)
           if p.strip() and (retrieval.looks_german(p)
                             or (force and not _looks_english(p)))]
    if not hit:
        return g, 0
    try:
        got = await translate([parts[i] for i in hit])
    except Exception as e:
        print(f"[cues] general translation failed: {e}", flush=True)
        return g, 0
    if len(got) == len(hit):
        for j, i in enumerate(hit):
            parts[i] = str(got[j]).strip()
        return " ".join(parts), len(hit)
    return g, 0
