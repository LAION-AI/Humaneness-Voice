"""The ten voice profiles: who the voice can be.

Each profile is a speaker identity, and it comes in two parts that have to agree:

  anchor   a real recording of that speaker, used as the reference clip
  lora     an adapter trained on that speaker across the whole expressive range

The two are complementary, not redundant.  The anchor tells the model what the
voice sounds like right now; the adapter moves the model itself towards that
speaker so the identity survives the emotions.  The upstream card is blunt about
the trade: "identity is bought with expressiveness" — the adapters raise speaker
similarity well above the base model and lower genuineness — so the dose stays a
dial rather than a fixed choice.

Ranks are not a menu: the top level of each voice folder is the rank that repo's
own ablation shipped for that voice (rank 4 for nine of the ten, rank 8 for
emolia_c0542), and that is what is loaded here.
"""
import glob
import json
import os

import config

# k325_age3_bg1 is the corpus speaker the whole reference matrix was rendered
# with, so it is the only one that also has 832 conditioned clips to retrieve.
CORPUS_VOICE = "k325_age3_bg1"

# Fallback descriptions; voice.json overrides where it exists.
_FALLBACK = {
    "k325_age3_bg1": ("Velvet Sage Baritone",
                      "a man's voice in his early fifties, a warm unhurried "
                      "baritone, dark and resonant with a soft gravel low down"),
    "k10_age3_bg1": ("Profile k10", "an adult voice, even and clear"),
    "k91_age5_bg0": ("Profile k91", "an older adult voice, settled and grounded"),
    "k395_age3_bg1": ("Profile k395", "an adult voice, bright and forward"),
    "anime_088": ("Anime 088", "a bright, animated, highly expressive voice"),
    "mediathek_0184": ("Mediathek 0184",
                       "a broadcast-trained German voice, clear and measured"),
    "emolia_c0542": ("Emolia c0542", "an expressive conversational voice"),
    "emolia_c1682": ("Emolia c1682", "an expressive conversational voice"),
    "emolia_c1699": ("Emolia c1699", "an expressive conversational voice"),
    "emolia_c2570": ("Emolia c2570", "an expressive conversational voice"),
}


def _card(vid):
    """Speaker card from voice.json, where the repo ships one."""
    p = os.path.join(config.PROFILE_REFS, "pilot", vid, "voice.json")
    if not os.path.exists(p):
        return {}
    try:
        d = json.load(open(p, encoding="utf-8"))
        # the card nests the interesting half under "identity"
        return {**d, **(d.get("identity") or {})}
    except Exception:
        return {}


_TIMBRE = ("gravelly", "raspy", "breathy", "smooth", "silky", "warm", "bright",
           "dark", "husky", "wispy", "velvet", "rich", "deep", "low", "clear",
           "nasal", "denasal", "thin", "resonant", "steady", "soft", "rough",
           "weary", "conspiratorial", "measured", "commanding")


def _short(card, vid):
    """A compact label: gender, rough age, one timbre word."""
    g = (card.get("gender") or "").strip().lower()
    g = {"male": "m", "man": "m", "female": "f", "woman": "f"}.get(g, "")
    age = (card.get("age") or "").strip()
    # "Late 40s to 60s" -> "40s-60s";  "Early to mid-20s" -> "20s"
    import re as _re
    d = _re.findall(r"(\d{2})s", age)
    if len(d) >= 2:
        agep = f"{d[0]}s-{d[-1]}s"
    elif d:
        agep = f"{d[0]}s"
    else:
        agep = "adult" if age else ""
    blob = ((card.get("tagline") or "") + " " + (card.get("description") or "")).lower()
    tim = next((w for w in _TIMBRE if w in blob), "")
    bits = [x for x in (g, agep, tim) if x]
    return ", ".join(bits)


_TRAITS = None


def traits():
    """Gender, age and timbre measured from each profile's own recordings.

    The shipped cards get the gender wrong for three of the ten voices, and that
    error does not stay cosmetic: the card text becomes "a woman's voice" in the
    GENERAL block, where it argues with the reference audio.  profile_traits.py
    derives these from the corpus's own VoiceNet predictions instead.
    """
    global _TRAITS
    if _TRAITS is None:
        p = os.path.join(config.RETRIEVAL_DIR, "profile_traits.json")
        try:
            _TRAITS = json.load(open(p, encoding="utf-8"))
        except Exception:
            _TRAITS = {}
    return _TRAITS


_AGE_WORDS = {"20s": "in their twenties", "30s": "in their thirties",
              "40s-50s": "middle-aged", "60s": "in their sixties",
              "70s+": "elderly"}


def _describe(vid, card):
    """One sentence for the GENERAL block: who this speaker is."""
    bits = []
    t = traits().get(vid) or {}
    g = (card.get("gender") or "").strip().lower()
    age = (card.get("age") or "").strip()
    if t:
        # measured beats declared
        g = {"f": "female", "m": "male", "n": ""}.get(t.get("sex"), g)
        age = _AGE_WORDS.get(t.get("age"), age)
    if g in ("male", "man"):
        bits.append("a man's voice")
    elif g in ("female", "woman"):
        bits.append("a woman's voice")
    elif g:
        bits.append(f"a {g} voice")
    if age:
        bits.append(age if age.startswith(("in their", "middle", "elderly"))
                    else f"aged {age}")
    acc = (card.get("accent") or "").strip()
    lang = (card.get("language") or "").strip()
    if acc:
        bits.append(f"speaking with {acc}")
    elif lang:
        bits.append(f"a native {lang} speaker")
    tag = (card.get("tagline") or card.get("description") or "").strip()
    if tag:
        bits.append(tag.rstrip("."))
    if bits:
        return ", ".join(bits) + "."
    return _FALLBACK.get(vid, (vid, "a natural speaking voice"))[1] + "."


def discover():
    """Every profile that has both an anchor and an adapter on disk."""
    out = {}
    for d in sorted(glob.glob(os.path.join(config.PROFILE_LORAS, "*"))):
        vid = os.path.basename(d)
        lora = os.path.join(d, "adapter_model.safetensors")
        if not os.path.exists(lora):
            continue
        anchor = None
        for cand in (os.path.join(config.PROFILE_REFS, "pilot", vid, "reference.wav"),
                     os.path.join(config.PROFILE_REFS, "pilot", vid, "reference.mp3")):
            if os.path.exists(cand):
                anchor = cand
                break
        if anchor is None and vid == CORPUS_VOICE:
            anchor = os.path.join(config.REF_DIR, "reference", "reference_target.mp3")
        if anchor is None:
            continue
        card = _card(vid)
        rec = {}
        rp = os.path.join(d, "RECOMMENDED.json")
        if os.path.exists(rp):
            try:
                rec = json.load(open(rp, encoding="utf-8"))
            except Exception:
                pass
        out[vid] = {
            "id": vid,
            "name": (traits().get(vid) or {}).get("name")
                    or card.get("name") or _FALLBACK.get(vid, (vid,))[0],
            "short": (traits().get(vid) or {}).get("short") or _short(card, vid),
            "accent": (card.get("accent") or "").strip(),
            "identity": _describe(vid, card),
            "anchor": anchor,
            "lora": f"{config.PROFILE_LORA_KIND}:{vid}",
            "rank": rec.get("rank"),
            "has_conditions": vid == CORPUS_VOICE,
        }
    return out


def listing(profiles):
    return [{k: p[k] for k in ("id", "name", "short", "accent", "identity",
                               "rank", "has_conditions")}
            for p in profiles.values()]
