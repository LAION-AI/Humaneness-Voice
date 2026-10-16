"""The release's exact trained packing surfaces, with an affect-only CFG pair."""
import json
import math
import re
import sys

import settings

sys.path.append(str(settings.PARENT))
import timed_script

FORMATS = {"structured", "caption", "tags", "transcript"}
REFERENCE_IDENTITY = "Use the same speaker as the reference audio. Preserve that speaker's voice identity."


def performance_direction(text):
    """Do not ask the style retriever/reward to score a reference instruction."""
    return str(text or '').replace(REFERENCE_IDENTITY, '').strip()


def plain_script(script):
    return " ".join(re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", script).split())


def neutral_script(script):
    """Remove directions, retaining even manually written untimed bursts."""
    def cue(match):
        body = match[1].strip()
        timed = re.search(r"\d+(?:\.\d+)?\s*(?:s|sec|seconds?)\b", body)
        return match[0] if timed or timed_script._is_burst_label(body) else ""
    return " ".join(re.sub(r"\(([^)]*)\)", cue, script).split())


def parse_literal(prompt, fmt):
    """Derive exact spoken words without sending a manual prompt through Luna."""
    if fmt == "structured":
        m = re.fullmatch(r"GENERAL:\s*(.*?)\nSCRIPT:\s*\n?(.*)", prompt.strip(), re.S)
        if not m:
            raise ValueError("GENERAL: und SCRIPT: fehlen im manuellen Prompt.")
        return {"general": m[1].strip(), "script": m[2].strip(), "caption": "", "tags": [], "text": plain_script(m[2])}
    m = re.search(r"(?:^|\n)TRANSCRIPT:\s*(.*)\Z", prompt.strip(), re.S)
    if not m:
        raise ValueError('TRANSCRIPT: "Wortgetreuer Text" fehlt.')
    try:
        text = json.loads(m[1])
    except json.JSONDecodeError as exc:
        raise ValueError("TRANSCRIPT muss eine mit doppelten Anführungszeichen zitierte Zeichenkette sein.") from exc
    if not isinstance(text, str):
        raise ValueError("TRANSCRIPT muss Text sein.")
    prefix = prompt[:m.start()].strip()
    if fmt != "transcript" and not prefix.startswith("CAPTION:"):
        raise ValueError("CAPTION: fehlt im manuellen Prompt.")
    return {"general": prefix.removeprefix("CAPTION:").strip(), "caption": prefix.removeprefix("CAPTION:").strip(), "script": "", "tags": [], "text": text}


def build(turn, fmt, language="en", seconds=None, literal=None, reference_identity=False):
    if fmt not in FORMATS:
        raise ValueError("Unbekanntes Prompt-Format.")
    text = str(turn.get("text", "")).strip()
    if not text:
        raise ValueError("Der gesprochene Text ist leer.")
    if len(text) > 1600 or len(text.split()) > 120:
        raise ValueError("Bitte höchstens 120 Wörter pro Take verwenden.")
    # Keep the actual spoken language separate from the English directions.
    lang = "de" if str(language).lower().startswith(("de", "ger")) else "en"
    natural = max(2.0, len(text.split()) * 0.36)
    tagged = ""
    edge_script = ""
    if fmt == "structured":
        tagged, frames, spoken = timed_script.render(turn.get("script") or text)
        if " ".join(spoken.split()) != " ".join(text.split()):
            raise ValueError("SCRIPT und gesprochenes Transkript stimmen nicht überein.")
        conditional = literal or f"GENERAL: {turn['general']}\nSCRIPT:\n{tagged}"
        # Retain timing, pauses, burst labels and reference in both branches.
        original = conditional.split("SCRIPT:", 1)[1].strip()
        # Legacy rendering adds 0.3s padding at both edges. Only cues actually
        # authored by Luna/the user protect edges from unwanted extra audio.
        edge_script = original if literal else (turn.get("script") or text)
        unconditional = f"GENERAL: plain neutral delivery at a normal conversational volume\nSCRIPT:\n{neutral_script(original)}"
        if literal:
            tagged = original
        natural = max(natural, frames / 12.5)
    else:
        quoted = json.dumps(text, ensure_ascii=False)
        if literal:
            conditional = literal.strip()
        elif fmt == "transcript":
            conditional = f"TRANSCRIPT: {quoted}"
        elif fmt == "tags":
            tags = ", ".join(str(t) for t in turn.get("tags", []))
            conditional = f"CAPTION: Tags: {tags}. Caption: {turn.get('caption') or turn['general']}\nTRANSCRIPT: {quoted}"
        else:
            conditional = f"CAPTION: {turn.get('caption') or turn['general']}\nTRANSCRIPT: {quoted}"
        # Unlike deleting Text, this keeps every literal transcript byte intact.
        transcript = conditional[conditional.index("TRANSCRIPT:"):]
        if fmt == "transcript":
            unconditional = transcript
        elif fmt == "tags":
            unconditional = f"CAPTION: Tags: neutral, conversational. Caption: Plain neutral delivery at a normal conversational volume.\n{transcript}"
        else:
            # Keep the caption surface in both branches, as in the published
            # Small CFG experiment. Deleting it also changes prompt format.
            unconditional = f"CAPTION: Voice and performance: plain neutral delivery at a normal conversational volume.\n{transcript}"
    # Opt-in/versioned for the live playground: frozen studies keep their exact
    # original CFG pair. Manual conditional prompts remain byte-for-byte intact.
    # Transcript-only has no direction channel and still has identical branches.
    separator = 'SCRIPT:' if fmt == 'structured' else 'TRANSCRIPT:'
    direction_prefix, divider, content = conditional.partition(separator)
    identity = reference_identity and fmt != 'transcript' and (literal is None or REFERENCE_IDENTITY in direction_prefix)
    if identity:
        marker = 'GENERAL: ' if fmt == 'structured' else 'CAPTION: '
        if literal is None:
            conditional = marker + REFERENCE_IDENTITY + ' ' + performance_direction(direction_prefix[len(marker):]) + '\n' + divider + content
        unconditional = marker + REFERENCE_IDENTITY + ' ' + unconditional[len(marker):]
    duration = float(seconds) if seconds is not None else natural
    frames = int(math.ceil(duration * 12.5))
    if not 12 <= frames <= settings.MAX_FRAMES:
        raise ValueError("Die Dauer muss zwischen 1 und 60 Sekunden liegen.")
    return {"prompt": conditional, "unconditional": unconditional, "text": text,
            "lang": lang, "frames": frames, "format": fmt, "tagged": tagged,
            "general": performance_direction(turn.get("general") or turn.get("caption", "")) if reference_identity else (turn.get("general") or turn.get("caption", "")), "edge_script": edge_script,
            **({"identity_cfg": bool(identity)} if reference_identity else {})}
