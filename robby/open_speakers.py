"""Optional wake-free session enrollment and explicit voice-variant claims.

These are user-asserted conversational identities, never authentication. A
spoken claim can be impersonated. No biometric persistence is added.
"""
import re
import numpy as np
from speaker_turns import self_name, NOT_NAMES, words

CLAIM=re.compile(r"\b(?:(?:ich\s+bin['’]?s|ich\s+bin\s+es)(?:\s+doch)?|it['’]?s\s+me|it\s+is\s+me)\s*[,!:]?\s+([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]{1,35})\b",re.I)

def claimed_name(text):
    match=CLAIM.search(text)
    if not match:return None
    name=match.group(1).strip("'’-")
    if name.casefold() in NOT_NAMES:return None
    return name[0].upper()+name[1:]

def eligible(track,text,coverage,background,playing,overlaps,memory):
    if playing or background or coverage<.65 or track.speech_s<1.28 or len(track.fast)<4:return False
    if len(words(text))<2 or not text.strip():return False
    if any(a<track.last_speech and b>track.start for a,b in overlaps):return False
    coherence=float(np.mean([memory.similarity(v,track.vector()) for v in track.fast]))
    return coherence>=max(.75,memory.threshold-.05)

def greeting_context(context,user):
    # Once per provisional person, no repetitive requests for a name.
    if not user.named and not getattr(user,'asked_name',False):
        context['needs_name']=True;user.asked_name=True
    return context
