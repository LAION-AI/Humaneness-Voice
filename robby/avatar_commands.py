"""Conservative, validated screen-action commands, independent of LLM JSON.

Only direct requests use this path. Quotes, conditions and explanations stay
with normal conversation/tools. No motor/device commands are ever produced.
"""
import re
from avatar_actions import VARIANTS,explicit_spin,validate
from performance import ALIASES

_BLOCK=re.compile(r'\b(?:wenn|falls|if|example|beispiel|zitiere|quote|erklär\w*|explain|beschreib\w*|describe|warum|why)\b',re.I)
_SPEECH_ONLY=re.compile(r'^\s*(?:(?:(?:bitte|please)|(?:(?:kannst|könntest|würdest)\s+du|(?:can|could|would)\s+you)(?:\s+mir)?(?:\s+bitte|\s+please)?)\s+)?(?:sag(?:e|en)?|say|repeat|wiederhol\w*)\b',re.I)
_REQUEST=re.compile(r'\b(?:bitte|please|kannst\s+du|könntest\s+du|würdest\s+du|can\s+you|could\s+you|would\s+you)\b|^(?:(?:ja|okay|also|hey|robb[yi]|well|now|first|erst)\s*[,!:]?\s+){0,3}(?:mach\w*|schneid\w*|zieh\w*|zeig\w*|guck\w*|schau\w*|schiel\w*|nick(?:e|en)?|zwinker\w*|schüttel\w*|shake|roll\w*|augenroll\w*|verdreh\w*|dreh\w*|grins\w*|make|show|do|look|act|nod|wink|cross|spin|rotate)\b',re.I)
_FACE=re.compile(r'\b(?:grim+ass\w*|grimace\w*|(?:funny|silly|cute|goofy|crazy)\s+faces?|fischgesicht|(?:lustig\w*|komisch\w*|verrückt\w*)\s+gesicht)\b',re.I)
_EMOTIONS={slug:re.compile(r'(?<!\w)(?:'+re.escape(slug)+'|'+aliases+r')(?:e|en|es|er|em)?(?!\w)',re.I) for slug,aliases in ALIASES.items()}

def unquoted(text):
    # A quoted command is data, but a quoted spoken phrase must not suppress
    # an independently requested action: 'Mach eine Grimasse und sag "Hallo"'.
    return re.sub(r'"[^"\n]*"|“[^”\n]*”|„[^“\n]*“|«[^»\n]*»|(?<!\w)\x27[^\x27\n]*\x27(?!\w)',' ',text)

def silent_requested(text):
    return bool(re.search(r'\b(?:stumm|lautlos|ohne\s+(?:etwas\s+)?(?:zu\s+)?(?:sagen|sprechen)|silent(?:ly)?|without\s+(?:speaking|talking)|no\s+speech)\b',unquoted(text),re.I))

def command_intent(text):
    if not isinstance(text,str) or not 1<=len(text)<=1200:return False
    text=unquoted(text)
    if _BLOCK.search(text):return False
    if _SPEECH_ONLY.search(text):
        # "Say something funny AND make a face" has a real second action;
        # "Please say make a face" alone is only a request for words.
        later=re.split(r'\b(?:und|and|dann|then)\b',text,flags=re.I)[1:]
        return any(_REQUEST.search(clause.strip()) for clause in later)
    return bool(_REQUEST.search(text) or _FACE.fullmatch(text.strip(' .!?')))

def explicit_actions(text,variant_index=0):
    if not command_intent(text):return []
    text=unquoted(text)
    steps=[]
    pose_request=bool(re.search(r'\b(?:guck\w*|schau\w*|look|ausseh\w*|gesicht\w*|emotion\w*|ausdruck|expression|pose)\b',text,re.I))
    normalized=re.sub(r'\b(roll\w*|schau\w*|guck\w*|mach\w*|make|look|show|schiel\w*|shake|schüttel\w*)\s+(?:dann|then)\b',r'\1',text,flags=re.I)
    clauses=re.split(r'[,;.!?]+|\b(?:und|and|dann|then|danach|erst|first|sondern|but)\b',normalized,flags=re.I)
    for clause in clauses:
        t=clause.casefold().strip()
        if not t or _SPEECH_ONLY.search(t) or re.search(r'\b(?:nicht|kein\w*|never|not|don.t|no)\b',t):continue
        spin=explicit_spin(t)
        if spin:steps.extend(spin)
        elif re.search(r'\b(?:schiel\w*|cross\w*\s+(?:your\s+)?eyes)\b',t):
            steps.append({'kind':'grimace','variant':'crossed','duration_s':3.})
        elif _FACE.search(t):
            variant=next((v for v in VARIANTS if re.search(r'\b'+re.escape(v)+r'\b',t)),None)
            if variant is None and re.search(r'\bfisch\w*',t):variant='fish'
            amount=2 if re.search(r'\b(?:grimassen|grimaces|mehrere|verschiedene|several)\b',t) else 1
            for offset in range(amount):
                steps.append({'kind':'grimace','variant':variant or VARIANTS[(variant_index+offset)%len(VARIANTS)],'duration_s':3.})
        elif re.search(r'\b(?:augen\s*roll\w*|(?:roll\w*|verdreh\w*)\b.*\b(?:augen|eyes)|eye\s+roll)\b',t):
            steps.append({'kind':'eye_roll','duration_s':3.})
        elif re.search(r'\b(?:zwinker\w*|wink\w*)\b',t):steps.append({'kind':'wink','duration_s':2.5})
        elif re.search(r'\b(?:schüttel\w*|shake\w*)\b.*\b(?:kopf|head)\b|\b(?:kopf|head)\b.*\b(?:schüttel\w*|shake\w*)\b',t):steps.append({'kind':'shake','duration_s':2.5})
        elif re.search(r'\b(?:nick(?:e|en)?|nod)\b',t):steps.append({'kind':'nod','duration_s':2.5})
        elif pose_request:
            found=[(m.start(),slug) for slug,pattern in _EMOTIONS.items() for m in [pattern.search(t)] if m]
            for label,slug in [('happy|fröhlich|glücklich|froh','elation'),('lustig|witzig|amüsiert','amusement'),('böse|boese','anger'),('frech','teasing')]:
                match=re.search(r'\b(?:'+label+r')(?:e|en|es|er|em)?\b',t)
                if match and not found:found=[(match.start(),slug)]
            if found:steps.append({'kind':'emotion','emotion':min(found)[1],'duration_s':3.})
        if len(steps)>=6:break
    if not steps:return []
    # The shared validator is the authority for all durations, kinds and fields.
    bounded=[];total=0.
    for step in steps[:6]:
        if total+step['duration_s']>24:break
        bounded.append(step);total+=step['duration_s']
    return validate({'steps':bounded})
