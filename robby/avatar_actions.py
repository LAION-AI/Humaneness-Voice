"""Allowlisted, bounded visual-only performances. Never controls robot hardware."""
import json
import math
import re
from pathlib import Path

EMOTIONS = tuple(p['slug'] for p in json.loads((Path(__file__).parent/'static/emotion-catalog.json').read_text())['profiles'])
KINDS = ('emotion', 'nod', 'shake', 'spin', 'carousel', 'afterglow', 'puppy', 'pet', 'eye_roll', 'grimace', 'wink', 'curious')
VARIANTS = ('crossed', 'opposite', 'wiggle', 'cheeky', 'goofy', 'squishy', 'fish', 'peekaboo')
TOOL_NAME = 'perform_avatar'

def explicit_spin(text):
    """Conservative direct-command fallback, never matches quotes or conditions.

    Complex choreography stays with the director tool. This only guarantees
    an unambiguous requested turn even if the model merely describes it.
    """
    if not isinstance(text,str) or len(text)>400:return []
    t=text.casefold().strip()
    if re.search(r'["“”„«»]|\b(?:nicht|kein\w*|never|not|don.t|wenn|falls|if|example|beispiel|zitiere|quote|sag|say|erklär\w*|explain)\b',t):return []
    direct=re.search(r'\b(?:dreh(?:e|en)?\s+(?:bitte\s+)?(?:dich|deinen?\s+kopf)|(?:kannst|könntest|würdest|willst)\s+du\b.{0,80}\b(?:dich|deinen?\s+kopf)\b.{0,60}\bdrehen|(?:please\s+)?spin(?:\s+(?:around|yourself))?(?:[.!?]|$)|(?:can|could|would)\s+you\b.{0,40}\b(?:spin|rotate)|(?:turn|rotate)\s+(?:your\s+head|around|yourself))',t)
    if not direct:return []
    direction='counterclockwise' if re.search(r'gegen\s+den\s+uhrzeigersinn|counter[- ]?clockwise|anti[- ]?clockwise',t) else 'clockwise'
    return [{'kind':'spin','duration_s':6.,'sentence':0,'direction':direction}]

TOOL = {'type':'function','function':{'name':TOOL_NAME,
    'description':'Perform a gentle visual-only robot action sequence, optionally at a spoken sentence. No hardware movement.',
    'parameters':{'type':'object','additionalProperties':False,'required':['steps'], 'properties':{
        'steps':{'type':'array','minItems':1,'maxItems':6,'items':{'type':'object','additionalProperties':False,
            'required':['kind'], 'properties':{
                'kind':{'type':'string','enum':list(KINDS)},
                'emotion':{'type':'string','enum':list(EMOTIONS)},
                'variant':{'type':'string','enum':list(VARIANTS)},
                'direction':{'type':'string','enum':['clockwise','counterclockwise']},
                'duration_s':{'type':'number','minimum':1,'maximum':6},
                'sentence':{'type':'integer','minimum':0,'maximum':11}}}}}}}}
NOTE = '''The perform_avatar tool is available for facial/head actions. Fulfil explicit
requests to nod, shake the head, rotate 360 degrees, roll the eyes, make a cute
grimace, wink or show any of the 40 authored facial emotions. Match German and
English requests to the English tool values. For example wütend=anger,
bitter=bitterness, traurig=sadness. Use ordered steps for "first angry, then a
funny face". emotion steps are full archetypical poses with no random variation.
spin is exactly one virtual yaw turn; carousel is three accelerating playful turns,
prefer duration_s=6, only on request or during permitted idle play. Always follow
carousel with afterglow (duration_s=6), a lingering euphoric smile that settles
into contentment. Puppy is a soft, upward-looking, affectionate big-eyed pose
for a genuinely optional invitation, never sadness or emotional pressure. pet is a soft,
contented, eyes-almost-closed grateful expression, not a physical sensation.
Grimace variants: crossed, opposite, wiggle, cheeky, goofy, squishy, fish, peekaboo.
clockwise/counterclockwise are viewed from
above. These are SCREEN animations, never real robot motor commands. Do not
claim a physical head can safely turn 360 degrees. Grimaces are soft, playful,
brief and non-threatening. Never mocking or creepy. Use a nod or curious gesture
sparingly to accompany a helpful sentence, not every reply. Large tricks and eye
rolling only on request; do not eye-roll at somebody's worries. sentence is a
zero-based spoken-sentence index (estimated timing), default 0. At most six steps
and 24 seconds total, no loops. Do not call the tool for negated requests or
quoted examples. A disabled client may suppress visuals; don't claim you saw them.
The tool schedules a performance; it has NOT finished when you receive its result.
Say "Schau mal" / "Watch this" rather than "Erledigt" / "Done". Answer in the
language of the user's current request unless they explicitly choose another.
Even for a visual request, put a brief acknowledgment OUTSIDE parentheses in
SCRIPT (e.g. "(amusement) Schau mal!"). A cue-only SCRIPT is not playable speech.
After tool execution return the usual three-field speech JSON; tool arguments
must never be spoken or put into SCRIPT. Available emotions: '''+', '.join(EMOTIONS)

def validate(raw):
    if isinstance(raw,str):raw=json.loads(raw)
    if not isinstance(raw,dict) or set(raw)!={'steps'}:raise ValueError('Invalid avatar plan')
    steps=raw['steps']
    if not isinstance(steps,list) or not 1<=len(steps)<=6:raise ValueError('Avatar step count')
    result=[];total=0.;last_sentence=0
    for item in steps:
        if not isinstance(item,dict) or set(item)-{'kind','emotion','variant','direction','duration_s','sentence'}:raise ValueError('Avatar fields')
        kind=item.get('kind')
        if kind not in KINDS:raise ValueError('Avatar kind')
        duration=item.get('duration_s',4. if kind=='spin' else 2.)
        sentence=item.get('sentence',last_sentence)
        if isinstance(duration,bool) or not isinstance(duration,(float,int)) or not math.isfinite(duration) or not 1<=duration<=6:raise ValueError('Avatar duration')
        if type(sentence) is not int or not last_sentence<=sentence<=11:raise ValueError('Avatar sentence order')
        if kind=='spin':duration=max(3.,duration)
        if kind=='carousel':duration=6.
        step={'kind':kind,'duration_s':duration,'sentence':sentence}
        for key,allowed,default in (('emotion',EMOTIONS,None),('variant',VARIANTS,'cheeky'),('direction',('clockwise','counterclockwise'),'clockwise')):
            value=item.get(key,default)
            if key in item and value not in allowed:raise ValueError('Avatar '+key)
            if (key=='emotion' and kind=='emotion') or (key=='variant' and kind=='grimace') or (key=='direction' and kind in ('spin','carousel')):
                if value not in allowed:raise ValueError('Missing avatar '+key)
                step[key]=value
        total+=duration;last_sentence=sentence;result.append(step)
    if total>24:raise ValueError('Avatar duration budget')
    return result
