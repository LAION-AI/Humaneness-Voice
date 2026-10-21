"""Luna's textual TTS directions -> authored robot cues. No audio classifier.

Sentence boundaries use the timed SCRIPT recipe, not acoustic word alignment.
The browser anchors these estimates to actual DAC playback/sample positions.
"""
import json,re
from pathlib import Path

PROFILES=json.loads((Path(__file__).parent/'static/emotion-catalog.json').read_text())['profiles']
SLUGS=[p['slug'] for p in PROFILES]
ALIASES={
 'amusement':'amused|amusement|playful|playfully|laughing|heiter|vergnügt',
 'elation':'elated|elation|joyful|joyfully|excited|excitedly|ecstatic|begeistert|freudig',
 'pleasure_ecstasy':'pleasure|ecstasy|blissful|genussvoll',
 'contentment':'contentment|contented|calm|relaxed|natural|neutral|sanft|ruhig|natürlich',
 'gratitude':'gratitude|grateful|thankful|dankbar', 'affection':'affection|affectionate|loving|tender|warm|liebevoll|zärtlich',
 'infatuation':'infatuation|infatuated|smitten|verliebt','hope':'hopeful|hope|optimistic|enthusiasm|hoffnungsvoll',
 'triumph':'triumph|triumphant|victorious|siegreich','pride':'pride|proud|proudly|confident|stolz',
 'interest':'interest|interested|curious|neugierig','awe':'awe|awestruck|wonder|ehrfürchtig',
 'surprise':'surprised|surprise|astonished|astonishment|startled|überrascht|erstaunt',
 'concentration':'concentration|concentrated|focused|konzentriert',
 'contemplation':'contemplation|contemplative|thoughtful|pensive|nachdenklich',
 'relief':'relief|relieved|erleichtert','longing':'longing|wistful|yearning|sehnsüchtig',
 'teasing':'teasing|tease|mischievous|cheeky|neckisch','impatience':'impatience|impatient|irritable|annoyed|ungeduldig|gereizt',
 'lust':'lust|lustful|seductive|sensual|sinnlich','doubt':'doubt|doubtful|skeptical|sceptical|skeptisch',
 'fear':'fear|fearful|afraid|scared|frightened|ängstlich','distress':'distress|distressed|anxious|worried|besorgt',
 'confusion':'confusion|confused|puzzled|verwirrt','embarrassment':'embarrassment|embarrassed|bashful|verlegen',
 'shame':'shame|ashamed|beschämt','disappointment':'disappointment|disappointed|enttäuscht',
 'sadness':'sadness|sad|sadly|sorrowful|melancholy|melancholic|tearful|crying|traurig','bitterness':'bitterness|bitter|resentful|verbittert',
 'contempt':'contempt|contemptuous|scornful|verächtlich','disgust':'disgust|disgusted|repulsed|angewidert',
 'anger':'anger|angry|angrily|furious|furiously|rage|wütend|zornig','malice':'malice|malevolent|malicious|villainous|bösartig',
 'sourness':'sourness|sour|sauer','pain':'pain|pained|hurting|schmerzerfüllt',
 'helplessness':'helplessness|helpless|desperate|hilflos','fatigue':'fatigue|exhausted|tired|weary|müde|erschöpft',
 'numbness':'numbness|numb|emotionless|flat|teilnahmslos','intoxication':'intoxication|intoxicated|tipsy|drunk|betrunken',
 'envy':'envy|envious|jealous|eifersüchtig|neidisch'}
PATTERNS={slug:re.compile(r'(?<!\w)(?:'+re.escape(slug)+'|'+ALIASES[slug]+r')(?!\w)',re.I) for slug in SLUGS}
DIRECTOR_NOTE=('For each spoken sentence, put a concise English acting direction in round brackets in SCRIPT, '
 'starting with the main emotion name. Those same directions also animate your robot face. '
 'You may change expression between sentences or combine a main and a secondary emotion. '
 'Describe intensity with words, never numbers inside acting brackets. Choose appropriately; do not exaggerate every line. '
 'Available facial emotions: '+', '.join(SLUGS)+'. Do not speak the emotion labels or describe the face. ')


def cue(direction,fallback='contentment'):
    found=[]
    for slug,pattern in PATTERNS.items():
        for m in pattern.finditer(direction):
            before=direction[max(0,m.start()-18):m.start()].lower()
            if re.search(r'(?:not|without|no|never|nicht|ohne)\s+(?:very\s+)?$',before):continue
            found.append((m.start(),slug));break
    found.sort();emotions=list(dict.fromkeys(slug for _,slug in found))
    strength=.9 if re.search(r'\b(strong(?:ly)?|intense(?:ly)?|furious(?:ly)?|deep(?:ly)?|very|vivid(?:ly)?|shouting)\b',direction,re.I) else .72
    if re.search(r'\b(subtle|slightly|gentle|soft|quiet)\b',direction,re.I):strength=.55
    return {'emotion':emotions[0] if emotions else fallback,'secondary':emotions[1] if len(emotions)>1 else None,
            'blend':.25 if len(emotions)>1 else 0.,'intensity':strength,'direction':direction[:400]}


def plan(spec):
    text=str(spec.get('tagged') or spec.get('edge_script') or spec.get('script') or spec.get('text') or '')
    general=str(spec.get('general') or 'natural');default=cue(general);current=default;duration=None;clock=0.;sentences=[]
    chunks=re.split(r'(\[[^\]]*\]|\([^)]*\))',text)
    for part in chunks:
        if not part.strip():continue
        if part.startswith('['):
            m=re.search(r'([\d.]+)\s*(?:s|seconds?)\s*(duration|pause)',part,re.I)
            if m:
                if m[2].lower()=='pause':clock+=float(m[1])
                else:duration=float(m[1])
        elif part.startswith('('):
            # Numeric bracket tags are vocal bursts, not spoken directives.
            if re.search(r'\d',part):
                m=re.search(r'([\d.]+)\s*(?:s|seconds?)',part)
                if m:clock+=float(m[1])
            else:current=cue(part[1:-1],default['emotion'])
        else:
            phrases=[s.strip() for s in re.split(r'(?<=[.!?])\s+',part) if s.strip()]
            lengths=[max(1,len(s.split())) for s in phrases];total=sum(lengths)
            budget=duration if duration is not None else max(.4,total*.34)
            for sentence,words in zip(phrases,lengths):
                length=budget*words/total
                sentences.append({**current,'text':sentence,'start_s':round(clock,4),'end_s':round(clock+length,4)})
                clock+=length
            duration=None
    if not sentences:
        sentences=[{**default,'text':str(spec.get('text') or ''),'start_s':0.,'end_s':max(1,float(spec.get('frames',25))*.08)}]
    return {'source':'Luna SCRIPT acting directions','timing':'Prompt-derived sentence estimates, anchored to DAC playback; not forced alignment',
            'sentences':sentences,'planned_duration_s':round(clock,4)}
