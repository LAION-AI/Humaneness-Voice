"""Small, deterministic acting catalogue and incremental sentence framing.

The local LLM generates cue keywords and words, never GENERAL prose. Catalogue
lookup is an in-memory nearest-neighbour search over sparse lexical features.
"""
from dataclasses import dataclass
import math
import re
from performance import SLUGS, cue

REFERENCE = "Use the same speaker as the reference audio. Preserve that speaker's voice identity."
DELIVERIES = ('natural', 'soft', 'calm', 'warm', 'playful', 'rapid', 'slow', 'whispered', 'firm', 'conversational')
ALIASES = {'happy':'elation', 'joy':'elation', 'neutral':'contentment', 'sad':'sadness',
           'angry':'anger', 'curious':'interest', 'grateful':'gratitude', 'tender':'affection'}

def features(text):
    tokens=re.findall(r'[^\W\d]+',text.casefold(),re.UNICODE)
    result={}
    for token in tokens:
        token=ALIASES.get(token,token)
        result[token]=result.get(token,0)+1
    return result

def cosine(a,b):
    return sum(v*b.get(k,0) for k,v in a.items())/max(1e-9,math.sqrt(sum(v*v for v in a.values())*sum(v*v for v in b.values())))

CATALOGUE=[]
for emotion in SLUGS:
    for level in ('soft','intense'):
        description=(f"Express {emotion.replace('_',' ')} with {'gentle, subtle' if level=='soft' else 'vivid, expressive'} acting. "
                     "Keep articulation clear and the delivery natural, spontaneous and conversational.")
        CATALOGUE.append({'id':f'{emotion}.{level}','emotion':emotion,'level':level,
                          'keywords':features(emotion+' '+level),'general':REFERENCE+' '+description})
for delivery in DELIVERIES:
    for level in ('soft','intense'):
        CATALOGUE.append({'id':f'delivery.{delivery}.{level}','emotion':'contentment','level':level,
            'keywords':features(delivery+' '+level),
            'general':REFERENCE+f" Use {delivery}, {'subtle' if level=='soft' else 'expressive'} delivery, clear articulation and natural spontaneous phrasing."})

def select_template(direction):
    direction=direction[:100]
    query=features(direction)
    # Emotion aliases also cover the existing 40-expression performance map.
    inferred=cue(direction)['emotion']
    explicit=any(k in SLUGS for k in query)
    if not explicit and not any(k in DELIVERIES for k in query):query[inferred]=1
    query.setdefault('intense' if re.search(r'\b(intense|strong|loud|very|rapid)\b',direction,re.I) else 'soft',1)
    row=max(CATALOGUE,key=lambda t:cosine(query,t['keywords']))
    general=row['general']
    # A second keyword selects a fixed modifier; it costs no LLM output prose.
    modifiers={
        'rapid':'Use brisk but intelligible pacing.', 'slow':'Use calm, unhurried pacing.',
        'whispered':'Use a soft, audible whisper.', 'warm':'Keep the delivery warm and friendly.',
        'playful':'Use light, playful phrasing.', 'calm':'Keep the tempo calm and unhurried.'}
    for key,value in modifiers.items():
        if key in query:general+=' '+value
    return {**row,'general':general,'score':round(cosine(query,row['keywords']),4)}

@dataclass
class Sentence:
    text:str
    direction:str='contentment, soft'
    def spec(self):
        template=select_template(self.direction)
        script=f'({self.direction}) {self.text}'
        return {'text':self.text,'script':script,'general':template['general'],
                'manual_prompt':'GENERAL: '+template['general']+'\nSCRIPT:\n'+script,
                'template_id':template['id'],'template_score':template['score'],
                'language':'de' if re.search(r'\b(ich|du|wir|ist|das|die|und|ein|eine|nicht|bitte)\b',self.text,re.I) else 'en'}

class SentenceFramer:
    """Frame complete streamed sentences, keeping cue brackets out of speech."""
    def __init__(self):self.buffer='';self.direction='contentment, soft'
    def feed(self,text,final=False):
        self.buffer+=text;out=[]
        if len(self.buffer)>12000:raise ValueError('Streamed response too long')
        while True:
            self.buffer=self.buffer.lstrip()
            if self.buffer.startswith('('):
                end=self.buffer.find(')')
                if end<0:
                    if len(self.buffer)>120:raise ValueError('Unclosed acting cue')
                    break
                raw=self.buffer[1:end]
                if not re.fullmatch(r'[A-Za-z_ ,\-]{1,100}',raw):raise ValueError('Invalid acting cue')
                self.direction=raw.strip();self.buffer=self.buffer[end+1:].lstrip()
            match=None
            for candidate in re.finditer(r'[.!?]+["”’»]*',self.buffer):
                end=candidate.end()
                if end==len(self.buffer) and not final:continue
                if end<len(self.buffer) and not self.buffer[end].isspace():continue
                prefix=self.buffer[:end]
                if re.search(r'\b(?:Dr|Prof|Mr|Mrs|Ms|Nr|bzw|ca|z|B)\.$',prefix):continue
                if candidate.start()>0 and self.buffer[candidate.start()-1].isdigit() and end<len(self.buffer) and self.buffer[end].isdigit():continue
                match=candidate;break
            if match is None:
                if final and self.buffer.strip():
                    text=self.buffer.strip();self.buffer=''
                else:break
            else:
                text=self.buffer[:match.end()].strip();self.buffer=self.buffer[match.end():]
            if text.casefold() in ('silent','[silent]'):continue
            if '(' in text or ')' in text:raise ValueError('Acting cue must precede its sentence')
            if text and len(text.split())<=90:out.append(Sentence(text,self.direction))
            elif text:raise ValueError('Sentence too long')
        return out

class ASRSentenceCommitter:
    """Commit stable punctuation once; tolerate cumulative ASR revisions.

    A committed prefix cannot be edited after playback starts. A shorter or
    revised preview is kept pending instead of re-submitting old words.
    """
    def __init__(self):self.committed=[];self.pending='';self.since=0.
    def update(self,text,now,final=False,allow_commit=True):
        text=' '.join(text.split())
        tokens=re.findall(r'\w+',text.casefold())
        if tokens[:len(self.committed)]!=self.committed:return []
        # Locate uncommitted words without reconstructing spelling/punctuation.
        matches=list(re.finditer(r'\w+',text))
        tail=text[matches[len(self.committed)].start():] if len(matches)>len(self.committed) else ''
        framer=SentenceFramer();ready=framer.feed(tail+' ',final=final)
        candidate=ready[0].text if ready else ''
        if not candidate:return []
        # ASR often punctuates a greeting before hearing its continuation.
        # "Hey Robby" must not start a reply before "ich bin Christoph".
        # Short genuine utterances still flush on the short-pause path.
        if not final and len(re.findall(r'\w+',candidate))<3:return []
        if candidate!=self.pending:self.pending=candidate;self.since=now
        if not final and now-self.since<.16:return []
        if not final and not allow_commit:return []
        self.committed.extend(re.findall(r'\w+',candidate.casefold()));self.pending=''
        return [candidate]
