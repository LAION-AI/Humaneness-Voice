"""Optional contextual 1–6-word reaction from the existing local Bolt E4B.

No fixed phrase bank, external provider key or extra model load. The main
answer is generated independently and always wins if it becomes ready first.
"""
import asyncio,json,re,time

SYSTEM='You write Robby\'s FIRST REACTION, not the real answer. Robby is a kind,\nwarm, playful little robot. The full answer will follow from another model.\nWrite ONE natural standalone utterance of 1–6 words in the requested language.\nAcknowledge this user\'s topic or tone gently, without answering the question,\nclaiming difficulty, asserting a feeling as fact, promising an action or lookup,\nor asking another question. Be empathetic, relevant and lightly noncommittal. For jokes and games, a small playful robot pun is welcome. Use natural grammar. Serious topics need sincere warmth without humor.\nDon\'t call an ordinary factual question difficult or mysterious. Don\'t say you\nare thinking, checking or waiting. Don\'t flatter every question. No bare yes/no.\nRespect the conversation\'s subject. Never invent a personal fact or a tool result.\nDo not announce a joke with "Ein Witz kommt" or say "kommt gleich".\nA visual phenomenon being discussed is not something you currently see.\nExamples of the KIND of reaction, not a menu to repeat:\nExhausting day -> Das klingt nach viel.\nPhilosophical question -> Da steckt eine große Frage drin.\nPlayful request -> Meine Quatschschraube sitzt locker.\nPersonal uncertainty -> Das klingt gerade ziemlich anstrengend.\nAffection -> Das ist lieb von dir.\nGood news -> Hey, herzlichen Glückwunsch!\nA joke in English -> My funny circuits are buzzing.\nReturn only the spoken words, with normal punctuation. No labels, quotes,\nparentheses, emojis, Markdown, instructions or reasoning. At most SIX words.\nIf a reaction would violate a request for silence or exact output, return NONE.'

TEMPERATURE=.35
TOP_P=.9

# Small, deterministic tone hints keep the tiny local model from transferring a
# joke example into grief or a lookup. These are prompting examples, never audio
# fallbacks or a bank of responses selected for playback.
TONE_EXAMPLES={
 'playful':('Be playfully amused. A tiny robot joke is welcome. Match the actual subject.',
    ['Na, das kann heiter werden!','Meine Quatschschraube sitzt locker.','Das wird ein herrlicher Blödsinn.'],
    ['My silly circuits are tingling.','Now that sounds delightfully ridiculous.']),
 'joke':('Enjoy the request without giving away a punchline. Keep the joke subject.',
    ['Na, dann her mit dem Quatsch!','Das kitzelt meine Lachschraube.','Noch eine Runde Lachmuskeltraining.'],
    ['My funny bone is tingling.','Another round of delightful nonsense.']),
 'serious':('Be sober and compassionate. Absolutely no humor or silver lining.',
    ['Das tut mir sehr leid.','Das ist wirklich schwer.'],
    ["I'm so sorry.",'That sounds really hard.']),
 'support':('Gently acknowledge what they said, without assuming unspoken emotions.',
    ['Puh, das klingt nach viel.','Das klingt gerade ziemlich anstrengend.'],
    ['That sounds like a lot.','Sounds like a tough day.']),
 'affection':('Respond warmly to kindness, no pressure, no silly joke or real-sensation claim.',
    ['Das ist lieb von dir.','So eine liebe Geste.'],
    ["That's really sweet of you.",'What a lovely gesture.']),
 'celebration':('Celebrate only the achievement they actually reported.',
    ['Hey, herzlichen Glückwunsch!','Das ist wirklich klasse!'],
    ['Hey, congratulations!','That is wonderful news!']),
 'correction':('Respect the correction or frustration. No teasing, excuses or promises.',
    ['Danke für die Klarstellung.','Das war unpassend von mir.'],
    ['Thanks for correcting me.','That was unhelpful of me.']),
 'social':('Be a friendly little robot in a casual conversation.',
    ['Meine Quatschlaune ist bestens.','Schön, dass du da bist.'],
    ['My silly mood is excellent.',"Lovely to have you here."]),
 'curious':('Be quietly interested in the topic. Do not answer it or praise the question.',
    ['Da steckt eine interessante Geschichte dahinter.','Das lohnt einen genaueren Blick.'],
    ["There's an interesting story there.",'That deserves a closer look.']),
}

def reaction_tone(text,history):
    """Cheap delivery guidance, not an emotion detector or source of facts."""
    t=text.casefold()
    if re.search(r'\b(?:gestorben|verstorben|trauere|trauer|died|died yesterday|passed away|verloren|lost my|vermisse|miss my)\b',t):return 'serious'
    if re.search(r'\b(?:korrigier\w*|nicht bob|nicht alice|falsch verstanden|klarstellung|weichst|ausweichen|endlich richtig|immer dasselbe|origineller|correct you|wrong name)\b',t):return 'correction'
    if re.search(r'\b(?:angst|ängstlich|müde|anstrengend\w*|erschöpft|stress\w*|afraid|scared|tired|exhaust\w*)\b',t):return 'support'
    if re.search(r'\b(?:streichel\w*|streichle|kraul\w*|cuddle|pet you|sweet robot)\b',t):return 'affection'
    if re.search(r'\b(?:bestanden|geschafft|befördert|gewonnen|passed my|got promoted|won the)\b',t):return 'celebration'
    from bolt_director import response_plan
    if response_plan(text,history)['kind']=='joke':return 'joke'
    if re.search(r'\b(?:quatsch|albern\w*|witzig\w*|lustig\w*|dreh\w*|grimass\w*|robotertanz|silly|funny|spin|dance)\b',t) or 'was wäre, wenn' in t:return 'playful'
    if re.search(r'\b(?:wie geht|how are|hallo|hello|guten morgen)\b',t):return 'social'
    return 'curious'

def recent_openings(history):
    openings=[]
    for h in history[-8:]:
        if h.get('role')!='assistant' or not isinstance(h.get('content'),str):continue
        # Do not use internal interruption metadata as a spoken opening.
        content=re.sub(r'^\s*\[[^\]]*\]\s*','',h['content'])
        first=re.split(r'(?<=[.!?])\s|\n',content,1)[0]
        if 1<=word_count(first)<=8:openings.append(first[:120])
    return openings[-4:]

def repeated_opening(value,history):
    norm=lambda s:' '.join(WORDS.findall(s.casefold()))
    return any(norm(value)==norm(old) for old in recent_openings(history))

WORDS=re.compile(r"[\w]+(?:['’\-][\w]+)*",re.UNICODE)

class AudioEnvelope:
    """Ten-ms fades, retaining only a ten-ms tail; no ASR lookahead."""
    def __init__(self):
        import numpy as np
        self.tail=np.zeros(0,np.float32);self.count=0;self.fade=240
    def feed(self,wave):
        import numpy as np
        wave=np.asarray(wave,np.float32).copy()
        n=min(len(wave),max(0,self.fade-self.count))
        if n:wave[:n]*=np.arange(self.count,self.count+n,dtype=np.float32)/self.fade
        self.count+=len(wave);out=np.r_[self.tail,wave]
        n=max(0,len(out)-self.fade);self.tail=out[n:].copy()
        return out[:n]
    def finish(self):
        import numpy as np
        out=self.tail.copy();self.tail=np.zeros(0,np.float32)
        if len(out):out*=np.linspace(1,0,len(out),dtype=np.float32)
        return out

def word_count(text):return len(WORDS.findall(text))

def validate(text):
    if not isinstance(text,str):return None
    value=text.strip()
    if value.upper()=='NONE' or not 1<=word_count(value)<=6 or len(value)>100:return None
    if '\n' in value or re.search(r'[\[\]{}<>():?"*#=]',value):return None
    if re.search(r"[^\w\s.,!…\-'\u2019]",value,re.UNICODE):return None
    if re.match(r'^(?:ja|nein|yes|no|natürlich|klar|sure|okay)[.!…]*$',value,re.I):return None
    if re.search(r'\b(?:kommt? (?:gleich|sofort)|ich (?:werde|prüfe|suche|mache|schaue|schau|erkläre|merke)|schau(?:e)? ich nach|lass mich|i (?:will|promise|am checking)|coming right up)\b',value,re.I):return None
    # Only a complete utterance can be played; never truncate a long answer.
    if not value.endswith(('.','!','…')):value+='.'
    return value

def eligible(text,history,context=None,preset=None,reply_words=None,remote=True):
    if not remote or preset is not None or reply_words or (context or {}).get('event'):return False
    from stop_commands import is_stop_command
    if is_stop_command(text):return False
    # Optional rapport must never postpone immediate help. This conservative
    # fast path only suppresses the preface, not the main response.
    if re.search(r'\b(?:keine luft|not breathing|can.t breathe|herzinfarkt|heart attack|notfall|emergency|starke brustschmerzen)\b',text,re.I):return False
    # Exact-output instructions and deliberate silence must remain exact.
    if re.search(r'\b(?:nur|only|genau|exactly|exakt|ausschließlich|wortwörtlich|verbatim|ohne einleitung|no preamble|schweig\w*|sei leise|be quiet)\b',text,re.I):return False
    # Keep verse/word/sentence-count tasks intact (including follow-up poems).
    from bolt_director import response_plan
    plan=response_plan(text,history)
    if plan['kind']=='poem' or plan.get('units'):return False
    return True

async def generate(client,text,identity,history,language='German',timeout=.7):
    from bolt_director import local_config
    base,model=local_config();start=time.perf_counter()
    recent=[{'role':h['role'],'content':h['content'][-500:]} for h in history[-4:]
            if h.get('role') in ('user','assistant') and isinstance(h.get('content'),str)]
    # Raw observations remain uncertain, and only this turn's short hint is used.
    hint=(identity or {}).get('voice_delivery',{})
    tone=reaction_tone(text,history);guidance,de,en=TONE_EXAMPLES[tone]
    examples=de if language=='German' else en
    openings=recent_openings(history)
    examples=[e for e in examples if not repeated_opening(e,history)]
    system=SYSTEM
    data={'language':language,'recent_conversation':recent,'user':text[:1800]}
    if hint.get('status')=='available':data['uncertain_voice_cue']=hint.get('caption','')[:200]
    body={'model':model,'messages':[{'role':'system','content':system},
          {'role':'user','content':json.dumps(data,ensure_ascii=False)}],
          'max_tokens':40,'temperature':TEMPERATURE,'top_p':TOP_P,'stop':['\n'],
          'chat_template_kwargs':{'enable_thinking':False,'conversational_mode':False}}
    try:
        async with asyncio.timeout(timeout):
            r=await client.post(base+'/v1/chat/completions',json=body,timeout=timeout)
            r.raise_for_status();result=r.json();raw=result['choices'][0]['message']['content']
            value=validate(raw)
            if value and repeated_opening(value,history):value=None
            return {'text':value,'status':'ready' if value else 'invalid_or_abstained',
                'ms':round((time.perf_counter()-start)*1000,2),'model':model,
                'word_count':word_count(value) if value else 0,'raw':raw[:300],
                'tone':tone,'direction':'amusement, warm, playful' if tone in ('playful','joke') else 'contentment, soft, warm',
                'timings':result.get('timings',{})}
    except asyncio.CancelledError:raise
    except Exception as exc:
        return {'text':None,'status':'timeout' if isinstance(exc,TimeoutError) else 'unavailable',
            'ms':round((time.perf_counter()-start)*1000,2),'model':model,'word_count':0}

def duplicate_acknowledgment(reaction,answer):
    """Only suppress a near-identical short first sentence, never useful prose."""
    if word_count(answer)>8:return False
    from difflib import SequenceMatcher
    norm=lambda s:' '.join(WORDS.findall(s.casefold()))
    return SequenceMatcher(None,norm(reaction),norm(answer)).ratio()>=.88
