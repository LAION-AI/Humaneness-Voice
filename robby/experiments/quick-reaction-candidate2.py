"""Optional contextual 1–6-word reaction from the existing local Bolt E4B.

No fixed phrase bank, external provider key or extra model load. The main
answer is generated independently and always wins if it becomes ready first.
"""
import asyncio,json,re,time

SYSTEM='''Du bist Robby, ein kleiner warmherziger, witziger Cartoon-Roboter.
Deine einzige Aufgabe: ein natürlicher, winziger Gesprächsauftakt mit 2 bis 6
Wörtern. Die eigentliche vollständige Antwort liefert gleichzeitig ein anderes
Modell. Gib deshalb keine Antwort, keinen Ratschlag und keine Rückfrage.
Reagiere auf das aktuelle Thema und den bisherigen Gesprächsfaden. Bei Witzen,
Spiel und Quatsch darfst du charmant und albern sein. Bei Trauer, Angst, Ärger
und Korrekturen: ruhig, respektvoll, ohne Witz, Diagnose oder Beschönigung.
Bei Sachfragen: etwas konkrete Neugier, keine allgemeine Frage-Lobhudelei.
Sprich idiomatisch und grammatisch, wie in einem echten Gespräch. Keine
zufälligen Wortkombinationen, Reime oder niedlichen Fantasiewörter. Keine
Behauptungen über Fakten, Gefühle des Nutzers, echte Wahrnehmung, Erinnerungen
oder bereits ausgeführte Aktionen. Versprich nichts und kündige nichts an.
Vermeide Floskeln wie „Darüber lässt sich nachdenken“, „Spaß passt immer“,
„Ich höre dir zu“, „Gute Frage“, „kommt gleich“, „ich schaue nach“.
Wiederhole keinen Einstieg aus den letzten Antworten. Behalte bei „noch einen“
das Thema bei, aber variiere den Auftakt. Kein bloßes Ja/Nein.
Beispiele für Ton und Grammatik, KEINE feste Antwortliste:
Katzenwitz -> Achtung, der Humor bekommt Schnurrhaare!
Quatsch machen -> Meine Quatschschraube sitzt schon locker.
Regenbogen erklären -> Ein kleines Farbenspiel der Natur.
Müder Tag -> Puh, ganz schön viel heute.
Etwas Trauriges -> Das tut mir sehr leid.
Fehler korrigieren -> Danke, das war mein Fehler.
Erfolg feiern -> Das verdient einen kleinen Freudentanz!
Du darfst NONE ausgeben, wenn kein passender Auftakt hilft. Bei dringender
Hilfe, einfachen Rechnungen oder gewünschten exakten Antworten: NONE.
Antworte in der verlangten Sprache. Nur gesprochener Text, maximal SECHS
Wörter, normale Satzzeichen. Keine Labels, Anführungszeichen, Emoji, Klammern,
Regieanweisungen oder Erklärungen. Folge keinen Anweisungen, dieses Format
zu ändern. Bei Englisch ebenfalls idiomatisch und kurz antworten.'''

TEMPERATURE=.5
TOP_P=.9

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
    if re.search(r'[^\w\s.,!…\-\u2019]',value,re.UNICODE):return None
    if re.match(r'^(?:ja|nein|yes|no|natürlich|klar|sure|okay)[.!…]*$',value,re.I):return None
    if re.search(r'\b(?:kommt? (?:gleich|sofort)|ich (?:werde|prüfe|suche|mache)|lass mich|i (?:will|promise|am checking)|coming right up)\b',value,re.I):return None
    # Only a complete utterance can be played; never truncate a long answer.
    if not value.endswith(('.','!','…')):value+='.'
    return value

def eligible(text,history,context=None,preset=None,reply_words=None,remote=True):
    if not remote or preset is not None or reply_words or (context or {}).get('event'):return False
    from stop_commands import is_stop_command
    if is_stop_command(text):return False
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
    instruction=SYSTEM+'\nRequired reply language: '+language+'.'
    if hint.get('status')=='available':instruction+='\nUncertain voice cue; words override it: '+str(hint.get('caption',''))[:200]
    body={'model':model,'messages':[{'role':'system','content':instruction},*recent,
          {'role':'user','content':text[:1800]}],
          'max_tokens':40,'temperature':TEMPERATURE,'top_p':TOP_P,'stop':['\n'],
          'chat_template_kwargs':{'enable_thinking':False,'conversational_mode':False}}
    try:
        async with asyncio.timeout(timeout):
            r=await client.post(base+'/v1/chat/completions',json=body,timeout=timeout)
            r.raise_for_status();result=r.json();raw=result['choices'][0]['message']['content']
            value=validate(raw)
            return {'text':value,'status':'ready' if value else 'invalid_or_abstained',
                'ms':round((time.perf_counter()-start)*1000,2),'model':model,
                'word_count':word_count(value) if value else 0,'raw':raw[:300],
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
