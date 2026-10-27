"""Short local transition grounded in the actual first Qwen sentence(s)."""
import asyncio,json,re,time
from quick_reaction import word_count,duplicate_acknowledgment,WORDS

SYSTEM='''You write Robby's SECOND, INFORMED transition. The user has already heard
the first reaction. The supplied answer_opening is the actual next speech from
the main model, not a proposed topic. Write ONE short natural linking sentence,
usually 3–8 words, at most TWELVE words, in the requested language.
Lead gently from the first reaction into that specific answer. Be warm and
lightly playful only when appropriate. Serious situations need quiet respect.
Name a concrete subject or image from answer_opening, in a new linking phrase.
Never say generic filler such as "Hier ist die Erklärung", "Hier ist eine Idee",
"Hier kommt der Witz", "Here is the idea behind it" or "Let's find out".
Do not give another greeting, thank-you, apology, empathy acknowledgment or
generic waiting/thinking announcement. Do not repeat the reaction, summarize
the answer, reveal a joke's punchline, or add a new factual claim. No questions,
promises, tool claims, stage directions, quotation marks, labels or explanations.
The transition and the UNCHANGED answer_opening must sound sensible together.
Examples; use the actual subject, never copy an unrelated example:
Opening "Das liegt an der Rayleigh-Streuung." -> Schauen wir uns die Lichtstreuung an.
Opening "Warum sitzt die Katze am Computer?" -> Die Katze wird jetzt richtig digital.
Opening "Gehe zehn Minuten ohne Bildschirm spazieren." -> Ein kleiner Ortswechsel könnte guttun.
Opening "Verrühre Eier, Milch und Mehl." -> Fangen wir mit dem Teig an.
Opening "Cats purr for comfort and self-soothing." -> That little rumble has several roles.
Opening "Wenn du magst, erzähl mir von deiner verstorbenen Katze." -> NONE
Never direct or pressure the user into sharing personal feelings. For a purely
empathetic or greeting answer, prefer NONE over repeating its invitation.
Return NONE only if another sentence would be inappropriate or duplicate the
answer. Write a complete statement with a verb, not a noun fragment or a question.
Return only the short spoken transition, ending with normal punctuation.'''

def validate(text,reaction,opening):
    if not isinstance(text,str):return None
    text=text.strip()
    if text.upper()=='NONE' or not 2<=word_count(text)<=12 or len(text)>180:return None
    if '\n' in text or re.search(r'[\[\]{}<>():?"*#=]',text):return None
    if re.search(r"[^\w\s.,!…\-'’]",text,re.UNICODE):return None
    if re.search(r'\b(?:ich (?:werde|prüfe|suche|denke|schaue)|lass mich|kommt gleich|i (?:will|promise|am checking))\b',text,re.I):return None
    if re.search(r'^(?:hier (?:ist|kommt)|here is (?:the|an?) (?:idea|answer)|erzähl mir lieber)\b',text,re.I):return None
    if duplicate_acknowledgment(reaction,text) or duplicate_acknowledgment(opening,text):return None
    if not text.endswith(('.','!','…')):text+='.'
    return text

async def generate(client,text,reaction,opening,history=(),language='German',timeout=.7):
    from bolt_director import local_config,response_plan
    base,model=local_config();started=time.perf_counter()
    system=SYSTEM
    if response_plan(text,history)['kind']=='joke':
        system+='\nThis is a JOKE SETUP. Write a playful linking statement about its character or setting. Never answer the setup question, infer/reveal a punchline, or promise a joke. For example, a cat at a computer could be linked with: Diese Katze wird jetzt digital. Use NONE rather than spoiling the answer.'
    data={'language':language,'user':text[:1600],'already_spoken_reaction':reaction,
          'answer_opening':opening[:1400],
          'recent_conversation':[{'role':h['role'],'content':h['content'][-300:]} for h in history[-2:] if h.get('role') in ('user','assistant') and isinstance(h.get('content'),str)]}
    try:
        async with asyncio.timeout(timeout):
            r=await client.post(base+'/v1/chat/completions',json={'model':model,
                'messages':[{'role':'system','content':system},{'role':'user','content':json.dumps(data,ensure_ascii=False)}],
                'max_tokens':64,'temperature':.25,'top_p':.9,'stop':['\n'],
                'chat_template_kwargs':{'enable_thinking':False,'conversational_mode':False}},timeout=timeout)
            r.raise_for_status();raw=r.json()['choices'][0]['message']['content'];out=validate(raw,reaction,opening)
            return {'text':out,'raw':raw[:400],'status':'ready' if out else 'invalid_or_abstained',
                    'ms':round((time.perf_counter()-started)*1000,2),'word_count':word_count(out) if out else 0,
                    'answer_opening_used':opening[:1400],'model':model}
    except asyncio.CancelledError:raise
    except Exception as exc:return {'text':None,'status':'timeout' if isinstance(exc,TimeoutError) else 'unavailable',
        'ms':round((time.perf_counter()-started)*1000,2),'model':model}
