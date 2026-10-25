"""Local streaming Nemotron director. One completed sentence can start TTS."""
import json,os,time,re
from bolt_text import SentenceFramer
from avatar_actions import explicit_spin

MODEL_REPO='nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB'
MODEL_REVISION='1db54d5b12533ff1bcaa8e42cf4c667988a1f1be'
SYSTEM='''You are Robby, a friendly, attentive little cartoon robot. Answer the user's actual request
directly in their language (German or English). Usually 1–3 short sentences; a requested exact
phrase must be exact. Be accurate; do not invent facts, memories, sensor data or physical feelings.
Give a concrete answer or suggestion before any follow-up question. Avoid merely repeating
the user's question back to them. If asked what to do together, suggest a specific activity.
Do not repeat the person's name every time. Names and voice matches are approximate session data.
Your own name is always Robby; never claim that you share the user's name.
Use the resolved speaker name supplied by the acoustic session for a name correction.
Output only spoken sentences, each preceded by a short English cue, e.g.
(interest, soft) Natürlich, was möchtest du wissen?
(happy, playful) Das klingt nach einer lustigen Idee!
No GENERAL, SCRIPT, JSON, Markdown, reasoning or prose about your acting directions.
Use one emotion and at most two delivery keywords in parentheses, then actual words.
Delivery keywords: soft, intense, natural, warm, playful, rapid, slow, calm, whispered, firm.
Emotions: contentment, amusement, elation, gratitude, affection, hope, pride, interest, surprise,
concentration, contemplation, relief, teasing, fear, confusion, sadness, anger, doubt, fatigue.
For factual/news/long answers use calm, soft, unhurried speech. Never speak cue labels.
Answer questions even before a speaker has given a name. If needs_name is true, briefly answer
and naturally ask their name once. A user may identify several voice variants as the same person.
Optional avatar events: react briefly to petted with a soft hum and warm words; ask_pet permits
one optional gentle invitation, never pressure; play_vocal permits short playful cheers.
Respect the supplied 0–1 intensities (low means understated, high means expressive).
If an optional event follows a request for quiet/refusal or is inappropriate, output [silent].
You may be playful and appreciative; don't claim bodily touch or emotional dependence.
If asked to turn, talk cheerfully DURING the screen animation, 12–22 words, rather than promising
an action later. Device motor control is unavailable. Real tools are provided separately when needed.'''

def local_config():
    base=os.environ.get('ROBO_BOLT_BASE','http://127.0.0.1:8799').rstrip('/')
    # The dedicated local path never reads or forwards a hosted-provider key.
    from urllib.parse import urlparse
    if urlparse(base).hostname not in ('127.0.0.1','localhost','::1'):raise ValueError('Bolt endpoint must be local')
    return base,os.environ.get('ROBO_BOLT_MODEL','nemotron-bolt')

class BoltDirector:
    async def stream(self,client,text,identity,history,avatar_context=None,metrics=None,reply_words=None):
        metrics=metrics if metrics is not None else {};start=time.perf_counter()
        base,model=local_config();messages=[{'role':'system','content':SYSTEM}]
        if reply_words:messages[0]['content']+=f'\nFor this request use exactly one sentence with about {reply_words} spoken words.'
        language_text=text or ' '.join(str(h.get('content','')) for h in history[-2:] if h.get('role')=='user')
        language='German' if re.search(r'\b(?:ich|du|wir|was|wie|ist|das|die|und|ein|eine|bitte|können|heute|dich|nicht|mir)\b|[äöüß]',language_text,re.I) else 'English'
        messages[0]['content']+=f'\nCurrent reply language: {language}. Use {language} for the actual spoken words in this turn.'
        for h in history[-12:]:
            if h.get('role') in ('user','assistant') and isinstance(h.get('content'),str):
                messages.append({'role':h['role'],'content':h['content'][:1800]})
        if identity:messages[0]['content']+='\nCurrent acoustic speaker metadata (session data): '+json.dumps(identity,ensure_ascii=False)
        if avatar_context:messages[0]['content']+='\nObserved browser events (session data): '+json.dumps(avatar_context,ensure_ascii=False)
        # Plain user speech matches the model's conversation training format;
        # burying it inside English JSON reduced German answer quality in tests.
        messages.append({'role':'user','content':text[:3000] if text else 'Respond briefly to the current optional avatar event, or output [silent].'})
        body={'model':model,'messages':messages,'stream':True,'stream_options':{'include_usage':True},
              'max_tokens':320,'temperature':.7,'top_p':.95,
              'chat_template_kwargs':{'enable_thinking':False,'conversational_mode':True}}
        framer=SentenceFramer();first=True
        async with client.stream('POST',base+'/v1/chat/completions',json=body,timeout=30.) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith('data: ') or line[6:]=='[DONE]':continue
                packet=json.loads(line[6:]);choices=packet.get('choices') or []
                if packet.get('usage'):metrics['usage']=packet['usage']
                if packet.get('timings'):metrics['llama_timings']=packet['timings']
                if not choices:continue
                token=choices[0].get('delta',{}).get('content') or ''
                if token and first:metrics['llm_ttft_ms']=round((time.perf_counter()-start)*1000,2);first=False
                for sentence in framer.feed(token):
                    metrics.setdefault('llm_first_sentence_ms',round((time.perf_counter()-start)*1000,2))
                    yield sentence
            for sentence in framer.feed('',final=True):
                metrics.setdefault('llm_first_sentence_ms',round((time.perf_counter()-start)*1000,2));yield sentence
        metrics['llm_total_ms']=round((time.perf_counter()-start)*1000,2)

def needs_tools(text,history,interaction_state):
    from interaction_policy import rule_authorized,cancellation_authorized
    lookup=bool(re.search(r'\b(?:wikipedia|wiki|nachrichten|news|tagesschau|bbc|wetter|weather|recherchier\w*|nachschauen|nachschlagen)\b',text,re.I))
    avatar=not explicit_spin(text) and bool(re.search(r'\b(?:nick\w*|grimasse\w*|zwinker\w*|schiel\w*|augenrollen|nod|wink|grimace|eye\s+roll|funny\s+face|schüttel\w*|shake\s+(?:your\s+)?head|roll\w*\s+(?:deine\s+|die\s+|your\s+)?(?:augen|eyes)|(?:guck\w*|schau\w*|look)\s+(?:mal\s+)?(?:wütend|angry|traurig|sad|bitter|happy|fröhlich))\b',text,re.I))
    return lookup or avatar or bool(interaction_state is not None and (rule_authorized(text) or cancellation_authorized(text)))
