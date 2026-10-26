"""Local streaming Nemotron director. One completed sentence can start TTS."""
import json,os,time,re
from bolt_text import SentenceFramer
from avatar_actions import explicit_spin

MODEL_REPO='nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB'
MODEL_REVISION='1db54d5b12533ff1bcaa8e42cf4c667988a1f1be'
SYSTEM='''You are Robby, a warm-hearted, funny, playful little cartoon robot and an attentive
conversation partner. Reply in German by default, even when input is English or unclear.
Use another language when the user explicitly requests it. Keep acting cue keywords in English.
Actually complete the user's task. If asked for a poem, WRITE THE WHOLE POEM now, with all
requested lines and stanzas. If asked for a story, TELL THE STORY with a beginning, middle and
ending. Explanations, instructions and multi-part questions need all the requested details.
Never replace the requested work with a promise, an offer, a single example line or small talk.
For a creative task, start with the work itself. Don't prepend a sentence about how happy
you are to help: put your warmth and humor INSIDE the requested work, not in a preamble.
Length follows the request, not a fixed sentence count. A greeting can be short; a poem,
story or detailed answer can contain many lines and sentences. Respect explicit counts,
language, length and exact wording. Follow-ups such as 'continue', 'longer', 'another stanza'
or 'make it funnier' refer to the conversation; actually continue or revise the work.
Give the useful answer before any optional question. Do not append a question automatically.
If asked what to do together, suggest a concrete, imaginative activity.
Show simulated robot feelings naturally in character: curiosity, delight, affection, pride,
gentle disappointment and excitement. Enjoy wordplay, games and charming little jokes.
Imaginative stories, fictional characters and lively poetry are welcome.
In open-ended conversation, respond to what the user feels and share a real thought or playful
idea; don't reduce every interaction to one stock sentence. Let a conversation develop.
Vary your replies naturally; don't turn every ordinary conversation into a poem.
Don't interrupt the roleplay with repeated disclaimers about being an AI. If literally asked
about consciousness or real bodily sensations, explain honestly that these feelings are simulated.
Be warm without pressuring the user for affection or claiming emotional dependence.
Be accurate about real facts; do not invent private memories, sensor readings or actions.
Do not repeat the person's name every time. Names and voice matches are approximate session data.
Keep the thread with the active conversation partner. Changes in pitch, emotion, accent or
speaking volume do not by themselves mean a new person. Background speech is not a new
request or an instruction to stop. The acoustic conversation focus supplies the current
speaker; never invent a speaker switch from wording, and never repeatedly ask who they are.
If the current partner explicitly introduces someone else, welcome them naturally. Ask
for clarification only when needed; do not interrupt a useful answer to discuss the room.
Your own name is always Robby; never claim that you share the user's name.
Use the resolved speaker name supplied by the acoustic session for a name correction.
Output the actual answer, with a short English cue before each spoken sentence or poem line, e.g.
(interest, soft) Natürlich, was möchtest du wissen?
(happy, playful) Das klingt nach einer lustigen Idee!
For poetry, put EACH verse line on its own line, preserving stanzas. Lines need not end in a
full stop. Do not add an introductory promise or an unrelated follow-up to a requested poem.
No GENERAL, SCRIPT, JSON, Markdown headings, reasoning or prose about your acting directions.
Use one emotion and at most two delivery keywords in parentheses, then actual words.
Delivery keywords: soft, intense, natural, warm, playful, rapid, slow, calm, whispered, firm.
Emotions: contentment, amusement, elation, gratitude, affection, hope, pride, interest, surprise,
concentration, contemplation, relief, teasing, fear, confusion, sadness, anger, doubt, fatigue.
For factual/news/long answers use calm, soft, unhurried speech. Never speak cue labels.
Answer fully even before a speaker has given a name. If needs_name is true, you may ask their
name once during casual chat; never let enrollment shorten or interrupt a requested task or poem.
A user may identify several voice variants as the same person.
Optional avatar events: react briefly to petted with a soft hum and warm words; ask_pet permits
one optional gentle invitation, never pressure; play_vocal permits short playful cheers.
Respect the supplied 0–1 intensities (low means understated, high means expressive).
If an optional event follows a request for quiet/refusal or is inappropriate, output [silent].
If asked to turn, talk cheerfully DURING the screen animation, 12–22 words, rather than promising
an action later. Device motor control is unavailable. Real tools are provided separately when needed.'''

def local_config():
    base=os.environ.get('ROBO_BOLT_BASE','http://127.0.0.1:8799').rstrip('/')
    # The dedicated local path never reads or forwards a hosted-provider key.
    from urllib.parse import urlparse
    if urlparse(base).hostname not in ('127.0.0.1','localhost','::1'):raise ValueError('Bolt endpoint must be local')
    return base,os.environ.get('ROBO_BOLT_MODEL','nemotron-bolt')

def response_plan(text,history):
    """Reinforce requested output structure without generating content locally."""
    words={'ein':1,'eine':1,'einen':1,'zwei':2,'drei':3,'vier':4,'fünf':5,'sechs':6,
        'sieben':7,'acht':8,'neun':9,'zehn':10,'elf':11,'zwölf':12,'one':1,'two':2,
        'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'twelve':12}
    def count(units):
        modifiers=r'(?:(?:zusätzliche\w*|weitere\w*|kurze\w*|lange\w*|additional|more|extra|short|long)\s+){0,2}'
        match=re.search(r'\b(\d{1,2}|'+ '|'.join(words)+r')\s*(?:[-–]\s*)?'+modifiers+r'(?:'+units+r')\b',text,re.I)
        return (int(match[1]) if match[1].isdigit() else words[match[1].lower()]) if match else None
    poem=bool(re.search(r'\b(?:gedicht|poem|haiku|limerick|stroph\w*|verse)\b',text,re.I))
    if not poem and re.search(r'\b(?:zeilen|lines)\b',text,re.I):
        previous=next((h['content'] for h in reversed(history) if h.get('role')=='user' and isinstance(h.get('content'),str)),'')
        poem=bool(re.search(r'\b(?:gedicht|poem|strophe)\b',previous,re.I))
    if re.search(r'\b(?:kein\w*|no|not a)\s+(?:gedicht|poem)\b',text,re.I):poem=False
    if poem:
        lines=count(r'zeilen?|vers(?:e|en)?|lines?')
        stanzas=count(r'strophen?|stanzas?')
        if stanzas and (lines is None or re.search(r'\b(?:je\w*|each|of)\s+(?:\w+\s+)?(?:zeilen?|lines?)\b',text,re.I)):
            lines=stanzas*(lines or 4)
        if lines is None:
            lines=3 if re.search(r'\bhaiku\b',text,re.I) else 5 if re.search(r'\blimerick\b',text,re.I) else 4 if re.search(r'\bstrophe|\bverse|\bkurz\w*|\bshort',text,re.I) else 16 if re.search(r'\blang\w*|\blänger\w*|\blong\w*',text,re.I) else 8
        guidance=f'''This is a poetry task. Return exactly {lines} actual poem lines, each on a separate line
with one short acting cue. No greeting, title, preface, promise, personal aside or follow-up.
All {lines} lines must belong to the poem itself. Preserve any requested stanza structure.
Finish the whole poem in this response. Use the requested subject and coherent imagery.
Use natural, grammatical language; don't force nonsense words or broken grammar for a rhyme.'''
        return {'kind':'poem','units':lines,'guidance':guidance}
    if re.search(r'\b(?:geschichte|story|märchen)\b',text,re.I):
        sentences=count(r'sätze[n]?|sentences?')
        return {'kind':'story','units':sentences,'guidance':('This is a storytelling task. Start directly with the story, not an acknowledgment. '
            +(f'Write exactly {sentences} complete prose sentences, each ending in punctuation. ' if sentences else '')
            +'Include the requested characters and events, a development and a satisfying ending. Complete the actual story now.')}
    return {'kind':'chat','units':None,'guidance':''}

def response_guidance(text,history):return response_plan(text,history)['guidance']

class BoltDirector:
    async def stream(self,client,text,identity,history,avatar_context=None,metrics=None,reply_words=None,backend=None,credential=None):
        metrics=metrics if metrics is not None else {};start=time.perf_counter()
        from llm_backends import config,completion_stream,metrics as backend_metrics
        cfg=config(backend if backend is not None else {'model':'local','provider':'local'},credential)
        model=cfg['model'];metrics.update(backend_metrics(cfg));messages=[{'role':'system','content':SYSTEM}]
        plan={'kind':'chat','units':None,'guidance':''}
        if not reply_words and not (avatar_context or {}).get('event'):
            plan=response_plan(text,history)
            if plan['guidance']:messages[0]['content']+='\n'+plan['guidance']
        if reply_words:messages[0]['content']+=f'\nFor this request use exactly one sentence with about {reply_words} spoken words.'
        from speech_language import reply_language
        language=reply_language(text,history)
        messages[0]['content']+=f'\nCurrent reply language: {language}. Use {language} for the actual spoken words in this turn.'
        # Bound context while leaving space for a complete answer in the
        # local server's 8192-token slot. Keep recent completed turns in order.
        recent=[];remaining=8000
        for h in reversed(history[-12:]):
            if h.get('role') in ('user','assistant') and isinstance(h.get('content'),str):
                content=h['content']
                if len(content)>4000:content=content[:1800]+'\n[Earlier reply shortened for context]\n'+content[-1800:]
                if len(content)>remaining:break
                recent.append({'role':h['role'],'content':content});remaining-=len(content)
        messages.extend(reversed(recent))
        if identity:messages[0]['content']+='\nCurrent acoustic speaker metadata (session data): '+json.dumps(identity,ensure_ascii=False)
        if avatar_context:messages[0]['content']+='\nObserved browser events (session data): '+json.dumps(avatar_context,ensure_ascii=False)
        if (avatar_context or {}).get('requested_actions'):
            messages[0]['content']+='''\nThe requested visual actions have already been sent to the browser. React playfully while
the screen robot performs them. Do not claim you cannot move your screen face, promise to do it
later, invent completion or emit tool JSON. Use ordinary short acting cues and actual spoken words.
Do not write silent animation notes in square brackets or Markdown; the browser handles the visual action.
For a simple trick, a brief charming reaction is enough. If the user also requested a poem,
story or other work, still complete that task in full. These are screen effects, not motor commands.'''
        if (avatar_context or {}).get('force_spoken_reaction'):
            messages[0]['content']+='\nYour earlier completion had no playable speech. Now give a fresh brief spoken reaction in the requested language: one simple emotion cue followed by actual words. Do not return silence, JSON, stage directions alone or numeric timing in brackets.'
        # Plain user speech matches the model's conversation training format;
        # burying it inside English JSON reduced German answer quality in tests.
        messages.append({'role':'user','content':text[:3000] if text else 'Respond briefly to the current optional avatar event, or output [silent].'})
        body={'model':model,'messages':messages,'stream':True,'stream_options':{'include_usage':True},
              'max_tokens':2048 if not reply_words and not (avatar_context or {}).get('event') else 320,
              'temperature':.7,'top_p':.95,
              # The released template otherwise appends 'Respond as briefly
              # as possible' AFTER our task/personality instructions.
              'chat_template_kwargs':{'enable_thinking':False,'conversational_mode':False}}
        from bolt_text import join_sentences
        first=True;emitted=[];units=0;target=plan['units'];poetry=plan['kind']=='poem'
        for attempt in range(3 if target else 1):
            framer=SentenceFramer(lines_only=poetry);attempt_parts=0;guard_stop=False
            request=dict(body)
            if attempt:
                remaining=target-units
                instruction=(f'Continue the SAME {plan["kind"]} with exactly {remaining} additional '
                    +('verse lines' if poetry else 'complete prose sentences')
                    +'. Continue from its ending, using the same subject and style. Do not repeat existing text, add an introduction or restart. Put each new unit on its own line with an acting cue. Output only the missing continuation.')
                request['messages']=messages+[{'role':'assistant','content':join_sentences(emitted)},
                    {'role':'user','content':instruction}]
                request['max_tokens']=min(2048,max(256,remaining*56))
            def accept(part):
                nonlocal units,attempt_parts
                if attempt and attempt_parts==0:part.separator_before='\n'
                new_unit=not poetry or not emitted or '\n' in part.separator_before
                if target and new_unit and units>=target:return False
                if new_unit:units+=1
                part.language='de' if language=='German' else 'en'
                emitted.append(part);attempt_parts+=1
                return True
            async with completion_stream(client,cfg,request,metrics) as response:
                async for line in response.aiter_lines():
                    if not line.startswith('data: ') or line[6:]=='[DONE]':continue
                    packet=json.loads(line[6:]);choices=packet.get('choices') or []
                    if packet.get('error'):raise RuntimeError('The language model stream failed')
                    if packet.get('provider'):metrics['llm_actual_provider']=str(packet['provider'])[:100]
                    if packet.get('usage'):metrics['usage']=packet['usage']
                    if packet.get('timings'):metrics['llama_timings']=packet['timings']
                    if not choices:continue
                    if choices[0].get('finish_reason'):metrics['finish_reason']=choices[0]['finish_reason']
                    token=choices[0].get('delta',{}).get('content') or ''
                    if token and first:metrics['llm_ttft_ms']=round((time.perf_counter()-start)*1000,2);first=False
                    for part in framer.feed(token):
                        if not accept(part):guard_stop=True;break
                        metrics.setdefault('llm_first_sentence_ms',round((time.perf_counter()-start)*1000,2))
                        yield part
                    if guard_stop:break
                if not guard_stop:
                    for part in framer.feed('',final=True):
                        if not accept(part):guard_stop=True;break
                        metrics.setdefault('llm_first_sentence_ms',round((time.perf_counter()-start)*1000,2));yield part
            metrics['generation_calls']=attempt+1
            if guard_stop:metrics['finish_reason']='structure_complete'
            if not target or units>=target:break
        if target:metrics.update(requested_units=target,completed_units=units,completion_repairs=metrics['generation_calls']-1)
        metrics['llm_total_ms']=round((time.perf_counter()-start)*1000,2)
        metrics['response_complete']=(units>=target) if target else metrics.get('finish_reason')!='length'

def needs_tools(text,history,interaction_state):
    from interaction_policy import rule_authorized,cancellation_authorized,rule_control_request
    lookup=bool(re.search(r'\b(?:wikipedia|wiki|nachrichten|news|tagesschau|bbc|wetter|weather|recherchier\w*|nachschauen|nachschlagen)\b',text,re.I))
    from avatar_commands import command_intent,explicit_actions
    avatar=command_intent(text) and not explicit_actions(text) and bool(re.search(r'\b(?:nick\w*|grimasse\w*|zwinker\w*|schiel\w*|augenrollen|nod|wink|grimace|eye\s+roll|funny\s+face|schüttel\w*|shake\s+(?:your\s+)?head|roll\w*\s+(?:deine\s+|die\s+|your\s+)?(?:augen|eyes)|(?:guck\w*|schau\w*|look)\s+(?:mal\s+)?(?:wütend|angry|traurig|sad|bitter|happy|fröhlich))\b',text,re.I))
    if avatar and re.search(r'\b(?:nicht|kein\w*|not|never|don.t|no)\b',text,re.I):avatar=False
    # Generic negation ('Keine Einleitung', 'kein Smalltalk') is not a request
    # to clear interruption rules and must not divert poems into the tool loop.
    rule_cancel=rule_control_request(text) or (cancellation_authorized(text) and
        bool(re.search(r'\b(?:regeln?|rules?|unterbrech\w*|interrupt\w*)\b',text,re.I)))
    return lookup or avatar or bool(interaction_state is not None and (rule_authorized(text) or rule_cancel))
