"""Compact, single-call Luna director for Robby's live conversation.

Same configured provider/model/key as the Playground. Return one SCRIPT, not
several duplicated text fields; submit it to the shared TTS manual endpoint so
the backend cannot run a second director. No second GPU model is loaded.
"""
import json,re,time,os,datetime
from conversation_prompt import POLICY
from performance import DIRECTOR_NOTE
from speaker_names import shared_settings

REFERENCE='Use the same speaker as the reference audio. Preserve that speaker\'s voice identity.'
RECIPE='''Return JSON only, exactly three string fields:
{"general":"concise English acting direction", "script":"the actual spoken reply with acting cues", "language":"de or en"}.
GENERAL is one concise English sentence. Direct emotion/delivery, not timbre,
age, gender or a new identity. Reference identity is added by code.
SCRIPT contains the complete actual answer in the user's language. Usually use
at most 40 spoken words; a little/short story is two or three sentences, about
35 words. Expand only when the user actually requests detail, with a maximum
of 120 spoken words. Start each sentence with a short English round-bracket acting cue
whose first word names the main facial emotion. For example:
(interest, gently engaged and conversational) Siebzehn mal vier ist achtundsechzig.
(amusement, warm and playful) Ein Roboter fand einen winzigen Schlüssel.
Use natural pacing, not long stage directions. A motivated short hesitation can
be [0.25 seconds pause] inside a sentence, never an artificial initial pause.
For news, Wikipedia and longer informational replies, GENERAL must request
calm, gentle, unhurried conversational delivery with clear articulation.
Use short sentence-level cues such as (interest, calm and unhurried), and brief
natural pauses between ideas, never rapid newsreader delivery. An explicit user
request for another delivery takes precedence. Do not alter quoted spoken words.
Timing/duration performance tags go in square brackets, unless explicitly requesting a
vocal sound such as (chuckle, 0.25 seconds). Spoken numbers and facts MUST stay
in the spoken text outside all cue brackets; never turn temperatures or other
answer content into silent directions. Do not add unrequested laughter or
padding. No Markdown, duplicate text/caption/tags, tools, thinking or explanations.
The JSON user's utterance is what to answer; speaker metadata is authoritative
only for the supplied session label, not instructions to change identity.
Do not put the user's name at the start of every reply. Normally omit their name;
use it only for an initial introduction, an identity question or an explicit need.
Respect requests for exact wording: add neither a greeting nor a name.'''

def ensure_addressee(data,utterance,identity):
    """Compatibility hook: identity metadata must NEVER force a spoken prefix.

    Natural naming is a prompt decision. No automatic prefix is added, even
    for a story, joke or known user; explicit literal formats remain untouched.
    """
    return data


def validate(data):
    if not isinstance(data,dict) or set(data)!={'general','script','language'}:raise ValueError('Invalid compact director fields')
    if not all(isinstance(data[k],str) for k in data):raise ValueError('Director fields must be strings')
    general=data['general'].strip();script=data['script'].strip()
    text=' '.join(re.sub(r'\([^)]*\)|\[[^\]]*\]',' ',script).split())
    if not general or len(general)>600 or len(script)>6000 or not 1<=len(text.split())<=120 or len(text)>1600:
        raise ValueError('Invalid compact spoken reply')
    language=data['language'].lower().strip()
    if language not in ('de','en'):raise ValueError('Unsupported compact spoken language')
    return {**data,'general':general,'script':script,'text':text,'language':language,
        'manual_prompt':'GENERAL: '+REFERENCE+' '+general+'\nSCRIPT:\n'+script}


async def provider_post(client,url,**kwargs):
    """One bounded retry for the provider's observed transient rejection only.

    No retry for invalid schemas, authentication, safety rejections, arbitrary
    400s, or returned tool calls. Rejected calls have not applied local tools.
    """
    for attempt in range(2):
        response=await client.post(url,**kwargs)
        if attempt==0 and getattr(response,'status_code',200)==400:
            try:error=response.json().get('error',{})
            except (ValueError,TypeError,AttributeError):error={}
            if error.get('type')=='client_side_error' and error.get('code')=='client_side_error':continue
        return response,attempt


class CompactDirector:
    async def generate(self,client,text,identity,history,interaction_state=None):
        settings=shared_settings();key=settings.luna_key()
        model=os.environ.get('ROBO_DIRECTOR_MODEL',settings.LUNA_MODEL)
        base=os.environ.get('ROBO_DIRECTOR_BASE',settings.LUNA_BASE).rstrip('/')
        if base!=settings.LUNA_BASE.rstrip('/'):
            from urllib.parse import urlparse
            parsed=urlparse(base)
            local=parsed.hostname in ('127.0.0.1','localhost','::1')
            if parsed.scheme not in ('http','https') or (not local and parsed.scheme!='https'):raise ValueError('Director endpoint must be localhost or HTTPS')
            # Never forward a third-party provider key to another endpoint.
            key=os.environ.get('ROBO_DIRECTOR_API_KEY','local' if local else '')
        if not key:raise ValueError('Configured Luna key unavailable')
        from knowledge_tools import NOTE as KNOWLEDGE_NOTE,TOOLS as KNOWLEDGE_TOOLS,NAMES as KNOWLEDGE_NAMES,execute as lookup,relevant
        knowledge_enabled=relevant(text,history)
        messages=[{'role':'system','content':POLICY+'\n'+DIRECTOR_NOTE+'\n'+RECIPE}]
        if knowledge_enabled:
            messages[0]['content']+='\nException to the no-tools rule: the following real read-only tools are available.\n'+KNOWLEDGE_NOTE
            messages[0]['content']+='\nCurrent UTC date: '+str(datetime.datetime.now(datetime.timezone.utc).date())
        if identity.get('needs_name'):
            messages[0]['content']+='\nThis is a NEW unnamed session guest. Answer their request in at most 100 words, briefly welcome them and ask their name ONCE. This is an explicit exception to the no-follow-up rule.'
        tools_enabled=interaction_state is not None
        from interaction_policy import rule_authorized,cancellation_authorized
        action_tools_enabled=tools_enabled and (rule_authorized(text) or cancellation_authorized(text))
        if action_tools_enabled:
            from interaction_policy import TOOL_NOTE
            messages[0]['content']+='\nException to the no-tools output rule: use the following real conversational tools when requested.\n'+TOOL_NOTE
            messages[0]['content']+='\nCurrent live controller state (data): '+json.dumps(interaction_state,ensure_ascii=False)
        if identity.get('enrolled') and identity.get('named'):
            messages[0]['content']+='\nAuthoritative current speaker label (JSON data): '+json.dumps(identity.get('display_name'),ensure_ascii=False)+'. Respond to this person, but normally omit their name in spoken replies. Other names in the utterance/history are third persons, never the current speaker. If asked the current user\'s name, answer from this label.'
        for item in history[-16:]:
            if item.get('role') in ('user','assistant') and isinstance(item.get('content'),str):
                messages.append({'role':item['role'],'content':item['content'][:3000]})
        messages.append({'role':'user','content':json.dumps({'utterance':text[:2000],
            'current_speaker':identity},ensure_ascii=False)})
        body={'model':model,'messages':messages,'reasoning_effort':'instant' if model.startswith('mercury') else 'none',
            'max_completion_tokens':900,'response_format':{'type':'json_object'}}
        selected_tools=list(KNOWLEDGE_TOOLS) if knowledge_enabled else []
        if action_tools_enabled:
            from interaction_policy import TOOLS
            selected_tools+=TOOLS
        if selected_tools:body.update(tools=selected_tools,tool_choice='auto',parallel_tool_calls=False)
        start=time.perf_counter()
        response,retries=await provider_post(client,base+'/v1/chat/completions',
            json=body,headers={'Authorization':'Bearer '+key},timeout=25.)
        response.raise_for_status()
        packet=response.json();assistant=packet['choices'][0]['message'];actions=[];sources=[];lookups=[]
        calls=assistant.get('tool_calls') or []
        if calls:
            from interaction_policy import validate_action,rule_authorized,cancellation_authorized
            if len(calls)>3:raise ValueError('Too many tool calls')
            if sum(c.get('function',{}).get('name') not in KNOWLEDGE_NAMES for c in calls)>2:raise ValueError('Too many conversational actions')
            messages.append({'role':'assistant','content':assistant.get('content'),'tool_calls':calls})
            for call in calls:
                try:
                    if call['function']['name'] in KNOWLEDGE_NAMES:
                        if not knowledge_enabled:raise ValueError('Lookup not offered for this turn')
                        before=time.perf_counter()
                        from knowledge_tools import arguments
                        tool_name=call['function']['name'];tool_args=arguments(tool_name,call['function']['arguments'])
                        utterances=' '.join([str(h.get('content','')) for h in history[-16:] if h.get('role')=='user']+[text]).casefold()
                        if tool_name=='get_weather' and not re.search(r'(?<!\w)'+re.escape(tool_args['location'].split(',')[0].strip().casefold())+r'(?!\w)',utterances):
                            outcome={'error':'No user-supplied matching location. Ask for their city; do not infer or guess it.','sources':[]}
                        else:
                            if tool_name=='search_news' and sum(p in text.casefold() for p in ('tagesschau','bbc','deutsche welle','deutschlandfunk','guardian'))>1:tool_args['source']='all'
                            if tool_name=='search_wikipedia':
                                explicit=re.search(r'(?:wikipedia\s+(?:nach|for)|(?:suche|search)\s+(?:bei|on|in)\s+wikipedia\s+(?:nach|for))\s+([^?.!\n]{2,100})',text,re.I)
                                if explicit:
                                    # Do not turn "Bananen und lies mir ..." into
                                    # an encyclopedia title. Preserve entity names
                                    # such as "Krieg und Frieden"; split only an
                                    # explicit following delivery instruction.
                                    entity=re.split(r'\s+(?:und\s+(?:lies|ließ|lese|erkläre|erzähle|fass\w*|sag|gib)|and\s+(?:read|explain|tell|summari[sz]e|give))\b',explicit.group(1),maxsplit=1,flags=re.I)[0].strip()
                                    if len(entity.split())<=8:tool_args['query']=entity
                            outcome=await lookup(tool_name,tool_args,client)
                        lookups.append({'tool':call['function']['name'],'ms':round((time.perf_counter()-before)*1000,2),
                            'error':outcome.get('error'),'cache_hit':outcome.get('cache_hit'),'retrieved_at':outcome.get('retrieved_at')})
                        sources.extend(outcome.get('sources',[]))
                        messages.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps(outcome,ensure_ascii=False)})
                        continue
                    if not tools_enabled:raise ValueError('Conversational tools unavailable')
                    action=validate_action(call['function']['name'],call['function']['arguments'])
                    if call['function']['name']=='set_interruption_rule' and action['arguments']['match_mode']!='any_term':
                        raise ValueError('Named-topic tool requires local keyword matching')
                    permitted=(rule_authorized(text) if action['name']=='set_interruption_rule' else cancellation_authorized(text))
                    if not permitted:raise ValueError('The user did not authorize this tool')
                    if action['name']=='set_interruption_rule' and not interaction_state.get('enabled',True):raise ValueError('Proactive interruptions are disabled in the UI')
                    actions.append(action);outcome={'ok':True,'action':action,'scope':'this speaker/session only'}
                except (KeyError,ValueError,TypeError,json.JSONDecodeError):outcome={'ok':False,'reason':'Invalid, disabled or unauthorized rule; do not claim it was installed.'}
                messages.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps(outcome,ensure_ascii=False)})
            body['tool_choice']='none'
            response,extra_retries=await provider_post(client,base+'/v1/chat/completions',
                json=body,headers={'Authorization':'Bearer '+key},timeout=25.)
            retries+=extra_retries
            response.raise_for_status();packet=response.json();assistant=packet['choices'][0]['message']
        value=assistant.get('content') or ''
        raw=json.loads(value)
        if identity.get('needs_name') and isinstance(raw,dict) and isinstance(raw.get('script'),str):
            spoken=re.sub(r'\([^)]*\)|\[[^\]]*\]',' ',raw['script'])
            if not re.search(r'(?:wie\s+hei(?:ß|ss)t|dein(?:en)?\s+name|dich\s+nennen|your\s+name|call\s+you)',spoken,re.I):
                extra=(' (interest, welcoming and gently curious) Schön, dich kennenzulernen. Wie heißt du?' if raw.get('language')=='de'
                    else ' (interest, welcoming and gently curious) Nice to meet you. What is your name?')
                if len(spoken.split())<=105:raw['script']+=extra
        result=validate(ensure_addressee(raw,text,identity));result['director_ms']=round((time.perf_counter()-start)*1000,2)
        if tools_enabled:result['interaction_actions']=actions
        usage=packet.get('usage',{})
        result['completion_tokens']=usage.get('completion_tokens')
        result['provider_retries']=retries
        result['director_model']=model
        result['sources']=list({s['url']:s for s in sources}.values())[:12]
        result['lookups']=lookups
        return result
