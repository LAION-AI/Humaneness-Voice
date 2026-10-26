"""Small task-resolution hints; all actual creative content comes from the LLM."""
import re

# Common compound joke nouns also establish the subject of "noch einen".
# Do not match arbitrary *witz strings (e.g. a place name or "schwitze").
JOKE=re.compile(r'\b(?:(?:(?:katzen|hunde|pinguin|roboter|tier|flach|kinder|schüler)?)witz(?:e|en)?|jokes?)\b',re.I)
OTHER_TASK=re.compile(r'\b(?:gedicht|poem|geschichte|story|märchen|nachrichten|news|erklär\w*)\b',re.I)
FOLLOWUP=re.compile(r'^\s*(?:(?:und|bitte|haha|ha|okay|ja)[,!\s]+)*(?:noch\s+(?:ein\w*|mehr)|(?:erzähl\w*\s+(?:mir\s+)?)?(?:ein\w*\s+)?(?:weiter\w*|ander\w*)|(?:den|diesen|das|ihn)\b|mach\w*\s+(?:ihn|den|das)\b)',re.I)

def joke_task(text,history):
    """Resolve only explicit joke requests or bounded anaphoric follow-ups.

    A new explicit topic/task wins; a random later question must not inherit
    joke mode. Do not carry a previous turn's exact sentence count forward.
    """
    if re.search(r'\b(?:kein\w*|no|not a)\s+(?:witz\w*|jokes?)\b',text,re.I):return None
    explicit=bool(JOKE.search(text));prior=None
    if not explicit:
        followup_text=re.sub(r'^\s*(?:der|das) war (?:lahm|langweilig|nicht lustig)[.!]\s*','',text,flags=re.I)
        if OTHER_TASK.search(text) or not FOLLOWUP.search(followup_text):return None
        for item in reversed(history):
            if item.get('role')!='user' or not isinstance(item.get('content'),str):continue
            previous=item['content']
            if JOKE.search(previous):prior=previous;break
            if OTHER_TASK.search(previous):return None
        if prior is None:return None
    retell=bool(re.search(r'\b(?:ausführlicher|verlänger\w*|länger\w*|umschreib\w*|überarbeit\w*|retell|expand|rewrite|longer)\b',text,re.I))
    return {'retell':retell,'guidance':'''This request is a JOKE task, including elliptical follow-ups such as "noch einen".
Tell the actual complete joke immediately, with a coherent setup and a clear finishing punchline.
Do not answer with a promise, small talk, a name question or an unrelated activity. No introductory
"here is another one". A tiny acknowledgment is not the joke. Use natural, grammatical German.
Keep the most recently requested subject/characters unless the user explicitly changes them.
Do not import robot/body details into a newly requested cat joke merely because you are a robot.
Humor should come from a recognizable twist, not arbitrary nonsensical events or a missing ending.
The final sentence must finish the joke, not leave a dialogue question waiting for its answer.
Use enough room for the setup AND payoff, normally a few sentences; follow an explicit request
for a one-liner, length or sentence count. A sentence-level streaming boundary is not answer end.
''' + ('''The user wants the PREVIOUS joke retold/expanded: preserve its premise and characters,
add meaningful dialogue/detail, and include its payoff. Finish the full requested retelling now.''' if retell else '''The user wants a NEW joke. Invent a different situation AND punchline. Do not repeat or
paraphrase an earlier joke from this conversation. "Another" means another complete joke,
not another line of the previous joke. Do not carry an earlier exact sentence count into a new task.''')}

INTERRUPTION_PREFIX='[My preceding reply was interrupted.'
INTERNAL_NOTE=re.compile(r'\[My preceding reply was interrupted\.[^\]]*(?:\]|$)',re.I)

def clean_reply(text):
    """Remove only the obsolete internal playback annotation, not user cues."""
    return INTERNAL_NOTE.sub('',text).strip()

def memory_content(record,interrupted=False):
    """Spoken/generated words only; playback status is separate metadata."""
    return clean_reply(str(record.get('reply') or ''))

def memory_entry(record,interrupted=False):
    return {'role':'assistant','content':memory_content(record),'playback_interrupted':bool(interrupted)}

def history_for_model(history):
    """Keep playback uncertainty out of assistant speech examples/API fields."""
    messages=[];interrupted=[]
    for item in history:
        if item.get('role') not in ('user','assistant') or not isinstance(item.get('content'),str):continue
        text=item['content']
        if item['role']=='assistant':
            if item.get('playback_interrupted') or INTERNAL_NOTE.search(text):interrupted.append(len(messages)+1)
            text=clean_reply(text)
        messages.append({'role':item['role'],'content':text})
    note=('\nINTERNAL PLAYBACK STATE, never spoken or quoted: assistant messages at positions '
          +', '.join(map(str,interrupted))+' in the following recent history were interrupted. '
          'Their generated content remains useful context, but only a prefix may have been heard. '
          'Do not assume the whole answer was heard or automatically repeat it.') if interrupted else ''
    return messages,note


def acknowledgement_only(text):
    """Bounded repair for a promise in place of a creative task, not a judge.

    A real short joke/story must remain short when requested. Only detect a
    small acknowledgment/offer; never classify arbitrary concise prose as bad.
    """
    value=' '.join(re.findall(r'\w+',text.casefold()))
    if not value:return True
    if len(value.split())>30:return False
    if re.search(r'\b(?:es war|eines tages|einmal|sagte|fragte|rief|antwortete|ging|fand|traf|once|asked|said)\b',value):return False
    return bool(re.search(r'\b(?:bitte|wunsch)\b.*\b(?:erfülle|erfüllen|nachkomme)\b',value)
        or re.fullmatch(r'(?:klar|gerne|natürlich|okay|ja|sehr gerne|of course|sure)(?: \w+){0,6}',value)
        or re.fullmatch(r'(?:ich (?:kann|werde|möchte)|i (?:can|will|would)) .{0,160}',value)
        or re.fullmatch(r'(?:hier ist|here is) .{0,100}',value))
