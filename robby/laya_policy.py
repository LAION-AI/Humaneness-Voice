"""Pure bounded context/typed-question adapter for local multilingual Laya."""
import re
from stop_commands import has_stop_intent
MODEL='convaiinnovations/laya-multilingual'
REVISION='e4e9ddf21a7b1903b7acffd8814ad4307bf63a67'
SOURCE_COMMIT='ffc142012af63d015fa19f05f29f9c5ff329a2cc'
THRESHOLD=.75

STOP={'type':'choice','instructions':'Should the agent stop speaking and listen to this user?',
    'criteria':{'A':'The user requests stop, pause, silence, or to let them speak; ignore preceding agent echo.',
                'B':'The user wants continuation, asks about/quotes the word stop, gives a backchannel, or does not ask to interrupt.'}}

def prepare(context):
    text=str(context.get('user_partial',''))[-1200:]
    candidates=[text]+[str(t)[-1200:] for t in context.get('asr_candidates',[])[:3]]
    text=max(candidates,key=lambda x:len(x.split()))
    speaking=bool(context.get('agent_speaking'))
    state=('AGENT: '+str(context.get('agent_full_reply',''))[:650]+'\nUSER: '+text) if speaking else 'USER: '+text
    if not speaking:
        history=context.get('history',[])[-4:]
        state='RECENT CONVERSATION (background only):\n'+'\n'.join(str(m.get('role',''))+': '+str(m.get('content',''))[:350] for m in history if isinstance(m,dict))+'\nCURRENT '+state
    if speaking:
        try:played=max(0.,min(3600.,float(context.get('agent_played_seconds',0.))))
        except (TypeError,ValueError):played=0.
        state=state.replace('\nUSER:',f'\nPLAYBACK: {played:.2f} seconds actually played; remaining agent text is planned.\nUSER:',1)
    questions={};ids=[]
    specifications=[('stop',STOP)] if speaking else []
    if not speaking and context.get('proactive_enabled'):
        for i,r in enumerate(context.get('rules',[])[:4]):
            if not isinstance(r,dict) or not isinstance(r.get('id'),str):continue
            key='rule_'+str(i);ids.append((key,r['id']))
            specifications.append((key,{'type':'choice','instructions':'Does this agreed condition apply to the current USER words: '+str(r.get('condition',''))[:400],
                'criteria':{'A':'The USER speech satisfies: '+str(r.get('condition',''))[:400],
                            'B':'The USER speech does NOT satisfy: '+str(r.get('condition',''))[:400]}}))
    for key,question in specifications:
        questions[key+'_forward']=question
        questions[key+'_reverse']={**question,'criteria':{'A':question['criteria']['B'],'B':question['criteria']['A']}}
    return state,questions,ids

def interpret(context,answers,ids):
    def probability(key):
        first=answers[key+'_forward']['probabilities']['A'];second=answers[key+'_reverse']['probabilities']['B']
        if not all(isinstance(x,(float,int)) and 0<=x<=1 for x in (first,second)):raise ValueError('Invalid local probabilities')
        return (first+second)/2
    scores={};action='listen';rule_id=None
    if context.get('agent_speaking'):
        scores['stop']=probability('stop')
        candidates=[str(context.get('user_partial',''))]+[str(t) for t in context.get('asr_candidates',[])[:3]]
        negated=lambda t:bool(re.search(r'\b(?:höre?\s+(?:bitte\s+)?nicht\s+auf|nicht\s+aufhören|(?:do\s+not|don.t|never)\s+stop|keep\s+(?:talking|speaking)|weiterreden)\b',t,re.I))
        if not all(negated(t) for t in candidates if t.strip()) and (any(has_stop_intent(t) for t in candidates) or scores['stop']>=THRESHOLD):action='stop_reply'
    else:
        for key,ident in ids:scores[ident]=probability(key)
        if scores and context.get('proactive_enabled'):
            ident=max(scores,key=scores.get)
            required=.90 if ident=='helpful-interjection' else THRESHOLD
            if scores[ident]>=required:action='interrupt_user';rule_id=ident
    return {'action':action,'rule_id':rule_id},scores
