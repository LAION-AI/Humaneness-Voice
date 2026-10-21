"""Validated session-only conversational tools; no device or identity tools."""
import hashlib,json,re
from dataclasses import dataclass
from collections import deque
from speaker_turns import words

TOOL_NOTE='''You can configure Robby's real live interruption controller.
Only when the user EXPLICITLY asks you to interrupt them under a future
condition, call set_interruption_rule. Convert the user's condition into a
clear description and a few literal trigger terms/inflections in their language.
Include common German AND English forms for named topics when possible (e.g.
Banane, Bananen, banana, bananas), because interim multilingual ASR can switch
languages. Never promise an interruption without installing the matching tool.
Use set_interruption_rule with any_term for named topics/words (bananas, etc.).
Use set_semantic_interruption_rule ONLY when the condition needs meaning rather
than named words. Named-topic games must not depend on another decision API call.
reply is the SHORT exact interjection to speak, not an explanation (e.g.
"Stopp, stopp!"). Never register rules just because a topic was mentioned.
When asked to stop/cancel the game or rule, call clear_interruption_rules.
Rules belong to the supplied speaker/session, never to background people.
After tool results, acknowledge the actual installed rule briefly using the
normal three-field acting JSON. Do not claim rules were installed without a
successful tool result. These tools do not control devices or authenticate
speakers. If proactive interruptions are disabled, do not promise to interrupt.'''

TOOLS=[{'type':'function','function':{'name':'set_interruption_rule','description':'Register an explicitly requested live condition for interrupting the current user BEFORE their turn ends.','strict':True,
    'parameters':{'type':'object','properties':{
        'condition':{'type':'string','description':'The agreed condition, e.g. user talks about bananas.'},
        'trigger_terms':{'type':'array','items':{'type':'string'},'description':'Literal words/inflections, e.g. Banane, Bananen, banana, bananas.'},
        'match_mode':{'type':'string','enum':['any_term']},
        'reply':{'type':'string','description':'Exact brief spoken interruption, e.g. Stopp, stopp!'}},
        'required':['condition','trigger_terms','match_mode','reply'],'additionalProperties':False}}},
    {'type':'function','function':{'name':'set_semantic_interruption_rule','description':'Register an explicitly requested meaning-dependent condition, ONLY when literal named topic/word triggers are insufficient. Requires the decision API during live speech.','strict':True,
        'parameters':{'type':'object','properties':{'condition':{'type':'string'},'reply':{'type':'string'}},
            'required':['condition','reply'],'additionalProperties':False}}},
    {'type':'function','function':{'name':'clear_interruption_rules','description':'Cancel the current user’s explicitly cancelled interruption game/rules.','strict':True,
        'parameters':{'type':'object','properties':{},'required':[],'additionalProperties':False}}}]

def validate_action(name,arguments):
    if isinstance(arguments,str):arguments=json.loads(arguments)
    if not isinstance(arguments,dict):raise ValueError('Tool arguments must be an object')
    if name=='set_semantic_interruption_rule':
        if set(arguments)!={'condition','reply'}:raise ValueError('Invalid semantic tool fields')
        name='set_interruption_rule';arguments={**arguments,'trigger_terms':[],'match_mode':'semantic'}
    if name=='clear_interruption_rules':
        if arguments:raise ValueError('Unexpected clear arguments')
        return {'name':name,'arguments':{}}
    if name!='set_interruption_rule' or set(arguments)!={'condition','trigger_terms','match_mode','reply'}:
        raise ValueError('Unknown or malformed conversational tool')
    condition,terms,mode,reply=(arguments[k] for k in ('condition','trigger_terms','match_mode','reply'))
    if (not isinstance(condition,str) or not 1<=len(condition.strip())<=400 or
        not isinstance(reply,str) or not 1<=len(reply.strip())<=180 or len(words(reply))>24 or
        mode not in ('any_term','semantic') or not isinstance(terms,list) or len(terms)>12 or
        (mode=='any_term' and not terms)):
        raise ValueError('Invalid rule bounds')
    if any(not isinstance(t,str) or not 1<=len(t.strip())<=60 or not words(t) for t in terms):raise ValueError('Invalid trigger term')
    return {'name':name,'arguments':{'condition':condition.strip(),'trigger_terms':list(dict.fromkeys(t.strip() for t in terms)),
        'match_mode':mode,'reply':reply.strip()}}

def rule_authorized(text):
    if re.search(r'\b(?:nicht(?:\s+mehr)?\s+unterbrech|unterbr\w*\s+(?:du\s+)?mich\s+(?:bitte\s+)?nicht|(?:do\s+not|don.t)\s+interrupt)',text,re.I):return False
    return bool(re.search(r'\b(?:unterbr(?:ich(?:st)?|ech(?:e|en|t)|icht)|interrupt|stop\s+me|stopp?e?\s+mich|fall\s+mir|dazwischen(?:reden|sprechen))\b',text,re.I))

def cancellation_authorized(text):
    return bool(re.search(r'\b(?:beenden|beende|aufhören|höre?\s+auf|keine?|nicht|vorbei|vergiss|lösche|cancel|stop|end|disable|forget|don.t|do\s+not)\b',text,re.I))

def rule_control_request(text):
    """Do not interject over a request to cancel/reconfigure the game itself."""
    return bool(re.search(r'\b(?:beende\s+(?:bitte\s+)?(?:das\s+)?(?:spiel|regel)|'
        r'(?:cancel|stop|end|disable|clear)\s+(?:the\s+)?(?:game|rules?)|'
        r'unterbr\w*\s+mich\s+(?:bitte\s+)?nicht|(?:do\s+not|don.t)\s+interrupt|'
        r'(?:lösche|vergiss)\s+(?:bitte\s+)?(?:die\s+)?(?:regel|spiel))\b',text,re.I))

class VoiceEvidence:
    """Bounded sequential evidence, not best-of-many lucky speaker windows."""
    def __init__(self):self.sid=None;self.total=0.;self.weight=0.;self.end=-100.;self.initial_score=None;self.observations=deque(maxlen=8);self.complete_total=0.;self.complete_weight=0.
    def update(self,sid,score,end,quiet=False,complete=True):
        if quiet or sid!=self.sid:
            # The first observation already covers a complete acoustic
            # context, not only the scheduler's first 320-ms hop. Giving it
            # quarter weight let a sparse later partial erase contrary proof.
            self.sid=sid;self.total=0.;self.weight=0.;self.end=end-1.28
            self.initial_score=None
            self.complete_total=0.;self.complete_weight=0.
            self.observations.clear()
        if complete and self.initial_score is None:self.initial_score=score
        weight=max(.32,min(1.28,end-self.end))
        self.total+=score*weight;self.weight+=weight;self.end=end
        if complete:
            qualified_weight=1.28 if not self.complete_weight else weight
            self.complete_total+=score*qualified_weight;self.complete_weight+=qualified_weight
            # Scheduler runs every 320 ms. Keep recovery observations at
            # >=600-ms spacing; otherwise its four-window 1.8-s evidence
            # requirement becomes impossible with the faster cadence.
            if not self.observations or end-self.observations[-1][0]>=.60:
                self.observations.append((end,score))
        return self.total/self.weight

    @property
    def complete_mean(self):
        return self.complete_total/self.complete_weight if self.complete_weight else None

    def recovered(self,threshold):
        # Recovery is sequential proof, not a lucky best-of-many window. Four
        # recent, separated high-quality checks must ALL support the person.
        tail=list(self.observations)[-4:]
        return len(tail)==4 and tail[-1][0]-tail[0][0]>=1.8 and all(s>=max(.60,threshold) for _,s in tail)

def live_voice_verified(score,margin,fast,sid,teacher_threshold,fast_threshold,long_vote=None,require_long=False,evidence_mean=None,initial_score=None,require_initial=False,recovered=False):
    """Proactive interjections need a strong teacher, never a weak-score rescue.

    Deliberately separate from the user's permissive ordinary-turn threshold.
    An early prototype's low-teacher/strong-student branch accepted a retained
    unknown synthetic voice; removing that branch is the conservative fix.
    """
    long_ok=not require_long or bool(long_vote and long_vote.get('candidate_id')==sid and
        long_vote.get('candidate_similarity',-1)>=.45 and long_vote.get('candidate_margin',0)>=.10)
    stable=score if evidence_mean is None else evidence_mean
    admitted=not require_initial or (initial_score is not None and initial_score>=max(.45,teacher_threshold)) or recovered
    return (admitted and long_ok and stable>=max(.50,teacher_threshold-.05) and score>=max(.55,teacher_threshold) and margin>=.10 and
        fast.get('candidate_id')==sid and fast.get('candidate_similarity',-1)>=max(.45,fast_threshold-.17))

@dataclass
class InterruptionRule:
    id:str
    owner:str
    condition:str
    trigger_terms:list
    match_mode:str
    reply:str

    def matches(self,text,partial=False):
        if self.match_mode!='any_term':return False
        tokens=words(text)
        for term in self.trigger_terms:
            needle=words(term)
            if any(tokens[i:i+len(needle)]==needle for i in range(len(tokens)-len(needle)+1)):return True
            if len(needle)==1 and len(needle[0])>=4:
                if any(re.fullmatch('(?:'+re.escape(needle[0])+'){2,}',token) for token in tokens):return True
                if partial and tokens and len(tokens[-1])>=5 and 0<len(needle[0])-len(tokens[-1])<=2 and needle[0].startswith(tokens[-1]):return True
        return False

def face_corroborated_voice(score,margin,fast,long_vote,sid,threshold,evidence_mean,initial_score):
    """Only called with a fresh positive Face-ID AND time-aligned mouth motion.
    Never a route for an audio-only unknown, static face, or expired face match.
    """
    floor=max(.32,threshold)
    return bool(score>=floor and margin>=.12 and evidence_mean is not None and evidence_mean>=floor
        and initial_score is not None and initial_score>=floor
        and fast.get('candidate_id')==sid and fast.get('candidate_similarity',-1)>=.30
        and long_vote.get('candidate_id')==sid and long_vote.get('candidate_similarity',-1)>=.50
        and long_vote.get('candidate_margin',0)>=.12)

class RuleBook:
    def __init__(self):self.rules={}
    def apply(self,actions,owner,utterance):
        if not isinstance(actions,list) or len(actions)>2:raise ValueError('At most two conversational actions')
        installed=[];pending=dict(self.rules)
        for raw in actions:
            if not isinstance(raw,dict):raise ValueError('Malformed conversational action')
            action=validate_action(raw.get('name'),raw.get('arguments'));name=action['name']
            if name=='clear_interruption_rules':
                if not cancellation_authorized(utterance):raise ValueError('No explicit cancellation')
                pending={k:r for k,r in pending.items() if r.owner not in (owner,'session')};continue
            if not rule_authorized(utterance):raise ValueError('No explicit interruption permission')
            args=action['arguments'];ident=hashlib.sha256((owner+json.dumps(args,sort_keys=True,ensure_ascii=False)).encode()).hexdigest()[:12]
            if ident not in pending and len(pending)>=4:raise ValueError('At most four live rules')
            rule=InterruptionRule(ident,owner,**args);pending[ident]=rule;installed.append(ident)
        self.rules=pending
        return installed
    def public(self):return [dict(id=r.id,owner=r.owner,condition=r.condition,reply=r.reply,match_mode=r.match_mode,trigger_terms=r.trigger_terms) for r in self.rules.values()]
    def for_owner(self,owner):return [r for r in self.rules.values() if r.owner in (owner,'session')]
