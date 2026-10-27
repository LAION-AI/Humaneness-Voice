"""Explicit, session-only invitations and background permissions.

Only an admitted conversation participant (or the typed UI) can change this
policy. A background transcript is never interpreted as an invitation.
"""
import re,time
from difflib import SequenceMatcher

NAME=r"([^\W\d_][\w’'-]{1,39})"
BAD={'ihm','ihr','ihnen','denen','dem','die','der','das','den','mit','him','her','them','someone','jemand','hier'}

def request(text):
    value=' '.join(str(text).split())
    if re.search(r'\b(?:ignorier\w*|ignore|red\w*\s+(?:(?:bitte|mal)\s+)?nur\s+mit\s+mir|talk\s+only\s+to\s+me)\b',value,re.I) and (
        re.search(r'\b(?:hintergrund|background|anderen|others?|die|sie|them)\b',value,re.I) or re.search(r'\bnur\s+mit\s+mir|only\s+to\s+me',value,re.I)):
        return {'action':'ignore_background'}
    invitation=re.search(r'\b(?:(?:red\w*|sprich|sprech\w*|talk|speak)\s+(?:(?:mal|bitte|auch|jetzt|please|also)\s+){0,4}(?:mit|to|with)|reagier\w*\s+(?:(?:mal|bitte|auch|jetzt)\s+){0,4}auf|lass\w*\b.{0,45}\bmitreden|(?:may|can)\s+(?:also\s+)?join)\b',value,re.I)
    if not invitation:return None
    # Negated/quoted future examples never grant access.
    if re.search(r'\b(?:nicht|kein\w*|don.t|do not|never|wenn|falls|if|zitiere|quote|beispiel)\b|["“”„]',value,re.I):return None
    patterns=[r'\b(?:kumpel|freund(?:in)?|kolleg[ein]+|friend)\s+'+NAME,
              r'\b(?:das\s+ist|this\s+is)\s+(?:hier\s+)?'+NAME,
              r'\b(?:mit|to|with|auf)\s+'+NAME]
    name=None
    for pattern in patterns:
        m=re.search(pattern,value,re.I)
        if m and m[1].casefold() not in BAD:name=m[1];break
    return {'action':'invite','name':name}


class SpeakerPermissions:
    def __init__(self,session,clock=time.monotonic):
        self.s=session;self.clock=clock;self.pending=None
        self.background_count=0;self.last_background=-1e9;self.questions=True;self.last_question=-1e9
        self.mentioned=None

    def command(self,text,actor=None,typed=False):
        action=request(text)
        s=self.s;sid=(actor or {}).get('speaker_id') or s.focus.owner
        if not typed and sid not in s.focus.members:return None
        if not action:
            mention=re.search(r'\b(?:kumpel|freund(?:in)?|friend)\s+'+NAME,text,re.I)
            if mention and mention[1].casefold() not in BAD:self.mentioned=(mention[1],self.clock(),sid)
            return None
        if action['action']=='ignore_background':
            self.pending=None;self.questions=False
            s.focus.enabled=True
            if sid and re.search(r'\bnur\s+mit\s+mir|only\s+to\s+me',text,re.I):s.focus.members={sid};s.focus.accept(sid)
            return {'action':'ignore_background','applied':True,'speaker_id':sid}
        name=action['name']
        if not name and self.mentioned and self.clock()-self.mentioned[1]<30 and self.mentioned[2]==sid:name=self.mentioned[0]
        known=next((key for key,u in s.memory.users.items() if name and u['named'] and u['name'].casefold()==name.casefold()),None)
        if known:
            s.focus.members.add(known);self.pending=None
            return {'action':'invite','applied':True,'name':s.memory.users[known]['name'],'speaker_id':known}
        self.pending={'name':name,'invited_by':sid,'until':self.clock()+45.}
        self.questions=True
        return {'action':'invite','applied':True,'name':name,'waiting_for_next_voice':True,'expires_seconds':45}

    def candidate(self,embedding,fast,level,text):
        s=self.s;p=self.pending
        if not p or self.clock()>p['until'] or embedding is None:return False
        inviter=s.memory.users.get(p['invited_by'],{})
        import numpy as np
        q=np.asarray(embedding,np.float32);q=q/max(1e-8,float(np.linalg.norm(q)))
        inviter_similarity=max((float(q@v) for v in inviter.get('vectors',[])),default=-1.)
        reference=inviter.get('level',0.)
        # Do not accidentally rename the main speaker finishing the invitation,
        # or enroll a much quieter TV while waiting for the guest's response.
        # A high match to an already heard unnamed GUEST is what we want to
        # name, not evidence that the inviter is still speaking.
        if inviter_similarity>=.65 or (reference and level<reference*.15):return False
        return True

    def enroll(self,embedding,text,fast,level):
        s=self.s;p=self.pending
        if not p:return None
        name=p['name'];spoken=('My name is '+name+'.') if name else text
        # Explicitly supplied guest names must not be interpreted as the
        # current speaker's self-name by the LLM; bind them acoustically here.
        import numpy as np
        q=np.asarray(embedding,np.float32);q=q/max(1e-8,float(np.linalg.norm(q)))
        matches=sorted(((max((float(q@v) for v in u['vectors']),default=-1.),sid)
            for sid,u in s.memory.users.items() if not u['named'] and sid!=p['invited_by']),reverse=True)
        reuse=matches[0][1] if matches and matches[0][0]>=.5 else None
        context=s.memory.assign(embedding,spoken,preferred=reuse,force_new=reuse is None)
        sid=context['speaker_id']
        if sid not in s.memory.users:return None
        if name:
            s.memory.users[sid]['name']=name;s.memory.users[sid]['named']=True
            context.update(display_name=name,named=True)
        s.memory.users[sid]['level']=level;s.focus.accept(sid)
        if fast is not None:s.fast_anchors.setdefault(sid,[]).append(fast.copy())
        self.pending=None
        return context

    def observe_background(self):
        self.background_count+=1;self.last_background=self.clock()

    def context(self):
        s=self.s
        return {'participants':s.focus.state(s.memory.users,s.c.speaking).get('participants',[]),
            'pending_invitation':{'name':self.pending['name'],'expires_seconds':max(0,round(self.pending['until']-self.clock()))} if self.pending and self.clock()<self.pending['until'] else None,
            'background_voice_detected':self.clock()-self.last_background<60,
            'may_ask_about_background':self.questions and self.background_count>=2 and self.clock()-self.last_background<45 and self.clock()-self.last_question>120}

    def note_reply(self,text):
        if re.search(r'\b(?:wer|who)\b.{0,100}\b(?:hintergrund|background|redet|talking|spricht|speaking)\b',text,re.I):
            self.last_question=self.clock()
