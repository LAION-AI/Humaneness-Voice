"""Wake-phrase admission with a short, acoustic-owner-only follow-up window."""
import time
import numpy as np
from speaker_turns import WAKE

class WakeMode:
    def __init__(self,clock=time.monotonic):
        self.enabled=True;self.clock=clock;self.seconds=10.;self.owner=None;self.anchor=None;self.until=-1e9
    def clear(self):self.owner=None;self.anchor=None;self.until=-1e9
    def configure(self,enabled):
        self.enabled=bool(enabled);self.clear()
    def active(self,at=None):return self.owner is not None and self.anchor is not None and (self.clock() if at is None else at)<=self.until
    def is_wake(self,text):return bool(WAKE.match(text.lstrip()))
    def match(self,embedding,users,threshold=.32,require_active=True,at=None):
        if (require_active and not self.active(at)) or self.anchor is None or embedding is None:
            return {'accepted':False,'reason':'Say Hey Robby to start, or speak clearly enough to identify your voice.'}
        q=np.asarray(embedding,np.float32).reshape(-1)
        if q.shape!=self.anchor.shape or not np.isfinite(q).all() or np.linalg.norm(q)<1e-8:return {'accepted':False,'reason':'Voice identity unavailable.'}
        q=q/np.linalg.norm(q);refs=[self.anchor,*users.get(self.owner,{}).get('vectors',[])]
        own=max(float(q@v) for v in refs);rival=max((float(q@v) for sid,u in users.items() if sid!=self.owner for v in u.get('vectors',[])),default=-1.)
        ok=own>=threshold and own>=rival+.06
        return {'accepted':bool(ok),'speaker_id':self.owner,'similarity':round(own,4),'rival_similarity':round(rival,4),
            'reason':'Wake-window speaker matched' if ok else 'Another or uncertain voice; say Hey Robby to start your own turn.'}
    def evaluate(self,text,embedding,users,threshold=.32,at=None,reply_active=False):
        if not self.enabled:return {'accepted':True,'wake':False,'reason':'Open conversation'}
        if self.is_wake(text):return {'accepted':True,'wake':True,'reason':'Explicit wake phrase'}
        return {**self.match(embedding,users,threshold,require_active=not reply_active,at=at),'wake':False}
    def accept(self,sid,embedding,at=None):
        if not self.enabled:return
        if embedding is None or not sid or sid=='unknown':self.clear();return
        v=np.asarray(embedding,np.float32).reshape(-1);n=float(np.linalg.norm(v))
        if not np.isfinite(v).all() or n<1e-8:self.clear();return
        if self.owner!=sid or self.anchor is None:self.anchor=v/n
        self.owner=sid;self.until=(self.clock() if at is None else at)+self.seconds
    def finish_reply(self,sid):
        # Do not spend the user's whole follow-up window while Robby is talking.
        # A reply for an old owner must never refresh a newer person's window.
        if self.enabled and self.anchor is not None and sid==self.owner:self.until=max(self.until,self.clock()+self.seconds)
    def state(self):
        return {'enabled':self.enabled,'active':self.active(),'window_s':self.seconds,
                'remaining_s':round(max(0.,self.until-self.clock()),1) if self.active() else 0.}
