"""Session-only approximate voice cues, independent from TYGER turn decisions.

No authentication, no voiceprints on disk. Never allow a short, uncertain frame
or an overlap to enroll somebody. The full wake utterance supplies enrollment.
"""
from dataclasses import dataclass
import re
import time
import numpy as np


def unit(value):
    x=np.asarray(value,np.float32)
    if x.shape!=(192,) or not np.isfinite(x).all():raise ValueError('Expected finite 192-dimensional embedding')
    norm=float(np.linalg.norm(x))
    if norm<1e-8:raise ValueError('Empty speaker embedding')
    return x/norm


@dataclass
class User:
    name:str
    embedding:np.ndarray
    seen:float


class IdentityManager:
    def __init__(self,threshold=.65,margin=.04,timeout_s=300,max_users=5,robo=None,clock=time.monotonic):
        self.threshold=threshold;self.margin=margin;self.timeout_s=timeout_s
        self.max_users=max_users;self.robo=unit(robo) if robo is not None else None
        self.clock=clock;self.users=[];self.utterance=[];self.voiced_ms=0

    def clear(self):
        self.users.clear();self.reset_utterance()

    def reset_utterance(self):
        self.utterance=[];self.voiced_ms=0

    def observe(self,prediction):
        kind=prediction.get('kind','uncertain');embedding=prediction.get('embedding')
        if kind!='speech' or embedding is None or prediction.get('confidence',0)<.6:
            return {'kind':kind,'name':None,'similarity':None}
        x=unit(embedding)
        match=self.match(x,refresh=True)
        if match['kind']=='robo':return match
        self.utterance.append(x)
        self.utterance=self.utterance[-150:]  # <=48 s of accepted cues, RAM only
        self.voiced_ms+=320
        return match

    def expire(self):
        now=self.clock()
        self.users=[u for u in self.users if self.timeout_s<=0 or now-u.seen<self.timeout_s]

    def match(self,embedding,refresh=False):
        self.expire();x=unit(embedding)
        entries=[(u.name,float(x@u.embedding),u) for u in self.users]
        if self.robo is not None:entries.append(('Robby',float(x@self.robo),None))
        entries.sort(key=lambda item:item[1],reverse=True)
        if not entries:return {'kind':'unknown','name':None,'similarity':None}
        name,score,user=entries[0];runner=entries[1][1] if len(entries)>1 else -1
        confident=score>=self.threshold and score-runner>=self.margin
        if not confident:return {'kind':'unknown','name':None,'similarity':score}
        if user is not None and refresh:user.seen=self.clock()
        return {'kind':'robo' if user is None else 'enrolled','name':name,'similarity':score}

    def finish(self,text,enabled=False):
        """Apply a gate to endpoint handling, not to the turn-taking network.

        Typed messages never pass through this function. Thresholds must come
        from validation; a weak pilot must leave this feature opt-in.
        """
        wake=bool(re.search(r'\b(?:hey|hi|hallo|hello)[,\s]+(?:robo|robby|robbie|robot)\b',text,re.I))
        try:
            if not self.utterance or self.voiced_ms<640:
                return {'accepted':not enabled,'kind':'uncertain','reason':'Not enough confident speech to identify or enroll'}
            vector=unit(np.mean(self.utterance,axis=0))
            match=self.match(vector)
            if match['kind']=='robo':return {**match,'accepted':not enabled,'reason':'Likely playback voice'}
            if wake and match['kind']!='enrolled':
                # Do not silently overwrite a similar user's voiceprint.
                similarities=[float(vector@u.embedding) for u in self.users]
                if similarities and max(similarities)>self.threshold-self.margin:
                    return {**match,'accepted':not enabled,'reason':'Voice too close to an existing user; uncertain enrollment'}
                if len(self.users)>=self.max_users:
                    return {**match,'accepted':not enabled,'reason':'Active-user limit reached'}
                name='User '+str(max([int(u.name.split()[-1]) for u in self.users],default=0)+1)
                self.users.append(User(name,vector,self.clock()))
                return {'kind':'enrolled','name':name,'similarity':1.,'accepted':True,'new_user':True}
            if match['kind']=='enrolled':match=self.match(vector,refresh=True)
            return {**match,'accepted':not enabled or match['kind']=='enrolled',
                'reason':'Say Hey Robo in a full sentence to join' if match['kind']=='unknown' else 'Approximate session speaker cue'}
        finally:self.reset_utterance()

    def status(self):
        self.expire()
        return {'users':[u.name for u in self.users],'timeout_s':self.timeout_s,'secure_authentication':False,
            'storage':'RAM for this conversation only; cleared on disconnect'}

