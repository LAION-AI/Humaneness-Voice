"""Speaker-owned audio, session enrollment and an independent inactivity clock.

Pure numpy/state logic: no GPU, ASR, HTTP, file writes or LLM. Unknown voices do
not refresh an enrolled user's deadline. No identity is accepted on a name
alone. A wake transcript must come from that candidate's own selected audio.
"""
from collections import deque
from dataclasses import dataclass,field
import re,time,uuid
import numpy as np
from identity import unit

# Nemotron occasionally renders the same spoken wake name as Rabi/Rabbi.
# Keep the greeting mandatory; never activate on a bare person's name.
WAKE=re.compile(r'\b(?:hey|he|ey|hi|hallo|hello|هي|هاي)[,\s.!]*r[oa]b{1,2}(?:y|ie|i|o)\b',re.I)
INTRO=re.compile(r"\b(?:ich\s+(?:bin|hei(?:ß|ss)e)|mein\s+name\s+ist|my\s+name\s+is|i\s+am|i['’]m)\s+([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]{1,35})\b",re.I)
NOT_NAMES={'auch','hier','da','der','die','das','ein','eine','nicht','sehr','so','heute','gerade','müde','froh','traurig','wütend','krank','hungrig','durstig','bereit','okay','gut','allein','zufrieden',
    'the','a','an','here','also','very','not','tired','happy','sad','angry','ready','fine','okay','sorry','going','feeling','sure','glad','hungry','thirsty','from','just','back','wondering'}
NOT_NAMES.update('aufgeregt enttäuscht begeistert glücklich neugierig ängstlich nervös gelangweilt entspannt verliebt überrascht erschöpft frustriert verwirrt dankbar einsam besorgt optimistisch pessimistisch unsicher skeptisch stolz eifersüchtig panisch hilflos verzweifelt gekommen zurück interessiert genervt überfordert robust mensch roboter deutsch deutscher student lehrer arzt'.split())
NOT_NAMES.update('excited surprised curious confused worried grateful proud lonely frustrated calm relaxed exhausted depressed upset delighted disappointed nervous interested overwhelmed jealous bored afraid human robot german american'.split())
NOT_NAMES.update({'es','doch'})

def self_name(text):
    match=INTRO.search(text)
    if not match:return None
    name=match.group(1).strip("'’-")
    if name.casefold() in NOT_NAMES:return None
    return name[0].upper()+name[1:]

ROBOT_CALL=re.compile(r'\br[oa]b{1,2}(?:y|ie|i|o)\b',re.I)
def wake_intent(text):
    if WAKE.search(text):return True
    match=ROBOT_CALL.search(text)
    if match is None:return False
    before=text[:match.start()].strip(' ,.!?:').casefold()
    after=text[match.end():].strip(' ,.!?:').casefold()
    # An ASR-dropped greeting is not a reason to lose an explicitly addressed
    # self-introduction or an isolated wake call. Ordinary third-person
    # mentions ("Robby tells a story") do not activate enrollment.
    return bool(self_name(after) or (after in ('','ich','ich bin','i','i am') and
        (not before or before.split()[-1] in ('hey','he','ey','hi','hallo','hello','هي','هاي'))))

def words(text):return re.findall(r"[^\W_]+",text.casefold(),re.UNICODE)

def word_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]

def choose_turn_text(texts,owned_text):
    """Keep explicit activation, otherwise require owned/raw ASR agreement.

    This is a conservative word-origin cross-check, NOT source separation.
    It prevents a nonverbal foreground voice from submitting background TV
    words merely because its speaker embedding passes verification.
    """
    if not texts:return None,None
    owned_name=self_name(owned_text)
    if (owned_name and wake_intent(owned_text) and any(self_name(t)==owned_name for t in texts)
        and not any(wake_intent(t) and self_name(t)==owned_name for t in texts)):
        matched=WAKE.search(owned_text)
        return owned_text[matched.start():] if matched else owned_text,-1
    for predicate in (lambda t:bool(WAKE.search(t) and self_name(t)),
        lambda t:bool(wake_intent(t) and self_name(t)),lambda t:bool(self_name(t)),wake_intent):
        choice=next((i for i,t in enumerate(texts) if predicate(t)),None)
        if choice is not None:
            text=texts[choice]
            match=WAKE.search(text) or (ROBOT_CALL.search(text) if wake_intent(text) else None)
            retained=text[match.start():] if match else text
            if match:
                same=[i for i,t in enumerate(texts) if WAKE.search(t)
                    and words(t[WAKE.search(t).start():])==words(retained)]
                choice=max(same) if same else choice
            return retained,choice
    own=words(owned_text)
    short={'ja','nein','yes','no','okay','ok','genau','danke','thanks','klar','gerne','bitte','sure','right'}
    if own and len(own)<=3 and all(w in short for w in own):return owned_text,-1
    if len(own)<2:return None,None
    choices=[]
    for i,text in enumerate(texts):
        raw=words(text);overlap=sum(min(raw.count(w),own.count(w)) for w in set(own))
        if overlap<2 or overlap/min(len(own),max(1,len(raw)))<.5:continue
        choices.append((word_distance(raw,own)/max(1,len(raw),len(own)),i))
    if not choices:return None,None
    distance,choice=min(choices)
    # ASR can ignore a quiet TV prefix while the speaker embedder still hears
    # it. Prefer the latest bounded crop only when it retains the EXACT same
    # recognized word sequence and the same independent agreement score.
    # Never trade a missing initial word for a superficially cleaner identity.
    retained=words(texts[choice])
    choice=max(i for d,i in choices if d==distance and words(texts[i])==retained)
    selected=words(texts[choice])
    # Post-roll can restore one quiet final word, but must not append a TV
    # phrase after a complete independently recognized user sentence.
    if len(own)>=3:
        prefixes=[(word_distance(selected[:end],own),-end)
            for end in range(max(3,len(own)-1),min(len(selected),len(own)+1)+1)]
        if prefixes:
            distance,negative_end=min(prefixes);end=-negative_end
            # One dropped article/inflection must not disable the suffix
            # guard. Prefer the best/longest matching prefix so one genuinely
            # quiet missing final word is still retained.
            if distance<=1 and len(selected)>end+1:return owned_text,-1
    return texts[choice],choice

@dataclass
class Speaker:
    id:str
    name:str
    fast:np.ndarray
    long:np.ndarray|None
    seen:float
    named:bool=False
    teacher:np.ndarray|None=None
    teacher_templates:tuple=()
    created:float=0.
    teacher_anchor:np.ndarray|None=None
    teacher_examples:deque=field(default_factory=lambda:deque(maxlen=2))
    confirmed_variants:list=field(default_factory=list)
    asked_name:bool=False
    def teacher_score(self,embedding):
        """Bounded multi-duration enrollment, not identity probability."""
        refs=((self.teacher,) if self.teacher is not None else ())+self.teacher_templates+tuple(v['teacher'] for v in self.confirmed_variants)
        return max((float(embedding@v) for v in refs),default=-1.)
    def adapt_teacher(self,embedding):
        if self.teacher is None:self.teacher=unit(embedding);return
        if self.teacher_anchor is None:self.teacher_anchor=self.teacher.copy()
        self.teacher_examples.append(unit(embedding).copy())
        # The original clean introduction always has twice the weight of the
        # bounded follow-up centroid. Never recursively replace the anchor.
        self.teacher=unit(self.teacher_anchor+.5*unit(np.mean(self.teacher_examples,axis=0)))

class SpeakerMemory:
    def __init__(self,threshold=.55,margin=.04,timeout_s=900,max_users=8,center=None,clock=time.monotonic):
        self.threshold=threshold;self.margin=margin;self.timeout_s=timeout_s;self.max_users=max_users
        self.center=np.asarray(center,np.float32) if center is not None else None
        self.clock=clock;self.users={};self.counter=0
    def similarity(self,a,b):
        a=unit(a);b=unit(b)
        if self.center is not None:
            a=unit(a-self.center);b=unit(b-self.center)
        return float(a@b)
    def expire(self):
        now=self.clock()
        for sid in [sid for sid,u in self.users.items() if self.timeout_s>0 and now-u.seen>self.timeout_s]:del self.users[sid]
    def match(self,vector,long=None):
        self.expire()
        if vector is None or not self.users:return {'kind':'unknown','id':None,'name':None,'similarity':None}
        entries=[]
        for sid,u in self.users.items():
            fast=self.fast_score(vector,u);slow=self.long_score(long,u)
            score=fast if slow is None else .8*fast+.2*slow
            entries.append((score,sid,fast,slow))
        entries.sort(reverse=True);score,sid,fast,slow=entries[0];runner=entries[1][0] if len(entries)>1 else -1.
        # Long history cannot rescue a poor current 320-ms match. This avoids
        # background speech inheriting the previous speaker's long context.
        fast_winner=max(entries,key=lambda item:item[2])[1]
        if sid!=fast_winner or fast<self.threshold or score-runner<self.margin:return {'kind':'unknown','id':None,'name':None,'similarity':fast}
        u=self.users[sid];return {'kind':'enrolled','id':sid,'name':u.name,'similarity':fast,'long_similarity':slow,'margin':score-runner}
    def fast_score(self,vector,user):
        return max(self.similarity(vector,v) for v in [user.fast,*[r['fast'] for r in user.confirmed_variants]])
    def long_score(self,vector,user):
        refs=[v for v in [user.long,*[r['long'] for r in user.confirmed_variants]] if v is not None]
        return max((self.similarity(vector,v) for v in refs),default=None) if vector is not None else None
    def confirm_variant(self,sid,fast,long,teacher):
        """Explicit user assertion only; bounded independent anchors, no averaging away identity."""
        user=self.users[sid]
        variant={'fast':unit(fast),'long':unit(long) if long is not None else None,'teacher':unit(teacher)}
        if user.teacher_score(variant['teacher'])>=.96:return False
        if len(user.confirmed_variants)>=3:return False
        user.confirmed_variants.append(variant);user.seen=self.clock();return True
    def enroll(self,fast,long=None,name=None,force_new=False):
        self.expire()
        match=self.match(fast)
        if match['id'] and not force_new:
            sid=match['id'];user=self.users[sid]
            if not user.named or (name and user.name.casefold()==name.casefold()):self.rename(sid,name)
            user.seen=self.clock();return user,False
        if len(self.users)>=self.max_users:return None,False
        if not force_new and self.users and max(self.fast_score(fast,u) for u in self.users.values())>self.threshold-self.margin:
            return None,False # don't duplicate an ambiguous existing identity
        self.counter+=1;sid='speaker-'+uuid.uuid4().hex[:12]
        u=Speaker(sid,name or f'Guest {self.counter}',unit(fast),unit(long) if long is not None else None,self.clock(),bool(name),created=self.clock())
        self.users[sid]=u;return u,True
    def rename(self,sid,name):
        if sid in self.users and name and len(name)<=40 and re.fullmatch(r"[^\W\d_][\w'’\-]{1,39}",name,re.UNICODE):
            self.users[sid].name=name;self.users[sid].named=True;return True
        return False
    def clear(self):self.users.clear();self.counter=0
    def status(self):
        self.expire();return {'users':[u.name for u in self.users.values()],
            'speakers':[{'id':u.id,'name':u.name,'named':u.named,'reference_ready':u.teacher is not None,'voice_variants':1+len(u.confirmed_variants)} for u in self.users.values()],
            'timeout_s':self.timeout_s,'threshold':self.threshold,'margin':self.margin,
            'similarity':'mean-centered cosine' if self.center is not None else 'cosine',
            'secure_authentication':False,'storage':'RAM for this session; cleared on disconnect'}

@dataclass
class Track:
    key:str
    start:float
    last_speech:float
    chunks:deque=field(default_factory=deque)
    fast:deque=field(default_factory=lambda:deque(maxlen=64))
    long:deque=field(default_factory=lambda:deque(maxlen=16))
    speech_s:float=0.
    wake:bool=False
    partial:str=''
    last_decode:float=-100.
    decoding:bool=False
    version:int=0
    last_decoded_version:int=-1
    last_quiet_decoded_version:int=-1
    request_version:int=0
    sid:str|None=None
    closed:bool=False
    channel:int|None=None
    submission_audio:np.ndarray|None=None
    verification_audio:np.ndarray|None=None
    voiced_start:float|None=None
    recognition_start:float|None=None
    last_direct_speech:float=-100.
    levels:deque=field(default_factory=lambda:deque(maxlen=64))
    name_heard:bool=False
    independent_activity_confirmed:bool=False
    token:str=field(default_factory=lambda:uuid.uuid4().hex)
    def vector(self):return unit(np.mean(self.fast,axis=0))
    def long_vector(self):return unit(np.mean(self.long,axis=0)) if self.long else None
    def waveform(self):
        if not self.chunks:return np.zeros(0,np.float32)
        # Merged candidates/pre-roll can arrive out of order. A deque's first
        # item is not necessarily the oldest: negative NumPy indices formerly
        # wrote an early background slice into the *end* of the sentence.
        start=min(a for a,b,x in self.chunks);end=max(b for a,b,x in self.chunks)
        wave=np.zeros(max(0,round((end-start)*16000)),np.float32)
        for a,b,x in sorted(self.chunks,key=lambda chunk:chunk[0]):
            at=round((a-start)*16000);n=min(len(x),len(wave)-at)
            if n>0:wave[at:at+n]=x[:n]
        return wave
    def owned_waveform(self):
        """Direct/bridged speech only; raw pre-roll is NOT word evidence."""
        wave=self.waveform()
        if not self.chunks or self.voiced_start is None:return wave
        start=min(a for a,b,x in self.chunks)
        return wave[max(0,round((self.voiced_start-start)*16000)):]

class SpeakerTurns:
    def __init__(self,memory,quiet_s=1.,max_turn_s=20.,candidate_s=8.):
        self.memory=memory;self.quiet_s=quiet_s;self.max_turn_s=max_turn_s;self.candidate_s=candidate_s
        self.tracks={};self.now=0.;self.last_match=None;self.last_match_at=-100.;self.previous=None
        self.events=[];self.ignored_hops=0;self.channels={}
    def clear(self):self.memory.clear();self.tracks.clear();self.channels.clear();self.previous=None;self.last_match=None;self.events=[]
    def rebind_channel(self,channel,sid):
        """Move an in-flight continuation when async verification fixes an ID.

        Updating only channels leaves an open old-ID track behind. Its audio
        then appears to be a DIFFERENT known person and pre-roll protection
        cuts the new person's own introduction out of the next track.
        """
        old=self.channels.get(channel);self.channels[channel]=sid;spans=[]
        for key,t in list(self.tracks.items()):
            if t.closed or t.channel!=channel or t.sid not in (old,sid):continue
            spans.append((t.sid,t.start,t.last_speech));self.tracks.pop(key,None)
            t.sid=t.key=sid
            if sid in self.tracks:
                target=self.tracks[sid];target.chunks.extend(t.chunks);target.fast.extend(t.fast);target.long.extend(t.long)
                target.start=min(target.start,t.start);target.last_speech=max(target.last_speech,t.last_speech)
                target.last_direct_speech=max(target.last_direct_speech,t.last_direct_speech)
                target.speech_s=min(target.last_speech-target.start,target.speech_s+t.speech_s)
                target.wake|=t.wake;target.name_heard|=t.name_heard;target.levels.extend(t.levels)
                if t.voiced_start is not None:target.voiced_start=min(target.voiced_start if target.voiced_start is not None else t.voiced_start,t.voiced_start)
                t=target
            else:self.tracks[sid]=t
            if sid in self.memory.users and not self.memory.users[sid].named:t.wake=True
        return spans
    def update(self,prediction,audio,end_s,*,speaking=False):
        self.now=float(end_s);x=np.asarray(audio,np.float32);dt=len(x)/16000;begin=end_s-dt
        channel=prediction.get('diarization_channel');diar=channel is not None
        vector=prediction.get('embedding');confidence=prediction.get('confidence',0.)
        if diar:
            vector=prediction.get('raw_embedding',vector)
            # Never turn classified silence into a fabricated matching voice.
            # Only a genuinely audible encoder warm-up can borrow continuity.
            if vector is None and prediction.get('kind')=='warming' and len(x) and float(np.sqrt(np.mean(x*x)))>=.002 and self.channels.get(channel) in self.memory.users:
                vector=self.memory.users[self.channels[channel]].fast
        rms=float(np.sqrt(np.mean(x*x))) if len(x) else 0.
        speech=vector is not None and (diar or (prediction.get('kind')=='speech' and confidence>=.60))
        # For enrollment the independent diarizer can recover audible speech
        # missed by the short student. Only an already-enrolled channel must
        # not prolong its clock with a silent short-window identity cue.
        match=self.memory.match(vector,prediction.get('long_embedding'))
        fast_match=self.memory.match(vector)
        if diar and fast_match.get('id') and not match.get('id'):
            # The long context can still describe the previous person after
            # a turn change. It must not veto a strong current 320-ms match
            # used for activity ownership (never final identity acceptance).
            match=fast_match
        bound=self.channels.get(channel)
        if (diar and prediction.get('foreground_energy') is not False and bound in self.memory.users
            and match.get('id') in self.memory.users and match['id']!=bound):
            # A strong independent known-person match may supersede a reused
            # activity channel. It cannot enroll anyone or trust an unbound
            # channel; final full-utterance verification still decides the ID.
            bound=match['id']
            # Carry this independently strong known-voice transition into
            # subsequent compatible (but less confident) phonemes. A local
            # override for just this hop reverted to the old channel owner
            # mid-sentence, splitting and masking the returning user's words.
            # This only changes activity ownership; ECAPA still verifies the
            # full utterance before any answer or reference adaptation.
            self.channels[channel]=bound
        if diar and bound in self.memory.users and vector is not None:
            if self.memory.users[bound].teacher is not None and prediction.get('foreground_energy') is False:speech=False
            strength=self.memory.fast_score(vector,self.memory.users[bound])
            floor=.15 if self.memory.users[bound].named and self.memory.users[bound].teacher is not None else .10
            track=self.tracks.get(bound)
            level=max(track.levels,default=.032) if track is not None else .032
            # Channel reuse must not override a clearly different current
            # 320-ms voice. Keep a permissive continuity floor; final ECAPA
            # remains the stricter independent identity decision.
            if strength<floor:speech=False
            silent=prediction.get('encoder_activity_kind',prediction.get('kind'))=='silence'
            if silent and not (strength>=floor and rms>=max(.003,level*.25)):speech=False
        if diar and rms<.0015:speech=False
        if diar and prediction.get('speaker_activity_confirmed') is False:speech=False
        if not speech:
            self.previous=None;return None
        sid=match['id']
        if diar:
            if bound in self.memory.users:
                sid=bound;u=self.memory.users[sid]
                match={'kind':'enrolled','id':sid,'name':u.name,
                    'similarity':self.memory.fast_score(vector,u),'source':'enrolled Nemotron session channel + predicted reference'}
            else:
                # A channel's old person can move to another channel after a
                # merge is corrected. That must NOT make a weak short-vector
                # match on an unregistered background channel authoritative.
                sid=None
                match={'kind':'unknown','id':None,'name':None,'similarity':match.get('similarity')}
        self.last_match=match
        if sid:self.last_match_at=end_s
        # Unknown speech while Robby talks cannot enroll Robby's own echo.
        if speaking and sid is None:self.ignored_hops+=1;self.previous=None;return None
        if sid:
            key=sid;self.memory.users[sid].seen=self.memory.clock()
        else:
            candidates=[t for t in self.tracks.values() if t.sid is None and not t.closed and t.fast and end_s-t.last_speech<1.6]
            closest=max(candidates,key=lambda t:self.memory.similarity(vector,t.vector()),default=None)
            if diar:key='channel-'+str(channel)
            elif closest is not None and self.memory.similarity(vector,closest.vector())>=self.memory.threshold-.12:key=closest.key
            else:key='candidate-'+uuid.uuid4().hex[:10]
        if key not in self.tracks:
            if len(self.tracks)>=12:
                disposable=[t for t in self.tracks.values() if t.sid is None and not t.decoding]
                if not disposable:self.ignored_hops+=1;return None
                self.tracks.pop(min(disposable,key=lambda t:t.last_speech).key,None)
            provisional=bool(sid and not self.memory.users[sid].named and self.memory.clock()-self.memory.users[sid].created<=15)
            track=Track(key,begin,end_s,sid=sid,channel=channel,voiced_start=begin,wake=provisional);self.tracks[key]=track
            # One bounded pre-roll hop, only if it was unassigned speech, never
            # known to belong to a different enrolled speaker.
            if not diar and self.previous is not None and self.previous[3] is None:
                a,b,old,_,old_vector=self.previous
                if begin-b<.05 and self.memory.similarity(vector,old_vector)>=self.memory.threshold-.12:
                    track.chunks.append((a,b,old.copy()));track.start=a
        track=self.tracks[key]
        track.independent_activity_confirmed|=prediction.get('speaker_activity_confirmed') is True
        track.chunks.append((begin,end_s,x.copy()));track.fast.append(unit(vector));track.speech_s+=dt
        if prediction.get('long_embedding') is not None:track.long.append(unit(prediction['long_embedding']))
        track.last_speech=max(track.last_speech,end_s);track.version+=1
        if not prediction.get('provisional_bridge'):
            track.last_direct_speech=max(track.last_direct_speech,end_s)
            if len(x):track.levels.append(float(np.sqrt(np.mean(x*x))))
        # Bound an unaddressed background speaker without losing a foreground
        # turn. Such a candidate never submits just because its buffer is full.
        limit=self.max_turn_s if sid else self.candidate_s
        while track.chunks and end_s-track.chunks[0][0]>limit:
            a,b,_=track.chunks.popleft();track.speech_s=max(0.,track.speech_s-(b-a))
        if not sid and track.chunks:
            track.start=max(track.start,track.chunks[0][0])
            if track.voiced_start is not None:track.voiced_start=max(track.voiced_start,track.start)
        self.previous=(begin,end_s,x.copy(),sid,unit(vector))
        return match
    def decode_candidates(self):
        result=[]
        for t in self.tracks.values():
            if t.sid or t.decoding or t.closed or t.speech_s<.64:continue
            quiet=self.now-t.last_speech
            quiet_retry=bool(t.channel is not None and self.memory.users and
                quiet>=self.quiet_s and t.version>t.last_quiet_decoded_version)
            if t.version<=t.last_decoded_version and not quiet_retry:continue
            if self.now-t.last_decode>=1.28 or quiet>=self.quiet_s:
                if quiet>=self.quiet_s:t.last_quiet_decoded_version=t.version
                t.last_decode=self.now;t.decoding=True;t.request_version=t.version;result.append(t)
                if len(result)>=1:break
        return result
    def candidate_text(self,key,text,*,observed_wake=False,allow_unprompted=False):
        t=self.tracks.get(key)
        if t is None or t.closed:return None
        t.decoding=False;t.last_decoded_version=t.request_version;t.partial=text[:2000]
        if not (wake_intent(text) or (observed_wake and self_name(text)) or allow_unprompted):return None
        if len(t.fast)<2 or t.speech_s<.64:return None
        vector=t.vector()
        # A mixed/unstable candidate is not a stable enrollment utterance.
        coherence=float(np.mean([self.memory.similarity(x,vector) for x in t.fast]))
        if coherence<self.memory.threshold-.10:return None
        user,new=self.memory.enroll(vector,t.long_vector(),self_name(text),force_new=t.channel is not None and t.channel not in self.channels)
        if user is None:return None
        oldkey=t.key;self.tracks.pop(oldkey,None);t.key=t.sid=user.id;t.wake=True
        if allow_unprompted:t.open_enrollment=True
        t.name_heard=bool(self_name(text))
        if t.channel is not None:self.channels[t.channel]=user.id
        if user.id in self.tracks:
            # A same-user duplicate candidate is not a second independent turn.
            self.tracks[user.id].chunks.extend(t.chunks);self.tracks[user.id].fast.extend(t.fast)
            t=self.tracks[user.id]
        else:self.tracks[user.id]=t
        t.name_heard=t.name_heard or bool(self_name(text))
        decision={'kind':'enrolled','id':user.id,'name':user.name,'new_user':new,'coherence':coherence,'accepted':True}
        self.events.append(decision);return decision
    def due(self):
        result=[]
        for key,t in list(self.tracks.items()):
            quiet=self.now-t.last_speech
            if t.sid is None:
                if quiet>max(4.,self.quiet_s) and not t.decoding:self.tracks.pop(key,None)
                continue
            if t.sid not in self.memory.users:self.tracks.pop(key,None);continue
            # Registration is not yet a recognized speaker: collect a complete
            # introduction/reference before enabling its one-second fast path.
            pending_reference=self.memory.users[t.sid].teacher is None
            quiet_limit=2.5 if t.wake and (not t.name_heard or pending_reference) else self.quiet_s
            if not t.decoding and (quiet>=quiet_limit or self.now-t.start>=self.max_turn_s):
                t.closed=True;self.tracks.pop(key,None);result.append(t)
        return result
    def speaker_context(self,track,text):
        user=self.memory.users.get(track.sid)
        if user is None:return None
        name=self_name(text)
        if name and user.named and user.name.casefold()!=name.casefold():return None
        if name and (not user.named or user.name.casefold()==name.casefold()):self.memory.rename(user.id,name)
        return {'speaker_id':user.id,'display_name':user.name,'named':user.named,'enrolled':True,
            'self_introduction':bool(name),'known_speakers':[{'speaker_id':u.id,'display_name':u.name} for u in self.memory.users.values()],
            'identity_source':'approximate session acoustic matching, not authentication'}
    def status(self):
        return {**self.memory.status(),'quiet_seconds':self.quiet_s,'buffered_speakers':len(self.tracks),
            'ignored_hops':self.ignored_hops,'active':self.last_match}
