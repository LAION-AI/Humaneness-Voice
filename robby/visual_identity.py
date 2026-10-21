"""Additive, session-local visual speech gate. No images/face vectors on server.

Client observations use the microphone's *sample timeline*, not packet arrival
or wall-clock timestamps. Visual speech is corroboration, never authentication
and never a replacement for ECAPA, the speaker student, diarization or TYGER.
"""
from collections import deque, defaultdict
import math
import re
import time
import uuid

DEFAULTS={'ignoreOffscreenSpeakers':False,'lookAtActiveSpeaker':True,'greetUnknownSpeakers':True,'faceIdEnabled':False}
TOKEN=re.compile(r'^[a-zA-Z0-9-]{1,70}$')


class VisualIdentity:
    def __init__(self,clock=time.monotonic):
        self.clock=clock;self.settings=DEFAULTS.copy();self.frames=deque(maxlen=1800)
        self.camera='off';self.last_received=-1e9;self.last_audio=-1.;self.bindings={}
        self.people={};self.greeted={};self.last_greeting=-1e9
        self.face_verified={}

    def configure(self,data):
        for key in DEFAULTS:
            if type(data.get(key)) is bool:self.settings[key]=data[key]
        if not self.settings['faceIdEnabled']:self.bindings.clear();self.face_verified.clear()

    def observe(self,data,audio_now):
        state=data.get('camera')
        if state not in ('on','off','unavailable'):raise ValueError('Invalid camera state')
        self.camera=state
        if state!='on':
            self.last_received=self.clock();self.frames.clear();self.face_verified.clear();self.last_audio=-1.;return
        at=data.get('audio_s')
        if type(at) not in (float,int) or not math.isfinite(at) or at<0 or abs(at-audio_now)>2:
            raise ValueError('Visual sample clock is not aligned to microphone')
        if at<self.last_audio:return
        faces=data.get('faces',[])
        if not isinstance(faces,list) or len(faces)>4:raise ValueError('Too many face tracks')
        clean=[]
        for face in faces:
            if not isinstance(face,dict) or not TOKEN.fullmatch(str(face.get('id',''))):raise ValueError('Invalid track')
            score=face.get('confidence')
            if type(score) not in (float,int) or not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid score')
            clean.append({'id':face['id'],'confidence':score,'speaking':face.get('speaking') is True})
            if face.get('ambiguous') is True:self.bindings.pop(face['id'],None);self.face_verified.pop(face['id'],None)
        self.frames.append((float(at),clean));self.last_audio=at;self.last_received=self.clock()

    def state(self):
        return {'type':'visual_identity','settings':self.settings.copy(),'camera':self.camera,
            'fallback':self.camera!='on','people':list(self.people.values()),
            'note':'Acoustic speaker recognition; face VAD disabled' if not self.settings['ignoreOffscreenSpeakers'] else 'Audio-only fallback: camera off/unavailable' if self.camera!='on' else 'Visible-speaker matching active; ambiguous/offscreen speech is not answered'}

    def select(self,start,end):
        if self.camera!='on':return {'allowed':True,'fallback':True,'track_id':None,'reason':'Camera unavailable/off: existing audio logic'}
        # A dead camera connection is not proof of offscreen speech. Fail closed
        # while it still claims ON; the browser explicitly reports track-ended.
        if self.clock()-self.last_received>2:return {'allowed':False,'reason':'Camera observations stale; restart camera or disable offscreen filtering'}
        totals=defaultdict(float);counts=defaultdict(int);ambiguous=0;valid=0;visible=0;peak=0.
        # Tolerate measured video-frame/microphone timestamp quantization, not a
        # hardcoded playback offset. Do not borrow another utterance's history.
        begin=max(start-.16,end-3.2);finish=end+.16
        for at,faces in self.frames:
            if not begin<=at<=finish:continue
            valid+=1;active=[f for f in faces if f['speaking'] and f['confidence']>=.45]
            visible+=bool(faces);peak=max(peak,max((f['confidence'] for f in faces),default=0.))
            if len(active)>1:ambiguous+=1;continue
            if len(active)==1:
                face=active[0];totals[face['id']]+=face['confidence'];counts[face['id']]+=1
        ranked=sorted(totals,key=totals.get,reverse=True)
        diagnostics={'frames':valid,'visible_frames':visible,'peak_motion':round(peak,3)}
        if not ranked or valid<3:
            code='visual_timing' if valid<3 else 'visual_no_face' if not visible else 'visual_no_motion'
            reasons={
                'visual_timing':'Camera and microphone observations did not overlap. Restart the camera, or use voice recognition without camera filtering.',
                'visual_no_face':'No visible face during this sentence. Face the camera, or use voice recognition without camera filtering.',
                'visual_no_motion':'I heard you, but could not confirm your mouth movements. Keep your mouth visible, or use voice recognition without camera filtering.'}
            return {'allowed':False,'code':code,'reason':reasons[code],'diagnostics':diagnostics}
        top=ranked[0];runner=totals[ranked[1]] if len(ranked)>1 else 0
        confident=counts[top]>=3 and counts[top]/valid>=.20 and ambiguous/valid<.25 and (totals[top]-runner)/max(1,sum(totals.values()))>.25
        return {'allowed':confident,'track_id':top if confident else None,'confidence':round(totals[top]/max(1,counts[top]),3),'diagnostics':diagnostics,
            'reason':'Time-aligned visible speaker' if confident else 'Several faces or insufficient visual speech; no identity guessed'}

    def gate(self,start,end):
        if not self.settings['ignoreOffscreenSpeakers']:
            return {'allowed':True,'filter_disabled':True,'fallback':self.camera!='on','track_id':None,'reason':'Face VAD disabled; acoustic speaker recognition unchanged'}
        return self.select(start,end)

    def corroborates(self,sid,start,end):
        if self.camera!='on' or not self.settings['faceIdEnabled']:return False
        view=self.select(start,end);track=view.get('track_id')
        return bool(view['allowed'] and track and self.bindings.get(track,{}).get('speaker_id')==sid
            and self.clock()-self.face_verified.get(track,-1e9)<6.)

    def bind(self,track_id,speaker):
        if not self.settings['faceIdEnabled']:return None
        if speaker.get('named') is False:return None
        sid=speaker['speaker_id'];existing=self.bindings.get(track_id)
        # A visual track cannot silently move between acoustically different
        # people. A fresh face descriptor/track is required after crossing.
        if existing and existing['speaker_id']!=sid:return None
        person=next((p for p in self.people.values() if p['speaker_id']==sid),None)
        if person is None:
            person={'person_id':'person-'+uuid.uuid4().hex[:16],'speaker_id':sid,'name':speaker['display_name']}
            self.people[person['person_id']]=person
        person['name']=speaker['display_name'];self.bindings[track_id]=person
        return {**person,'track_id':track_id}

    def greeting(self,track_id,text,language='de'):
        from speaker_turns import self_name,WAKE
        if not self.settings['greetUnknownSpeakers'] or not track_id or track_id in self.bindings:return None
        # Answer requests/introductions through the existing conversation flow,
        # never replace them with a repetitive greeting.
        if self_name(text) or '?' in text or re.search(r'\b(wie|was|warum|wo|wann|kannst|bitte|erklär|what|why|how|can|could|please|tell)\b',text,re.I):return None
        now=self.clock()
        if now-self.greeted.get(track_id,-1e9)<45 or now-self.last_greeting<15:return None
        self.greeted[track_id]=now;self.last_greeting=now
        return 'Hallo, wer bist du?' if language.startswith('de') else 'Hi, who are you?'

    def forget(self):
        self.bindings.clear();self.people.clear();self.greeted.clear();self.frames.clear();self.face_verified.clear()
