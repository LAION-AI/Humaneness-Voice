"""Optional, session-local environmental awareness alongside existing ASR.

No second microphone, raw-audio persistence, identity mutation or caption queue.
Only idle slots may produce a proactive reply; real user turns take priority.
"""
import asyncio
import math
import os
import time
from collections import deque
import numpy as np
from ambient_audio import AmbientDetector, caption_kind

class EnvironmentSession:
    def __init__(self, conversation, clock=time.monotonic):
        self.c=conversation; self.clock=clock; self.enabled=False; self.voices=True
        self.impressions=True; self.events=True; self.closed=False; self.epoch=0
        self.background_notices=[]
        self.detector=AmbientDetector(); self.job=None; self.worker=None; self.pending=None
        self.last_reply=-1e9; self.last_voice=-1e9; self.last_playback=-1e9
        self.last_audio=-1e9; self.last_user=-1e9; self.recent={}; self.greeted=False
        self.active_reply=None
        self.audio_ring=deque(maxlen=320);self.audio_end=0.;self.question_mark=None;self.listening=False

    def state(self):
        return {'type':'environment_settings','enabled':self.enabled,'voices':self.voices,
            'impressions':self.impressions,'events':self.events,'prominence_db':self.detector.prominence_db,
            'minimum_dbfs':self.detector.minimum_db,'cooldown_s':self.detector.cooldown,
            'recordings_saved':False,'caption_worker':'sound-effect-captioning-whisper · GPU'}

    async def configure(self, data):
        for key in ('enabled','voices','impressions','events'):
            if key in data and type(data[key]) is not bool:raise ValueError('Expected a boolean')
        prominence=data.get('prominence_db',self.detector.prominence_db)
        if type(prominence) not in (float,int) or not np.isfinite(prominence) or not 6<=prominence<=24:
            raise ValueError('Prominence must be 6–24 dB')
        self.enabled=data.get('enabled',self.enabled);self.voices=data.get('voices',self.voices)
        self.impressions=data.get('impressions',self.impressions);self.detector.prominence_db=float(prominence)
        self.events=data.get('events',self.events)
        self.epoch+=1;self.pending=None
        if self.job:self.job.cancel()
        self.detector.reset()
        if not self.enabled:await self.cancel_reply()
        if self.worker is None:self.worker=asyncio.create_task(self.run())
        await self.c.ws.send_json(self.state())

    def welcome(self):
        if self.greeted:return
        self.greeted=True
        # The browser calls this once AFTER audio playback is actually enabled.
        # Environment sensing may be disabled independently of the welcome.
        if self.clock()-self.last_user<5:return
        self.pending={'kind':'welcome','created':self.clock(),'epoch':self.epoch,
            'caption':'Page enabled; no user utterance or sound event.', 'not_before':self.clock()+.6}

    def feed(self, audio, end, playback=False):
        if self.closed or not self.enabled:return
        self.last_audio=self.clock()
        if playback or self.c.speaking:self.last_playback=self.clock()
        stream=getattr(self.c,'sentence_stream',None)
        speech=bool(stream and getattr(stream,'vad',None) and stream.vad.active)
        blocked=playback or self.c.speaking or self.clock()-self.last_playback<1.0
        self.audio_end=end
        if blocked:self.audio_ring.clear()
        else:self.audio_ring.append((end,np.asarray(audio,np.float32).copy()))
        event=self.detector.feed(audio,end,speech=speech,playback=blocked)
        if event and not (self.job and not self.job.done()):
            self.job=asyncio.create_task(self.caption(event,self.epoch))

    async def caption(self,event,epoch,requested=False):
        import httpx
        start=self.clock(); wave=event.pop('audio')
        from sound_events import detect
        tag_task=asyncio.create_task(detect(wave,enabled=self.events))
        try:
            async with httpx.AsyncClient(timeout=2.5,trust_env=False) as client:
                response=await client.post(os.environ.get('ROBO_SOUND_URL','http://127.0.0.1:8803')+'/caption',
                    content=wave.astype('<f4').tobytes(),headers={'Content-Type':'application/octet-stream'})
                response.raise_for_status();result=response.json()
            if result.get('status')!='available' or not isinstance(result.get('caption'),str):raise ValueError('Invalid caption')
            result['caption']=result['caption'][:450]
        except asyncio.CancelledError:
            tag_task.cancel();await asyncio.gather(tag_task,return_exceptions=True);raise
        except (httpx.HTTPError,ValueError):result={'status':'unavailable'}
        tags=await tag_task
        if epoch!=self.epoch or self.closed or not self.enabled:return
        result['sound_events']=tags
        result.update(type='sound_caption',request_ms=round((self.clock()-start)*1000,2),
            capture_ms=event['capture_ms'],event_age_ms=round((self.clock()-start)*1000+event['capture_ms'],2))
        if result['status']=='available':
            kind=caption_kind(result['caption']);result['kind']=kind
            if tags.get('status')=='available':
                music=tags.get('music_score',0);speech=tags.get('speech_score',0)
                if kind=='voice' and music>=.65 and music>2*speech:
                    kind='music';result.update(kind=kind,routing_source='dominant_music_event_hint')
                elif kind=='sound' and speech>=.8 and music<.2:
                    kind='voice';result.update(kind=kind,routing_source='dominant_speech_event_hint')
            signature=' '.join(result['caption'].lower().split()[:12])
            duplicate=self.clock()-self.recent.get(signature,-1e9)<180
            # Different descriptions of the same playing music must not cause
            # repeated comments. Ordinary sounds retain their own signatures.
            music_repeat=kind=='music' and self.clock()-self.recent.get('category:music',-1e9)<180
            result['eligible']=(kind in ('sound','music') or (requested and kind=='voice')) and (requested or (not duplicate and not music_repeat))
            result['suppression_reason']='music_cooldown' if music_repeat else 'repeated_caption' if duplicate else 'speech_or_quiet' if kind not in ('sound','music') else None
            if result['eligible']:
                item={'kind':kind,'caption':result['caption'],'created':start,
                    'sound_events':{k:v for k,v in tags.items() if k in ('status','model','candidates','frames','music_score','speech_score','uncertainty')},
                    'epoch':epoch,'timing':{k:result[k] for k in ('request_ms','capture_ms','event_age_ms','inference_ms') if k in result},
                    'prominence_db':event['prominence_db'],'peak_dbfs':event['peak_dbfs']}
                if requested:
                    item['requested']=True;result['requested']=True;result['suppression_reason']=None
                    await self.c.ws.send_json(result)
                    return self.public(item)
                self.recent={k:v for k,v in self.recent.items() if self.clock()-v<180}
                self.recent[signature]=self.clock()
                if kind=='music':self.recent['category:music']=self.clock()
                self.pending=item
        await self.c.ws.send_json(result)

    def note_text(self,text,at):
        from ambient_audio import sound_question
        if self.enabled and sound_question(text) and (not self.question_mark or self.clock()-self.question_mark[1]>10):
            self.question_mark=(at,self.clock())

    async def context_for_turn(self,text):
        """Bounded, explicitly requested after-question audio; ordinary turns never wait."""
        from ambient_audio import sound_question
        existing=self.take_context()
        if not self.enabled or not sound_question(text):return existing
        epoch=self.epoch;start=self.clock();self.listening=True
        marker=self.question_mark;self.question_mark=None
        anchor=marker[0] if marker and start-marker[1]<10 else self.audio_end
        try:
            await self.c.ws.send_json({'type':'environment_observation','kind':'requested_sound','status':'listening_after_question'})
            # The ASR preview can arm this before endpointing, so much of the
            # 2.4 s is usually already captured. Silence is not hallucinated.
            while self.audio_end<anchor+2.4 and self.clock()-start<3.:
                if self.closed or epoch!=self.epoch or not self.enabled:return None
                await asyncio.sleep(.05)
            chunks=[x for end,x in self.audio_ring if anchor<end<=anchor+3.]
            wave=np.concatenate(chunks)[-48000:] if chunks else np.zeros(0,np.float32)
            levels=[20*np.log10(max(1e-7,float(np.sqrt(np.mean(x*x))))) for x in chunks]
            peak=max(levels,default=-140.)
            if len(wave)>=8000 and peak>=self.detector.minimum_db:
                # Don't race the single-flight GPU worker against a proactive
                # caption already in progress. It is bounded to 2.5 seconds.
                if self.job and not self.job.done():
                    await asyncio.gather(self.job,return_exceptions=True)
                result=await self.caption({'audio':wave,'capture_ms':round(len(wave)/16),
                    'prominence_db':round(peak-self.detector.floor,1),'peak_dbfs':round(peak,1)},epoch,requested=True)
                self.pending=None  # consumed by this answer, never a second comment
                if result:return result
            return {**existing,'requested':True} if existing else {'kind':'requested_sound','requested':True,'status':'no_clear_sound',
                'caption':'No identifiable non-speech event was captured in this short listening window.'}
        finally:
            self.listening=False

    def notice_background(self,audio,decision,level):
        """Repeated audible background speech may invite one cautious question.

        Never enroll an ignored voice, interrupt a real answer or infer that a
        TV is a person. Levels are just salience, not an identity classifier.
        """
        if not self.enabled or not self.voices or self.c.speaking or level<.003 or len(audio)<16000:return
        if decision.get('known_voice_similarity',-1.)>=.5:return
        prominence=decision.get('estimated_level_above_floor_db')
        if not isinstance(prominence,(int,float)) or not math.isfinite(prominence) or prominence<8:return
        now=self.clock();self.background_notices=[t for t in self.background_notices if now-t<25]+[now]
        if len(self.background_notices)<2 or now-self.last_voice<120 or now-self.last_playback<1 or (self.job and not self.job.done()):return
        stream=getattr(self.c,'sentence_stream',None)
        if stream and not stream.permissions.questions:return
        self.last_voice=now;self.background_notices=[]
        self.job=asyncio.create_task(self.describe_voice(audio[-48000:].copy(),self.epoch))

    def background_voice(self,audio,decision,level,owner_level):
        stream=getattr(self.c,'sentence_stream',None)
        if (not self.enabled or not self.voices or not stream or not stream.permissions.questions
            or decision.get('quiet_background') or decision.get('similarity',1.)>.25
            or level<max(.008,owner_level*.5) or len(audio)<16000
            or self.clock()-self.last_voice<120 or self.clock()-self.last_playback<1.
            or self.c.speaking or (self.job and not self.job.done())):return
        # Repeated unaccepted speech alone is not proof of another person.
        if stream.permissions.background_count<2:return
        self.last_voice=self.clock();self.job=asyncio.create_task(self.describe_voice(audio[-48000:].copy(),self.epoch))

    async def describe_voice(self,audio,epoch):
        from voice_captioning import describe
        hint=await describe(audio,timeout=.8,impression=self.impressions)
        if epoch!=self.epoch or not self.enabled or self.closed:return
        self.pending={'kind':'unfamiliar_voice','caption':'A salient voice did not confidently match the current partner. It might be a TV, recording or the same person speaking differently.',
            'voice_delivery':hint if hint.get('status')=='available' else None,
            'created':self.clock(),'epoch':epoch}
        await self.c.ws.send_json({'type':'environment_observation','kind':'unfamiliar_voice',
            'status':'awaiting_idle','voice_caption_ms':hint.get('request_ms')})

    def take_context(self):
        """Consume a fresh observation with a real user turn, without waiting."""
        self.last_user=self.clock(); item=self.pending;self.pending=None
        if item and item['kind']!='welcome' and self.clock()-item['created']<8:
            return self.public(item)
        return None

    @staticmethod
    def public(item):return {k:v for k,v in item.items() if k not in ('created','epoch','not_before')}

    def idle(self):
        s=getattr(self.c,'sentence_stream',None)
        if self.listening or self.c.speaking or self.c.halting or self.c.ending or not self.c.have_mouth:return False
        if s:
            if s.preview or not s.queue.empty():return False
            if s.mic_now-s.last_speech<max(.8,s.pause_ms/1000):return False
        return True

    async def cancel_reply(self):
        if self.active_reply is not None and self.active_reply==self.c.playback.rid and self.c.speaking:
            await self.c._on_barge({'onset_s':0.,'yield_prob':1.,'source':'environment_cancel'},self.c.hop_pos)
        self.active_reply=None

    async def run(self):
        while not self.closed:
            await asyncio.sleep(.1)
            item=self.pending
            if not item:continue
            if item['epoch']!=self.epoch or self.clock()-item['created']>8:
                self.pending=None
                await self.c.ws.send_json({'type':'environment_observation','kind':item['kind'],'status':'expired_while_busy'})
                continue
            if self.clock()<item.get('not_before',0) or not self.idle():continue
            if item['kind']!='welcome' and (not self.enabled or self.clock()-self.last_reply<30):
                self.pending=None
                await self.c.ws.send_json({'type':'environment_observation','kind':item['kind'],'status':'cooldown'})
                continue
            # No await between idle check and claiming the conversational slot.
            self.pending=None
            if item['kind']!='welcome':self.last_reply=self.clock()
            self.c._start_speaking();self.c.avatar_optional=True;self.active_reply=self.c.playback.rid
            context={**self.c.avatar_personality.snapshot(),'environment':self.public(item),'environment_proactive':True}
            await self.c.ws.send_json({'type':'environment_observation','kind':item['kind'],'status':'responding'})
            await self.c.ws.send_json({'type':'latency_start','reply_id':self.c.playback.rid,
                'input':{'kind':'environment','sound_caption_ms':item.get('timing',{}).get('request_ms')}})
            await self.c.to_mouth(b'T',{'transcript':'','speaker':{**(self.c.current_speaker or {}),'needs_name':False,'self_introduction':False},
                'avatar_context':context,'quick_replies':self.c.quick_replies and item['kind']!='welcome',**self.c.llm.snapshot()})

    async def close(self):
        self.closed=True;self.epoch+=1;self.pending=None;self.detector.reset();self.audio_ring.clear()
        tasks=[t for t in (self.job,self.worker) if t]
        for t in tasks:t.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
