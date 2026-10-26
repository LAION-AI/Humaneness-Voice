"""Wake-free, speaker-labelled turns after a configurable neural speech pause.

New voices can join without a wake phrase while no conversation is focused.
During a conversation, acoustic ownership guards turns and interruptions.
Voice profiles and explicit aliases live only in the current session.
"""
import asyncio,math,os,re,time
from difflib import SequenceMatcher
from collections import deque
import numpy as np
from speaker_turns import self_name,NOT_NAMES
from open_speakers import claimed_name

def name_claim(text):
    cleaned=re.sub(r'\bich\s+bin\s+(?:auch|doch|wirklich|wieder)\s+', 'Ich bin ',text,flags=re.I)
    cleaned=re.sub(r'\bI\s+am\s+(?:also|really|still)\s+', 'I am ',cleaned,flags=re.I)
    candidate=claimed_name(cleaned) or self_name(cleaned)
    return candidate if candidate and candidate.casefold() not in NOT_NAMES else None

def unit(v):
    v=np.asarray(v,np.float32).reshape(-1);return v/max(1e-8,float(np.linalg.norm(v)))

class OpenVoiceMemory:
    def __init__(self,threshold=.38,max_users=32,max_variants=8,clock=time.monotonic):
        self.threshold=threshold;self.max_users=max_users;self.max_variants=max_variants
        self.users={};self.counter=0;self.turn=0;self.clock=clock;self.timeout_s=900
    def expire(self):
        for sid in list(self.users):
            if self.timeout_s>0 and self.clock()-self.users[sid].get('seen',self.clock())>self.timeout_s:del self.users[sid]
    def clear(self):self.users.clear();self.counter=0;self.turn=0
    def assign(self,embedding,text,preferred=None,adapt=False):
        self.expire()
        self.turn+=1;name=name_claim(text);vector=unit(embedding) if embedding is not None else None
        name_resolution=None
        if name and re.search(r"\b(?:nein|no|verwechselt|doch)\b|it['’]?s me",text,re.I):
            matches=sorted(((SequenceMatcher(None,name.casefold(),u['name'].casefold()).ratio(),u['name'])
                for u in self.users.values() if u['named']),reverse=True)
            if matches and matches[0][0]>=.78 and (len(matches)==1 or matches[0][0]-matches[1][0]>=.12):
                if name!=matches[0][1]:name_resolution={'heard':name,'canonical':matches[0][1],'reason':'unique close name in an explicit correction'}
                name=matches[0][1]
        ranking=sorted(((max((float(vector@v) for v in u['vectors']),default=-1),sid)
            for sid,u in self.users.items()),reverse=True) if vector is not None else []
        score,sid=ranking[0] if ranking else (-1,None)
        # An explicit self-identification can attach another acoustic mode to
        # the same named person; a mere third-person name mention cannot.
        named=next((sid for sid,u in self.users.items() if name and u['named'] and u['name'].casefold()==name.casefold()),None)
        if named:
            if sid and sid!=named and score>=self.threshold and not self.users[sid]['named']:
                for v in self.users[sid]['vectors']:
                    if len(self.users[named]['vectors'])<self.max_variants:self.users[named]['vectors'].append(v.copy())
                del self.users[sid]
            sid=named
        elif preferred in self.users:sid=preferred
        elif score<self.threshold:sid=None
        elif name and self.users[sid]['named'] and self.users[sid]['name'].casefold()!=name.casefold():sid=None
        if sid is None:
            if vector is None and name is None:
                return {'speaker_id':'unknown','display_name':'Speaker (uncertain)','named':False,
                        'enrolled':False,'similarity':None,'needs_name':False}
            if len(self.users)>=self.max_users:
                # Still answer when memory is full; don't force a false identity.
                return {'speaker_id':'unknown','display_name':name or 'Unknown speaker','named':bool(name),
                        'enrolled':False,'similarity':score,'needs_name':False}
            self.counter+=1;sid=f'speaker-{self.counter}'
            self.users[sid]={'name':name or f'Speaker {self.counter}','named':bool(name),
                             'vectors':[],'asked_turn':-100,'turns':0,'seen':self.clock()}
        user=self.users[sid]
        user['seen']=self.clock()
        if name:user['name']=name;user['named']=True
        if vector is not None:
            refs=user['vectors']
            similarity=max((float(vector@v) for v in refs),default=-1.)
            if (not refs or named or similarity>=(.48 if adapt else .70)) and similarity<.96:
                if len(refs)<self.max_variants:refs.append(vector.copy())
                elif named or adapt:refs[1:]=[*refs[2:],vector.copy()]  # preserve original anchor
        user['turns']+=1
        needs=not user['named'] and self.turn-user['asked_turn']>=3
        if needs:user['asked_turn']=self.turn
        return {'speaker_id':sid,'display_name':user['name'],'named':user['named'],'enrolled':True,
                'needs_name':needs,'similarity':round(score,4),'voice_variants':len(user['vectors']),
                'self_introduction':bool(name),'name_resolution':name_resolution,'source':'wake-free nearest-prototype ECAPA'}
    def status(self):
        self.expire()
        return [{'id':sid,'name':u['name'],'named':u['named'],'reference_ready':bool(u['vectors']),
                 'voice_variants':len(u['vectors'])} for sid,u in self.users.items()]

class SentenceSession:
    def __init__(self,conversation):
        self.c=conversation;self.memory=OpenVoiceMemory();self.records=deque(maxlen=1000)
        self.preview='';self.preview_at=-100.;self.segment_floor=0.
        self.now=0.;self.begin=None;self.last_speech=-100.;self.reset_at=-100.;self.fresh_asr_at=-100.;self.probe_text='';self.probe_at=-100.
        self.queue=asyncio.Queue(maxsize=12);self.tasks=set();self.worker=None;self.probe=None
        self.last_probe=-100.;self.closed=False;self.epoch=0;self.last_context=None
        self.embedding_task=None;self.prefetched=None;self.last_embedding=-100.;self.sentence_begin=None;self.pause_checked=False
        self.pause_ms=400
        self.set_pause(float(os.environ.get('ROBO_SPEECH_PAUSE_MS','400')))
        self.vad=None;self.mic_driven=False;self.mic_now=0.;self.native_parts=[];self.native_preview=''
        self.last_activity_notice=-100.;self.last_probe_error=-100.
        from conversation_focus import ConversationFocus
        self.focus=ConversationFocus()
        self.barge_task=None;self.last_barge_check=-100.;self.bot_voice=None
        self.reply_seen=None;self.reply_started=0.
        self.activity_task=None;self.last_focus_check=-100.;self.focus_activity=None
        self.fast_records=deque(maxlen=100);self.fast_anchors={}
        predictor=getattr(getattr(self.c,'speaker_ext',None),'predictor',None)
        self.fast_center= predictor.fast.teacher_mean.detach().numpy().copy() if predictor is not None else None
        from asr_choice import ASRChoice
        self.asr=ASRChoice(self)
    def set_pause(self,value):
        if not math.isfinite(value) or not 200<=value<=2000:raise ValueError('Speech pause must be 200–2000 ms')
        self.pause_ms=round(value)
    def settings(self):
        return {'type':'asr_settings',**self.asr.settings(),'speech_pause_ms':self.pause_ms,'min_ms':200,'max_ms':2000,
                'conversation_focus':self.focus.state(self.memory.users,self.c.speaking),
                'vad':'Silero ONNX CPU · neural speech activity','submission':'All pending sentences after a speech pause'}
    def start(self):self.worker=asyncio.create_task(self.deliver())
    async def close(self):
        self.closed=True
        tasks=[t for t in [self.worker,self.probe,self.embedding_task,self.barge_task,self.activity_task,*self.tasks] if t]
        for t in tasks:t.cancel()
        await asyncio.gather(*tasks,return_exceptions=True);self.memory.users.clear();self.records.clear();self.fast_records.clear();self.fast_anchors.clear()
        await self.asr.close()
    def audio(self,start,end):
        result=[]
        for a,b,w in self.records:
            lo=max(0,round((start-a)*16000));hi=min(len(w),round((end-a)*16000))
            if hi>lo:result.append(w[lo:hi])
        return np.concatenate(result) if result else np.zeros(0,np.float32)
    async def microphone(self,audio,end,reference=None):
        """Classify the existing mic packets without waiting for a GPU hop.

        The network runs off the event loop on CPU; ASR still uses the original
        audio. Only strongly correlated, timestamp-aligned bot echo is muted
        for this pause clock. Stop recognition remains independent.
        """
        self.mic_driven=True
        await self._speech_audio(audio,end,reference)
        self._maybe_probe()
    async def _speech_audio(self,audio,end,reference=None):
        from pause_vad import SpeechClock
        if self.vad is None:self.vad=SpeechClock()
        wave=np.asarray(audio,np.float32)
        if reference is not None and len(reference)==len(wave):
            ref=np.asarray(reference,np.float32);norm=float(np.linalg.norm(wave)*np.linalg.norm(ref))
            if norm>1e-6 and float(wave@ref)/norm>.7:wave=np.zeros_like(wave)
        before=self.last_speech
        detected=await asyncio.to_thread(self.vad.feed,wave,end)
        activity=self.focus_activity
        # A confidently different voice must not keep the focused person's
        # pause clock open. Identity is sampled from recent speech, not energy.
        background=activity and 0<=end-activity['at']<=1.0 and not activity['accepted']
        self.last_speech=max(self.last_speech, min(detected,activity['last_owned'])) if background else detected
        self.mic_now=end
        if self.last_speech>before+1e-6:
            if self.begin is None:self.begin=max(self.segment_floor,self.last_speech-.64)
            self.pause_checked=False
        if end-self.last_activity_notice>=.25:
            self.last_activity_notice=end
            await self.c.ws.send_json({'type':'speech_activity','speech':self.vad.active,
                'probability':round(self.vad.probability,3),'quiet_ms':round(max(0,end-self.last_speech)*1000),
                'speech_pause_ms':self.pause_ms,'pending':bool(self.preview),'source':'neural_vad'})
    def native_text(self):return ' '.join([*self.native_parts,self.native_preview]).strip()
    def quiet_ready(self):
        candidate=self.vad is not None and (self.vad.active or self.vad.probability>=.35)
        if self.focus_activity and 0<=self.mic_now-self.focus_activity['at']<=1.0 and not self.focus_activity['accepted']:candidate=False
        return not candidate and (self.mic_now-self.last_speech)*1000+1e-5>=self.pause_ms
    def agent_echo(self,text):
        tokens=re.findall(r'\w+',text.casefold())
        agent=set(re.findall(r'\w+',getattr(self.c,'open_agent_text','').casefold()))
        return self.c.speaking and len(tokens)>=4 and agent and sum(w in agent for w in tokens)/len(tokens)>.8
    def reset_segment(self):
        self.segment_floor=max(self.segment_floor,self.now)
        self.preview='';self.native_preview='';self.native_parts=[]
        self.begin=None;self.sentence_begin=None;self.prefetched=None;self.pause_checked=False
        self.probe_text='';self.epoch+=1
    async def preview_text(self,text,at,final=False,fresh=True,source='native'):
        if not text.strip():return
        if fresh:self.fresh_asr_at=at
        # Playback can leak into the microphone after acoustic delays. Reject
        # a near-copy of the current assistant's words, but never veto a short
        # Stop merely because the assistant also happened to say that word.
        if self.agent_echo(text):return
        # Punctuation is preview-only. Even a native TYGER endpoint must wait
        # for the neural speech-pause clock; consecutive sentences form one turn.
        if source=='native':self.native_preview=text;self.preview=self.native_text()
        else:self.preview=text
        self.preview_at=at
        from stop_commands import is_stop_command
        if self.c.speaking and is_stop_command(text):
            self.propose_barge({'yield_prob':1.,'onset_s':0.,'source':'open_sentence_stop','stop_text':text},round(at*self.c.ears.sr))
            return

    def identity_state(self):
        return {'type':'identity','enabled':self.focus.enabled,'mode':'open-sentences','sentence_mode':True,
                'open_enrollment':True,'speakers':self.memory.status(),'users':[u['name'] for u in self.memory.users.values()],
                'threshold':self.memory.threshold,'conversation_focus':self.focus.state(self.memory.users,self.c.speaking),
                'storage':'Session RAM'}

    def propose_barge(self,event,hop_end):
        """Single-flight acoustic confirmation, never block a microphone hop."""
        at=hop_end/self.c.ears.sr
        if not self.c.speaking or at-self.last_barge_check<.24:return
        if self.barge_task and not self.barge_task.done():return
        rid=getattr(getattr(self.c,'playback',None),'rid',None)
        if self.reply_seen!=rid:self.reply_seen=rid;self.reply_started=max(0.,at-.64)
        self.last_barge_check=at
        owner_at_proposal=self.focus.owner
        async def verify():
            query_at=min(at,self.last_speech+.16) if event.get('stop_text') and 0<=at-self.last_speech<=1.2 else at
            decision=await self.verify_barge(query_at,rid)
            if self.focus.enabled:
                fast=self.fast_match(at,self.focus.owner)
                evidence=self.fast_evidence(at,self.focus.owner)
                decision.update(fast_similarity=fast,fast_consistency=evidence['consistency'],fast_frames=evidence['frames'])
                # A short changed-vowel Stop can have weak ECAPA similarity.
                # Rescue only explicit commands with sustained, independent
                # native-hop evidence; never rescue a competing voice or echo.
                separate=decision.get('similarity',-1.)>=decision.get('rival_similarity',-1.)+.06 and (decision.get('bot_similarity',-1.)<.4 or decision.get('similarity',-1.)>=decision.get('bot_similarity',-1.)+.06)
                if (not decision['accepted'] and event.get('stop_text') and separate and evidence['frames']>=2
                    and evidence['consistency']>=.65 and fast is not None and fast>=.25
                    and decision.get('similarity',-1.)>=max(.25,self.focus.threshold-.07)):
                    decision.update(accepted=True,reason='Explicit Stop with temporally consistent voice features')
                if (not decision['accepted'] and event.get('stop_text') and separate and fast is not None and fast>=.78
                    and evidence['frames']>=3 and evidence['consistency']>=.8):
                    from speaker_teacher import get_cached_teacher
                    whole=self.audio(max(self.reply_started,at-3.04),at)
                    try:
                        vector=await asyncio.to_thread(get_cached_teacher().embed,whole)
                        longer=self.focus.check(vector,self.memory.users,True,self.bot_voice[1] if self.bot_voice and self.bot_voice[0]==rid else None)
                        if longer['accepted'] and longer.get('similarity',-1.)>=max(.45,self.focus.threshold):
                            decision.update(accepted=True,reason='Explicit Stop with strong fast features and longer voice context',long_similarity=longer['similarity'])
                    except Exception:pass
            if decision['accepted'] and self.focus.enabled:
                # Free-form turn-taking proposals need stronger agreement
                # than a recognized explicit Stop. Mere background speech is
                # never enough to cancel the agent's answer.
                if fast is not None and fast<.25:decision.update(accepted=False,reason='Recent speaker features disagree with the active partner')
                if evidence['frames']>=3 and evidence['consistency']<.60:decision.update(accepted=False,reason='Unstable recent speaker identity; background cannot cancel the reply')
                if not event.get('stop_text') and (decision.get('similarity',-1.)<max(.55,self.focus.threshold) or (fast is not None and fast<.45)):
                    decision.update(accepted=False,reason='Background conversation is not an addressed interruption')
            if self.closed or not self.c.speaking or getattr(getattr(self.c,'playback',None),'rid',None)!=rid or owner_at_proposal!=self.focus.owner:return
            await self.c.ws.send_json({'type':'speaker_focus','operation':'interrupt',**decision})
            if decision['accepted']:
                await self.c._on_barge({**event,'source':'focused_stop' if event.get('stop_text') else 'focused_tyger'},hop_end)
                if event.get('stop_text'):
                    self.reset_segment();self.c.session.reset()
                    await self.c.ws.send_json({'type':'stop_command','speaker_name':self.memory.users.get(self.focus.owner,{}).get('name')})
        self.barge_task=asyncio.create_task(verify())

    def fast_vector(self,start,end):
        rows=[v for at,v in self.fast_records if start<=at<=end]
        return unit(np.mean(rows,axis=0)) if rows else None

    def fast_match(self,at,sid):
        refs=self.fast_anchors.get(sid,[])
        q=self.fast_vector(max(self.reply_started,self.segment_floor,at-3.04),at)
        if q is None or not refs:return None
        center=self.fast_center if self.fast_center is not None else 0.
        return round(max(float(unit(q-center)@unit(v-center)) for v in refs),4)

    def fast_evidence(self,at,sid):
        refs=self.fast_anchors.get(sid,[])
        rows=[v for t,v in self.fast_records if max(self.reply_started,self.segment_floor,at-3.04)<=t<=at]
        if not refs or not rows:return {'frames':0,'consistency':0.}
        center=self.fast_center if self.fast_center is not None else 0.
        scores=[max(float(unit(v-center)@unit(r-center)) for r in refs) for v in rows]
        return {'frames':len(rows),'consistency':sum(v>=.25 for v in scores)/len(scores)}

    async def verify_barge(self,at,rid):
        if not self.focus.enabled:return {'accepted':True,'reason':'Conversation focus disabled'}
        if self.last_speech<at-.48:return {'accepted':False,'reason':'No fresh speech'}
        # Only the current reply's recent microphone audio; no old enrollment
        # speech may authorize a new background speaker to cancel playback.
        audio=self.audio(max(self.reply_started,self.segment_floor,at-1.60),at)
        if len(audio)<10240:return {'accepted':False,'reason':'Collecting a short voice confirmation'}
        try:
            from speaker_teacher import get_cached_teacher,enrollment_reference
            teacher=get_cached_teacher()
            embedding=await asyncio.to_thread(teacher.embed,audio)
            bot=None
            playback=getattr(self.c,'playback',None)
            wave=getattr(playback,'audio',None)
            if wave is not None:wave=wave[:getattr(playback,'n16',len(wave))]
            if self.bot_voice and self.bot_voice[0]==rid:bot=self.bot_voice[1]
            elif wave is not None and len(wave)>=10240:
                bot=await asyncio.to_thread(teacher.embed,enrollment_reference(wave))
                self.bot_voice=(rid,bot)
            return self.focus.check(embedding,self.memory.users,True,bot)
        except Exception:return {'accepted':False,'reason':'Voice confirmation unavailable; manual Stop remains available'}

    def prepare_focus_activity(self,at):
        """Rolling voice check separates background speech from owner silence.

        Uses 1.28 s already heard at 640-ms cadence, independent of ASR. Longer
        3.04-s utterance embeddings remain the final identity anchors.
        """
        if not self.focus.active(self.memory.users,self.c.speaking):self.focus_activity=None;return
        detected=getattr(self.vad,'last_speech',self.last_speech) if self.vad is not None else self.last_speech
        if at-self.last_focus_check<.64 or at-detected>.4:return
        if self.activity_task and not self.activity_task.done():return
        audio=self.audio(max(self.segment_floor,at-1.28),at)
        if len(audio)<10240:return
        self.last_focus_check=at;owner=self.focus.owner
        async def classify():
            try:
                from speaker_teacher import get_cached_teacher
                v=await asyncio.to_thread(get_cached_teacher().embed,audio)
                if owner!=self.focus.owner:return
                result=self.focus.check(v,self.memory.users,self.c.speaking)
                # Require a clear rejection, not merely a small prosody dip,
                # before overriding neural VAD's speech clock.
                if result.get('similarity',1.)>=self.focus.threshold-.10 and result.get('rival_similarity',-1.)<result.get('similarity',1.)+.10:
                    result['accepted']=True
                previous=self.focus_activity
                last_owned=at if result['accepted'] else previous['last_owned'] if previous else max(self.segment_floor,at-1.28)
                self.focus_activity={**result,'at':at,'last_owned':last_owned}
            except Exception:pass
        self.activity_task=asyncio.create_task(classify())
    async def native_endpoint(self,text,at):
        await self.preview_text(text,at)
        if self.native_preview:self.native_parts.append(self.native_preview)
        self.native_preview='';self.preview=self.native_text()
    def prepare_embedding(self,at):
        """Compute a real ECAPA anchor off the microphone/event-loop thread.

        Reuses 1.28–3.04 seconds already heard, never waits to collect future
        audio. Only same-utterance, <=1-second-old results can be consumed.
        """
        source_start=self.sentence_begin if self.sentence_begin is not None else self.begin
        if source_start is None or at-source_start<1.28 or at-self.last_embedding<.64:return
        if self.embedding_task and not self.embedding_task.done():return
        if at-self.last_speech>.64:return
        audio=self.audio(max(source_start,at-3.2),at)
        if len(audio)<20480:return
        entry={'epoch':self.epoch,'at':at,'embedding':None,'compute_ms':None}
        self.last_embedding=at;self.prefetched=entry
        async def prepare():
            started=time.perf_counter()
            try:
                from speaker_teacher import get_cached_teacher,enrollment_reference
                entry['embedding']=await asyncio.to_thread(lambda:get_cached_teacher().embed(enrollment_reference(audio)))
                entry['compute_ms']=(time.perf_counter()-started)*1000
            except asyncio.CancelledError:raise
            except Exception:entry['failed']=True
        self.embedding_task=asyncio.create_task(prepare());entry['task']=self.embedding_task
    async def observe(self,event,hop_end):
        at=hop_end/self.c.ears.sr;wave=np.asarray(event['audio16'],np.float32)
        begin=at-len(wave)/16000;self.records.append((begin,at,wave.copy()));self.now=at
        predictor=getattr(getattr(self.c,'speaker_ext',None),'predictor',None)
        if predictor is not None:
            prediction=await asyncio.to_thread(predictor.predict,event)
            if prediction.get('kind')=='speech' and prediction.get('embedding') is not None:
                self.fast_records.append((at,prediction['embedding']))
        rid=getattr(getattr(self.c,'playback',None),'rid',None)
        if self.c.speaking and self.reply_seen!=rid:self.reply_seen=rid;self.reply_started=at-len(wave)/16000
        if not self.mic_driven:await self._speech_audio(wave,at,event.get('playback_audio16'))
        self.prepare_embedding(at)
        self.prepare_focus_activity(at)
    def _maybe_probe(self):
        at=self.now;due=self.quiet_ready()
        if (self.begin is not None and at-self.begin>=.32 and self.mic_now-self.last_probe>=(.16 if due else .48)
            and not self.pause_checked and (self.probe is None or self.probe.done())
            and (due or self.c.speaking or at-self.fresh_asr_at>.48)):
            self.last_probe=self.mic_now
            audio_end=min(at,self.last_speech+.12) if due else at
            audio=self.audio(max(self.begin,audio_end-28),audio_end)
            epoch=self.epoch;speech_end=self.last_speech
            if len(audio)<3200:return
            async def recognize():
                try:
                    model=self.asr.selected;receipt={'model':'nemotron'}
                    if due and model!='nemotron':
                        await self.c.ws.send_json({'type':'asr_finalizing','model':model,'label':'Transcribing the completed speech segment…'})
                        receipt=await self.asr.transcribe(model,self.audio(self.begin,audio_end))
                        result=[receipt['text']]
                    else:
                        async with self.c.speaker_ext.asr_slot(5):
                            began=time.perf_counter()
                            result=await asyncio.to_thread(self.c.ears.service.engine.asr.transcribe,[np.pad(audio,(0,max(0,10240-len(audio))))])
                            receipt['transcribe_ms']=round((time.perf_counter()-began)*1000,2)
                    if not self.closed and epoch==self.epoch and result:
                        self.probe_text=result[0];self.probe_at=at
                        await self.c.ws.send_json({'type':'asr_preview','text':result[0],'status':'sentence_stream','source':'local_sentence_asr'})
                        from stop_commands import is_stop_command
                        if self.c.speaking and is_stop_command(result[0]):
                            await self.preview_text(result[0],at,source='probe');return
                        # No dispatch based on the captured quiet value: the
                        # user may have resumed while the GPU probe was running.
                        still_quiet=self.quiet_ready()
                        unchanged=self.last_speech<=speech_end+1e-6
                        if due and still_quiet and unchanged:
                            text=self.native_text() if model=='nemotron' and at-self.begin>28 and self.native_text() else result[0]
                            if epoch!=self.epoch or self.last_speech>speech_end+1e-6 or not self.quiet_ready():return
                            if not text.strip() or self.agent_echo(text):
                                if self.mic_now-self.last_speech>=max(1.28,self.pause_ms/1000+.32):
                                    self.c.session.reset();self.reset_segment()
                                return
                            await self.commit_pause(text,at,epoch,speech_end,receipt)
                except asyncio.CancelledError:raise
                except Exception:
                    # A transient probe failure must neither discard text nor
                    # turn background energy into activity. Retry on later audio.
                    if self.mic_now-self.last_probe_error>3:
                        self.last_probe_error=self.mic_now
                        await self.c.ws.send_json({'type':'warning','text':'Checking the last spoken words again before sending. If the selected ASR is unavailable, switch to Nemotron original.'})
                    self.last_probe=self.mic_now+.8
            self.probe=asyncio.create_task(recognize())
    async def commit_pause(self,text,at,epoch,speech_end,asr=None):
        if (epoch!=self.epoch or self.last_speech>speech_end+1e-6
            or not self.quiet_ready()):return False
        text=' '.join(text.split())
        if not text or self.begin is None:return False
        start=self.begin;audio=self.audio(start,min(at,speech_end+.12))
        asr=dict(asr or {'model':self.asr.selected})
        asr['speech_pause_ms']=self.pause_ms;asr['observed_quiet_ms']=round((self.mic_now-self.last_speech)*1000)
        cached=self.prefetched if self.prefetched and self.prefetched['epoch']==epoch and 0<=self.mic_now-self.prefetched['at']<=1.2 and self.prefetched['at']<=speech_end+.16 else None
        fast=self.fast_vector(start,speech_end)
        try:self.queue.put_nowait((text,audio,at,time.perf_counter(),cached,asr or {'model':self.asr.selected},fast))
        except asyncio.QueueFull:
            await self.c.ws.send_json({'type':'warning','text':'Several turns are queued. Please let Robby catch up.'});return False
        await self.c.ws.send_json({'type':'asr_sentence','text':text,'audio_s':at,'queued':self.c.speaking,
            'asr':asr or {'model':self.asr.selected},
            'reason':'neural speech pause','speech_pause_ms':self.pause_ms,
            'observed_quiet_ms':round((self.mic_now-self.last_speech)*1000)})
        self.c.session.reset();self.reset_segment();return True
    async def advance(self):
        """ASR previews never submit; only a neural speech pause can flush."""
        self._maybe_probe()
    async def deliver(self):
        try:
            while True:
                item=await self.queue.get();text,audio,at,queued_at,cached=item[:5]
                asr=item[5] if len(item)>5 else {'model':'nemotron'}
                start=time.perf_counter();embedding=None
                if cached and cached['task'].done() and not cached['task'].cancelled():embedding=cached.get('embedding')
                prefetched=embedding is not None
                if embedding is None and len(audio)>=10240:
                    try:
                        from speaker_teacher import get_cached_teacher,enrollment_reference
                        embedding=await asyncio.to_thread(lambda:get_cached_teacher().embed(enrollment_reference(audio)))
                    except Exception:pass
                embedding_wait_ms=(time.perf_counter()-start)*1000
                lookup_start=time.perf_counter()
                from speaker_turns import wake_intent
                owner=self.focus.active(self.memory.users,self.c.speaking)
                claim=name_claim(text)
                correction=claim and owner and self.memory.users[owner]['name'].casefold()==claim.casefold()
                handover=not self.c.speaking and bool(wake_intent(text))
                decision=self.focus.check(embedding,self.memory.users,self.c.speaking)
                fast=item[6] if len(item)>6 else None
                anchors=self.fast_anchors.get(owner,[])
                if fast is not None and anchors:
                    center=self.fast_center if self.fast_center is not None else 0.
                    fast_score=max(float(unit(fast-center)@unit(v-center)) for v in anchors)
                    decision['fast_similarity']=round(fast_score,4)
                    if fast_score<.10:decision.update(accepted=False,reason='Independent recent voice features indicate a different speaker')
                if not decision['accepted'] and not (correction and embedding is not None) and not handover:
                    await self.c.ws.send_json({'type':'speaker_focus','operation':'turn',**decision})
                    await self.c.ws.send_json({'type':'ignored','source':'conversation_focus','reason':decision['reason']})
                    continue
                preferred=owner if decision['accepted'] and not handover else None
                context=self.memory.assign(embedding,text,preferred=preferred,adapt=bool(preferred and len(audio)>=16000))
                self.focus.accept(context['speaker_id'])
                if owner!=self.focus.owner:self.focus_activity=None
                if fast is not None and context['speaker_id']!='unknown':
                    refs=self.fast_anchors.setdefault(context['speaker_id'],[])
                    if len(refs)<8:refs.append(fast.copy())
                context['conversation_focus']=self.focus.state(self.memory.users,self.c.speaking)
                lookup_ms=(time.perf_counter()-lookup_start)*1000;self.last_context=context
                await self.c.ws.send_json({**self.identity_state(),
                    'prediction':{'name':context['display_name'],'similarity':context.get('similarity')},
                    'storage':'Session RAM'})
                from stop_commands import is_stop_command
                if self.c.speaking and is_stop_command(text):
                    self.propose_barge({'yield_prob':1.,'onset_s':0.,'source':'open_sentence_stop','stop_text':text},round(at*self.c.ears.sr));continue
                while (self.c.speaking or self.c.halting) and not self.closed:await asyncio.sleep(.02)
                if self.closed:return
                self.c.current_speaker=context
                self.c.pending_input_timing={'kind':'voice','asr':asr,
                    'speaker_assignment_ms':round(embedding_wait_ms+lookup_ms,3),
                    'speaker_prefetch_compute_ms':round(cached['compute_ms'],2) if prefetched else None,
                    'speaker_prefetch_used':prefetched,'sentence_queue_ms':round((time.perf_counter()-queued_at)*1000,2)}
                await self.c.ws.send_json({'type':'endpoint','transcript':text,'speaker_id':context['speaker_id'],
                    'asr':asr,
                    'speaker_name':context['display_name'],'by':'asr_complete_sentence','timing':{
                        'speaker_assignment_ms':round(embedding_wait_ms+lookup_ms,3),
                        'speaker_embedding_wait_ms':round(embedding_wait_ms,3),'speaker_lookup_ms':round(lookup_ms,3),
                        'speaker_prefetch_used':prefetched,'speaker_prefetch_compute_ms':round(cached['compute_ms'],2) if prefetched else None,
                        'sentence_queue_ms':round((time.perf_counter()-queued_at)*1000,2),'sentence_audio_s':at}})
                await self.c._send_turn(text)
        except asyncio.CancelledError:pass
