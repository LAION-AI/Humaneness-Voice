"""Wake-free, speaker-labelled turns after a configurable neural speech pause.

All intelligible voices are admitted. Similarity assigns a label, never rejects
words. Voice profiles and explicit aliases live only in the current session.
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
    def assign(self,embedding,text):
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
            if (not refs or named or similarity>=.70) and similarity<.96:
                if len(refs)<self.max_variants:refs.append(vector.copy())
                elif named:refs[-1]=vector.copy()  # preserve the original anchor
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
        from asr_choice import ASRChoice
        self.asr=ASRChoice(self)
    def set_pause(self,value):
        if not math.isfinite(value) or not 200<=value<=2000:raise ValueError('Speech pause must be 200–2000 ms')
        self.pause_ms=round(value)
    def settings(self):
        return {'type':'asr_settings',**self.asr.settings(),'speech_pause_ms':self.pause_ms,'min_ms':200,'max_ms':2000,
                'vad':'Silero ONNX CPU · neural speech activity','submission':'All pending sentences after a speech pause'}
    def start(self):self.worker=asyncio.create_task(self.deliver())
    async def close(self):
        self.closed=True
        tasks=[t for t in [self.worker,self.probe,self.embedding_task,*self.tasks] if t]
        for t in tasks:t.cancel()
        await asyncio.gather(*tasks,return_exceptions=True);self.memory.users.clear();self.records.clear()
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
        self.last_speech=await asyncio.to_thread(self.vad.feed,wave,end)
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
            await self.c._on_barge({'yield_prob':1.,'onset_s':0.,'source':'open_sentence_stop'},round(at*self.c.ears.sr))
            self.reset_segment();self.c.session.reset();return
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
        if not self.mic_driven:await self._speech_audio(wave,at,event.get('playback_audio16'))
        self.prepare_embedding(at)
    def _maybe_probe(self):
        at=self.now;due=self.quiet_ready()
        if (self.begin is not None and at-self.begin>=.32 and self.mic_now-self.last_probe>=(.16 if due else .48)
            and not self.pause_checked and (self.probe is None or self.probe.done())
            and (due or self.c.speaking or at-self.fresh_asr_at>.48)):
            self.last_probe=self.mic_now;audio=self.audio(max(self.begin,at-28),at)
            epoch=self.epoch;speech_end=self.last_speech
            if len(audio)<3200:return
            async def recognize():
                try:
                    model=self.asr.selected;receipt={'model':'nemotron'}
                    if due and model!='nemotron':
                        await self.c.ws.send_json({'type':'asr_finalizing','model':model,'label':'Transcribing the completed speech segment…'})
                        receipt=await self.asr.transcribe(model,self.audio(self.begin,at))
                        result=[receipt['text']]
                    else:
                        async with self.c.speaker_ext.asr_slot(5):
                            result=await asyncio.to_thread(self.c.ears.service.engine.asr.transcribe,[np.pad(audio,(0,max(0,10240-len(audio))))])
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
        start=self.begin;audio=self.audio(start,at)
        cached=self.prefetched if self.prefetched and self.prefetched['epoch']==epoch and 0<=self.mic_now-self.prefetched['at']<=1.2 else None
        try:self.queue.put_nowait((text,audio,at,time.perf_counter(),cached,asr or {'model':self.asr.selected}))
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
                lookup_start=time.perf_counter();context=self.memory.assign(embedding,text)
                lookup_ms=(time.perf_counter()-lookup_start)*1000;self.last_context=context
                await self.c.ws.send_json({'type':'identity','enabled':False,'mode':'open-sentences',
                    'open_enrollment':True,'speakers':self.memory.status(),'users':[u['name'] for u in self.memory.users.values()],
                    'prediction':{'name':context['display_name'],'similarity':context.get('similarity')},
                    'storage':'Session RAM','threshold':self.memory.threshold,'sentence_mode':True})
                from stop_commands import is_stop_command
                if self.c.speaking and is_stop_command(text):
                    await self.c._on_barge({'yield_prob':1.,'onset_s':0.,'source':'open_sentence_stop'},round(at*self.c.ears.sr));continue
                while (self.c.speaking or self.c.halting) and not self.closed:await asyncio.sleep(.02)
                if self.closed:return
                self.c.current_speaker=context
                await self.c.ws.send_json({'type':'endpoint','transcript':text,'speaker_id':context['speaker_id'],
                    'asr':asr,
                    'speaker_name':context['display_name'],'by':'asr_complete_sentence','timing':{
                        'speaker_assignment_ms':round(embedding_wait_ms+lookup_ms,3),
                        'speaker_embedding_wait_ms':round(embedding_wait_ms,3),'speaker_lookup_ms':round(lookup_ms,3),
                        'speaker_prefetch_used':prefetched,'speaker_prefetch_compute_ms':round(cached['compute_ms'],2) if prefetched else None,
                        'sentence_queue_ms':round((time.perf_counter()-queued_at)*1000,2),'sentence_audio_s':at}})
                await self.c._send_turn(text)
        except asyncio.CancelledError:pass
