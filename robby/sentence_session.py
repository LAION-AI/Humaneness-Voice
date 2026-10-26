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

def voice_level(audio):
    """Upper speech-frame RMS; don't dilute the reference with quiet pauses."""
    wave=np.asarray(audio,np.float32).reshape(-1)
    if not len(wave):return 0.
    n=len(wave)//1280
    if not n:return float(np.sqrt(np.mean(wave.astype(np.float64)**2)))
    rms=np.sqrt(np.mean(wave[:n*1280].reshape(n,1280).astype(np.float64)**2,axis=1))
    return float(np.quantile(rms,.8))

class OpenVoiceMemory:
    def __init__(self,threshold=.38,max_users=32,max_variants=8,clock=time.monotonic):
        self.threshold=threshold;self.max_users=max_users;self.max_variants=max_variants
        self.users={};self.counter=0;self.turn=0;self.clock=clock;self.timeout_s=900
    def expire(self):
        for sid in list(self.users):
            if self.timeout_s>0 and self.clock()-self.users[sid].get('seen',self.clock())>self.timeout_s:del self.users[sid]
    def clear(self):self.users.clear();self.counter=0;self.turn=0
    def assign(self,embedding,text,preferred=None,adapt=False,force_new=False):
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
        elif force_new:sid=None
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
            if (not refs or name or named or similarity>=(.42 if adapt else .70)) and similarity<.96:
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
        self.pause_ms=700
        self.set_pause(float(os.environ.get('ROBO_SPEECH_PAUSE_MS','700')))
        self.vad=None;self.mic_driven=False;self.mic_now=0.;self.native_parts=[];self.native_preview=''
        self.last_activity_notice=-100.;self.last_probe_error=-100.
        from conversation_focus import ConversationFocus
        self.focus=ConversationFocus()
        self.barge_task=None;self.last_barge_check=-100.;self.bot_voice=None
        self.reply_seen=None;self.reply_started=0.
        self.activity_task=None;self.last_focus_check=-100.;self.focus_activity=None
        self.fast_records=deque(maxlen=100);self.fast_anchors={}
        self.admission_epoch=0;self.handover_until=0.;self.handover_floor=0.
        self.word_at=-100.;self.word_signature='';self.max_word_count=0;self.last_owned_at=-100.;self.background_streak=0
        self.foreground_last=-100.;self.foreground_peak=0.;self.foreground_now=False;self.foreground_first=None
        self.raw_loud_at=-100.;self.stop_candidate=None
        predictor=getattr(getattr(self.c,'speaker_ext',None),'predictor',None)
        self.fast_center= predictor.fast.teacher_mean.detach().numpy().copy() if predictor is not None else None
        from asr_choice import ASRChoice
        self.asr=ASRChoice(self)
        from sentence_decisions import SentenceDecisionIdentity
        self.decision_identity=SentenceDecisionIdentity(self)
        from speaker_management import SpeakerPermissions
        self.permissions=SpeakerPermissions(self)
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
        # Identity uncertainty must NEVER move the audio boundary backwards:
        # mixed TV speech and changing vowels used to truncate real sentences.
        self.last_speech=detected
        level=float(np.sqrt(np.mean(wave.astype(np.float64)**2))) if len(wave) else 0.
        previous_peak=self.foreground_peak
        if self.vad.active:self.foreground_peak=max(self.foreground_peak,level)
        reference=self.memory.users.get(self.focus.owner,{}).get('level',0.)
        baseline=reference or self.foreground_peak
        if baseline>1e-5 and level>=max(.0005,baseline*.25):self.raw_loud_at=end
        self.foreground_now=bool(self.vad.active and baseline>1e-5 and level>=baseline*.25)
        if self.foreground_now:
            if self.foreground_first is None or (not reference and previous_peak>0 and level>previous_peak*2.5 and end-self.foreground_first<2.):
                self.foreground_first=max(self.segment_floor,end-len(wave)/16000)
            self.foreground_last=end
        self.mic_now=end
        if self.last_speech>before+1e-6:
            if self.begin is None:self.begin=max(self.segment_floor,self.last_speech-.64)
            self.pause_checked=False
        if end-self.last_activity_notice>=.25:
            self.last_activity_notice=end
            await self.c.ws.send_json({'type':'speech_activity','speech':self.vad.active,
                'foreground_speech':end-max(self.foreground_last,self.raw_loud_at)<.5,
                'probability':round(self.vad.probability,3),'quiet_ms':round(max(0,end-self.last_speech)*1000),
                'speech_pause_ms':self.pause_ms,'pending':bool(self.preview),'source':'neural_vad'})
    def native_text(self):return ' '.join([*self.native_parts,self.native_preview]).strip()
    def quiet_ready(self):
        candidate=self.vad is not None and (self.vad.active or self.vad.probability>=.35)
        neural=not candidate and (self.mic_now-self.last_speech)*1000+1e-5>=self.pause_ms
        # Quiet TV can remain speech-like to VAD while ASR has finished the
        # foreground sentence. Confirm word stability again with a full probe.
        foreground_pause=(self.foreground_last>=self.segment_floor and not self.foreground_now
            and (self.mic_now-self.foreground_last)*1000+1e-5>=self.pause_ms)
        word_pause=bool(self.preview.strip()) and self.word_at>=self.segment_floor and self.mic_now-self.word_at>=max(1.6,self.pause_ms/1000)
        if self.foreground_now or (self.foreground_last>=self.segment_floor and self.mic_now-self.foreground_last<self.pause_ms/1000):word_pause=False
        # Do not build the initial voice reference from an isolated wake word
        # while the person is about to say their name or first full sentence.
        if self.focus.owner is None and 0<len(self.preview.split())<4:
            quiet_for=self.mic_now-max(self.last_speech,self.word_at)
            if quiet_for<1.2:return False
        return neural or foreground_pause or word_pause

    def final_audio_end(self,at):
        if self.foreground_last>=self.segment_floor and not self.foreground_now and self.mic_now-self.foreground_last>=self.pause_ms/1000:
            return min(at,self.foreground_last+.32)
        return min(at,self.last_speech+.12)

    def final_audio_start(self,start):
        return max(start,self.foreground_first-.40) if self.foreground_first is not None else start

    def note_words(self,text,at):
        signature=' '.join(re.findall(r'\w+',text.casefold()))
        count=len(signature.split());self.word_signature=signature
        # Native and full-window decoders may disagree on punctuation or an
        # article indefinitely. Only additional recognized words reset this
        # fallback pause clock; ordinary neural VAD remains the primary clock.
        if signature and count>self.max_word_count:
            self.max_word_count=count;self.word_at=max(at,self.mic_now)
            return True
        return False

    def complete_text(self,probe,allow_native=True):
        """Keep an independently decoded full prefix when a probe is a suffix."""
        if not allow_native:return probe
        native=self.native_text();a=re.findall(r'\w+',native.casefold());b=re.findall(r'\w+',probe.casefold())
        if len(a)>=4 and b and len(b)<len(a)*.6:
            overlap=sum(min(a.count(w),b.count(w)) for w in set(b))
            if overlap/len(b)>=.75:return native
        return probe or native

    def invite_speaker(self):
        """Explicit handover invalidates stale audio, pending ASR and turns."""
        self.admission_epoch+=1;self.focus.release();self.focus_activity=None
        self.permissions.pending=None;self.decision_identity.cached=None
        self.handover_until=time.monotonic()+30.;self.handover_floor=max(self.now,self.mic_now)
        for task in (self.probe,self.embedding_task,self.activity_task):
            if task and not task.done():task.cancel()
        while not self.queue.empty():self.queue.get_nowait()
        self.reset_segment();self.segment_floor=self.handover_floor
        self.last_speech=-100.;self.word_at=-100.;self.word_signature='';self.vad=None
        self.c.session.reset();self.c.current_speaker=None

    def clear_identity(self):
        self.invite_speaker();self.memory.clear();self.fast_anchors.clear();self.fast_records.clear();self.records.clear()
        self.focus.members.clear();self.focus.primary=None
        from speaker_management import SpeakerPermissions
        self.permissions=SpeakerPermissions(self)
    def agent_echo(self,text):
        tokens=re.findall(r'\w+',text.casefold())
        agent=set(re.findall(r'\w+',getattr(self.c,'open_agent_text','').casefold()))
        return self.c.speaking and len(tokens)>=4 and agent and sum(w in agent for w in tokens)/len(tokens)>.8
    def reset_segment(self):
        self.segment_floor=max(self.segment_floor,self.now)
        self.preview='';self.native_preview='';self.native_parts=[]
        self.begin=None;self.sentence_begin=None;self.prefetched=None;self.pause_checked=False
        self.probe_text='';self.epoch+=1
        self.word_signature='';self.word_at=-100.;self.max_word_count=0
        self.foreground_last=-100.;self.foreground_peak=0.;self.foreground_now=False;self.foreground_first=None
        self.stop_candidate=None
    async def preview_text(self,text,at,final=False,fresh=True,source='native'):
        if not text.strip():return
        if at<self.handover_floor:return
        if fresh:self.fresh_asr_at=at
        # Playback can leak into the microphone after acoustic delays. Reject
        # a near-copy of the current assistant's words, but never veto a short
        # Stop merely because the assistant also happened to say that word.
        if self.agent_echo(text):return
        # Punctuation is preview-only. Even a native TYGER endpoint must wait
        # for the neural speech-pause clock; consecutive sentences form one turn.
        if source=='native':self.native_preview=text;self.preview=self.native_text()
        else:self.preview=text
        self.note_words(self.preview,at)
        self.preview_at=at
        decisions=getattr(self.c,'decisions',None)
        if decisions:
            hop=round(at*self.c.ears.sr)
            if self.c.speaking:decisions.offer_agent([text],hop)
            else:decisions.offer(self.preview,hop)
        from stop_commands import is_stop_command
        if self.c.speaking and is_stop_command(text):
            rid=getattr(getattr(self.c,'playback',None),'rid',None)
            old=self.stop_candidate
            if (old is None or old['rid']!=rid or old['epoch']!=self.epoch or
                (at-old['at']>1.6 and self.raw_loud_at>old['at']+.32)):
                self.stop_candidate={'at':at,'rid':rid,'epoch':self.epoch}
            self.propose_barge({'yield_prob':1.,'onset_s':0.,'source':'open_sentence_stop','stop_text':text},round(at*self.c.ears.sr))
            return

    def identity_state(self):
        return {'type':'identity','enabled':self.focus.enabled,'mode':'open-sentences','sentence_mode':True,
                'open_enrollment':True,'speakers':[{**u,'admitted':u['id'] in self.focus.members} for u in self.memory.status()],'users':[u['name'] for u in self.memory.users.values()],
                'threshold':self.memory.threshold,'conversation_focus':self.focus.state(self.memory.users,self.c.speaking),
                'handover_pending':time.monotonic()<self.handover_until,'storage':'Session RAM'}

    def turn_decision(self,embedding,fast=None,level=None):
        """Tolerant turn continuity; strong rivals still require handover."""
        decision=self.focus.check(embedding,self.memory.users,self.c.speaking)
        owner=decision.get('speaker_id') or self.focus.owner;anchors=self.fast_anchors.get(owner,[])
        reference_level=self.memory.users.get(owner,{}).get('level',0.)
        if (self.focus.enabled and reference_level>0 and level is not None and level<reference_level*.35
            and decision.get('similarity',-1.)<.65):
            return {**decision,'accepted':False,'quiet_background':True,'reason':'Much quieter competing voice; keeping the foreground conversation'}
        if fast is not None and anchors:
            center=self.fast_center if self.fast_center is not None else 0.
            score=max(float(unit(fast-center)@unit(v-center)) for v in anchors)
            decision['fast_similarity']=round(score,4)
            # A weak short student estimate is not an identity veto. Allow
            # modest ECAPA dips only with a recent established conversation,
            # supportive student evidence and no better known rival.
            own=decision.get('similarity',-1.);rival=decision.get('rival_similarity',-1.)
            if (not decision['accepted'] and owner and self.focus.clock()-self.focus.seen<60.
                and own>=max(.18,self.focus.threshold-.10) and own>=rival+.10 and score>=.30):
                decision.update(accepted=True,provisional=True,reason='Current speaker retained across a supported voice variation')
        return decision

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
            decision=await self.assess_interrupt(event,at,rid)
            if self.closed or not self.c.speaking or getattr(getattr(self.c,'playback',None),'rid',None)!=rid or owner_at_proposal!=self.focus.owner:return
            await self.c.ws.send_json({'type':'speaker_focus','operation':'interrupt',**decision})
            if decision['accepted']:
                target=decision.get('speaker_id')
                if target in self.memory.users:
                    self.focus.accept(target)
                    self.c.current_speaker={'speaker_id':target,'display_name':self.memory.users[target]['name'],'enrolled':True}
                await self.c._on_barge({**event,'source':'focused_stop' if event.get('stop_text') else 'focused_tyger'},hop_end)
                if event.get('stop_text'):
                    self.reset_segment();self.c.session.reset()
                    await self.c.ws.send_json({'type':'stop_command','speaker_name':None if decision.get('uncertain_literal_control') else self.memory.users.get(self.focus.owner,{}).get('name'),
                        'identity_uncertain':bool(decision.get('uncertain_literal_control'))})
        self.barge_task=asyncio.create_task(verify())

    async def assess_interrupt(self,event,at,rid):
        candidate=self.stop_candidate
        lexical=bool(event.get('stop_text') and candidate and candidate['rid']==rid and candidate['epoch']==self.epoch
            and 0<=self.mic_now-candidate['at']<=2.0 and self.raw_loud_at>=self.reply_started)
        if lexical:
            # Keep the original candidate expiry, but include newly received
            # foreground phonemes. Freezing the query at the first partial
            # "Warte" used to keep rejecting the later complete "Warte bitte".
            query_at=max(candidate['at'],min(at,self.raw_loud_at+.12))
        else:query_at=min(at,self.last_speech+.16) if event.get('stop_text') and 0<=at-self.last_speech<=1.2 else at
        # A single imperative has much less voiced audio than a sentence.
        # A 1.28-s window can contain mostly competing speech around "Stopp".
        brief=lexical and len(re.findall(r'\w+',event.get('stop_text','')))==1
        decision=await self.verify_barge(query_at,rid,lexical=lexical,**({'brief':True} if brief else {}))
        decision['lexical_audio_evidence']=lexical
        if self.focus.enabled and not decision.get('quiet_background'):
            target=decision.get('speaker_id') or self.focus.owner
            fast=self.fast_match(at,target)
            evidence=self.fast_evidence(at,target)
            decision.update(fast_similarity=fast,fast_consistency=evidence['consistency'],fast_frames=evidence['frames'])
            # A short changed-vowel Stop can have weak ECAPA similarity.
            # Rescue only explicit commands with sustained, independent
            # native-hop evidence; never rescue a competing voice or echo.
            separate=decision.get('similarity',-1.)>=decision.get('rival_similarity',-1.)+.06 and (decision.get('bot_similarity',-1.)<.4 or decision.get('similarity',-1.)>=decision.get('bot_similarity',-1.)+.06)
            # A one-word imperative has too few phonemes for reliable identity.
            # Prefer a clear foreground control over continuing unwanted speech,
            # without assigning/training an identity from that uncertain cue.
            # Strong known rivals, echo, quiet TV and stale words still veto it.
            from stop_commands import is_stop_command
            reference_level=self.memory.users.get(self.focus.owner,{}).get('level',0.)
            cue_level=voice_level(self.audio(max(self.reply_started,query_at-1.28),query_at)) if lexical else 0.
            if (lexical and is_stop_command(event.get('stop_text',''))
                and reference_level>0 and cue_level>=reference_level*.45
                and decision.get('similarity',-1.)>=.12 and decision.get('rival_similarity',-1.)<.45
                and (decision.get('bot_similarity',-1.)<.4 or decision.get('similarity',-1.)>=decision.get('bot_similarity',-1.)+.06)):
                decision['foreground_literal_control']=True
                if not decision['accepted'] or (fast is not None and fast<.25) or evidence['consistency']<.60:
                    decision.update(accepted=True,uncertain_literal_control=True,reason='Fresh foreground Stop accepted; short speaker cue is uncertain')
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
            if fast is not None and fast<.25 and not decision.get('foreground_literal_control'):decision.update(accepted=False,reason='Recent speaker features disagree with the active partner')
            if evidence['frames']>=3 and evidence['consistency']<.60 and not decision.get('foreground_literal_control'):decision.update(accepted=False,reason='Unstable recent speaker identity; background cannot cancel the reply')
            if not event.get('stop_text') and (decision.get('similarity',-1.)<max(.55,self.focus.threshold) or (fast is not None and fast<.45)):
                decision.update(accepted=False,reason='Background conversation is not an addressed interruption')
        return decision

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

    async def verify_barge(self,at,rid,lexical=False,brief=False):
        if not self.focus.enabled:return {'accepted':True,'reason':'Conversation focus disabled'}
        if self.last_speech<at-.48 and not lexical:return {'accepted':False,'reason':'No fresh speech'}
        # Only the current reply's recent microphone audio; no old enrollment
        # speech may authorize a new background speaker to cancel playback.
        audio=self.audio(max(self.reply_started,at-1.60),at)
        if lexical:
            from speaker_teacher import enrollment_reference
            audio=enrollment_reference(self.audio(max(self.reply_started,at-3.04),at),seconds=.64 if brief else 1.28)
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
            decision=self.focus.check(embedding,self.memory.users,True,bot)
            level=voice_level(audio)
            reference_level=self.memory.users.get(decision.get('speaker_id') or self.focus.owner,{}).get('level',0.)
            if reference_level>0 and level<reference_level*.35 and decision.get('similarity',-1.)<.65:
                decision.update(accepted=False,quiet_background=True,reason='Quiet background voice cannot cancel the current reply')
            return decision
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
                # Diagnostic activity only: a weak or wrong short embedding
                # must never crop ASR or discard the beginning of a sentence.
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

        Selects the most audible 3.04 seconds of the current utterance, never
        waits for future audio. A trailing TV-only window must not replace
        the actual request's voice. Only recent same-epoch results are reused.
        """
        source_start=self.sentence_begin if self.sentence_begin is not None else self.begin
        if source_start is None or at-source_start<1.28 or at-self.last_embedding<.64:return
        if self.embedding_task and not self.embedding_task.done():return
        if at-self.last_speech>.64:return
        # Match the final-utterance selector instead of always taking its tail.
        # Quiet background speech can continue for seconds after the owner;
        # using that tail produced a fresh but wrong-speaker cached embedding.
        audio=self.audio(max(source_start,self.segment_floor,at-28.),at)
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
            audio_end=self.final_audio_end(at) if due else at
            audio_start=self.final_audio_start(self.begin) if due else self.begin
            audio=self.audio(max(audio_start,audio_end-28),audio_end)
            epoch=self.epoch;speech_end=self.last_speech;words_at=self.word_at
            if len(audio)<3200:return
            async def recognize():
                try:
                    model=self.asr.selected;receipt={'model':'nemotron'}
                    if due and model!='nemotron':
                        await self.c.ws.send_json({'type':'asr_finalizing','model':model,'label':'Transcribing the completed speech segment…'})
                        receipt=await self.asr.transcribe(model,self.audio(audio_start,audio_end))
                        result=[receipt['text']]
                    else:
                        async with self.c.speaker_ext.asr_slot(5):
                            began=time.perf_counter()
                            result=await asyncio.to_thread(self.c.ears.service.engine.asr.transcribe,[np.pad(audio,(0,max(0,10240-len(audio))))])
                            receipt['transcribe_ms']=round((time.perf_counter()-began)*1000,2)
                    if not self.closed and epoch==self.epoch and result:
                        self.probe_text=result[0];self.probe_at=at
                        await self.c.ws.send_json({'type':'asr_preview','text':result[0],'status':'sentence_stream','source':'local_sentence_asr'})
                        complete=self.complete_text(result[0],allow_native=audio_start-self.begin<.3) if model=='nemotron' else result[0]
                        neural_quiet=not (self.vad is not None and (self.vad.active or self.vad.probability>=.35))
                        if not neural_quiet:
                            if self.note_words(complete,at):
                                self.preview=complete
                                await self.preview_text(complete,at,source='probe')
                                # A full recognition pass found new words:
                                # continue collecting rather than cutting them.
                                return
                        decisions=getattr(self.c,'decisions',None)
                        if decisions:
                            hop=round(at*self.c.ears.sr)
                            if self.c.speaking:decisions.offer_agent([result[0]],hop)
                            else:decisions.offer(complete,hop)
                        from stop_commands import is_stop_command
                        if self.c.speaking and is_stop_command(result[0]):
                            await self.preview_text(result[0],at,source='probe');return
                        # No dispatch based on the captured quiet value: the
                        # user may have resumed while the GPU probe was running.
                        still_quiet=self.quiet_ready()
                        foreground_quiet=self.foreground_last>=self.segment_floor and not self.foreground_now and self.mic_now-self.foreground_last>=self.pause_ms/1000
                        unchanged=self.last_speech<=speech_end+1e-6 or foreground_quiet or (self.word_at<=words_at+1e-6 and self.mic_now-self.word_at>=max(1.6,self.pause_ms/1000))
                        if due and still_quiet and unchanged:
                            text=complete
                            if epoch!=self.epoch or not self.quiet_ready():return
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
        word_quiet=self.word_at>=self.segment_floor and self.mic_now-self.word_at>=max(1.6,self.pause_ms/1000)
        foreground_quiet=self.foreground_last>=self.segment_floor and not self.foreground_now and self.mic_now-self.foreground_last>=self.pause_ms/1000
        if (epoch!=self.epoch or (self.last_speech>speech_end+1e-6 and not word_quiet and not foreground_quiet)
            or not self.quiet_ready()):return False
        text=' '.join(text.split())
        if not text or self.begin is None:return False
        if self.focus.owner is None and len(text.split())<4 and self.mic_now-max(self.last_speech,self.word_at)<1.2:
            if not self.preview:self.preview=text;self.note_words(text,at)
            return False
        decisions=getattr(self.c,'decisions',None)
        if decisions and decisions.consumed:
            from types import SimpleNamespace
            track=SimpleNamespace(sid=self.focus.owner,start=self.begin,voiced_start=self.begin,last_speech=self.last_speech)
            if decisions.suppresses(track,text):
                self.c.session.reset();self.reset_segment();return True
        start=self.final_audio_start(self.begin);audio=self.audio(start,self.final_audio_end(at))
        asr=dict(asr or {'model':self.asr.selected})
        asr['speech_pause_ms']=self.pause_ms;asr['observed_quiet_ms']=round((self.mic_now-self.last_speech)*1000)
        cached=self.prefetched if self.prefetched and self.prefetched['epoch']==epoch and 0<=self.mic_now-self.prefetched['at']<=1.2 and self.prefetched['at']<=speech_end+.16 else None
        fast=self.fast_vector(start,speech_end)
        try:self.queue.put_nowait((text,audio,at,time.perf_counter(),cached,asr or {'model':self.asr.selected},fast,self.admission_epoch))
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
                admission=item[7] if len(item)>7 else self.admission_epoch
                if admission!=self.admission_epoch:continue
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
                if admission!=self.admission_epoch:continue
                lookup_start=time.perf_counter()
                from speaker_turns import wake_intent
                owner=self.focus.active(self.memory.users,self.c.speaking)
                claim=name_claim(text)
                correction=claim and owner and self.memory.users[owner]['name'].casefold()==claim.casefold()
                handover=not self.c.speaking and bool(wake_intent(text))
                fast=item[6] if len(item)>6 else None
                from speaker_teacher import enrollment_reference
                # Identity and its level gate must describe the same audible
                # context. Long TV pre-roll otherwise dilutes the 80th-percentile
                # level and rejects an otherwise correctly matched short request.
                level=voice_level(enrollment_reference(audio))
                decision=self.turn_decision(embedding,fast,level)
                guest=None
                if self.permissions.candidate(embedding,fast,level,text):
                    guest=self.permissions.enroll(embedding,text,fast,level)
                    if guest:decision={'accepted':True,'speaker_id':guest['speaker_id'],'reason':'Explicitly invited guest'}
                initial_intro=claim and owner and not self.memory.users[owner]['named'] and self.focus.clock()-self.focus.seen<20.
                old_level=self.memory.users.get(owner,{}).get('level',0.)
                foreground_reclaim=owner and not self.memory.users[owner]['named'] and old_level>0 and level>old_level*2.5 and len(text.split())>=4 and embedding is not None
                if decision.get('quiet_background') or (not decision['accepted'] and not ((correction or initial_intro or foreground_reclaim) and embedding is not None) and not handover):
                    await self.c.ws.send_json({'type':'speaker_focus','operation':'turn',**decision})
                    await self.c.ws.send_json({'type':'ignored','source':'conversation_focus','reason':decision['reason']})
                    self.permissions.observe_background()
                    continue
                preferred=decision.get('speaker_id',owner) if decision['accepted'] and not handover else None
                reference=self.memory.users.get(preferred,{}).get('level',0.)
                safe_update=(not decision.get('provisional') and decision.get('similarity',1.)>=.5 and len(audio)>=20480 and (not reference or level>=reference*.45))
                context=guest or self.memory.assign(embedding,text,preferred=preferred,adapt=bool(preferred and safe_update and decision.get('fast_similarity',1.)>=.35),force_new=bool(foreground_reclaim))
                if (initial_intro or foreground_reclaim) and owner and context['speaker_id']!=owner:
                    self.focus.members.discard(owner)
                    if self.focus.primary==owner:self.focus.primary=context['speaker_id']
                self.focus.accept(context['speaker_id'])
                if context['speaker_id'] in self.memory.users:
                    user=self.memory.users[context['speaker_id']]
                    if not user.get('level') and len(audio)>=16000:user['level']=level
                self.handover_until=0.;self.last_owned_at=at
                if owner!=self.focus.owner:self.focus_activity=None
                if fast is not None and context['speaker_id']!='unknown' and (context['speaker_id'] not in self.fast_anchors or owner is None or correction or safe_update):
                    refs=self.fast_anchors.setdefault(context['speaker_id'],[])
                    if len(refs)<8:refs.append(fast.copy())
                context['conversation_focus']=self.focus.state(self.memory.users,self.c.speaking)
                context['speaker_management']=self.permissions.context()
                lookup_ms=(time.perf_counter()-lookup_start)*1000;self.last_context=context
                await self.c.ws.send_json({**self.identity_state(),
                    'prediction':{'name':context['display_name'],'similarity':context.get('similarity')},
                    'storage':'Session RAM'})
                from stop_commands import is_stop_command
                if self.c.speaking and is_stop_command(text):
                    self.propose_barge({'yield_prob':1.,'onset_s':0.,'source':'open_sentence_stop','stop_text':text},round(at*self.c.ears.sr));continue
                while (self.c.speaking or self.c.halting) and not self.closed:await asyncio.sleep(.02)
                if self.closed:return
                if admission!=self.admission_epoch:continue
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
