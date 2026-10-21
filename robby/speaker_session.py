"""Async bridge from frozen live features to speaker-relative ASR turns."""
import asyncio,json,time,logging,re,copy
from collections import deque
from pathlib import Path
import numpy as np
from speaker_runtime import LivePredictor
from speaker_turns import SpeakerMemory,SpeakerTurns,self_name,WAKE,wake_intent,choose_turn_text,INTRO,words,word_distance
from identity import unit
from asr_priority import PriorityGate
from enrollment_cues import aligned_reference,foreground_context_allowed
from recovery_rules import known_voice_consensus
from stop_commands import is_stop_command,is_brief_stop_command,has_stop_intent

class SpeakerSession:
    def __init__(self,conversation,long_context='long3',use_diarization=True):
        self.c=conversation;self.predictor=LivePredictor(long_context)
        self.identity_context_s=self.predictor.frames*.08 if self.predictor.use_long else 3.04
        config=Path(__file__).resolve().parent/'speaker_live/calibration.json'
        values=json.loads(config.read_text()) if config.exists() else {}
        self.memory=SpeakerMemory(threshold=float(values.get('threshold',.55)),margin=float(values.get('margin',.04)),
            center=self.predictor.fast.teacher_mean.detach().numpy().copy() if values.get('centered',True) else None)
        self.turns=SpeakerTurns(self.memory);self.tasks=set();self.queue=asyncio.Queue(maxsize=4)
        self.asr_lock=PriorityGate();self.worker=None;self.closed=False;self.last_notice=0.;self.last_barge=-100.
        self.prepared={};self.prepare_quiet_s=.24;self.pending_turns={}
        self.pending_names={};self.decode_count=0;self.last_activity=None
        self.epoch=0;self.records=deque(maxlen=80);self.diar_offset=None;self.owners=deque(maxlen=2048)
        self.overlaps=deque(maxlen=512)
        self.asr_wakes=deque(maxlen=64)
        self.background_channels=set();self.channel_levels={};self.bridged_seconds=0.
        from diarization_client import DiarizationClient
        self.diar=DiarizationClient() if use_diarization else None
        self.diarization_active=False
        self.seen_channels=set();self.foreground_channel=None
        self.utterance_threshold=.32;self.utterance_margin=.08
        self.interrupt_checked_at=-100.;self.interrupt_reference=None
        self.activity_checked_at=-100.;self.activity_scores={}
        self.stop_task=None;self.stop_checked_at=-100.
        self.open_enrollment=True
        self.claim_observations=deque(maxlen=32)
    def start(self):
        if self.worker is None:self.worker=asyncio.create_task(self.deliver_loop())
    async def close(self):
        self.closed=True
        for task in [*self.tasks,*([self.worker] if self.worker else [])]:task.cancel()
        await asyncio.gather(*self.tasks,*([self.worker] if self.worker else []),return_exceptions=True)
        if self.diar is not None:await self.diar.close()
        self.turns.clear();self.pending_names.clear();self.records.clear();self.owners.clear();self.overlaps.clear();self.asr_wakes.clear();self.background_channels.clear();self.channel_levels.clear()
        while not self.queue.empty():self.queue.get_nowait()
        self.worker=None;self.last_activity=None
        self.interrupt_reference=None
        self.prepared.clear()
        self.pending_turns.clear()
        if hasattr(self,'claim_observations'):self.claim_observations.clear()
    async def clear(self):
        self.epoch+=1
        if hasattr(self,'claim_observations'):self.claim_observations.clear()
        for task in list(self.tasks):task.cancel()
        await asyncio.gather(*list(self.tasks),return_exceptions=True)
        self.tasks.clear();self.turns.clear();self.pending_names.clear()
        self.prepared.clear()
        self.pending_turns.clear()
        while not self.queue.empty():self.queue.get_nowait()
        self.records.clear();self.diar_offset=None
        self.owners.clear()
        self.overlaps.clear()
        self.asr_wakes.clear()
        self.background_channels.clear();self.channel_levels.clear();self.bridged_seconds=0.
        self.seen_channels.clear();self.foreground_channel=None
        self.interrupt_reference=None;self.interrupt_checked_at=-100.
        self.activity_checked_at=-100.;self.activity_scores={}
        if self.diar is not None:
            await self.diar.close()
            from diarization_client import DiarizationClient
            self.diar=DiarizationClient()
    def spawn(self,coroutine):
        task=asyncio.create_task(coroutine);self.tasks.add(task);task.add_done_callback(self.tasks.discard)
        return task

    def asr_slot(self,priority=10):
        # Plain locks remain supported for offline/mocked contracts.
        return self.asr_lock.slot(priority) if isinstance(self.asr_lock,PriorityGate) else self.asr_lock
    async def notice(self,**fields):
        if not self.c.ws.closed:
            await self.c.ws.send_json({'type':'identity','enabled':self.c.identity_enabled,'mode':self.c.identity_mode,
                **self.turns.status(),'diarization_active':self.diarization_active,'utterance_threshold':self.utterance_threshold,
                'identity_context_s':self.identity_context_s,'open_enrollment':getattr(self,'open_enrollment',False),**fields})
            if fields.get('warning'):await self.c.ws.send_json({'type':'warning','text':fields['warning']})
    async def transcribe(self,wave):
        if len(wave)<2560:return ''
        # A known person's short "yes" is valid. Right padding is silence,
        # never borrowed background audio from a different speaker.
        wave=np.pad(wave,(0,max(0,10240-len(wave))))
        async with self.asr_slot(20):
            self.decode_count+=1
            texts=await asyncio.to_thread(self.c.ears.service.engine.asr.transcribe,[np.asarray(wave,np.float32)])
            return (texts[0] if texts else '').strip()[:3000]

    async def transcribe_stop(self,wave):
        """One bounded native batch: continuous + two acoustic stop views.

        Continuous audio protects quiet consonants; fixed padding protects the
        decoder from sub-second input. Neither view is selected by voice score.
        The caller still requires independent current-owner/echo verification.
        """
        if len(wave)<2560:return ''
        views=[wave];n=len(wave)//320
        levels=np.sqrt(np.mean(wave[:n*320].reshape(n,320)**2,axis=1))
        if float(levels.max())<max(.003,self.foreground_floor()):return ''
        for fraction in (.05,.20):
            active=np.flatnonzero(levels>=max(.002,self.foreground_floor(),float(levels.max())*fraction))
            if len(active):
                lo=max(0,int(active[0])*320-1920);hi=min(len(wave),(int(active[-1])+1)*320+1920)
                views.append(wave[lo:hi])
        batch=[np.pad(v,(3200,max(0,40960-len(v)-3200))).astype(np.float32) for v in views]
        async with self.asr_slot(0):
            self.decode_count+=1
            texts=await asyncio.to_thread(self.c.ears.service.engine.asr.transcribe,batch)
        self.last_stop_texts=[str(t).strip()[:300] for t in texts]
        return next((t for t in self.last_stop_texts if has_stop_intent(t)),'')

    def note_asr(self,text,at):
        if getattr(self,'open_enrollment',False):
            from open_speakers import claimed_name
            claim=claimed_name(text)
            if claim:self.claim_observations.append((float(at),claim.casefold()))
        # Keep explicit, same-name activation evidence across ASR revisions.
        # A later full-window decode may lose "Hey" although the incremental
        # decoder already recognized it. This is not an LLM-inferred wake.
        calls=list(WAKE.finditer(text))
        activation=text[calls[-1].start():] if calls else text
        name=self_name(activation)
        activated=wake_intent(activation)
        if name and activated:self.asr_wakes.append((float(at),name.casefold()))
        active=[t for t in self.turns.tracks.values() if t.sid and not t.closed and at-t.last_speech<1.6]
        if len(active)==1:
            track=active[0]
            if activated:track.wake=True
            user=self.memory.users.get(track.sid)
            same_name=bool(name and (self_name(track.partial) or (user.name if user and user.named else '')).casefold()==name.casefold())
            # A room-wide partial cannot give a background-channel Guest the
            # name spoken by a different, not-yet-diarized foreground person.
            if same_name and (activated or track.wake):track.name_heard=True

    def observed_wake(self,track,text):
        name=self_name(text)
        return bool(name and self.observed_intro_names(track,name))
    def observed_intro_names(self,track,name):
        name=name.casefold()
        matches={value for at,value in self.asr_wakes if track.start-.32<=at<=track.last_speech+.96
            and (value==name or (min(len(value),len(name))>=4 and (value.startswith(name) or name.startswith(value))))}
        return matches if len(matches)==1 else set()

    async def confirm_interrupt(self,proposal,hop_end):
        """Confirm a native GRU proposal with known-user vs actual bot voice.

        No enrollment, ASR or invented BARGE. A reusable diarization channel
        alone is not sufficient: echoes can occupy an old user's channel.
        """
        end=hop_end/self.c.ears.sr
        addressee=(getattr(self.c,'current_speaker',None) or {}).get('speaker_id')
        if addressee not in self.memory.users:return None
        if end-self.interrupt_checked_at<.30:return None
        self.interrupt_checked_at=end
        references=[u for u in self.memory.users.values() if u.teacher is not None]
        if not references:return None
        context=getattr(self,'identity_context_s',3.04)
        begin=max(0.,end-context,float(getattr(self.c,'speaker_reply_start_s',0.)))
        wave=self.audio_between(begin,end)
        literal=proposal.get('source')=='verified_stop_command'
        if literal:
            from audio_sync import suppress_aligned_leakage
            wave=suppress_aligned_leakage(wave,self.playback_between(begin,end))
        if len(wave)<(2560 if literal else 7680):return None
        # Remove quiet leading leakage before speaker verification, not from
        # the captured turn or native GRU input. Double talk is not separated.
        n=len(wave)//320;levels=np.sqrt(np.mean(wave[:n*320].reshape(n,320)**2,axis=1))
        active=np.flatnonzero(levels>=max(.003,float(levels.max())*(.15 if literal else .25)))
        if len(active)<(8 if literal else 24):
            if getattr(self.c,'synthetic_diagnostics',False):
                await self.c.ws.send_json({'type':'synthetic_barge_verification','accepted':False,
                    'reason':'insufficient_voiced_duration','voiced_seconds':len(active)*.02,
                    'fast_vote':self.stop_fast_vote(begin,end)})
            return None # at least 480 ms of audible evidence
        if literal and (len(wave)-(int(active[-1])+1)*320)/16000>.80:
            if getattr(self.c,'synthetic_diagnostics',False):
                await self.c.ws.send_json({'type':'synthetic_barge_verification','accepted':False,'reason':'stale_stop_audio'})
            return None
        brief=literal and len(active)<24 and is_brief_stop_command(proposal.get('stop_text',''))
        # A sub-second word needs its weak consonants; on a longer phrase,
        # expanding to quiet surrounding playback dilutes the voice cue.
        context_active=(np.flatnonzero(levels>=max(.001,float(levels.max())*.03)) if brief else active)
        pad=1280 if brief else 160
        lo=max(0,int(context_active[0])*320-pad);hi=min(len(wave),(int(context_active[-1])+1)*320+pad)
        query=wave[lo:hi]
        query=np.pad(query,(0,max(0,10240-len(query))))
        floor=self.foreground_floor()
        if floor and float(np.sqrt(np.mean(query*query)))<floor:return None
        aligned_playback=self.playback_between(begin+lo/16000,begin+hi/16000)
        waveform_correlation=None
        if len(query)>=20480 and len(aligned_playback)==len(query) and float(np.sqrt(np.mean(aligned_playback**2)))>=.001:
            from audio_sync import estimate_echo_delay
            waveform_correlation=estimate_echo_delay(query,aligned_playback,symmetric=True,absolute=True)['confidence']
        playback=self.c.playback;rid=playback.rid;epoch=self.epoch
        size=min(playback.n16,4*16000)
        reference=playback.audio[:size].copy()
        cached=self.interrupt_reference
        def score():
            from speaker_teacher import get_cached_teacher
            teacher=get_cached_teacher()
            reuse=bool(cached and cached[0]==rid and size<cached[1]*1.8)
            # While waiting for admission/first TTS audio there is no new
            # playback to echo. Retain the previous bot cue for acoustic tails
            # if available, without preventing known users cancelling a wait.
            bot=(cached[2] if cached else None) if size<10240 else cached[2] if reuse else unit(teacher.embed(reference))
            user=unit(teacher.embed(query))
            values=sorted(((u.teacher_score(user),u.id) for u in references),reverse=True)
            return bot,cached[1] if reuse else size,float(user@bot) if bot is not None else -1.,values
        try:bot,bot_size,bot_score,values=await asyncio.to_thread(score)
        except Exception:return None
        if (self.closed or self.epoch!=epoch or not self.c.speaking or playback.rid!=rid or
            (getattr(self.c,'current_speaker',None) or {}).get('speaker_id')!=addressee):return None
        if size>=10240:self.interrupt_reference=(rid,bot_size,bot)
        best,sid=values[0];runner=max(bot_score,values[1][0] if len(values)>1 else -1.)
        if sid!=addressee:
            if getattr(self.c,'synthetic_diagnostics',False):
                await self.c.ws.send_json({'type':'synthetic_barge_verification','accepted':False,
                    'reason':'not_current_addressee','winner_id':sid,'addressee_id':addressee,'user_cosine':best})
            return None
        # Preserve quick, strong native interruptions when the already-bound
        # person's independent fast cue agrees. Use the same bounded acoustic
        # continuity floor as full-turn verification, never for new users.
        recent=self.turns.tracks.get(sid);user=self.memory.users.get(sid)
        recently_owned=bool(recent is not None and not recent.closed and end-recent.last_speech<=.64
            and recent.speech_s>=.48)
        # A short ECAPA window alone can confuse two male voices. Require
        # independent live ownership, or a much stronger full known-voice match
        # when the diarizer assigned a genuinely new channel.
        barge_threshold=max(.50,self.utterance_threshold)
        accepted=bool(best>=barge_threshold and best-runner>=self.utterance_margin
            and (recently_owned or (best>=max(.70,barge_threshold) and best-runner>=.20)))
        continuity=bool(float(proposal.get('yield_prob',0))>=.8 and recent is not None and user is not None
            and not recent.closed and end-recent.last_speech<=.64 and len(recent.fast)>=2
            and self.memory.fast_score(recent.vector(),user)>=self.memory.threshold-.12
            and best>=max(.42,barge_threshold-.08) and best-runner>=max(.12,self.utterance_margin))
        accepted=accepted or continuity
        # Similar timbre is not proof of an echo. A native BARGE proposal can
        # be corroborated by full known-voice verification and actual aligned
        # playback-content disagreement, even when the bot uses a similar
        # voice. Exact/delayed waveform echo retains the ordinary bot veto.
        known_runner=values[1][0] if len(values)>1 else -1.
        different_playback=bool(waveform_correlation is not None and waveform_correlation<.12)
        content_confirmed=bool(different_playback and best>=barge_threshold and
            best-known_runner>=max(.12,self.utterance_margin))
        accepted=accepted or content_confirmed
        # A literal stop can contain far less than 480 ms of speech. Only
        # that already-ASR-confirmed path may use a shorter teacher cue,
        # AND it requires an independently strong current-person fast vote.
        fast=self.stop_fast_vote(begin,end) if literal else {}
        fast_score=fast.get('candidate_similarity',-1)
        # A longer unrelated voice can weakly resemble the only enrolled user.
        # Do not grant that entire phrase the low-evidence short-word rule.
        fast_floor=.40 if brief else max(.58,self.memory.threshold-.04)
        short_command=bool(literal and fast.get('candidate_id')==addressee and
            fast_score>=fast_floor and fast.get('candidate_margin',0)>=.12 and
            best>=max(.25,self.utterance_threshold-.15) and best-runner>=max(.10,self.utterance_margin)
            and best*fast_score>=.12)
        long_vote=self.stop_long_vote(begin,end) if literal and not brief else {}
        # A clearly recognized short phrase is often 0.5–1.2 s, too long for
        # the one-word duration exception but too short for stable fast vectors.
        # Require agreement from two ASR crops and BOTH acoustic models, with
        # a stronger teacher floor/margin than the existing one-word path.
        # No change to ordinary turn matching or uncertain semantic requests.
        compact_command=bool(literal and proposal.get('stop_views_agree') and
            is_stop_command(proposal.get('stop_text','')) and len(words(proposal['stop_text']))<=6 and
            len(active)<=64 and fast.get('candidate_id')==addressee and fast_score>=.35 and
            fast.get('candidate_margin',0)>=.15 and best>=max(.35,self.utterance_threshold) and
            best-runner>=max(.15,self.utterance_margin) and best*fast_score>=.14)
        # For a sub-480-ms single imperative, zero-padding can make the
        # full-utterance teacher weak. A STRONGER fast cue may compensate, but
        # only with two agreeing ASR crops, teacher/rival separation and the
        # existing joint-score floor. Never apply this to a longer sentence.
        brief_consensus=bool(brief and proposal.get('stop_views_agree') and
            fast.get('candidate_id')==addressee and fast_score>=.50 and
            fast.get('candidate_margin',0)>=.15 and best>=max(.20,self.utterance_threshold-.15) and
            best-runner>=max(.15,self.utterance_margin) and best*fast_score>=.12)
        accepted=accepted or short_command or compact_command or brief_consensus
        if getattr(self.c,'synthetic_diagnostics',False):
            await self.c.ws.send_json({'type':'synthetic_barge_verification','accepted':accepted,
                'user_cosine':best,'bot_cosine':bot_score,'margin':best-runner,'evidence_seconds':len(query)/16000,
                'strong_native_continuity':continuity,'waveform_correlation':waveform_correlation,
                'different_playback_content':content_confirmed,'literal_stop':literal,'short_command_consensus':short_command,
                'compact_command_consensus':compact_command,
                'brief_crop_consensus':brief_consensus,
                'long_student_consensus':False,'long_vote':long_vote,'fast_vote':fast})
        return sid if accepted and sid in self.memory.users else None

    def stop_fast_vote(self,begin,end):
        rows=[(p['raw_embedding'],float(np.sqrt(np.mean(w*w)))) for a,b,p,w in self.records if b>begin and a<end
            and p.get('raw_embedding') is not None and len(w)]
        if not rows:return {}
        floor=max(.003,self.foreground_floor(),max(r[1] for r in rows)*.35)
        rows=[r for r in rows if r[1]>=floor]
        if not rows:return {}
        # Otherwise 1.4 s of quiet agent leakage overwhelms a 200-ms word.
        vector=np.average([r[0] for r in rows],axis=0,weights=[r[1]**2 for r in rows])
        vector=unit(vector);result=self.memory.match(vector)
        scores=sorted(((self.memory.fast_score(vector,u),u.id) for u in self.memory.users.values()),reverse=True)
        if scores:
            score,sid=scores[0];runner=scores[1][0] if len(scores)>1 else -1.
            result.update(candidate_id=sid,candidate_similarity=score,candidate_margin=score-runner)
        return result

    def schedule_stop_probe(self,hop_end):
        """Bounded ASR fallback for literal stop requests while TYGER speaks.

        Native two-ear BARGE remains primary. No continuous second ASR model:
        reuse the frozen decoder only after independent current-user evidence.
        """
        end=hop_end/self.c.ears.sr
        if not self.c.speaking or (self.stop_task and not self.stop_task.done()) or end-self.stop_checked_at<.32:return
        target=(getattr(self.c,'current_speaker',None) or {}).get('speaker_id')
        if target not in self.memory.users:return
        begin=max(0.,end-max(3.04,getattr(self,'identity_context_s',3.04)),float(getattr(self.c,'speaker_reply_start_s',0.)))
        if end-begin<.48:return
        vote=self.stop_fast_vote(begin,end)
        if getattr(self.c,'synthetic_diagnostics',False):
            self.spawn(self.c.ws.send_json({'type':'synthetic_stop_probe','audio_s':end,'fast_vote':vote,'target':target}))
        # This is a cheap decode-admission hint, NOT interruption permission.
        # ASR must still see a short stop when a phase-sensitive 320-ms voice
        # cue is weak. The full current-addressee verifier runs AFTER decoding.
        strong_fast=vote.get('candidate_id')==target and vote.get('candidate_similarity',-1)>=max(.20,self.memory.threshold-.45) and vote.get('candidate_margin',0)>=.10
        pending=getattr(self.c,'pending_barge_proposal',None)
        native=bool(pending and 0<=hop_end-pending[1]<=.96*self.c.ears.sr)
        if not strong_fast and not native:return
        self.stop_checked_at=end;epoch=self.epoch;rid=self.c.playback.rid
        async def probe():
            try:
                wave=self.audio_between(begin,end)
                from audio_sync import suppress_aligned_leakage
                wave=suppress_aligned_leakage(wave,self.playback_between(begin,end))
                text=await self.transcribe_stop(wave)
                controller=getattr(self.c,'decisions',None)
                if controller:controller.offer_agent(getattr(self,'last_stop_texts',[]),hop_end)
                if getattr(self.c,'synthetic_diagnostics',False):
                    await self.c.ws.send_json({'type':'synthetic_stop_decode','audio_s':end,'text':text,'views':getattr(self,'last_stop_texts',[])})
                if (self.closed or epoch!=self.epoch or not self.c.speaking or self.c.playback.rid!=rid or
                    (getattr(self.c,'current_speaker',None) or {}).get('speaker_id')!=target or not has_stop_intent(text)):return
                proposal={'kind':'BARGE','onset_s':min(2.,end-begin),'yield_prob':1.,'source':'verified_stop_command','stop_text':text,
                    'stop_views_agree':sum(is_stop_command(t) for t in getattr(self,'last_stop_texts',[]))>=2}
                self.interrupt_checked_at=-100.
                confirmed=await self.confirm_interrupt(proposal,hop_end)
                if confirmed==target:
                    await self.c._on_barge(proposal,hop_end)
                    await self.c.ws.send_json({'type':'stop_command','speaker_name':self.memory.users[target].name,
                        'text':'Paused. I’m listening.','source':'verified_stop_command'})
            except asyncio.CancelledError:raise
            except Exception:
                logging.getLogger('robo.speakers').warning('Optional stop-command probe unavailable')
        self.stop_task=self.spawn(probe())

    async def playback_voice(self):
        playback=getattr(getattr(self,'c',None),'playback',None)
        if playback is None or playback.n16<10240:return None
        rid=playback.rid;size=min(playback.n16,64000);epoch=self.epoch
        cached=self.interrupt_reference
        if cached and cached[0]==rid and size<cached[1]*1.8:return cached[2]
        reference=playback.audio[:size].copy()
        def embed():
            from speaker_teacher import get_cached_teacher
            return unit(get_cached_teacher().embed(reference))
        try:bot=await asyncio.to_thread(embed)
        except Exception:return None
        if self.closed or self.epoch!=epoch or playback.rid!=rid:return None
        self.interrupt_reference=(rid,size,bot);return bot

    def live_speech_recent(self,hop_end):
        """No late proactive interjection after the live microphone went quiet."""
        end=hop_end/self.c.ears.sr
        return any(b>end-.64 and p.get('kind')=='speech' and len(w) and
            float(np.sqrt(np.mean(w*w)))>=.003 for a,b,p,w in self.records)

    def stop_long_vote(self,begin,end):
        rows=[(p['long_embedding'],float(np.sqrt(np.mean(w*w)))) for a,b,p,w in self.records
            if b>begin and a<end and p.get('long_embedding') is not None and len(w)]
        if not rows:return {}
        floor=max(.003,self.foreground_floor(),max(r[1] for r in rows)*.35)
        rows=[r for r in rows if r[1]>=floor]
        if not rows:return {}
        vector=unit(np.average([r[0] for r in rows],axis=0,weights=[r[1]**2 for r in rows]))
        values=sorted(((self.memory.long_score(vector,u),u.id) for u in self.memory.users.values()
            if u.named and u.long is not None),reverse=True)
        if not values:return {}
        best,who=values[0];other=values[1][0] if len(values)>1 else -1.
        return {'candidate_id':who,'candidate_similarity':best,'candidate_margin':best-other}

    async def confirm_live_speaker(self,hop_end):
        """Read-only attribution for an agent interjection, never enrollment."""
        end=hop_end/self.c.ears.sr;begin=max(0.,end-3.04)
        refs=[u for u in self.memory.users.values() if u.named and u.teacher is not None]
        if not refs:return None
        wave=self.audio_between(begin,end)
        if len(wave)<10240:return None
        n=len(wave)//320;levels=np.sqrt(np.mean(wave[:n*320].reshape(n,320)**2,axis=1))
        active=np.flatnonzero(levels>=max(.003,self.foreground_floor(),float(levels.max())*.04))
        if len(active)<16:return None
        lo=max(0,int(active[0])*320-1600);hi=min(len(wave),(int(active[-1])+1)*320+1600)
        query=np.pad(wave[lo:hi],(0,max(0,10240-(hi-lo))))
        epoch=self.epoch
        from speaker_teacher import get_cached_teacher
        try:embedding=await asyncio.to_thread(lambda:unit(get_cached_teacher().embed(query)))
        except Exception:return None
        if self.closed or self.epoch!=epoch:return None
        scores=sorted(((u.teacher_score(embedding),u.id) for u in refs),reverse=True)
        score,sid=scores[0];runner=scores[1][0] if len(scores)>1 else -1.
        fast=self.stop_fast_vote(begin,end)
        # The long student is an independent trained model, not an average
        # of the same fast predictions. Do not silently discard it in this
        # new path: one lucky short-window winner can be an unknown voice.
        rows=[(p['long_embedding'],float(np.sqrt(np.mean(w*w)))) for a,b,p,w in self.records
            if b>begin and a<end and p.get('long_embedding') is not None and len(w)]
        long_vote={}
        if rows:
            floor=max(.003,self.foreground_floor(),max(r[1] for r in rows)*.35)
            rows=[r for r in rows if r[1]>=floor]
            if rows:
                vector=unit(np.average([r[0] for r in rows],axis=0,weights=[r[1]**2 for r in rows]))
                values=sorted(((self.memory.long_score(vector,u),u.id) for u in refs if u.long is not None),reverse=True)
                if values:
                    best,who=values[0];other=values[1][0] if len(values)>1 else -1.
                    long_vote={'candidate_id':who,'candidate_similarity':best,'candidate_margin':best-other}
        from interaction_policy import live_voice_verified,VoiceEvidence
        evidence=getattr(self,'live_voice_evidence',None)
        if evidence is None:evidence=self.live_voice_evidence=VoiceEvidence()
        # Keep weak early evidence instead of repeatedly drawing a fresh
        # identity lottery until a later window happens to pass. Reset at a
        # genuine acoustic pause, a changed candidate or a long inactivity.
        gap=self.audio_between(max(0.,evidence.end),end) if evidence.end>=0 else np.zeros(0,np.float32)
        quiet=False
        if len(gap)>=16000:
            levels_gap=np.sqrt(np.mean(gap[:len(gap)//320*320].reshape(-1,320)**2,axis=1))
            quiet=bool(np.any(np.convolve((levels_gap<.003).astype(int),np.ones(50,dtype=int),'valid')>=50))
        mean_score=evidence.update(sid,score,end,quiet,complete=len(active)>=64)
        recovery=evidence.recovered(self.utterance_threshold)
        verified=live_voice_verified(score,score-runner,fast,sid,self.utterance_threshold,self.memory.threshold,
            long_vote,require_long=self.predictor.use_long,evidence_mean=mean_score,
            initial_score=evidence.initial_score,require_initial=True,recovered=recovery)
        visual=getattr(self.c,'visual',None)
        face_confirmed=bool(visual and visual.corroborates(sid,begin,end))
        from interaction_policy import face_corroborated_voice
        face_route=bool(face_confirmed and len(active)>=64 and face_corroborated_voice(score,score-runner,fast,long_vote,
            sid,self.utterance_threshold,evidence.complete_mean,evidence.initial_score))
        verified=verified or face_route
        self.live_verification={'teacher':round(score,3),'required':round(max(.55,self.utterance_threshold),3),
            'fast':round(fast.get('candidate_similarity',-1),3),'long':round(long_vote.get('candidate_similarity',-1),3),
            'initial':evidence.initial_score,'recovered':bool(recovery),'accepted':bool(verified)}
        self.live_verification['face_confirmed']=face_confirmed
        if getattr(self.c,'synthetic_diagnostics',False):
            await self.c.ws.send_json({'type':'synthetic_live_speaker_verification','audio_s':end,
                'teacher_cosine':score,'rival_margin':score-runner,'candidate_id':sid,'fast_vote':fast,
                'strong_teacher_required':max(.55,self.utterance_threshold),'teacher_evidence_mean':mean_score,
                'initial_reference_cosine':evidence.initial_score,
                'long_vote':long_vote,'verified':verified,'face_route':face_route})
        if not verified:return None
        user=self.memory.users.get(sid)
        if user is None:return None
        return {'speaker_id':sid,'display_name':user.name,'named':True,'enrolled':True,
            'identity_source':'independent live teacher + fast cue','teacher_cosine':round(score,4)}

    async def transcribe_turn(self,track):
        wave=self.submission_wave(track)
        start=max(track.start,track.recognition_start or track.start)
        # Post-roll rescues word endings, but must not turn quiet trailing TV
        # into the enrollment voiceprint. Identity keeps the original span.
        identity_length=max(0,round((track.last_speech-start)*16000))
        track.verification_audio=wave[:identity_length].copy()
        track.enrollment_source=wave.copy() # includes already-received quiet consonants/word endings
        # A reused channel can become confident only after the wake phrase.
        # Decode several bounded views of that one-second pre-roll as ONE
        # native batch. Prefix interference otherwise hides a perfectly
        # audible self-introduction and silently maps the new person to the
        # old channel. Never invent a wake/name or remove ordinary content.
        preroll=(track.voiced_start-track.start) if track.voiced_start is not None else 0.
        current=self.memory.users.get(track.sid)
        known=bool(current and current.named and current.teacher is not None)
        offsets=([0.,.32,.64,.96,1.28,1.6] if known and preroll>=1.5 else
            [0.,.16,.32,.80]) if preroll>=.8 and len(wave)>=25600 else [0.]
        views=[np.pad(wave[round(offset*16000):],(0,max(0,10240-len(wave[round(offset*16000):])))) for offset in offsets]
        owned=track.owned_waveform();owned=np.pad(owned,(0,max(0,10240-len(owned))))
        def decode():
            output=[];batch=views+[owned]
            for at in range(0,len(batch),4):output.extend(self.c.ears.service.engine.asr.transcribe(batch[at:at+4]))
            return output
        async with self.asr_slot(0):
            self.decode_count+=len(views)+1
            results=await asyncio.to_thread(decode)
        texts=[str(text).strip()[:3000] for text in results[:len(views)]]
        owned_text=str(results[-1]).strip()[:3000] if results else ''
        text,chosen=choose_turn_text(texts,owned_text)
        native=hasattr(self.c.ears.service.engine.asr,'rnnt')
        if (known and native and len(views)>1 and not wake_intent(owned_text) and
            not wake_intent(text or '') and not self_name(text or '') and
            not any(wake_intent(t) for t in texts)):
            verified=await self.known_prefix_view(track,wave,views,texts,owned_text,offsets)
            if verified is not None:text,chosen=verified
        track.asr_crop_verified=bool(text and chosen is not None and chosen>0)
        if chosen is not None and chosen>0:track.submission_audio=views[chosen].copy()
        if chosen==-1:track.submission_audio=owned.copy()
        # Names are verified from the complete wake/introduction. Build the
        # voice anchor from its latest crop that STILL contains that explicit
        # name and every following word, not the preceding TV/old voice.
        name=self_name(text or '')
        if name and (wake_intent(text or '') or self.observed_wake(track,text or '')):
            match=INTRO.search(text);suffix=words(text[match.start():]) if match else []
            candidates=[]
            for i,value in enumerate(texts):
                intro=INTRO.search(value)
                tail=words(value[intro.start():]) if intro else []
                agrees=bool(tail and (tail[:len(suffix)]==suffix or suffix[:len(tail)]==tail))
                if intro and self_name(value)==name and agrees:
                    length=len(views[i])
                    if length>=20480:candidates.append((i,length))
            if candidates:
                i,length=candidates[-1];track.enrollment_audio=views[i][:length].copy()
                track.enrollment_crop_s=offsets[i]
        if getattr(self.c,'synthetic_diagnostics',False):
            await self.c.ws.send_json({'type':'synthetic_asr_views','texts':texts,'owned_text':owned_text,
                'offsets':offsets,'chosen':chosen,'word_origin_consistent':text is not None})
        return text
    async def known_prefix_view(self,track,wave,views,texts,owned_text,offsets):
        """Discard only an independently verified quiet OTHER-voice pre-roll.

        Never maximize cosine by deleting arbitrary first words. The removed
        prefix must be quiet relative to retained speech AND fail every known
        voice reference. The remaining full view must independently match the
        current speaker and agree with owned-word ASR. Search is bounded to the
        two-second protective pre-roll; the full original span remains a second
        acoustic verification view. No new model or early LLM call.
        """
        # A self-introduction may be a NEW person reusing an old person's
        # channel. Never repair/enroll it using that old person's references.
        # In particular, owned ASR can retain the wake when raw views lose it.
        if wake_intent(owned_text) or self_name(owned_text) or any(wake_intent(t) or self_name(t) for t in texts):return None
        refs=[u for u in self.memory.users.values() if u.teacher is not None]
        baseline,_=choose_turn_text(texts,owned_text)
        own=words(owned_text)
        def word_error(value):
            seq=words(value or '')
            return word_distance(seq,own)/max(1,len(seq),len(own))
        best_word_error=word_error(baseline)
        def levels(x):
            n=len(x)//320
            return np.sqrt(np.mean(x[:n*320].reshape(n,320)**2,axis=1)) if n else np.zeros(1)
        def verify():
            from speaker_teacher import get_cached_teacher
            teacher=get_cached_teacher()
            for i in reversed(range(1,len(views))):
                cut=round(offsets[i]*16000);prefix=wave[:cut]
                span=max(0,round((track.last_speech-max(track.start,track.recognition_start or track.start)-offsets[i])*16000))
                retained=views[i][:span]
                if len(prefix)<10240 or len(retained)<20480:continue
                ratio=float(np.quantile(levels(prefix),.95))/max(float(np.quantile(levels(retained),.75)),1e-7)
                if ratio>.6:continue
                prefix_vector=unit(teacher.embed(prefix))
                prefix_score=max(u.teacher_score(prefix_vector) for u in refs)
                if prefix_score>=min(.45,self.utterance_threshold-.05):continue
                vector=unit(teacher.embed(retained))
                scores=sorted(((u.teacher_score(vector),u.id) for u in refs),reverse=True)
                score,sid=scores[0];runner=scores[1][0] if len(scores)>1 else -1.
                if sid!=track.sid or score<self.utterance_threshold or score-runner<self.utterance_margin:continue
                text,which=choose_turn_text([texts[i]],owned_text)
                if not text:continue
                # Acoustic prefix repair may never replace a better complete
                # transcript with one containing extra unowned room words.
                if word_error(text)>best_word_error+1e-8:continue
                whole=track.verification_audio
                original=unit(teacher.embed(whole)) if whole is not None and len(whole)>=10240 else vector
                agreement=float(vector@original)
                if agreement<.35:continue
                merged=unit(.8*vector+.2*original)
                return text,i,merged,agreement,{'offset_s':offsets[i],'prefix_cosine':round(prefix_score,4),
                    'prefix_to_speech_energy':round(ratio,4),'retained_cosine':round(score,4),
                    'original_view_cosine':round(agreement,4),'owned_word_error':round(word_error(text),4)}
            return None
        try:result=await asyncio.to_thread(verify)
        except Exception:return None
        if result is None:return None
        text,i,embedding,agreement,proof=result
        track.precomputed_identity=(text,(embedding,agreement,2));track.known_prefix_proof=proof
        return text,i
    @staticmethod
    def foreground_crop(wave):
        """Rejected offline ablation; NOT invoked by the live path.

        Only strong level contrast qualifies. Keep 320-ms onset and 160-ms
        tail protection and bridge one-second internal gaps. Equally loud
        mixtures keep the full-span verification/abstention path.
        """
        if len(wave)<10240:return wave,0.
        levels=np.array([np.sqrt(np.mean(wave[a:a+2560]**2)) for a in range(0,len(wave)-2559,640)])
        peak=float(levels.max());floor=float(np.quantile(levels,.25))
        if peak<max(.008,floor*3.):return wave,0.
        active=np.flatnonzero(levels>=max(.004,peak*.35,floor*2.5))
        if not len(active):return wave,0.
        # Never cut leading words or choose only the last loud sentence:
        # pauses/soft openings can be genuine foreground speech.
        lo=0
        hi=min(len(wave),round((float(active[-1])*.04+.16+.16)*16000))
        if hi-lo<10240:return wave,0.
        return wave[lo:hi].copy(),lo/16000
    async def candidate(self,track,wave,coverage=0.,continuous=False):
        try:
            cut=0.
            if continuous and len(wave)>=25600:
                offsets=[0.,.32,.64,.96]
                views=[np.pad(wave[round(offset*16000):],(0,max(0,10240-len(wave[round(offset*16000):])))) for offset in offsets]
                async with self.asr_slot(20):
                    self.decode_count+=len(views)
                    texts=await asyncio.to_thread(self.c.ears.service.engine.asr.transcribe,views)
                # Trim only verified pre-wake context; take the latest view
                # that STILL contains the greeting, never fabricate one.
                choices=[i for i,text in enumerate(texts) if wake_intent(text) and self_name(text)]
                if not choices:choices=[i for i,text in enumerate(texts) if wake_intent(text)]
                chosen=choices[-1] if choices else 0;text=(texts[chosen] if texts else '').strip()[:3000];cut=offsets[chosen]
            else:text=await self.transcribe(wave)
            if self.closed or not self.c.identity_enabled or self.c.speaking:return
            was_background=track.channel in self.background_channels
            recovered_start=None
            if getattr(self,'open_enrollment',False) and continuous and coverage>=.65 and not was_background:
                from open_speakers import claimed_name
                if not claimed_name(text):
                    recovered=await self.recover_voice_claim(track)
                    if recovered is not None:text,recovered_start=recovered
            if coverage>=.4 and self_name(text) and not wake_intent(text) and (
                (continuous and not was_background) or self.observed_wake(track,text)):
                recovered=await self.recover_intro_prefix(track,text)
                if recovered is not None:text,recovered_start=recovered
            background_wake_only=track.channel in self.background_channels and wake_intent(text) and not self_name(text)
            eligible=coverage>=.4
            if was_background and wake_intent(text) and self_name(text) and eligible and self.records:
                raw=self.audio_between(max(track.start,track.last_speech-8.),min(self.records[-1][1],track.last_speech+.8))
                room_text=await self.transcribe(raw)
                eligible=bool(self_name(room_text)==self_name(text) and wake_intent(room_text))
            # A long-observed TV channel can briefly own a new person's "Hey"
            # during a channel transition. Wait for an addressed complete
            # introduction instead of enrolling the TV as a permanent Guest.
            observed_wake=not was_background and continuous and coverage>=.4 and self.observed_wake(track,text)
            decision=self.turns.candidate_text(track.key,'' if background_wake_only or not eligible else text,observed_wake=observed_wake and eligible)
            # Diarizer-owned voiced hops can be sparse even when the bounded
            # utterance is several seconds long. Eligibility is not identity:
            # let ECAPA inspect that existing context after .64 s of owned
            # evidence; its 1.28-s minimum window and strict cosine/margin
            # still apply. Otherwise a correctly transcribed returning user
            # never even reaches verification.
            if not decision and not wake_intent(text) and track.speech_s>=.64-1e-6:
                context=wave if continuous and coverage>=.4 else None
                if context is None and self.records and foreground_context_allowed(self.records,track.levels):
                    # Independent known-person verification may inspect the
                    # actual loud utterance even when sparse diarizer masks
                    # ruined its ASR. This NEVER enables new-user enrollment.
                    context=self.audio_between(track.start,min(self.records[-1][1],track.last_speech+.8))
                decision=await self.recover_existing_channel(track,was_background,context=context)
            if not decision and getattr(self,'open_enrollment',False):
                from open_speakers import eligible
                if eligible(track,text,coverage,was_background,self.c.speaking,self.overlaps,self.memory):
                    decision=self.turns.candidate_text(track.key,text,allow_unprompted=True)
            if not decision and track.channel is not None and track.channel not in self.turns.channels:
                # Lack of a wake/too little reference context is NOT evidence
                # of background speech. Marking a loud new foreground this
                # early poisoned its energy baseline, then required it to be
                # 2.5 times louder than itself before known-voice recovery.
                if coverage<.35 and not foreground_context_allowed(self.records,track.levels):
                    self.background_channels.add(track.channel)
            if decision and (continuous or recovered_start is not None):
                if recovered_start is not None:
                    track.start=min(track.start,recovered_start);track.recognition_start=recovered_start
                else:track.recognition_start=track.start+cut
            controller=getattr(self.c,'decisions',None)
            if controller and decision and not self.c.speaking:
                controller.offer(text,round(track.last_speech*self.c.ears.sr))
            await self.c.ws.send_json({'type':'asr_preview','text':text,
                    'status':'enrolled' if decision else 'awaiting_wake_or_stable_voice',
                    'source':'speaker_candidate','track':track.key,'audio_seconds':round(len(wave)/16000,2),
                    'voiced_seconds':round(track.speech_s,2),'channel':track.channel,
                    'energy_coverage':round(coverage,3),'continuous_audio':continuous,
                    'identity_probe':getattr(track,'recovery_probe',None)})
            if decision:await self.notice(decision=decision)
            elif self.turns.now-track.last_speech>=.64 and hasattr(self.c,'maybe_visual_greeting'):
                await self.c.maybe_visual_greeting(text,track.voiced_start if track.voiced_start is not None else track.start,track.last_speech)
            elif not decision and hasattr(self.c,'maybe_visual_greeting'):
                epoch=self.epoch
                async def greet_after_pause():
                    await asyncio.sleep(.8)
                    if self.closed or epoch!=self.epoch or track.sid or self.c.speaking or self.turns.now-track.last_speech<.64:return
                    await self.c.maybe_visual_greeting(text,track.voiced_start if track.voiced_start is not None else track.start,track.last_speech)
                self.spawn(greet_after_pause())
        except asyncio.CancelledError:raise
        except Exception:
            track.decoding=False
            await self.notice(warning='Speaker-owned ASR was unavailable; enrollment not guessed.')
        finally:
            # Especially a stale job that completes after TTS starts must not
            # leave this channel permanently stuck as "decoding".
            track.decoding=False

    async def recover_voice_claim(self,track):
        """A channel transition must not detach an explicit identity claim.

        Native text only requests this bounded probe. Re-decode real captured
        audio, require the same literal name, and compare two non-overlapping
        acoustic windows before admitting any previously unowned prefix.
        """
        if not self.records or track.speech_s<1.28:return None
        observations=[(at,name) for at,name in getattr(self,'claim_observations',())
            if track.start-3.2<=at<=track.last_speech+.8 and track.last_speech-at<6.]
        if not observations or len({name for _,name in observations})!=1:return None
        start=max(self.records[0][0],min(track.start,observations[0][0]-3.2))
        end=min(self.records[-1][1],track.last_speech+.16)
        if end-start>8. or any(a<end and b>start for a,b in self.overlaps):return None
        if any(a<track.start and b>start and sid in self.memory.users and self.memory.users[sid].named
            and sid!=track.sid for a,b,sid,*_ in self.owners):return None
        wave=self.audio_between(start,end)
        if len(wave)<40960:return None
        text=await self.transcribe(wave)
        from open_speakers import claimed_name
        if (claimed_name(text) or '').casefold()!=observations[0][1]:return None
        levels=np.sqrt(np.mean(wave[:len(wave)//320*320].reshape(-1,320)**2,axis=1))
        voiced=np.flatnonzero(levels>=max(.003,float(levels.max())*.15))
        if not len(voiced):return None
        lo=max(0,int(voiced[0])*320-1600);hi=min(len(wave),(int(voiced[-1])+1)*320+1600)
        clean=wave[lo:hi]
        if len(clean)<40960:return None
        from speaker_teacher import get_cached_teacher
        def compare():
            teacher=get_cached_teacher();return float(unit(teacher.embed(clean[:20480]))@unit(teacher.embed(clean[-20480:])))
        agreement=await asyncio.to_thread(compare)
        if agreement<.55 or self.closed or self.c.speaking:return None
        proven_start=start+lo/16000;old=track.voiced_start if track.voiced_start is not None else track.start
        if proven_start<old:
            track.chunks.appendleft((proven_start,old,self.audio_between(proven_start,old)))
            track.verified_claim_prefix=(proven_start,old)
            track.voiced_start=proven_start;track.start=min(track.start,proven_start)
        track.claim_prefix_cosine=agreement
        return text,proven_start

    async def recover_intro_prefix(self,track,text):
        """Verify a real greeting immediately before a new channel's name.

        A channel transition can split "Hey Robbie" from "I am ...". Do not
        invent a wake or transfer an unrelated room-wide partial: re-decode
        the bounded original microphone prefix and require the SAME explicit
        self-introduction. Only energy-eligible non-background candidates call
        this, and final owned-word/acoustic checks remain mandatory.
        """
        if not self.records:return None
        start=max(self.records[0][0],track.start-1.6)
        end=min(self.records[-1][1],track.last_speech+.8)
        if any(a<end and b>start for a,b in self.overlaps):return None
        wave=self.audio_between(start,end)
        if len(wave)<10240:return None
        offsets=[0.,.32,.64,.96]
        views=[np.pad(wave[round(offset*16000):],(0,max(0,10240-len(wave[round(offset*16000):])))) for offset in offsets]
        async with self.asr_slot(15):
            self.decode_count+=len(views)
            texts=await asyncio.to_thread(self.c.ears.service.engine.asr.transcribe,views)
        name=self_name(text)
        observed=self.observed_intro_names(track,name) if name else set()
        expected=next(iter(observed)) if observed else (name or '').casefold()
        choices=[i for i,value in enumerate(texts) if wake_intent(value) and (self_name(value) or '').casefold()==expected]
        if not choices:return None
        chosen=choices[-1]
        return texts[chosen].strip()[:3000],start+offsets[chosen]

    async def recover_existing_channel(self,track,was_background=False,*,context=None):
        """A returning named voice may acquire a new diarizer channel.

        Reassociate a known, acoustically verified identity only; never enroll
        an unrelated speaker from ordinary speech without a wake phrase.
        """
        if self.c.speaking or track.channel is None:return None
        references=[u for u in self.memory.users.values() if u.named and u.teacher is not None]
        if not references:return None
        query=track.owned_waveform()[-round(getattr(self,'identity_context_s',3.04)*16000):]
        if context is not None:
            # The energy-dominant candidate can have sparse diarizer holes.
            # Zero-masking those holes is not a longer acoustic observation.
            # Use its captured continuous view, excluding most protective
            # pre-roll, only for known-reference verification. No enrollment
            # or final word-origin/full-span guard is bypassed.
            offset=max(0.,(track.voiced_start or track.start)-track.start-.32)
            query=np.asarray(context)[round(offset*16000):][-round(getattr(self,'identity_context_s',3.04)*16000):]
        if len(query)<20480:return None
        levels=np.sqrt(np.mean(query[:len(query)//320*320].reshape(-1,320)**2,axis=1))
        if was_background and not foreground_context_allowed(self.records,track.levels):return None
        if was_background and context is None:
            history=self.channel_levels.get(track.channel,deque())
            floor=float(np.median(history)) if history else 0.
            if float(np.quantile(levels,.75))<max(.008,floor*2.5):return None
        from speaker_teacher import get_cached_teacher
        epoch=self.epoch
        def embed():return unit(get_cached_teacher().embed(query))
        try:embedding=await asyncio.to_thread(embed)
        except Exception:return None
        if self.closed or epoch!=self.epoch or track.key not in self.turns.tracks or self.c.speaking:return None
        scores=sorted(((u.teacher_score(embedding),u.id) for u in references),reverse=True)
        best,sid=scores[0];runner=scores[1][0] if len(scores)>1 else -1.
        track.recovery_probe={'best_cosine':round(best,4),'nearest_margin':round(best-runner,4),
            'threshold':self.utterance_threshold,'context_seconds':round(len(query)/16000,3)}
        vote=self.memory.match(track.vector())
        consensus=known_voice_consensus(sid,best,best-runner,vote,teacher_threshold=self.utterance_threshold,
            teacher_margin=self.utterance_margin,fast_threshold=self.memory.threshold)
        track.recovery_probe.update(fast_vote=vote,corroborated=consensus)
        # A diarizer may assign a returning person a new channel. Honor the
        # selected cosine there, but a sub-0.50 reassignment needs independent
        # same-person fast evidence. Otherwise weak TV audio can steal the
        # foreground channel and contaminate the next real question.
        relaxed_corroborated=(vote.get('id')==sid and
            vote.get('similarity',-1)>=max(.55,self.memory.threshold) and vote.get('margin',0)>=.10)
        if best<.50 and not relaxed_corroborated:return None
        if not consensus and (best<self.utterance_threshold or best-runner<max(.12,self.utterance_margin)):return None
        if self.records and track.voiced_start is not None and track.last_speech-track.voiced_start<=4.:
            # Once the voice is independently known, give its ASR the same
            # two-second protection as an already-bound speaker. Never borrow
            # another enrolled person's preceding words.
            pre=max(self.records[0][0],track.voiced_start-2.)
            for a,b,who,*_ in self.owners:
                if who and who!=sid and b>pre and a<track.start:pre=max(pre,b)
            if pre<track.start:
                track.chunks.appendleft((pre,track.start,self.audio_between(pre,track.start)))
                track.start=pre
        self.rebind(track,sid)
        # A channel that was already observed as TV/background cannot retain
        # this person's identity after a verified move to a new channel.
        for channel,who in list(self.turns.channels.items()):
            if who==sid and channel!=track.channel and channel in self.background_channels:self.turns.channels.pop(channel,None)
        self.background_channels.discard(track.channel)
        return {'kind':'enrolled','id':sid,'name':self.memory.users[sid].name,'new_user':False,
            'accepted':True,'source':'Full known-voice match on a new diarization channel','cosine':best}
    def recovery_start(self,track):
        """Rejected offline ablation; NOT invoked by the live path.

        Locate the latest louder episode in at most eight seconds of already
        captured audio. Smooth over 320 ms, bridge sub-second hesitations, and
        keep 320 ms of phoneme pre-roll. This is not an identity decision or
        source separation; equally loud mixtures retain the full-span guard.
        """
        start=max(track.start,track.last_speech-8.)
        if track.last_speech-track.start<=5.:return start
        wave=self.audio_between(start,track.last_speech)
        if len(wave)<5120:return start
        levels=np.array([np.sqrt(np.mean(wave[a:a+5120]**2)) for a in range(0,len(wave)-5119,1280)])
        active=np.flatnonzero(levels>=max(.004,float(levels.max())*.35))
        if not len(active):return start
        first=active[0]
        return max(start,start+float(first)*.08-.32)
    async def observe(self,event,hop_end):
        c=self.c;at=hop_end/c.ears.sr
        prediction=await asyncio.to_thread(self.predictor.predict,event)
        prediction['playback_audio16']=event.get('playback_audio16')
        floor=self.foreground_floor()
        prediction['foreground_energy']=bool(float(np.sqrt(np.mean(event['audio16']**2)))>=floor) if floor else None
        begin=at-len(event['audio16'])/16000
        self.records.append((begin,at,prediction,event['audio16'].copy()))
        self.native_continuity(prediction,event['audio16'],begin,at)
        # Short CPU-teacher activity veto was tested but is not deployed: it
        # split soft speech across ordinary sub-second hesitations.
        diarized=False
        if self.diar is not None:
            if self.diar_offset is None:self.diar_offset=begin
            try:
                response=await self.diar.feed(event['audio16']);self.diarization_active=True;diarized=True
                values=np.asarray(response['probabilities'],np.float32).reshape(-1,8)
                if len(values):
                    labels=[]
                    for row in values:
                        active=[int(ch) for ch in np.flatnonzero(row>=.55)]
                        new=[ch for ch in active if ch not in self.seen_channels]
                        self.seen_channels.update(active)
                        if len(new)==1:self.foreground_channel=new[0]
                        known=[ch for ch in active if self.turns.channels.get(ch) in self.memory.users]
                        known_ids={self.turns.channels[ch] for ch in known}
                        if len(known_ids)==1:label=max(known,key=lambda ch:row[ch])
                        elif len(known_ids)>1:label=-1 # two enrolled people: don't guess mixed words
                        elif self.foreground_channel in active:label=self.foreground_channel
                        elif len(active)==1:label=active[0]
                        else:label=-1
                        labels.append(label)
                    labels=np.asarray(labels)
                    boundaries=np.r_[0,np.flatnonzero(np.diff(labels)!=0)+1,len(labels)]
                    for a,b in zip(boundaries[:-1],boundaries[1:]):
                        start=self.diar_offset+(response['start_frame']+int(a))*.01
                        end=self.diar_offset+(response['start_frame']+int(b))*.01
                        source=next((p for s,t,p,x in self.records if s<end<=t+.0001),prediction)
                        cue=dict(source);cue['encoder_activity_kind']=source.get('kind')
                        label=int(labels[a]);wave=self.audio_between(start,end)
                        bridged=self.bridge_gap(label,cue,wave,start,end,values[a:b].mean(0))
                        if bridged is not None:
                            label=bridged;cue['provisional_bridge']=True;self.bridged_seconds+=end-start
                        if label>=0:
                            cue.update(kind='speech',confidence=max(.65,cue.get('confidence',0.)),diarization_channel=label)
                            self.consider_new_address(label,cue,end)
                            sid=self.turns.channels.get(label)
                            if sid in self.memory.users and self.activity_scores:
                                cue['speaker_activity_confirmed']=self.activity_scores.get(sid,False)
                        else:
                            cue.update(kind='silence',confidence=0.,embedding=None)
                            # Ambiguous overlaps do not put mixed speech into
                            # ASR, but a genuinely active enrolled person has
                            # not yet stopped talking just because they overlap.
                            active=np.flatnonzero(values[a:b].mean(0)>=.55)
                            if len(active)>1:
                                known_ids=[self.turns.channels.get(int(ch)) for ch in active
                                    if self.turns.channels.get(int(ch)) in self.memory.users]
                                if len(set(known_ids))>1:self.overlaps.append((start,end))
                                for ch in active:
                                    sid=self.turns.channels.get(int(ch));track=self.turns.tracks.get(sid)
                                    vector=cue.get('raw_embedding',cue.get('embedding'))
                                    owner=self.memory.match(vector).get('id') if vector is not None and source.get('kind')=='speech' else None
                                    if track is not None and owner==sid:track.last_speech=end
                        await self.step(cue,wave,end)
            except asyncio.CancelledError:raise
            except Exception:
                await self.diar.close();self.diar=None;self.diarization_active=False
                await c.ws.send_json({'type':'warning','text':'Diarization unavailable: using the less reliable V4 embedding-only fallback. Speaker strictness is adjustable.'})
        if not diarized:await self.step(prediction,event['audio16'],at)
        if getattr(c,'synthetic_diagnostics',False):
            vector=prediction.get('raw_embedding',prediction.get('embedding'))
            await c.ws.send_json({'type':'synthetic_activity','audio_s':at,'kind':prediction['kind'],
                'rms':round(float(np.sqrt(np.mean(event['audio16']**2))),5),
                'scores':{u.name:round(self.memory.fast_score(vector,u),4) for u in self.memory.users.values()} if vector is not None else {},
                'channels':{str(k):self.memory.users[v].name for k,v in self.turns.channels.items() if v in self.memory.users},
                'tracks':[{'name':self.memory.users[t.sid].name if t.sid in self.memory.users else t.key,
                    'last_speech':t.last_speech,'last_direct':t.last_direct_speech,'speech_s':t.speech_s,
                    'peak_rms':max(t.levels,default=0.)} for t in self.turns.tracks.values()]})
        if time.monotonic()-self.last_notice>.65:
            self.last_notice=time.monotonic();await self.notice(prediction=self.last_activity or {'kind':prediction['kind']})

    def consider_new_address(self,channel,cue,at):
        """Quarantine reused-channel ownership, never enroll from room ASR.

        A fresh explicit DIFFERENT name plus an incompatible current fast voice
        may start a new candidate. Its own ASR and full voice reference remain
        mandatory. Do not do this during bot playback or for provisional users.
        """
        if self.c.speaking:return
        sid=self.turns.channels.get(channel);user=self.memory.users.get(sid)
        vector=cue.get('raw_embedding',cue.get('embedding'))
        if user is None or not user.named or user.teacher is None or vector is None:return
        name=next((name for when,name in reversed(self.asr_wakes) if 0<=at-when<=2.5),None)
        if name and name!=user.name.casefold() and self.memory.fast_score(vector,user)<.30:
            self.turns.channels.pop(channel,None)

    def native_continuity(self,prediction,wave,begin,end):
        """Use an already-arrived native timestamp before delayed diarizer rows.

        Positive continuity only: no new identity, no BARGE, no API/model call.
        Background must not prolong a voice's clock. Require a separated known
        winner, foreground level and <=1.6 s since direct diarizer evidence.
        """
        if prediction.get('foreground_energy') is False:return
        vector=prediction.get('raw_embedding',prediction.get('embedding'))
        if vector is None:return
        refs=[u for u in self.memory.users.values() if u.named and u.teacher is not None]
        scores=sorted(((self.memory.fast_score(vector,u),u.id) for u in refs),reverse=True)
        if not scores:return
        best,sid=scores[0];runner=scores[1][0] if len(scores)>1 else -1.
        track=self.turns.tracks.get(sid)
        if (best<max(.35,self.memory.threshold-.20) or best-runner<.08 or track is None
            or track.closed or end-track.last_direct_speech>1.6 or end<=track.last_speech):return
        level=float(np.sqrt(np.mean(wave*wave))) if len(wave) else 0.
        reference=max(track.levels,default=.032)
        if level<max(.004,reference*.3):return
        track.chunks.append((begin,end,wave.copy()))
        track.last_speech=end;track.version+=1;track.native_continuity_used=True

    def foreground_floor(self):
        """2x measured background amplitude, only from observed room channels.

        No extra wait/model and no absolute microphone loudness assumption.
        Unknown new-user enrollment remains independent; this gates known
        activity/interruptions so a matching quiet TV cannot hold the clock.
        """
        levels=[float(np.median(self.channel_levels[ch])) for ch in self.background_channels
            if len(self.channel_levels.get(ch,()))>=4]
        return 2.*float(np.median(levels)) if levels else 0.

    @staticmethod
    def install_reference(user,embedding,track):
        user.teacher=unit(embedding);user.teacher_templates=getattr(track,'enrollment_templates',())
        fast=getattr(track,'enrollment_fast',None)
        if fast is not None:
            user.fast=unit(fast)
            long=getattr(track,'enrollment_long',None)
            user.long=unit(long) if long is not None else None

    async def check_activity(self,at):
        """Rejected offline ablation; NOT invoked by the live path.

        Uses the same frozen CPU teacher, no extra GPU encoder. This is only
        an activity veto; final full-span identity thresholds stay stricter.
        Long references improve the comparison without delaying microphone
        capture to collect a new three-second utterance.
        """
        if at-getattr(self,'activity_checked_at',-100.)<.64:return
        refs=[u for u in self.memory.users.values() if u.teacher is not None]
        if not refs:return
        self.activity_checked_at=at;epoch=self.epoch
        wave=self.audio_between(max(0.,at-.96),at)
        if len(wave)<10240:return
        from speaker_teacher import get_cached_teacher
        try:embedding=await asyncio.to_thread(lambda:unit(get_cached_teacher().embed(wave)))
        except Exception:return
        if self.closed or epoch!=self.epoch:return
        scores={u.id:u.teacher_score(embedding) for u in refs}
        self.activity_scores={sid:score>=.35 and score-max((v for k,v in scores.items() if k!=sid),default=-1.)>=.04
            for sid,score in scores.items()}

    def bridge_gap(self,label,cue,wave,start,end,probabilities):
        """Repair a short foreground-channel dropout, never permanent binding.

        Only established quiet background channels (or silence with a strong
        speech cue) are eligible. Require louder energy, same recent fast
        speaker vector and <=1.6 s since DIRECT diarizer evidence. This avoids
        both splitting a sentence and letting room noise hold its clock open.
        Full-utterance identity and word-origin checks still decide submission.
        """
        rms=float(np.sqrt(np.mean(wave*wave))) if len(wave) else 0.
        previous=self.channel_levels.get(label,deque())
        floor=float(np.median(previous)) if previous else 0.
        new_channel=False
        if label>=0:
            self.channel_levels.setdefault(label,deque(maxlen=128)).append(rms)
            if label in self.turns.channels:return None
            new_channel=label not in self.background_channels
        elif cue.get('kind')!='speech' or cue.get('confidence',0)<.8:return None
        vector=cue.get('raw_embedding',cue.get('embedding'))
        if vector is None:return None
        active=[t for t in self.turns.tracks.values() if t.sid in self.memory.users and t.channel is not None
            and not t.closed and t.fast and end-t.last_direct_speech<=1.6 and start>=t.start]
        if new_channel:
            scores=sorted(((self.memory.fast_score(vector,u),u.id) for u in self.memory.users.values()),reverse=True)
            if not scores:return None
            best,sid=scores[0];runner=scores[1][0] if len(scores)>1 else -1.
            if best<max(.4,self.memory.threshold-.12) or best-runner<.12:return None
            active=[t for t in active if t.sid==sid]
        if len(active)!=1:return None
        track=active[0];reference=max(track.levels,default=.01)
        similarity=self.memory.similarity(vector,track.vector())
        energy_floor=0. if new_channel else floor*2.5
        similarity_floor=max(.4,self.memory.threshold-.12) if new_channel else .55
        if rms<max(.004,energy_floor,reference*.3) or similarity<similarity_floor:return None
        if not new_channel and probabilities[track.channel]<.15 and similarity<.75:return None
        return track.channel

    def audio_between(self,start,end):
        output=np.zeros(max(0,round((end-start)*16000)),np.float32)
        for a,b,_,wave in self.records:
            lo=max(start,a);hi=min(end,b)
            if hi<=lo:continue
            at=round((lo-start)*16000);source=round((lo-a)*16000);n=min(round((hi-lo)*16000),len(output)-at,len(wave)-source)
            if n>0:output[at:at+n]=wave[source:source+n]
        return output

    def playback_between(self,start,end):
        """Existing second-ear reference, aligned by capture/DAC positions."""
        output=np.zeros(max(0,round((end-start)*16000)),np.float32)
        for a,b,p,_ in self.records:
            wave=p.get('playback_audio16')
            if wave is None:continue
            lo=max(start,a);hi=min(end,b)
            if hi<=lo:continue
            at=round((lo-start)*16000);source=round((lo-a)*16000)
            n=min(round((hi-lo)*16000),len(output)-at,len(wave)-source)
            if n>0:output[at:at+n]=wave[source:source+n]
        return output

    def submission_wave(self,track):
        """Keep phonemes through short diarizer dropouts, not source separation.

        Channel activity owns the *turn clock*, but zeroing every uncertain
        10-ms frame corrupts wake words/names and biases ECAPA. Reconstruct the
        bounded continuous sentence, excluding another enrolled user's frames
        and ambiguous enrolled overlap. Freeze it before the RAM ring rotates.
        """
        if track.submission_audio is not None:return track.submission_audio
        start=max(track.start,track.recognition_start or track.start)
        # The diarizer can stop owning a quiet final consonant before the
        # sentence ends. Use already-received quiet-clock audio as bounded
        # post-roll; do not delay the speaker-relative one-second endpoint.
        # Other known users/ambiguous overlaps are still masked below, and
        # independent owned/raw recognition still guards background words.
        available=self.records[-1][1] if self.records else track.last_speech
        end=min(available,track.last_speech+.80)
        wave=self.audio_between(start,end)
        provisional=self.memory.users.get(track.sid)
        introduction=bool(provisional and provisional.teacher is None and track.wake and self_name(track.partial))
        excluded=[] if introduction else [(owner[0],owner[1]) for owner in self.owners if owner[2] and owner[2]!=track.sid]
        proof=getattr(track,'verified_claim_prefix',None)
        if proof:excluded=[(a,b) for a,b in excluded if not (proof[0]<=a and b<=proof[1])]
        excluded.extend(self.overlaps)
        for a,b in excluded:
            lo=max(0,round((a-start)*16000));hi=min(len(wave),round((b-start)*16000))
            if hi>lo:wave[lo:hi]=0
        return wave if len(wave) else track.waveform()

    def candidate_wave(self,track):
        selected=track.owned_waveform().copy();continuous=self.submission_wave(track).copy()
        energy=float(np.sum(continuous.astype(np.float64)**2))
        coverage=min(1.,float(np.sum(selected.astype(np.float64)**2))/max(energy,1e-10))
        # Only the candidate that owns most of the acoustic energy may repair
        # its internal channel gaps. A quiet TV/background channel cannot
        # borrow a louder person's wake phrase from the same raw microphone.
        coherent=bool(len(track.fast)>=2 and np.mean([self.memory.similarity(v,track.vector()) for v in track.fast])>=.8)
        competitor=max((float(np.sum(t.owned_waveform().astype(np.float64)**2)) for t in self.turns.tracks.values()
            if t is not track and not t.closed),default=0.)
        dominant=float(np.sum(selected.astype(np.float64)**2))>=1.25*competitor
        use_continuous=energy>1e-8 and (coverage>=.65 or
            (coverage>=.4 and coherent and dominant and track.channel not in self.background_channels))
        if use_continuous:
            # A coherent energy-winning new foreground can span an old/new
            # diarizer channel transition. Recover its full greeting; known
            # name/acoustic checks still decide whether enrollment is valid.
            if self.records:
                end=min(self.records[-1][1],track.last_speech+.8)
                continuous=self.audio_between(max(track.start,track.last_speech-8.),end)
                # Eligibility must describe the audio actually sent to ASR,
                # not an easier masked view that omitted a competing voice.
                energy=float(np.sum(continuous.astype(np.float64)**2))
                coverage=min(1.,float(np.sum(selected.astype(np.float64)**2))/max(energy,1e-10))
                if coverage<.4:use_continuous=False
        return (continuous if use_continuous else selected),coverage,use_continuous

    def rebind(self,track,sid):
        old=track.sid
        continuation=any(not t.closed and t.channel==track.channel and t.sid==old
            and (t.voiced_start if t.voiced_start is not None else t.start)<=track.last_speech+.32
            for t in self.turns.tracks.values())
        may_bind=not track.closed or continuation
        spans=self.turns.rebind_channel(track.channel,sid) if track.channel is not None and may_bind else []
        spans.append((old,track.voiced_start if track.voiced_start is not None else track.start,track.last_speech))
        updated=deque(maxlen=self.owners.maxlen)
        for owner in self.owners:
            a,b,who=owner[:3];channel=owner[3] if len(owner)>3 else track.channel
            if channel==track.channel and any(who==previous and a<end and b>start for previous,start,end in spans):who=sid
            updated.append((a,b,who,channel))
        self.owners=updated;track.sid=sid

    async def prepare_turn(self,track,entry):
        """Read-only speculation during a pause, never an early LLM request.

        Snapshot arrays/deques are frozen; a resumed syllable increments the
        live track version and makes the result ineligible for submission.
        """
        try:
            if track.version!=entry['version']:return
            snapshot=copy.copy(track)
            snapshot.chunks=deque(track.chunks);snapshot.fast=deque(track.fast);snapshot.long=deque(track.long)
            snapshot.submission_audio=self.submission_wave(snapshot).copy()
            start=time.perf_counter();text=await self.transcribe_turn(snapshot)
            entry['asr_ms']=(time.perf_counter()-start)*1000
            if not text:return
            start=time.perf_counter();embedding=await self.identity_embedding(snapshot,text)
            entry['embedding_ms']=(time.perf_counter()-start)*1000
            entry.update(snapshot=snapshot,text=text,embedding=embedding)
        except asyncio.CancelledError:raise
        except Exception:
            # Speculation failure is not an enrollment or a failed user turn:
            # the ordinary final path still runs after the one-second pause.
            entry['failed']=True

    def schedule_preparation(self):
        for track in self.turns.tracks.values():
            user=self.memory.users.get(track.sid)
            quiet=self.turns.now-track.last_speech
            if (self.c.speaking or not user or not user.named or user.teacher is None or
                track.speech_s<.64 or not self.prepare_quiet_s<=quiet<self.turns.quiet_s):continue
            key=track.token;old=self.prepared.get(key)
            if old and (old['version']==track.version or not old['task'].done()):continue
            entry={'version':track.version,'epoch':self.epoch,'started':time.perf_counter()}
            self.prepared[key]=entry;entry['task']=self.spawn(self.prepare_turn(track,entry))
        live={t.token for t in self.turns.tracks.values()}
        for key,entry in list(self.prepared.items()):
            if key not in live and not entry.get('queued') and entry['task'].done():self.prepared.pop(key,None)

    async def use_preparation(self,track,epoch):
        entry=self.prepared.pop(track.token,None)
        if not entry or entry['version']!=track.version or entry['epoch']!=epoch:return None
        await entry['task']
        if self.closed or self.epoch!=epoch or not entry.get('text'):return None
        snapshot=entry['snapshot']
        track.submission_audio=snapshot.submission_audio;track.verification_audio=snapshot.verification_audio
        track.asr_crop_verified=getattr(snapshot,'asr_crop_verified',False)
        track.precomputed_identity=(entry['text'],entry['embedding'])
        if hasattr(snapshot,'known_prefix_proof'):track.known_prefix_proof=snapshot.known_prefix_proof
        if hasattr(snapshot,'enrollment_templates'):track.enrollment_templates=snapshot.enrollment_templates
        if hasattr(snapshot,'enrollment_fast'):track.enrollment_fast=snapshot.enrollment_fast;track.enrollment_long=snapshot.enrollment_long
        track.preparation_metrics={'asr_ms':round(entry['asr_ms'],2),'embedding_ms':round(entry['embedding_ms'],2)}
        return entry['text']

    async def step(self,prediction,wave,at):
        c=self.c
        old_keys=set(self.turns.tracks)
        match=self.turns.update(prediction,wave,at,speaking=c.speaking)
        for key in set(self.turns.tracks)-old_keys:
            track=self.turns.tracks[key];begin=at-len(wave)/16000
            preroll=2. if track.sid in self.memory.users and self.memory.users[track.sid].teacher is not None else 1.
            pre=max(self.records[0][0] if self.records else begin,min(track.start,begin-preroll))
            # Preserve a clipped initial "Hey" while avoiding another known
            # person's preceding words. The pre-roll is not identity evidence.
            # The streaming diarizer can recognize a new foreground channel
            # late. Preserve its wake phrase, but never borrow a previous
            # enrolled person's words (including already closed turns).
            for owner in self.owners:
                a,b,sid=owner[:3]
                if sid and sid!=track.sid and b>pre and a<begin:pre=max(pre,b)
            conflict=any(t.sid and t.key!=key and t.last_speech>pre for t in self.turns.tracks.values())
            if pre<begin and not conflict:
                track.chunks.appendleft((pre,begin,self.audio_between(pre,begin)));track.start=pre
        self.owners.append((at-len(wave)/16000,at,match.get('id') if match else None,prediction.get('diarization_channel')))
        self.last_activity=match
        visual=getattr(c,'visual',None)
        if visual and visual.camera=='on' and prediction.get('kind')=='speech':
            view=visual.select(max(0,at-.8),at)
            if view.get('track_id'):await c.ws.send_json({'type':'visual_active','track_id':view['track_id'],'confidence':view.get('confidence')})
        # Original TYGER/second-ear proposals own interruptions. Identity only
        # gates/replays a recent native proposal in server.py; it must never
        # create its own barge from a diarizer channel (including bot echo).
        for track in ([] if c.speaking else self.turns.decode_candidates()):
            wave,coverage,continuous=self.candidate_wave(track)
            self.spawn(self.candidate(track,wave,coverage,continuous))
        self.schedule_preparation()
        for track in self.turns.due():
            try:
                track.due_audio_s=self.turns.now
                track.submission_audio=self.submission_wave(track)
                self.queue.put_nowait(track)
                self.pending_turns[track.token]=track
                if track.token in self.prepared:self.prepared[track.token]['queued']=True
            except asyncio.QueueFull:
                self.prepared.pop(track.token,None)
                await c.ws.send_json({'type':'warning','text':'Several people spoke at once; please repeat the last turn.'})
    def merge_continuation(self,track):
        """Do not submit a clipped prefix when its same-owner tail is heard.

        Short diarizer dropouts can close a track while ASR is still working.
        Transfer only contiguous, already-bound speech, never a new wake/name
        or another person's words. The successor keeps its own quiet deadline.
        """
        user=self.memory.users.get(track.sid)
        if not user or not user.named or user.teacher is None:return False
        candidates={t.token:t for t in [*self.turns.tracks.values(),*getattr(self,'pending_turns',{}).values()]}
        newer=[t for t in candidates.values() if t.token!=track.token and t.sid==track.sid
            and t.last_speech>track.last_speech+.08
            and (t.voiced_start if t.voiced_start is not None else t.start)<=track.last_speech+.64
            and (t.voiced_start if t.voiced_start is not None else t.start)>=track.start-.32]
        if not newer:return False
        target=min(newer,key=lambda t:t.last_speech)
        if target.wake and self_name(target.partial):return False
        target.start=min(target.start,track.start)
        target.chunks=deque(sorted([*track.chunks,*target.chunks],key=lambda chunk:chunk[0]))
        target.fast=deque([*track.fast,*target.fast],maxlen=target.fast.maxlen)
        target.long=deque([*track.long,*target.long],maxlen=target.long.maxlen)
        target.speech_s=min(target.last_speech-target.start,target.speech_s+track.speech_s)
        target.last_direct_speech=max(target.last_direct_speech,track.last_direct_speech)
        if track.voiced_start is not None:target.voiced_start=min(target.voiced_start if target.voiced_start is not None else track.voiced_start,track.voiced_start)
        target.levels.extend(track.levels);target.version+=1
        # Cached crops/teacher scores refer to the old partial sentence.
        target.submission_audio=None;target.verification_audio=None
        target.recognition_start=None
        for attr in ('precomputed_identity','known_prefix_proof','asr_crop_verified'):
            if hasattr(target,attr):delattr(target,attr)
        for token in (track.token,target.token):
            entry=getattr(self,'prepared',{}).pop(token,None)
            if entry and not entry['task'].done():entry['task'].cancel()
        return True

    async def stop_turn(self,track,text,context):
        """Queued stop transcripts must not bypass the interruption guard.

        Normal turns deliberately use the permissive UI threshold. A stop
        during playback must independently pass the SAME current-owner gate
        as native/probe interruptions, even after normal identity refinement.
        """
        target=(getattr(self.c,'current_speaker',None) or {}).get('speaker_id')
        if not self.c.speaking or context['speaker_id']!=target:return False
        proposal={'kind':'BARGE','onset_s':min(2.,max(0.,self.c.hop_pos/self.c.ears.sr-track.start)),
            'yield_prob':1.,'source':'verified_stop_command','stop_text':text}
        self.interrupt_checked_at=-100.
        confirmed=await self.confirm_interrupt(proposal,round(track.last_speech*self.c.ears.sr))
        if confirmed!=target:return False
        await self.c._on_barge(proposal,self.c.hop_pos)
        self.c.session.reset()
        await self.c.ws.send_json({'type':'stop_command','speaker_name':context['display_name'],
            'text':'Paused. I’m listening.','source':'verified_turn'})
        return True

    def remember_stop(self,sid,text,end):
        if not sid or not is_stop_command(text):return
        if not hasattr(self,'recent_stops'):self.recent_stops=deque(maxlen=8)
        self.recent_stops.append((sid,end-3.04,end+.64,set(words(text))))

    def stopped_fragment(self,track,text):
        tokens=words(text)
        if not tokens:return False
        start=track.voiced_start if track.voiced_start is not None else track.start
        return any(sid==track.sid and start<=end and track.last_speech>=begin and
            len(tokens)<=len(stop_words) and set(tokens)<=stop_words
            for sid,begin,end,stop_words in getattr(self,'recent_stops',()))

    async def deliver_loop(self):
        try:
            while True:
                track=await self.queue.get()
                self.pending_turns.pop(track.token,None)
                epoch=self.epoch
                try:
                    if self.merge_continuation(track):continue
                    if getattr(self.c,'synthetic_diagnostics',False):
                        import base64
                        await self.c.ws.send_json({'type':'synthetic_audio_debug','start':track.start,'end':track.last_speech,
                            'audio':base64.b64encode(self.submission_wave(track).astype('<f2').tobytes()).decode(),
                            'track':track.key,'channel':track.channel})
                    final_started=time.perf_counter()
                    text=await self.use_preparation(track,epoch)
                    prefetched=text is not None
                    if not prefetched:text=await self.transcribe_turn(track)
                    if self.closed or not self.c.identity_enabled or epoch!=self.epoch:continue
                    if self.merge_continuation(track):continue
                    if text is None:
                        await self.c.ws.send_json({'type':'ignored','reason':'Speech could not be separated confidently from background words. Please repeat; nothing was sent to Luna.'})
                        continue
                    text=text or track.partial
                    visual=getattr(self.c,'visual',None)
                    view=visual.gate(track.voiced_start if track.voiced_start is not None else track.start,track.last_speech) if visual else {'allowed':True}
                    if not view['allowed'] and not is_stop_command(text):
                        await self.c.ws.send_json({'type':'ignored','reason':view['reason'],'source':'visual_filter',
                            'code':view.get('code','visual_uncertain'),'diagnostics':view.get('diagnostics',{})});continue
                    if self.observed_wake(track,text):track.wake=True
                    await self.c.ws.send_json({'type':'asr_preview','text':text,'status':'checking_identity',
                        'source':'speaker_turn','track':track.key,'audio_seconds':round(len(self.submission_wave(track))/16000,2),
                        'voiced_seconds':round(track.speech_s,2),'channel':track.channel})
                    await self.refine_identity(track,text,epoch=epoch)
                    if self.closed or epoch!=self.epoch:continue
                    if self.merge_continuation(track):continue
                    context=self.turns.speaker_context(track,text)
                    if not text or context is None:
                        if visual and text and hasattr(self.c,'maybe_visual_greeting'):
                            await self.c.maybe_visual_greeting(text,track.start,track.last_speech)
                        await self.c.ws.send_json({'type':'ignored','reason':getattr(track,'identity_rejection',None) or 'Speaker identity or words were uncertain. Please repeat or say Hey Robbie and introduce yourself again.'})
                        continue
                    if getattr(self,'open_enrollment',False):
                        from open_speakers import greeting_context
                        context=greeting_context(context,self.memory.users[context['speaker_id']])
                    if is_stop_command(text):
                        await self.stop_turn(track,text,context)
                        continue
                    if self.stopped_fragment(track,text):continue
                    controller=getattr(self.c,'decisions',None)
                    if controller and controller.suppresses(track,text):continue
                    # Wait for a halt acknowledgment rather than racing another
                    # generation. Microphone hops continue in their own task.
                    while (self.c.speaking or self.c.halting) and not self.closed:
                        await asyncio.sleep(.05)
                    if self.closed or epoch!=self.epoch or context['speaker_id'] not in self.memory.users:continue
                    if self.merge_continuation(track):continue
                    if visual and visual.settings.get('faceIdEnabled') and view.get('track_id'):
                        if not context.get('named') and hasattr(self.c,'maybe_visual_greeting'):
                            if await self.c.maybe_visual_greeting(text,track.start,track.last_speech):continue
                        previous=visual.bindings.get(view['track_id'])
                        if previous and previous['speaker_id'] not in self.memory.users:visual.bindings.pop(view['track_id'],None)
                        person=visual.bind(view['track_id'],context)
                        if person is None and context.get('named') and visual.settings['ignoreOffscreenSpeakers']:
                            await self.c.ws.send_json({'type':'ignored','reason':'Visible face and acoustic speaker disagree; no identity forced.'});continue
                        if person:await self.c.ws.send_json({'type':'visual_person',**person})
                    self.c.current_speaker=context;self.pending_names[context['speaker_id']]=text
                    self.c.session.reset()
                    await self.notice(decision={'accepted':True,'kind':'enrolled','id':context['speaker_id'],'name':context['display_name'],
                        'reason':'One second without this speaker; background voices do not extend the turn'})
                    await self.c.ws.send_json({'type':'endpoint','transcript':text,'speaker_id':context['speaker_id'],
                        'speaker_name':context['display_name'],'by':'speaker_silence_1s','turn_s':round(track.last_speech-track.start,2),
                        'timing':{'speaker_last_audio_s':round(track.last_speech,3),
                            'silence_detected_audio_s':round(getattr(track,'due_audio_s',self.turns.now),3),
                            'prefetch_used':prefetched,'finalization_ms':round((time.perf_counter()-final_started)*1000,2),
                            'native_timestamp_continuity':bool(getattr(track,'native_continuity_used',False)),
                            'verified_background_prefix':getattr(track,'known_prefix_proof',None),
                            **getattr(track,'preparation_metrics',{})}})
                    await self.c._send_turn(text)
                except asyncio.CancelledError:raise
                except Exception as exc:
                    logging.getLogger('robo.speakers').warning('Speaker submit failed: %s',type(exc).__name__)
                    await self.c.ws.send_json({'type':'warning','text':'Speaker turn could not be submitted; please repeat.'})
        except asyncio.CancelledError:pass

    async def identity_embedding(self,track,text):
        cached=getattr(track,'precomputed_identity',None)
        if cached is not None and cached[0]==text:return cached[1]
        current=self.memory.users.get(track.sid)
        if current is None:return None,None,0
        introduction_name=self_name(text)
        enrollment_probe=current.teacher is None or bool(introduction_name and (wake_intent(text)
            or self.observed_wake(track,text) or (track.wake and not current.named)))
        if enrollment_probe:
            start=max(track.start,track.recognition_start or track.start)+getattr(track,'enrollment_crop_s',0.)
            source=getattr(track,'enrollment_audio',getattr(track,'enrollment_source',self.submission_wave(track)))
            track.enrollment_fast,track.enrollment_long=aligned_reference(list(self.records),start,start+len(source)/16000)
        from speaker_teacher import get_cached_teacher,enrollment_reference
        wave=self.submission_wave(track);wave=np.pad(wave,(0,max(0,10240-len(wave))))
        whole=track.verification_audio
        # Retain the independent FULL-span check. Looking only at the final
        # three seconds can validate a long TV prefix just because the actual
        # user speaks at the end. Long student/recovery contexts remain 3.04 s.
        def embed_views():
            teacher=get_cached_teacher()
            if enrollment_probe and whole is not None and len(whole)>=10240:
                reference=getattr(track,'enrollment_audio',whole)
                if introduction_name:reference=getattr(track,'enrollment_audio',getattr(track,'enrollment_source',whole))
                windows=[enrollment_reference(reference,s) for s in (2.,3.04,4.)]
                unique={len(x):x for x in windows}
                templates={n:unit(teacher.embed(x)) for n,x in unique.items()}
                track.enrollment_templates=tuple(templates.values())
                primary=templates[len(windows[1])]
                return primary,None,1
            vectors=[unit(teacher.embed(wave))]
            if whole is not None and len(whole)>=10240 and (len(whole)!=len(wave) or not np.array_equal(whole,wave)):
                vectors.append(unit(teacher.embed(whole)))
            cosine=float(vectors[0]@vectors[1]) if len(vectors)>1 else None
            merged=(.8*vectors[0]+.2*vectors[1] if len(vectors)==2 and getattr(track,'asr_crop_verified',False)
                and len(wave)>=20480 else np.mean(vectors,axis=0))
            return unit(merged),cosine,len(vectors)
        return await asyncio.to_thread(embed_views)

    async def refine_identity(self,track,text,epoch=None):
        """Full-utterance ECAPA checks channel reuse/merges, not fast timing.

        A diarization channel is NOT a permanent biometric identity: two similar
        voices can share it. Never overwrite Christoph's name with Robert's
        introduction just because the diarizer reused a channel.
        """
        current=self.memory.users.get(track.sid)
        if current is None:return
        embedding=None
        view_cosine=None;view_count=0
        try:
            embedding,view_cosine,view_count=await self.identity_embedding(track,text)
        except Exception:pass
        if epoch is not None and (self.closed or epoch!=self.epoch):track.sid=None;return
        view_floor=.35 if getattr(track,'known_prefix_proof',None) else .5
        if view_cosine is not None and view_cosine<view_floor:
            track.sid=None
            track.identity_rejection=f'Speaker views disagree ({view_cosine:.2f} < {view_floor:.2f}). Please repeat with less overlap; lowering voice-match strictness does not disable this overlap check.'
            await self.notice(warning=track.identity_rejection)
            return
        if embedding is None:
            track.sid=None
            await self.notice(warning='Full-utterance speaker verification is unavailable. Please retry or explicitly use Talk without speaker filter.')
            return
        bot=await self.playback_voice();bot_score=float(embedding@bot) if bot is not None else None
        if epoch is not None and (self.closed or epoch!=self.epoch):track.sid=None;return
        known_best=max((u.teacher_score(embedding) for u in self.memory.users.values() if u.teacher is not None),default=-1.)
        if bot_score is not None and bot_score>=.65 and bot_score-known_best>=self.utterance_margin:
            track.sid=None
            await self.notice(warning='Likely playback echo rather than an enrolled user; no turn was submitted.')
            return
        if getattr(self,'open_enrollment',False):
            from open_speakers import claimed_name
            claim=claimed_name(text)
            if claim:
                matches=[u for u in self.memory.users.values() if u.named and u.name.casefold()==claim.casefold()]
                if len(matches)!=1:
                    track.identity_rejection='This name is unknown or ambiguous. Introduce yourself with “My name is …” or use a distinct name.'
                    track.sid=None;await self.notice(warning=track.identity_rejection);return
                target=matches[0]
                if current.named and current.id!=target.id:
                    track.sid=None;await self.notice(warning='Two named people are not merged automatically. Start with an unnamed guest voice variant.');return
                if track.speech_s<1.28 or (view_cosine is not None and view_cosine<.75):
                    track.sid=None;await self.notice(warning='Please repeat the identity claim in a longer, clear sentence without overlapping speech.');return
                added=self.memory.confirm_variant(target.id,track.vector(),track.long_vector(),embedding)
                if not added and target.teacher_score(embedding)<.96:
                    track.sid=None;await self.notice(warning='The four-variant limit is reached. No additional voiceprint was stored.');return
                previous=current.id;self.rebind(track,target.id)
                if previous!=target.id:self.memory.users.pop(previous,None)
                await self.notice(voice_variant={'speaker_id':target.id,'name':target.name,'added':added,
                    'count':1+len(target.confirmed_variants),'source':'explicit spoken user assertion; NOT authentication'})
                return
            if (current.teacher is None and not current.named and getattr(track,'open_enrollment',False)
                and known_best<self.utterance_threshold):
                # A genuinely new guest cannot agree with an OLD person's
                # teacher. The candidate's clean/owned/coherent speech, final
                # ASR views and echo veto already qualified this new identity.
                # Do not run the known-person cross-model-disagreement veto
                # against a low-scoring, unrelated reference winner.
                self.install_reference(current,embedding,track)
                return
        quality_score=None;quality_margin=None
        name=self_name(text)
        # A provisional Guest flag can outlive the introduction while final
        # ASR runs. It is NOT a fresh wake from a later background speaker.
        wake=bool(wake_intent(text) or self.observed_wake(track,text) or (track.wake and not current.named)
            or (getattr(self,'open_enrollment',False) and name))
        incomplete=bool(re.search(r'\b(?:ich\s+(?:bin|heiße)|i\s+am|my\s+name\s+is)\s*[.!?…]*$',text,re.I))
        other_introduction=any(t.sid and t.sid!=current.id and t.wake and t.name_heard and not t.closed
            and t.start<=track.last_speech+1. and t.last_speech>=track.start-.32 for t in self.turns.tracks.values())
        other_introduction=other_introduction or any(track.start-.32<=at<=track.last_speech+1.6
            and value!=current.name.casefold() for at,value in self.asr_wakes)
        if wake and not name and (incomplete or (current.named and other_introduction)):
            # One introduction can straddle old/new diarizer channels. Its
            # clipped "Hey, I am…" prefix is not an additional anonymous user.
            # Leave the complete foreground introduction to establish its ID.
            track.sid=None;return
        # Incremental ASR labels are provisional until the first complete
        # acoustic reference exists (e.g. partial "Selber" -> final "Selma").
        # Correct that label in place, not by creating an orphan second user.
        if current.teacher is None and name and wake:
            existing=next((u for u in self.memory.users.values() if u.id!=current.id and u.named
                and u.name.casefold()==name.casefold() and u.teacher is not None),None)
            if existing is not None:
                if existing.teacher_score(embedding)<self.utterance_threshold:track.sid=None;return
                previous=current.id;self.rebind(track,existing.id);self.memory.users.pop(previous,None)
            else:
                self.memory.rename(current.id,name);self.install_reference(current,embedding,track)
            return
        # A wake-only Guest is provisional. Use the immediately following
        # full self-introduction as its reliable teacher reference instead of
        # rejecting it against a 640-ms wake fragment. Never overwrite a named
        # person. A repeated known name must still agree acoustically.
        if not current.named and name and (wake or self.memory.clock()-current.created<=15):
            existing=next((u for u in self.memory.users.values() if u.id!=current.id and u.named and u.name.casefold()==name.casefold()),None)
            if existing is not None:
                if existing.teacher is None or existing.teacher_score(unit(embedding))<self.utterance_threshold:track.sid=None;return
                old=current.id;current=existing;self.rebind(track,existing.id);self.memory.users.pop(old,None)
            else:
                self.memory.rename(current.id,name);self.install_reference(current,embedding,track)
            return
        if name and current.named and current.name.casefold()!=name.casefold() and not wake:
            track.sid=None
            await self.notice(warning='A different name was heard without the wake phrase. Say “Hey Robbie, I am …” to join; an existing user will not be renamed.')
            return
        student_vote=self.memory.match(track.vector(),track.long_vector())
        scores=[{'name':u.name,'cosine':round(u.teacher_score(unit(embedding)),4)} for u in self.memory.users.values()
            if embedding is not None and u.teacher is not None]
        await self.notice(verification={'candidate':current.name,'reference_scores':scores,'wake':wake,
            'self_introduction':bool(name),'threshold':self.utterance_threshold,'views':view_count,'view_cosine':view_cosine,
            'student_vote':student_vote,'independent_activity_confirmed':track.independent_activity_confirmed,
            'playback_cosine':bot_score})
        if name and wake and current.named and current.name.casefold()!=name.casefold():
            existing=next((u for u in self.memory.users.values() if u.named and u.name.casefold()==name.casefold()),None)
            if existing is not None:
                # A repeated self-introduction may reuse a name only if its
                # acoustic reference also agrees; a spoken name is not proof.
                agree=embedding is not None and existing.teacher is not None and existing.teacher_score(unit(embedding))>=self.utterance_threshold
                if not agree:track.sid=None;return
                user=existing
            else:
                user,_=self.memory.enroll(track.vector(),track.long_vector(),name,force_new=True)
                if user is None:track.sid=None;return
            self.rebind(track,user.id)
            current=user
        elif embedding is not None:
            references=[(u.teacher_score(unit(embedding)),u) for u in self.memory.users.values() if u.teacher is not None]
            references.sort(key=lambda x:x[0],reverse=True)
            if references:
                score,user=references[0];runner=references[1][0] if len(references)>1 else -1.
                if (student_vote.get('id') and student_vote['id']!=user.id and
                    student_vote.get('similarity',-1)>=max(.60,self.memory.threshold) and
                    student_vote.get('margin',0)>=.12 and score<max(.60,self.utterance_threshold) and
                    (view_cosine is None or view_cosine<.85)):
                    track.sid=None
                    await self.notice(warning='Independent short and full speaker cues disagree; no identity was guessed.')
                    return
                if wake and not name and current.named and score<self.utterance_threshold:
                    guest,_=self.memory.enroll(track.vector(),track.long_vector(),force_new=True)
                    if guest is None:track.sid=None;return
                    self.install_reference(guest,embedding,track);self.rebind(track,guest.id)
                    return
                continuity=(len(references)>=2 and user.id==current.id and track.channel is not None
                    and self.turns.channels.get(track.channel)==current.id and
                    score>=max(.4,self.utterance_threshold-.08) and score-runner>=max(.12,self.utterance_margin))
                single_continuity=(len(references)==1 and user.id==current.id and current.named
                    and track.channel is not None and self.turns.channels.get(track.channel)==current.id
                    and track.independent_activity_confirmed and track.speech_s>=1.28
                    and score>=max(.45,self.utterance_threshold-.05)
                    and view_cosine is not None and view_cosine>=.85
                    and bot_score is not None and score-bot_score>=.12)
                consensus=(score>=max(.55,self.utterance_threshold) and score-runner>=max(.04,self.utterance_margin/2)
                    and view_cosine is not None and view_cosine>=.9 and track.speech_s>=1.5
                    and (bot_score is None or score-bot_score>=.12))
                # A nearly tied teacher must not be resolved by reusable
                # diarizer-channel ownership. A separately calibrated student
                # can corroborate the SAME teacher winner, never overrule it.
                corroborated=(score>=self.utterance_threshold and score-runner>=0 and
                    view_cosine is not None and view_cosine>=.85 and track.speech_s>=1.28 and
                    student_vote.get('id')==user.id and
                    student_vote.get('similarity',-1)>=max(.70,self.memory.threshold) and
                    student_vote.get('margin',0)>=.10 and
                    (bot_score is None or score-bot_score>=.12))
                if (score>=self.utterance_threshold and score-runner>=self.utterance_margin) or continuity or single_continuity or consensus or corroborated:
                    if (continuity or single_continuity) and score<self.utterance_threshold:
                        await self.notice(continuity={'name':user.name,'cosine':score,'margin':score-runner,
                            'reason':'Independent short activity + two agreeing full-span views + separated playback' if single_continuity else 'Established diarization channel + separated reference winner; bounded threshold relaxation'})
                    current=user;self.rebind(track,user.id)
                    quality_score=score;quality_margin=score-runner
                elif current.teacher is not None:
                    track.sid=None
                    track.identity_rejection=(f'Voice match for {user.name}: {score:.2f} / required {self.utterance_threshold:.2f}; '
                        f'gap to other users {score-runner:.2f} / required {self.utterance_margin:.2f}. '
                        'Try a slightly lower Voice match strictness if this is you, or repeat with less overlap.')
                    await self.notice(warning=track.identity_rejection)
                    return
        if embedding is not None and current.teacher is None:
            self.install_reference(current,embedding,track)
        elif (quality_score is not None and current.named and track.speech_s>=1.5
              and quality_score>=max(.4,self.utterance_threshold-.08) and quality_margin>=max(.20,self.utterance_margin)
              and view_cosine is not None and (view_cosine>=.9 or
                  (quality_score>=max(.55,self.utterance_threshold) and view_cosine>=.85))
              and (bot_score is None or quality_score-bot_score>=.12)):
            anchor=current.teacher_anchor if current.teacher_anchor is not None else current.teacher
            if float(embedding@anchor)>=max(.4,self.utterance_threshold-.08):current.adapt_teacher(embedding)
    async def metadata(self,data):
        update=data.get('speaker_update')
        if not isinstance(update,dict):return
        sid=update.get('speaker_id');name=update.get('display_name');text=self.pending_names.pop(sid,None)
        user=self.memory.users.get(sid)
        if text and name==self_name(text) and user is not None and (not user.named or user.name.casefold()==name.casefold()):
            self.memory.rename(sid,name);await self.notice(metadata=update)
