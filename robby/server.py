"""Robo page + unchanged TYGER Conversation and EARS, sharing the Playground TTS."""
import asyncio
import json
import logging
import math
import os
from pathlib import Path
import time
import hashlib
import numpy as np

from aiohttp import web, WSMsgType
import httpx

import paths
from app.ears.conversation import Conversation
from app.plugins.base import PluginContext
from app.plugins.ears.tyger.plugin import Tyger
from tts_bridge import MouthBridge
from app.ears.playback import mic_block
from audio_sync import estimate_echo_delay
from feature_tap import attach_feature_tap,local_only,extract,extract_context,extract_layers,transcribe_check
from gpu_guard import audio_failure
from identity import IdentityManager
from speaker_teacher import preserve_encoder_precision

preserve_encoder_precision()

ROOT=paths.ROOT
log=logging.getLogger('robo')
_signature=hashlib.sha256()
for _name in ('server.py','social_profiles.py','voice_caption_worker.py','visual_identity.py','static/social-profiles.js','static/multimodal-ui.js','addressed_speech.py','sound_events.py','sound_event_worker.py','bootstrap_sound_events.py','ambient_audio.py','environment_session.py','sound_caption_worker.py','bootstrap_sound_caption.py','static/environment-controls.js','llm_backends.py','llm_session.py','credential_store.py','conversation_focus.py','sentence_session.py','sentence_decisions.py','speaker_management.py','live_decisions.py','laya_policy.py','stop_commands.py','voice_captioning.py','quick_reaction.py','informed_bridge.py','live_speech.py','two_stage_pipeline.py','knowledge_memory.py','memory_session.py','static/memory-controls.js','interaction_policy.py','conversation_director.py','asr_choice.py','asr_catalog.py','speech_language.py','pause_vad.py','avatar_commands.py','avatar_actions.py','bolt_text.py','bolt_director.py','conversation_tasks.py','bolt_pipeline.py','tts_bridge.py','static/client.js','static/live-overview.js','static/ui.js','static/avatar-actions.js'):
    _signature.update(_name.encode());_signature.update((ROOT/_name).read_bytes())
STATE={'ready':False,'error':None,'phase':'Starting Nemotron / TYGER','ears':None,'source_sha256':_signature.hexdigest()}


class RoboConversation(Conversation):
    async def _on_text(self,payload):
        data=json.loads(payload)
        if data.get('memory_event'):
            await self.ws.send_json(data['memory_event']);return
        if data.get('avatar_action'):
            await self.ws.send_json({'type':'avatar_action',**data['avatar_action']});return
        if data.get('pipeline_metrics'):
            await self.ws.send_json({'type':'pipeline_metrics','reply_id':self.playback.rid,'metrics':data['pipeline_metrics']});return
        if data.get('component_metrics'):
            await self.ws.send_json({'type':'latency_update','reply_id':self.playback.rid,'metrics':data['component_metrics']});return
        if data.get('text') and not data.get('status_only'):self.open_agent_text=data['text']
        if data.get('text') and not data.get('status_only') and getattr(self,'sentence_stream',None):self.sentence_stream.permissions.note_reply(data['text'])
        if data.get('error_type'):
            await self.ws.send_json({'type':'error','error':data['text'],'stage':data.get('error_stage'),
                'error_type':data['error_type'],'http_status':data.get('http_status')})
            return
        if data.get('status_only'):
            await self.ws.send_json({'type':'service_status','text':str(data.get('text',''))[:300],'stage':data.get('stage')})
            return
        if getattr(self,'decisions',None):
            try:await self.decisions.note_agent(data)
            except (ValueError,TypeError,KeyError):await self.decisions.notice(warning='A conversational rule was invalid and was not installed.')
            if data.get('interaction_actions') and self.decisions.book.rules:
                await self.to_mouth(b'W',{'replies':[r.reply for r in self.decisions.book.rules.values()][:4],
                    'language':self.decisions.language})
        if self.speaker_ext is not None:await self.speaker_ext.metadata(data)
        await self.ws.send_json({'type':'backchannel' if data.get('bc') else 'reply','text':data['text'],
            'reply_id':self.playback.rid,'performance':data.get('performance'),
            'avatar_actions':data.get('avatar_actions',[]),
            'avatar_action_id':data.get('avatar_action_id'),
            'avatar_event':data.get('avatar_event'),
            'sentence_index':data.get('sentence_index'),'sentence_text':data.get('sentence_text'),'sentence_role':data.get('sentence_role'),
            'speaker_id':(data.get('speaker') or {}).get('speaker_id'),'speaker_name':(data.get('speaker') or {}).get('display_name'),
            'speaker_update':data.get('speaker_update'),'timing':data.get('timing'),
            'sources':data.get('sources',[]),'lookups':data.get('lookups',[])})

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from llm_session import LLMSession
        self.llm=LLMSession(self)
        self.quick_replies=os.environ.get('ROBO_QUICK_REPLIES','0')=='1'
        from avatar_personality import AvatarPersonality
        self.avatar_personality=AvatarPersonality();self.avatar_optional=False
        self.identity_enabled=False;self.predictor=None;self.last_identity_notice=0
        self.identity_mode='ecapa';self.teacher=None;self.teacher_stream=None
        self.identity_audio=np.zeros(0,np.float32)
        selection={};robo=None
        ref=ROOT/'speaker_v2/artifacts/robo_reference.npz'
        if ref.exists():
            try:robo=np.load(ref)['embedding']
            except Exception:pass
        checkpoint=ROOT/'speaker_v2/artifacts/best.pt'
        if checkpoint.exists():
            try:
                from speaker_v2.inference import Predictor
                self.predictor=Predictor(checkpoint)
                path=ROOT/'speaker_v2/artifacts/selection.json'
                if path.exists():selection=json.loads(path.read_text())
            except Exception as exc:log.warning('Speaker pilot unavailable: %s',type(exc).__name__)
        self.student_threshold=float(selection.get('verification_threshold',.65))
        self.teacher_threshold=.58  # conservative validation-EER operating point, not authentication
        self.identity=IdentityManager(threshold=self.teacher_threshold,robo=robo)
        from live_decisions import LiveDecisions
        self.decisions=LiveDecisions(self)
        from stop_feedback import StopFeedback
        self.stop_feedback=StopFeedback(self)
        from visual_identity import VisualIdentity
        self.visual=VisualIdentity()
        self.speaker_ext=None;self.current_speaker=None
        self.input_samples=0;self.input_blocks=0;self.last_input_notice=0.
        self.pending_barge_proposal=None
        self.native_barge_settings=(self.session.barge_thr,self.session.barge_deb)
        try:
            from speaker_session import SpeakerSession
            from speaker_runtime import available
            long='long3' if 'long3' in available() else 'long'
            self.speaker_ext=SpeakerSession(self,long_context=long)
            self.identity_mode='v4-long3' if long=='long3' else 'v4-long';self.identity_enabled=True
            self._speaker_barge_sensitivity(True)
        except Exception as exc:log.warning('Completed speaker student unavailable: %s',type(exc).__name__)
        from environment_session import EnvironmentSession
        self.environment=EnvironmentSession(self)
        self.sentence_stream=None
        if os.environ.get('ROBO_SENTENCE_STREAM','0')=='1' and self.speaker_ext is not None:
            from sentence_session import SentenceSession
            self.sentence_stream=SentenceSession(self)
            self._speaker_barge_sensitivity(False)

    def _speaker_barge_sensitivity(self,enabled):
        # The frozen GRU is unchanged. With full acoustic confirmation we can
        # use its supported sensitive operating point without trusting weak
        # short speaker cues or turning bot echo into an interruption.
        threshold,debounce=(.20,1) if enabled else self.native_barge_settings
        self.ears.set_threshold(self.session,'barge_thr',threshold)
        self.ears.set_threshold(self.session,'barge_deb',debounce)

    def _ready_message(self):
        return {**super()._ready_message(),'identity_enabled':self.identity_enabled,'identity_mode':self.identity_mode,
            'llm_settings':self.llm.settings(),'quick_replies_enabled':self.quick_replies,'environment':self.environment.state(),
            'interaction':self.decisions.state(),
            'visual_identity':self.visual.state(),
            'sentence_stream':self.sentence_stream is not None,
            'asr_settings':self.sentence_stream.settings() if self.sentence_stream else None,
            'speaker_instruction':'Speak normally. All audible speakers can join; unknown voices receive session labels and can introduce themselves.' if self.sentence_stream else 'Say: Hey Robbie, I am Christoph. Only enrolled voices keep their turn open.'}

    async def _send_turn(self,transcript):
        self.avatar_optional=False;self.avatar_personality.user()
        environment_epoch=self.environment.epoch if getattr(self,'environment',None) else None
        observation=await self.environment.context_for_turn(transcript) if getattr(self,'environment',None) else None
        if environment_epoch is not None and environment_epoch!=self.environment.epoch:return
        if (getattr(self,'pending_input_timing',None) or {}).get('kind')=='typed' and self.current_speaker:
            self.current_speaker={k:v for k,v in self.current_speaker.items() if k!='voice_delivery'}
        if self.sentence_stream:
            typed=(getattr(self,'pending_input_timing',None) or {}).get('kind')=='typed'
            if typed:
                from sentence_session import name_claim
                if name_claim(transcript):
                    # An explicit typed introduction names the current texter,
                    # but supplies no acoustic prototype for another person.
                    self.current_speaker=self.sentence_stream.memory.assign(None,transcript)
                    await self.ws.send_json(self.sentence_stream.identity_state())
            result=self.sentence_stream.permissions.command(transcript,self.current_speaker,typed=typed)
            if result:
                await self.ws.send_json({'type':'speaker_permission',**result})
                await self.ws.send_json(self.sentence_stream.identity_state())
            self.current_speaker={**(self.current_speaker or {}),'speaker_management':{
                **self.sentence_stream.permissions.context(),'permission_result':result}}
        self.decisions.note_user(transcript,self.current_speaker)
        from interaction_policy import explicit_literal_action
        compiled=explicit_literal_action(transcript)
        preapplied=False
        if compiled and (self.decisions.enabled or compiled['name']=='clear_interruption_rules'):
            installed=self.decisions.book.apply([compiled],self.decisions.source_owner,transcript)
            self.decisions.blocker='Listening for an agreed trigger' if self.decisions.book.rules else 'No agreed rules yet'
            preapplied=True;await self.decisions.notice(installed=installed)
        self._start_speaking()
        await self.ws.send_json({'type':'latency_start','reply_id':self.playback.rid,
            'input':getattr(self,'pending_input_timing',None) or {'kind':'unknown'}})
        self.pending_input_timing=None
        await self.to_mouth(b'T',{'transcript':transcript,'speaker':self.current_speaker,'interaction_state':{**self.decisions.state(),'current_rule_applied':preapplied},
            'avatar_context':{**self.avatar_personality.snapshot(),**({'environment':observation} if observation else {})},'quick_replies':getattr(self,'quick_replies',False),**self.llm.snapshot()})
        if preapplied and compiled['name']=='set_interruption_rule':
            await self.to_mouth(b'W',{'replies':[compiled['arguments']['reply']],'language':self.decisions.language})

    async def _finish_turn(self,barged):
        self.avatar_optional=False
        await super()._finish_turn(barged)

    async def _send_interjection(self,transcript,speaker,reply,rule_id):
        if self.speaking or self.halting:return False
        end=self.hop_pos/self.ears.sr
        if not self.visual.gate(max(0,end-3.2),end)['allowed']:return False
        self.current_speaker=speaker;self.decisions.note_user(transcript,speaker)
        self.session.reset()
        await self.ws.send_json({'type':'endpoint','transcript':transcript,'speaker_id':speaker['speaker_id'],
            'speaker_name':speaker['display_name'],'by':'agreed_interruption','rule_id':rule_id})
        self._start_speaking()
        await self.to_mouth(b'T',{'transcript':transcript,'speaker':speaker,'interaction_state':self.decisions.state(),
            'interaction_reply':reply,'interaction_rule_id':rule_id,'interaction_language':self.decisions.language})
        return True

    async def maybe_visual_greeting(self,text,start,end):
        if self.speaking or self.halting or self.visual.camera!='on':return False
        view=self.visual.select(start,end)
        if not view['allowed']:return False
        reply=self.visual.greeting(view.get('track_id'),text,self.decisions.language)
        if not reply:return False
        await self.ws.send_json({'type':'visual_greeting','track_id':view['track_id'],'text':reply})
        self.current_speaker=None;self.session.reset();self._start_speaking()
        await self.to_mouth(b'T',{'transcript':text,'interaction_reply':reply,'interaction_language':self.decisions.language})
        return True

    def _start_speaking(self):
        # Identity evidence for an interruption must be NEW speech, not the
        # tail of the very utterance whose answer is now being prepared.
        if getattr(self,'stop_feedback',None):self.stop_feedback.cancel()
        self.speaker_reply_start_s=self.hop_pos/self.ears.sr
        super()._start_speaking()

    async def run(self):
        if self.sentence_stream:
            self.sentence_stream.start()
            from asr_catalog import default_model
            # Keep original Nemotron as the truthful active fallback while
            # the selected final recognizer activates in the background.
            if default_model()!='nemotron':await self.sentence_stream.asr.select(default_model())
        elif self.speaker_ext is not None:self.speaker_ext.start()
        try:await super().run()
        finally:
            self.llm.credential=None
            typed_task=getattr(self,'typed_turn_task',None)
            if typed_task:
                typed_task.cancel();await asyncio.gather(typed_task,return_exceptions=True)
            await self.environment.close()
            if self.sentence_stream:await self.sentence_stream.close()
            self.stop_feedback.cancel()
            await self.decisions.close()
            if self.speaker_ext is not None:await self.speaker_ext.close()
            task=getattr(self,'calibration_task',None)
            if task is not None:task.cancel()
            self.identity.clear();self._reset_identity_context()
            self.cal_mic=[];self.cal_ref=[]

    async def _teacher(self):
        if self.teacher is None:
            from speaker_teacher import get_cached_teacher,StreamingTeacher
            self.teacher=await asyncio.to_thread(get_cached_teacher)
            self.teacher_stream=StreamingTeacher(self.teacher)
        return self.teacher

    def _reset_identity_context(self):
        self.identity.reset_utterance();self.identity_audio=np.zeros(0,np.float32)
        if self.predictor is not None:self.predictor.reset()
        if self.teacher_stream is not None:self.teacher_stream.reset()

    async def _on_events(self,events,hop_end):
        stream=getattr(self,'sentence_stream',None)
        foreground=not stream or hop_end/self.ears.sr-max(getattr(stream,'foreground_last',-100.),getattr(stream,'raw_loud_at',-100.))<.8
        if foreground and getattr(self,'avatar_personality',None) and any(e.get('kind')=='PARTIAL' and e.get('text','').strip() for e in events):
            self.avatar_personality.last_asr=time.monotonic()
        if getattr(self,'sentence_stream',None):
            # Ingest waveform/activity first, then newest decoder words, then
            # commit. Neither stale text nor an old quiet clock may win a hop.
            for event in events:
                if event['kind']=='SPEAKER_FEATURES':await self.sentence_stream.observe(event,hop_end)
            for event in events:
                if event['kind']=='PARTIAL':
                    await self.ws.send_json({'type':'asr_preview','text':event['text'],'status':'sentence_stream','source':'native_asr'})
                    await self.sentence_stream.preview_text(event['text'],hop_end/self.ears.sr)
                elif event['kind']=='ENDPOINT':
                    await self.sentence_stream.native_endpoint(event.get('transcript',''),hop_end/self.ears.sr)
                elif event['kind']=='BARGE' and self.speaking and not self.await_onset and not getattr(getattr(self,'decisions',None),'proactive_active',False):
                    self.sentence_stream.propose_barge(event,hop_end)
            await self.sentence_stream.advance()
            if getattr(self,'decisions',None):self.decisions.tick(hop_end)
            await super()._on_events([e for e in events if e['kind']=='FRAME'],hop_end)
            return
        if self.identity_enabled and self.speaker_ext is not None and self.identity_mode.startswith('v4'):
            proposals=[e for e in events if e['kind']=='BARGE']
            if proposals and self.speaking:self.pending_barge_proposal=(max(proposals,key=lambda e:e.get('yield_prob',0)),hop_end)
            if proposals and getattr(self,'synthetic_diagnostics',False):
                await self.ws.send_json({'type':'synthetic_barge_proposal','audio_s':hop_end/self.ears.sr,
                    'proposals':proposals,'speaker_cue':self.speaker_ext.last_activity})
            # Recognition is useful feedback even when enrollment rejects a
            # turn. A preview is NEVER an endpoint and never calls the LLM.
            if not self.speaking:
                for event in events:
                    if event['kind']=='PARTIAL':
                        self.speaker_ext.note_asr(event['text'],hop_end/self.ears.sr)
                        if getattr(self,'decisions',None):self.decisions.offer(event['text'],hop_end)
                        await self.ws.send_json({'type':'asr_preview','text':event['text'],
                            'status':'not_submitted','source':'native_asr'})
            else:
                # If the native decoder emits a partial during playback, do
                # not discard it. It is only a candidate, NEVER a voice bypass.
                for event in events:
                    if event['kind']=='PARTIAL':self.decisions.offer_agent([event['text']],hop_end)
            for event in events:
                if event['kind']=='SPEAKER_FEATURES':await self.speaker_ext.observe(event,hop_end)
            self.speaker_ext.schedule_stop_probe(hop_end)
            if getattr(self,'decisions',None):self.decisions.tick(hop_end)
            # Native GRU still runs unchanged. It no longer controls the
            # enrolled speaker's submit clock or transcribes background voices
            # into a submitted turn. Known users retain native interruption.
            filtered=[e for e in events if e['kind']=='FRAME']
            pending=self.pending_barge_proposal
            hold_native=bool(getattr(getattr(self,'decisions',None),'proactive_active',False))
            if pending and self.speaking and not hold_native and 0<=hop_end-pending[1]<=1.28*self.ears.sr:
                # A teacher/echo check must not hold the next microphone hop.
                # Single flight, current reply + identity epoch checked again.
                task=getattr(self,'native_barge_task',None)
                if task is None or task.done():
                    ext=self.speaker_ext;epoch=ext.epoch;rid=self.playback.rid
                    async def verify(proposal=pending[0],at=hop_end):
                        confirmed=await ext.confirm_interrupt(proposal,at)
                        if (confirmed and ext is self.speaker_ext and epoch==ext.epoch and self.speaking
                            and self.playback.rid==rid and not self.decisions.proactive_active):
                            self.pending_barge_proposal=None
                            await self._on_barge(proposal,at)
                    self.native_barge_task=ext.spawn(verify())
            elif pending and (not self.speaking or hop_end-pending[1]>1.28*self.ears.sr):self.pending_barge_proposal=None
            if getattr(self,'calibrating',False):filtered=[e for e in filtered if e['kind']=='FRAME']
            await super()._on_events(filtered,hop_end);return
        for event in events:
            if event['kind']=='SPEAKER_FEATURES':
                energy=float(np.sqrt(np.mean(10**(np.asarray(event['energy_db'])/10))))
                prediction=self.predictor.update(event['features'],energy=energy) if self.predictor is not None else {'kind':'warming'}
                audio=event['audio16']
                if self.identity_enabled or self.identity_mode=='student':
                    self.identity_audio=np.r_[self.identity_audio,audio][-(32000 if self.speaking else 320000):]
                identity=None
                if self.identity_mode=='student':identity=self.identity.observe(prediction)
                elif self.identity_enabled:
                    try:
                        await self._teacher()
                        cue=await asyncio.to_thread(self.teacher_stream.update,audio,speech=bool(self.session.in_turn and not self.speaking))
                        identity=self.identity.observe(cue)
                    except Exception as exc:
                        self.identity_enabled=False
                        await self.ws.send_json({'type':'identity','enabled':False,'mode':self.identity_mode,**self.identity.status()})
                        await self.ws.send_json({'type':'warning','text':'ECAPA unavailable; speaker filter disabled ('+type(exc).__name__+').'})
                if identity is None:continue
                if time.monotonic()-self.last_identity_notice>.8:
                    self.last_identity_notice=time.monotonic()
                    await self.ws.send_json({'type':'identity','prediction':identity,'enabled':self.identity_enabled,**self.identity.status()})
        filtered=[e for e in events if e['kind']!='SPEAKER_FEATURES']
        if getattr(self,'calibrating',False):filtered=[e for e in filtered if e['kind']=='FRAME']
        await super()._on_events(filtered,hop_end)

    async def _on_endpoint(self,event):
        end=self.hop_pos/self.ears.sr
        visual=self.visual.gate(max(0,end-3.2),end)
        if not visual['allowed']:
            await self.ws.send_json({'type':'ignored','reason':visual['reason']});return
        if self.identity_enabled or self.identity_mode=='student':
            if self.identity_mode=='ecapa' and self.identity_enabled:
                try:
                    teacher=await self._teacher()
                    audio=self.identity_audio
                    # The complete recognized utterance, not just the wake words.
                    active=np.flatnonzero(np.abs(audio)>.003)
                    if len(active):audio=audio[max(0,active[0]-320):min(len(audio),active[-1]+321)]
                    embedding=await asyncio.to_thread(teacher.embed,audio)
                    self.identity.reset_utterance()
                    # Whole-utterance evidence is one embedding, not repeated frames.
                    self.identity.observe({'embedding':embedding,'kind':'speech','confidence':.8})
                    self.identity.voiced_ms=round(len(audio)/16)
                except Exception:
                    self.identity.reset_utterance()
            decision=self.identity.finish(event.get('transcript') or '',enabled=self.identity_enabled)
            if self.predictor is not None:self.predictor.reset()
            if self.teacher_stream is not None:self.teacher_stream.reset()
            self.identity_audio=np.zeros(0,np.float32)
            await self.ws.send_json({'type':'identity','decision':decision,'enabled':self.identity_enabled,**self.identity.status()})
            if not decision['accepted']:
                await self.ws.send_json({'type':'ignored','reason':decision.get('reason','Unenrolled or uncertain speaker')})
                return
        await super()._on_endpoint(event)

    async def _on_barge(self,event,hop_end):
        if getattr(self,'environment',None):self.environment.pending=None;self.environment.last_reply=time.monotonic()+30
        # Stopping is not a conversational reply. Preserve the manual stop and
        # existing acoustically verified barge paths even without visible lips.
        if not self.speaking:return
        # Claim cancellation before the first socket await: native TYGER,
        # lexical ASR and the semantic controller may commit in the same hop.
        log.info('Reply cancellation: source=%s, before_audio=%s',event.get('source','manual'),getattr(self.playback,'n16',0)==0)
        self.speaking=False;self.halting=True
        if event.get('source')=='verified_stop_command' and self.speaker_ext is not None:
            self.speaker_ext.remember_stop((self.current_speaker or {}).get('speaker_id'),event.get('stop_text',''),hop_end/self.ears.sr)
        self.pending_barge_proposal=None
        await self.ws.send_json({'type':'reply_cancelled','reply_id':self.playback.rid})
        generated=self.mouth_done
        if generated and self.have_mouth:await self.to_mouth(b'H',{'interrupted':True})
        if getattr(self,'calibrating',False):
            self.calibrating=False;self.cal_mic=[];self.cal_ref=[]
            if getattr(self,'calibration_task',None) is not None:self.calibration_task.cancel()
        await super()._on_barge(event,hop_end)
        onset=max(0.,min(2.,float(event.get('onset_s') or 0.)))
        samples=int(onset*16000)
        self.identity_audio=self.identity_audio[-samples:] if samples else np.zeros(0,np.float32)
        self.identity.reset_utterance()
        if self.teacher_stream is not None:self.teacher_stream.reset()
        self.stop_feedback.schedule(event)

    async def _on_browser(self,msg):
        if msg.type==WSMsgType.BINARY:
            block=mic_block(msg.data)
            if block is not None:
                pcm=block[2];self.input_samples+=len(pcm);self.input_blocks+=1
                if getattr(self,'sentence_stream',None):
                    rid,at24,_=block;reference=None
                    if rid==self.playback.rid and at24>=0:
                        at=round(at24*16000/24000);reference=np.zeros(len(pcm),np.float32)
                        count=max(0,min(len(pcm),self.playback.n16-at))
                        if count:reference[:count]=self.playback.audio[at:at+count]
                    await self.sentence_stream.microphone(pcm,self.input_samples/16000,reference)
                if getattr(self,'environment',None):self.environment.feed(pcm,self.input_samples/16000,playback=block[1]>=0)
                now=time.monotonic()
                if now-self.last_input_notice>.5:
                    self.last_input_notice=now
                    await self.ws.send_json({'type':'input_status','blocks':self.input_blocks,
                        'seconds':round(self.input_samples/16000,2),
                        'rms_dbfs':round(20*np.log10(max(1e-7,float(np.sqrt(np.mean(pcm**2))))),1)})
        if getattr(self,'calibrating',False) and msg.type==WSMsgType.BINARY:
            block=mic_block(msg.data)
            if block is not None:
                rid,at24,pcm=block;ref=np.zeros(len(pcm),np.float32)
                if rid==self.playback.rid and at24>=0:
                    at=round(at24*16000/24000);length=max(0,min(len(pcm),self.playback.n16-at))
                    ref[:length]=self.playback.audio[at:at+length]
                self.cal_mic.append(pcm);self.cal_ref.append(ref)
        await super()._on_browser(msg)

    async def _on_command(self,data):
        if data in ('identity:background-filter:on','identity:background-filter:off') and self.sentence_stream:
            self.sentence_stream.addressed.enabled=data.endswith(':on')
            await self.ws.send_json(self.sentence_stream.settings());return
        if data=='interrupt' and getattr(self,'environment',None) and self.environment.listening:
            self.environment.epoch+=1;self.environment.pending=None
            await self.ws.send_json({'type':'environment_observation','kind':'requested_sound','status':'cancelled'})
            return
        if data.startswith('environment:'):
            try:
                if len(data)>1000:raise ValueError('Environment command too long')
                message=json.loads(data[12:])
                if not isinstance(message,dict):raise ValueError('Expected settings')
                if message.get('welcome') is True:self.environment.welcome()
                else:await self.environment.configure(message)
            except (ValueError,TypeError):await self.ws.send_json({'type':'warning','text':'Invalid environment settings'})
            return
        if data.startswith('text:'):self.pending_input_timing={'kind':'typed'}
        if data.startswith('memory:'):
            try:
                if len(data)>5000:raise ValueError('Memory command too long')
                command=json.loads(data[7:])
                if not isinstance(command,dict):raise ValueError('Memory command must be an object')
                await self.to_mouth(b'M',command)
            except (ValueError,TypeError,json.JSONDecodeError):
                await self.ws.send_json({'type':'memory_status','state':'error','error':'Invalid memory command'})
            return
        if data in ('reply:quick:on','reply:quick:off'):
            self.quick_replies=data.endswith(':on')
            await self.ws.send_json({'type':'quick_reply_settings','enabled':self.quick_replies});return
        if self.sentence_stream and data.startswith('profiles:'):
            try:
                from social_profiles import export_profiles,import_profiles
                if len(data)>384000:raise ValueError('Profiles too large')
                command=json.loads(data[9:])
                if not isinstance(command,dict):raise ValueError('Expected profile command')
                if command.get('action')=='export' and command.get('consent') is True:
                    await self.ws.send_json({'type':'speaker_profiles','profiles':export_profiles(self.sentence_stream.memory)})
                elif command.get('action')=='import' and command.get('consent') is True:
                    import_profiles(self.sentence_stream.memory,command.get('profiles'))
                    await self.ws.send_json(self.sentence_stream.identity_state())
                elif command.get('action')=='forget':
                    self.sentence_stream.clear_identity();self.current_speaker=None;self.visual.forget()
                    await self.ws.send_json(self.sentence_stream.identity_state())
                else:raise ValueError('Consent required')
            except (ValueError,TypeError,KeyError):
                await self.ws.send_json({'type':'warning','text':'Speaker profile update rejected; check explicit consent and profile format.'})
            return
        if data.startswith('llm:'):
            await self.llm.command(data);return
        if self.sentence_stream and data in ('captions:on','captions:off'):
            self.sentence_stream.captions_enabled=data.endswith(':on')
            await self.ws.send_json(self.sentence_stream.settings());return
        if self.sentence_stream and data.startswith('asr:model:'):
            await self.sentence_stream.asr.select(data.split(':',2)[2]);return
        if self.sentence_stream and data.startswith('asr:pause_ms:'):
            try:self.sentence_stream.set_pause(float(data.split(':',2)[2]))
            except ValueError:
                await self.ws.send_json({'type':'warning','text':'Speech pause must be between 200 and 2000 ms.'});return
            await self.ws.send_json(self.sentence_stream.settings());return
        if self.sentence_stream and data.startswith('identity:'):
            command=data[9:]
            if command=='clear':
                self.sentence_stream.clear_identity();self.current_speaker=None
                await self.decisions.clear();await self.decisions.notice()
            if command=='focus:release':
                if self.speaking:await self._on_barge({'yield_prob':1.,'onset_s':0.,'source':'manual_handover'},self.hop_pos)
                self.sentence_stream.invite_speaker()
            if command.startswith('focus:enabled:'):
                self.sentence_stream.focus.enabled=command.endswith(':on');self.sentence_stream.focus_activity=None
            if command.startswith('focus:threshold:'):
                try:
                    value=float(command.split(':')[-1])
                    if not math.isfinite(value) or not .20<=value<=.65:raise ValueError()
                    self.sentence_stream.focus.threshold=value
                except ValueError:pass
            if command.startswith('timeout:'):
                try:self.sentence_stream.memory.timeout_s=max(0,min(3600,float(command.split(':',1)[1])))
                except ValueError:pass
            if command.startswith('utterance_threshold:'):
                try:self.sentence_stream.memory.threshold=max(.25,min(.9,float(command.split(':',1)[1])))
                except ValueError:pass
            await self.ws.send_json(self.sentence_stream.identity_state())
            return
        if data=='avatar:cancel':
            if self.avatar_optional and self.speaking:
                self.avatar_optional=False
                await self._on_barge({'onset_s':0.,'yield_prob':1.},self.hop_pos)
            return
        if data.startswith('avatar:'):
            try:
                if len(data)>600:raise ValueError('Avatar message too large')
                event=json.loads(data[7:])
                context=self.avatar_personality.observe(event,busy=self.speaking or self.halting or self.ending,language=self.decisions.language)
                accepted=bool(context and self.have_mouth)
                # Claim the optional turn before yielding to the ASR hop task.
                if accepted:self._start_speaking();self.avatar_optional=True
                await self.ws.send_json({'type':'avatar_event','event':event.get('event'),'accepted':accepted})
                if accepted:
                    await self.to_mouth(b'T',{'transcript':'','speaker':{},'avatar_context':context,**self.llm.snapshot()})
            except (ValueError,TypeError,KeyError):
                await self.ws.send_json({'type':'avatar_event','accepted':False,'reason':'Invalid or unavailable avatar event'})
            return
        if data.startswith('visual:'):
            try:
                if len(data)>160000:raise ValueError('Visual message too large')
                message=json.loads(data[7:]);kind=message.get('type')
                if kind=='settings':self.visual.configure(message)
                elif kind=='frame':self.visual.observe(message,self.mic_pos/self.ears.sr)
                elif kind=='forget':
                    self.visual.forget()
                    if self.sentence_stream:self.sentence_stream.clear_identity();self.current_speaker=None
                    if self.speaker_ext is not None:await self.speaker_ext.clear();self.current_speaker=None
                elif kind=='identify':
                    if not self.visual.settings['faceIdEnabled']:return
                    from visual_identity import TOKEN
                    track_id=message.get('track_id','');pid=message.get('person_id','')
                    if not isinstance(track_id,str) or not TOKEN.fullmatch(track_id) or pid not in self.visual.people:raise ValueError('Unknown person')
                    self.visual.bindings[track_id]=self.visual.people[pid]
                    age=message.get('embedding_age_ms')
                    if type(age) in (int,float) and np.isfinite(age) and 0<=age<=5000:self.visual.face_verified[track_id]=time.monotonic()-age/1000
                elif kind=='export' and (message.get('consent') is True or message.get('session') is True):
                    profiles=[]
                    if self.sentence_stream:
                        for person in self.visual.people.values():
                            user=self.sentence_stream.memory.users.get(person['speaker_id'])
                            if user and user.get('vectors'):
                                profiles.append({**person,'schema':'sentence-ecapa-v1','vectors':[v.tolist() for v in user['vectors'][:8]]})
                    if self.speaker_ext is not None:
                        for person in self.visual.people.values():
                            user=self.speaker_ext.memory.users.get(person['speaker_id'])
                            if user is not None and user.teacher is not None:
                                profiles.append({**person,'fast':user.fast.tolist(),'long':user.long.tolist() if user.long is not None else None,'teacher':user.teacher.tolist()})
                    await self.ws.send_json({'type':'visual_profiles','profiles':profiles})
                elif kind=='import' and (message.get('consent') is True or message.get('session') is True):
                    from identity import unit
                    profiles=message.get('profiles',[])
                    if not isinstance(profiles,list) or len(profiles)>8:raise ValueError('Invalid profiles')
                    checked=[]
                    for p in profiles:
                        if not isinstance(p,dict) or not isinstance(p.get('name'),str) or not 1<=len(p['name'])<=60:raise ValueError('Invalid profile')
                        if not __import__('visual_identity').TOKEN.fullmatch(str(p.get('person_id',''))):raise ValueError('Invalid person ID')
                        if self.sentence_stream and p.get('schema')=='sentence-ecapa-v1':
                            refs=[np.asarray(v,np.float32) for v in p.get('vectors',[])]
                            if not 1<=len(refs)<=8 or any(v.shape!=(192,) or not np.isfinite(v).all() or not .5<np.linalg.norm(v)<1.5 for v in refs):raise ValueError('Invalid voice prototypes')
                            if p['person_id'] not in self.visual.people:
                                context=self.sentence_stream.memory.assign(refs[0],'Ich bin '+p['name']+'.')
                                self.sentence_stream.memory.users[context['speaker_id']]['vectors']=[v/np.linalg.norm(v) for v in refs]
                                self.visual.people[p['person_id']]={'person_id':p['person_id'],'speaker_id':context['speaker_id'],'name':p['name']}
                            continue
                        checked.append((p,unit(p['fast']),unit(p['long']) if p.get('long') is not None else None,unit(p['teacher'])))
                    if self.speaker_ext is not None:
                        for p,fast,long,teacher in checked:
                            if p['person_id'] in self.visual.people:continue
                            if len(self.speaker_ext.memory.users)>=self.speaker_ext.memory.max_users:break
                            user,_=self.speaker_ext.memory.enroll(fast,long,p['name'],force_new=True)
                            user.teacher=teacher;user.teacher_anchor=teacher.copy()
                            self.visual.people[p['person_id']]={'person_id':p['person_id'],'speaker_id':user.id,'name':p['name']}
                        await self.speaker_ext.notice()
                else:raise ValueError('Invalid visual command')
                if kind!='frame':await self.ws.send_json(self.visual.state())
            except (ValueError,TypeError,KeyError,AttributeError) as exc:
                clock_error=str(exc)=='Visual sample clock is not aligned to microphone'
                await self.ws.send_json({'type':'visual_warning','text':
                    'Camera/microphone timing mismatch. Restart the camera or use voice recognition without camera filtering.'
                    if clock_error else 'Invalid visual identity update ignored.'})
            return
        if data.startswith('interaction:'):
            command=data.split(':',1)[1]
            if command in ('on','off'):await self.decisions.toggle(command=='on')
            elif command in ('gestures:on','gestures:off'):
                self.decisions.listening_gestures=command.endswith(':on')
            elif command in ('stop-feedback:on','stop-feedback:off'):
                self.stop_feedback.enabled=command.endswith(':on')
                if not self.stop_feedback.enabled:self.stop_feedback.cancel()
                await self.ws.send_json({'type':'stop_feedback_setting','enabled':self.stop_feedback.enabled})
            elif command in ('helpful:on','helpful:off'):
                self.decisions.helpful=command=='helpful:on';self.decisions.epoch+=1
                if self.decisions.task:self.decisions.task.cancel()
                await self.decisions.notice()
            elif command=='clear':await self.decisions.clear();await self.decisions.notice()
            return
        if data.startswith('identity:'):
            command=data.split(':',1)[1]
            if command.startswith('mode:v4'):
                mode=command.split(':',1)[1]
                if mode not in ('v4','v4-long','v4-long3'):return
                from speaker_runtime import available
                if mode=='v4-long3' and 'long3' not in available():
                    await self.ws.send_json({'type':'warning','text':'The 3.04-s optimized study has not finished; no winner is claimed yet.'});return
                if self.speaker_ext is not None:await self.speaker_ext.close()
                await self.decisions.clear();await self.decisions.notice()
                from speaker_session import SpeakerSession
                self.current_speaker=None;self.speaker_ext=SpeakerSession(self,None if mode=='v4' else 'long3' if mode=='v4-long3' else 'long',use_diarization=mode!='v4')
                self.speaker_ext.start();self.identity_mode=mode;self.identity_enabled=True
                self._speaker_barge_sensitivity(True)
                await self.speaker_ext.notice();return
            if self.speaker_ext is not None and self.identity_mode.startswith('v4'):
                if command in ('open:on','open:off'):
                    self.speaker_ext.open_enrollment=command=='open:on'
                    await self.speaker_ext.notice();return
                if command in ('on','off'):
                    await self.decisions.clear();await self.decisions.notice()
                    self.identity_enabled=command=='on';await self.speaker_ext.clear();self.current_speaker=None
                    self._speaker_barge_sensitivity(self.identity_enabled)
                    self.session.reset();await self.speaker_ext.notice();return
                if command=='clear':
                    await self.decisions.clear()
                    await self.decisions.notice();await self.speaker_ext.clear();self.current_speaker=None;await self.speaker_ext.notice();return
                if command.startswith('timeout:'):
                    try:
                        value=float(command.split(':',1)[1])
                        if np.isfinite(value) and 0<=value<=3600:self.speaker_ext.memory.timeout_s=value
                    except ValueError:pass
                    await self.speaker_ext.notice();return
                if command.startswith('threshold:'):
                    try:
                        value=float(command.split(':',1)[1])
                        if np.isfinite(value) and .1<=value<=.95:self.speaker_ext.memory.threshold=value
                    except ValueError:pass
                    await self.speaker_ext.notice();return
                if command.startswith('utterance_threshold:'):
                    try:
                        value=float(command.split(':',1)[1])
                        if np.isfinite(value) and .3<=value<=.9:self.speaker_ext.utterance_threshold=value
                    except ValueError:pass
                    await self.speaker_ext.notice();return
            if command.startswith('mode:') and command.split(':')[1] in ('ecapa','student'):
                await self.decisions.clear();await self.decisions.notice()
                if self.speaker_ext is not None:await self.speaker_ext.close();self.speaker_ext=None
                self._speaker_barge_sensitivity(False)
                self.current_speaker=None
        if data.startswith('identity:'):
            command=data.split(':',1)[1]
            if command=='on':
                self._reset_identity_context()
                if self.identity_mode=='ecapa':
                    try:await self._teacher();self.identity_enabled=True
                    except Exception as exc:await self.ws.send_json({'type':'warning','text':'ECAPA local weights unavailable; filter remains off ('+type(exc).__name__+').'})
                else:await self.ws.send_json({'type':'warning','text':'The TYGER V2 pilot failed held-out reliability checks. Diagnostic cues only; use ECAPA to try enrollment.'})
            elif command=='off':self.identity_enabled=False;self._reset_identity_context()
            elif command=='clear':
                self.identity.clear();self.identity_audio=np.zeros(0,np.float32)
                if self.teacher_stream is not None:self.teacher_stream.reset()
                if self.predictor is not None:self.predictor.reset()
            elif command.startswith('mode:') and command.split(':')[1] in ('ecapa','student'):
                self.identity_mode=command.split(':')[1];self.identity_enabled=False;self.identity.clear()
                self._reset_identity_context()
                self.identity.threshold=self.teacher_threshold if self.identity_mode=='ecapa' else self.student_threshold
            elif command.startswith('timeout:'):
                try:
                    timeout=float(command.split(':')[1])
                    if np.isfinite(timeout) and 0<=timeout<=3600:self.identity.timeout_s=timeout
                except ValueError:pass
            await self.ws.send_json({'type':'identity','enabled':self.identity_enabled,'mode':self.identity_mode,**self.identity.status()})
        elif data=='calibrate' and not self.speaking:
            self.calibrating=True;self.cal_mic=[];self.cal_ref=[];self._start_speaking()
            calibration_rid=self.playback.rid
            t=np.arange(24000*.45)/24000
            probe=(.025*np.sin(2*np.pi*(250*t+1600*t*t))*np.sin(np.pi*t/.45)**2).astype(np.float32)
            await self._on_reply_audio(probe.tobytes())
            async def finish_calibration():
                await asyncio.sleep(2.5)
                if not self.calibrating or self.playback.rid!=calibration_rid:return
                self.calibrating=False
                report=estimate_echo_delay(np.concatenate(self.cal_mic) if self.cal_mic else [],np.concatenate(self.cal_ref) if self.cal_ref else [])
                self.cal_mic=[];self.cal_ref=[]
                if not self.ws.closed:
                    await self.ws.send_json({'type':'calibration',**report})
                    await self._on_done(b'{}')
            self.calibration_task=asyncio.create_task(finish_calibration())
        elif data=='interrupt' and self.speaking:
            await self._on_barge({'onset_s':0.,'yield_prob':1.},self.hop_pos)
        elif data.startswith('text:') and not self.speaking and not self.halting and not (getattr(self,'typed_turn_task',None) and not self.typed_turn_task.done()):
            text=data[5:].strip()[:2000]
            if text:
                self.current_speaker=None
                from ambient_audio import sound_question
                if sound_question(text):
                    # Keep accepting binary microphone packets while the
                    # explicitly requested after-question window is open.
                    self.typed_turn_task=asyncio.create_task(self._send_turn(text))
                else:await self._send_turn(text)
        elif data.startswith('text:'):await self.ws.send_json({'type':'busy','text':'Please stop or finish this reply before sending another typed turn.'})
        else:await super()._on_command(data)

    async def _finish_turn(self,barged):
        if getattr(self,'decisions',None):self.decisions.finish_interjection(self.hop_pos)
        self.avatar_optional=False
        if getattr(self,'environment',None):self.environment.active_reply=None
        offset=self.playback.offset_ms()
        await super()._finish_turn(barged)
        if not barged:
            self.identity_audio=np.zeros(0,np.float32);self.identity.reset_utterance()
            if self.predictor is not None:self.predictor.reset()
            if self.teacher_stream is not None:self.teacher_stream.reset()
        await self.ws.send_json({'type':'sync','arrival_order_error_ms':offset,
            'method':'capture timestamps + actual scheduled PCM sample positions',
            'hardware_accuracy':'not independently measured; inspect device latency availability'})


async def startup(app):
    import torch
    torch.set_num_threads(4)
    bridge=MouthBridge();app['bridge']=bridge;app['tasks']=[]
    app['mouth_server']=await asyncio.start_server(bridge.accept,'127.0.0.1',8795)
    cfg={'server':{'max_sessions':1,'max_session_s':int(os.environ.get('ROBO_MAX_SESSION_S','1800')),'mouth_host':'127.0.0.1','mouth_port':8795},'mouth':{}}
    app['cfg']=cfg
    from llm_backends import refresh_providers
    async def providers():
        try:await refresh_providers()
        except Exception:log.warning('OpenRouter provider catalogue unavailable; retaining Cerebras and local options')
    app['tasks'].append(asyncio.create_task(providers()))
    async def load():
        try:
            block={'plugin':'tyger','model':str(paths.BUNDLE),'ep_thr':.915,'ep_deb':2,
                'barge_thr':.5,'barge_deb':2,'bc_thr':.4,'bc_deb':2,'cpu_threads':4}
            STATE['ears']=await asyncio.to_thread(Tyger,block,PluginContext(None,cfg,'robo-ears'))
            attach_feature_tap(STATE['ears'])
            if os.environ.get('ROBO_SENTENCE_STREAM')=='1':
                from pause_vad import get_model
                await asyncio.to_thread(get_model)
                from speaker_teacher import get_cached_teacher
                await asyncio.to_thread(lambda:get_cached_teacher().embed(np.ones(16000,np.float32)*.01))
            STATE.update(ready=True,phase='Ready: Small RL + Nemotron 3.5 + TYGER')
            app['tasks'].append(asyncio.create_task(bridge.prepare_ack()))
        except Exception as exc:
            log.exception('EARS startup failed');STATE.update(error=type(exc).__name__+': '+str(exc),phase='EARS error')
    app['load_task']=asyncio.create_task(load())


async def cleanup(app):
    for task in app['tasks']:task.cancel()
    app['mouth_server'].close();await app['mouth_server'].wait_closed()


async def state(request):
    import torch
    trained=(ROOT/'speaker_v2/artifacts/best.pt').exists()
    try:
        from speaker_teacher import resolve_model_root
        resolve_model_root();teacher=True
    except Exception:teacher=False
    from speaker_runtime import available
    students=available()
    failure=audio_failure(STATE.get('ears'))
    status={k:v for k,v in STATE.items() if k!='ears'}
    if failure:status.update(ready=False,error=failure,phase='Audio GPU worker unavailable')
    from llm_backends import default_selection,MODELS
    chosen=default_selection()
    return web.json_response(status|{
        'audio_gpu_serialized':bool(STATE.get('ears') and getattr(STATE['ears'].service.engine,'robo_gpu_serialized',False)),
        'audio_worker_error':failure,
        'llm':MODELS[chosen['model']]['model'],
        'llm_default':chosen,
        'sentence_stream':os.environ.get('ROBO_SENTENCE_STREAM','0')=='1',
        'llm_selection_available':True,
        'reply_language_default':'de','asr_model_selection':True,
        'asr_default_model':__import__('asr_catalog').default_model(),'camera_auto_default':False,'voice_caption_default':True,'auto_start_default':True,'environment_default':True,
        'sound_caption_model':'laion/sound-effect-captioning-whisper','sound_caption_device':'cuda:0',
        'voice_caption_model':'Humaneness Ears Medium + procedural-voice-captions (GPU, parallel to ASR)',
        'speech_pause_default_ms':float(os.environ.get('ROBO_SPEECH_PAUSE_MS','700')),
        'pause_detector':'Silero ONNX CPU neural speech activity',
        'tts':'Humaneness Voice Small 1.1 Alpha RL r128 v1','tts_shared':True,
        'privacy':'Camera inference stays in your browser. Audio and voice matching run on this server. OpenRouter mode sends conversation text to OpenRouter and the selected provider; local mode keeps it on this server. No microphone recordings are saved.',
        'identity_model':'V4 layer-mix 320-ms + session enrollment; experimental, not authentication' if students else 'legacy ECAPA baseline',
        'identity_available':bool(students) or teacher,'identity_student_available':bool(students),
        'identity_models':students,'identity_quiet_seconds':1.,
        'identity_teacher_available':teacher,'identity_default_enabled':bool(students),
        'interaction_available':True,'interaction_default_enabled':True,'decision_interval_ms':320,
        'tf32_matmul':bool(torch.backends.cuda.matmul.allow_tf32),'tf32_cudnn':bool(torch.backends.cudnn.allow_tf32)})


async def asr_models(request):
    from asr_catalog import MODELS,worker_url
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response=await client.get(worker_url()+'/models');response.raise_for_status()
            return web.json_response(response.json())
    except Exception:
        return web.json_response({'active_gpu_model':None,'models':[{'id':key,'label':s['label'],'repo':s['repo'],
            'ready':key=='nemotron','phase':'resident' if key=='nemotron' else 'worker_unavailable'} for key,s in MODELS.items()]})


async def llm_models(request):
    from llm_backends import catalog,refresh_providers
    try:await refresh_providers()
    except Exception:pass
    return web.json_response(catalog(),headers={'Cache-Control':'no-store'})


async def socket(request):
    synthetic_diagnostics=request.query.get('synthetic_diagnostics')=='1'
    if synthetic_diagnostics:local_only(request)
    failure=audio_failure(STATE.get('ears'))
    if not STATE['ready'] or failure:raise web.HTTPServiceUnavailable(text=failure or STATE['error'] or STATE['phase'])
    ears=STATE['ears'];ears.browser_waiting=getattr(ears,'browser_waiting',0)+1;sess=None
    try:
        for _ in range(100):
            sess=ears.open_session(ears.defaults)
            if sess is not None:break
            await asyncio.sleep(.02)
    finally:ears.browser_waiting-=1
    if sess is None:raise web.HTTPTooManyRequests(text='One microphone session at a time on this pilot.')
    ws=web.WebSocketResponse(max_msg_size=65536,heartbeat=30)
    try:
        await ws.prepare(request)
        conv=RoboConversation(ws,ears,sess,request.app['cfg'],voices=['emolia_c1699'],
            default_voice='emolia_c1699',labels={'emolia_c1699':'Mira · fixed reference'})
        conv.synthetic_diagnostics=synthetic_diagnostics
        conv.synthetic_stop_audio=synthetic_diagnostics and request.query.get('synthetic_stop_audio')=='1'
        await conv.run()
    finally:ears.close(sess)
    return ws


async def index(request):return web.FileResponse(ROOT/'static/index.html')


async def desk_robot(request):
    # Share the actual conversation UI, not a second diverging chat client.
    html=(ROOT/'static/index.html').read_text()
    html=html.replace('<body>','<body data-robot-view="desk3d">')
    html=html.replace('<title>Robby · a little more human</title>','<title>Robby · 3D desk robot prototype</title>')
    html=html.replace('href="/desk-robot/">3D desk robot ↗','href="/">Original avatar ↗')
    html=html.replace('A FACE. A VOICE. A CONVERSATION.','TWO AXES. FORTY EXPRESSIONS. ONE LITTLE ROBOT.')
    html=html.replace('<h1>A little company.<br><span>A little expression.</span></h1>',
        '<h1>A little desk companion.<br><span>Made to turn towards you.</span></h1>')
    html=html.replace('id="avatar" class="avatar"','id="avatar" class="avatar desk-avatar"')
    html=html.replace('smoothly varying brows, eyelids, gaze, mouth, head and arms.',
        'smoothly varying brows, eyelids, gaze and mouth. This construction has pan and tilt only, no arms or roll axis.')
    html=html.replace('Show the robot’s 101 animation anchors','Show display anchors and mechanical axes')
    return web.Response(text=html,content_type='text/html',headers={'Cache-Control':'no-store'})


async def validate(request):
    local_only(request)
    if not STATE['ready'] or STATE['ears'].active():raise web.HTTPConflict(text='EARS must be ready and idle')
    from sync_validation import model_checks
    return web.json_response(await model_checks(STATE['ears']))


async def features(request):return await extract(request,STATE['ears'])


async def context_features(request):return await extract_context(request,STATE['ears'])


async def layer_features(request):return await extract_layers(request,STATE['ears'])


async def live_speaker_features(request):
    from speaker_live.research import extract_predictions
    return await extract_predictions(request,STATE['ears'])


async def asr_check(request):return await transcribe_check(request,STATE['ears'])


async def v4_status(request):
    path=ROOT/'speaker_v4/progress.json'
    return web.json_response(json.loads(path.read_text()) if path.exists() else {'phase':'Preparing'},headers={'Cache-Control':'no-store'})


async def v4_report(request):
    path=Path.home()/'.cache/robo-v4/report.html'
    return web.FileResponse(path) if path.exists() else web.Response(text='Study preparing; please reload shortly.')


async def v5_status(request):
    path=ROOT/'speaker_v5/progress.json'
    return web.json_response(json.loads(path.read_text()) if path.exists() else {'phase':'Queued'},headers={'Cache-Control':'no-store'})


async def v5_report(request):
    path=Path.home()/'.cache/robo-v5/report.html'
    return web.FileResponse(path) if path.exists() else web.Response(text='Context study preparing; please reload shortly.')


async def training_status(request):
    path=ROOT/'speaker_v3/progress.json'
    data=json.loads(path.read_text()) if path.exists() else {'phase':'Not started'}
    data.pop('results_path',None)
    return web.json_response(data,headers={'Cache-Control':'no-store'})


async def training_page(request):return web.FileResponse(ROOT/'static/speaker-training.html')


async def large_training_report(request):
    path=Path(os.environ.get('ROBO_V3_WORK',str(Path.home()/'.cache/robo-v3')))/'models/report.html'
    if not path.exists():return web.HTTPFound('/speaker-training/')
    return web.FileResponse(path)


async def expressions(request):return web.FileResponse(ROOT/'static/expressions.html')


async def speaker_report(request):
    variant=request.match_info.get('variant')
    path=ROOT/'speaker_v2'/variant/'report.html' if variant else ROOT/'speaker_v2/report.html'
    if not path.exists():raise web.HTTPServiceUnavailable(text='Pilot report is still being generated')
    return web.FileResponse(path)


async def release_download(request):
    path=ROOT/'speaker_v2/release/tyger-v2-public-pilot.zip'
    if not path.exists():raise web.HTTPNotFound()
    return web.FileResponse(path,headers={'Content-Disposition':'attachment; filename="tyger-v2-public-pilot.zip"'})


async def source_download(request):
    path=ROOT/'releases/robby-source.zip'
    if not path.exists():raise web.HTTPNotFound()
    return web.FileResponse(path,headers={'Content-Disposition':'attachment; filename="robby-source.zip"'})


def main():
    app=web.Application(client_max_size=2*1024*1024)
    app.router.add_get('/',index);app.router.add_get('/robo/',index)
    app.router.add_get('/desk-robot/',desk_robot)
    app.router.add_get('/desk-robot',lambda request:web.HTTPFound('/desk-robot/'))
    app.router.add_get('/api/state',state);app.router.add_get('/ws',socket)
    app.router.add_post('/internal/sync-check',validate)
    app.router.add_post('/internal/features',features)
    app.router.add_post('/internal/context-features',context_features)
    app.router.add_post('/internal/layer-features',layer_features)
    app.router.add_post('/internal/speaker-predictions',live_speaker_features)
    app.router.add_post('/internal/transcribe',asr_check)
    app.router.add_get('/api/speaker-v4',v4_status)
    app.router.add_get('/speaker-training/320ms-study/',v4_report)
    app.router.add_get('/api/speaker-v5',v5_status)
    app.router.add_get('/speaker-training/context-study/',v5_report)
    app.router.add_get('/api/speaker-training',training_status)
    app.router.add_get('/speaker-training/',training_page)
    app.router.add_get('/speaker-training/report.html',large_training_report)
    app.router.add_get('/expressions/',expressions)
    app.router.add_get('/speaker-study/',speaker_report)
    app.router.add_get('/speaker-study/{variant:results|results_centered|results_v2_raw|results_v2_centered|results_v2_gru}/report.html',speaker_report)
    app.router.add_get('/downloads/tyger-v2-public-pilot.zip',release_download)
    app.router.add_get('/downloads/robby-source.zip',source_download)
    app.router.add_get('/api/asr/models',asr_models)
    app.router.add_get('/api/llm/models',llm_models)
    app.router.add_static('/static/',ROOT/'static')
    app.on_startup.append(startup);app.on_cleanup.append(cleanup)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)s %(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    web.run_app(app,host='127.0.0.1',port=8793,access_log=None)


if __name__=='__main__':main()
