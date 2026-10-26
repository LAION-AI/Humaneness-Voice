"""Adapt sentence-stream acoustic memory to the existing live controller.

No second recognizer, identity store or decision policy. The same current
speaker, waveform, teacher, student and playback guard serve both turn modes.
"""
import time
from types import SimpleNamespace
import numpy as np


class SentenceDecisionIdentity:
    def __init__(self,session):
        self.s=session
        self.live_verification={}
        self.interrupt_checked_at=-100.
        self.cached=None

    @property
    def memory(self):
        return SimpleNamespace(users={sid:SimpleNamespace(name=u['name']) for sid,u in self.s.memory.users.items()})

    @property
    def turns(self):
        s=self.s;sid=s.focus.owner
        return SimpleNamespace(tracks={sid:SimpleNamespace(sid=sid,start=s.begin or s.segment_floor,
            voiced_start=s.begin,last_speech=s.last_speech)} if sid else {})

    def live_speech_recent(self,hop_end):
        s=self.s;at=hop_end/s.c.ears.sr
        return at-max(s.last_speech,s.word_at)<=.96 and at>=s.handover_floor

    def stop_fast_vote(self,start,end):
        s=self.s;sid=s.focus.owner or (s.c.current_speaker or {}).get('speaker_id')
        score=s.fast_match(end,sid) if sid else None
        # This only admits a semantic decision, never cancels playback.
        # The final waveform/echo/rival guard still has to accept the Stop.
        return {'candidate_id':sid,'candidate_similarity':score if score is not None else .30}

    async def confirm_interrupt(self,proposal,hop_end):
        s=self.s;at=hop_end/s.c.ears.sr;rid=s.c.playback.rid;owner=s.focus.owner
        result=await s.assess_interrupt(proposal,at,rid)
        self.live_verification=result
        if result['accepted'] and owner==s.focus.owner and rid==s.c.playback.rid:
            target=result.get('speaker_id') or owner or (s.c.current_speaker or {}).get('speaker_id')
            if target in s.memory.users:
                s.c.current_speaker={'speaker_id':target,'display_name':s.memory.users[target]['name'],'enrolled':True}
            return target
        return None

    async def confirm_live_speaker(self,hop_end):
        import asyncio
        from speaker_teacher import get_cached_teacher,enrollment_reference
        s=self.s;at=hop_end/s.c.ears.sr
        if not self.live_speech_recent(hop_end):return None
        owner=s.focus.active(s.memory.users,s.c.speaking)
        epoch=s.admission_epoch
        if self.cached and self.cached[0]==owner and self.cached[1]==epoch and 0<=at-self.cached[2]<=.48:
            return self.cached[3]
        wave=s.audio(max(s.segment_floor,s.begin if s.begin is not None else at-3.04,at-3.04),at)
        if len(wave)<10240:return None
        cached=s.prefetched
        vector=(cached.get('embedding') if cached and cached['epoch']==s.epoch and 0<=at-cached['at']<=.96 else None)
        try:
            if vector is None:vector=await asyncio.to_thread(get_cached_teacher().embed,enrollment_reference(wave))
        except Exception:return None
        if epoch!=s.admission_epoch or owner!=s.focus.active(s.memory.users,s.c.speaking):return None
        fast=s.fast_vector(max(s.segment_floor,at-3.04),at)
        from sentence_session import voice_level
        level=voice_level(wave)
        decision=s.turn_decision(vector,fast,level)
        self.live_verification={'teacher':decision.get('similarity'),'required':s.focus.threshold,
            'fast':decision.get('fast_similarity'),'long':None,'reason':decision['reason']}
        if not decision['accepted']:return None
        target=decision.get('speaker_id') or owner
        if owner is None:
            context=s.memory.assign(vector,s.preview)
            s.focus.accept(context['speaker_id'])
            if context['speaker_id'] in s.memory.users:s.memory.users[context['speaker_id']]['level']=level
            if fast is not None:s.fast_anchors.setdefault(context['speaker_id'],[]).append(fast.copy())
            await s.c.ws.send_json(s.identity_state())
        else:
            u=s.memory.users[target]
            context={'speaker_id':target,'display_name':u['name'],'named':u['named'],'enrolled':True,
                     'similarity':decision.get('similarity'),'conversation_focus':s.focus.state(s.memory.users)}
        self.cached=(s.focus.owner,epoch,at,context)
        return context

    def reset_after_stop(self):
        self.s.c.session.reset()
        self.s.reset_segment()
