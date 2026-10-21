import unittest,asyncio
from collections import deque
from unittest.mock import patch,AsyncMock,Mock
from types import SimpleNamespace
import numpy as np
from speaker_turns import SpeakerMemory,SpeakerTurns,Track
from speaker_session import SpeakerSession

class RefineTests(unittest.IsolatedAsyncioTestCase):
    def test_new_session_defaults_to_requested_threshold_and_three_second_context(self):
        tensor=SimpleNamespace(detach=lambda:SimpleNamespace(numpy=lambda:np.zeros(192,np.float32)))
        predictor=SimpleNamespace(frames=38,use_long=True,fast=SimpleNamespace(teacher_mean=tensor))
        with patch('speaker_session.LivePredictor',return_value=predictor):
            session=SpeakerSession(SimpleNamespace(),use_diarization=False)
        self.assertEqual(session.utterance_threshold,.32)
        self.assertEqual(session.identity_context_s,3.04)
        self.assertEqual(session.turns.quiet_s,1.)
    def setUp(self):
        self.a,self.b=np.eye(192,dtype=np.float32)[:2]
        self.s=SpeakerSession.__new__(SpeakerSession);self.s.memory=SpeakerMemory();self.s.turns=SpeakerTurns(self.s.memory);self.s.notice=AsyncMock()
        self.s.utterance_threshold=.5;self.s.utterance_margin=.08
        self.s.records=deque();self.s.owners=deque();self.s.overlaps=deque()
        self.s.asr_wakes=deque()
        self.s.background_channels=set();self.s.channel_levels={}
        self.christoph,_=self.s.memory.enroll(self.a,name='Christoph');self.christoph.teacher=self.a
    async def test_independent_activity_veto_is_cached_and_has_no_enrollment(self):
        s=self.s;s.epoch=0;s.closed=False;s.activity_checked_at=-100.;s.activity_scores={}
        s.audio_between=lambda a,b:np.ones(round((b-a)*16000),np.float32)*.04
        teacher=Mock(return_value=self.a)
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=teacher)):
            await s.check_activity(2.);await s.check_activity(2.3)
        self.assertTrue(s.activity_scores[self.christoph.id]);self.assertEqual(teacher.call_count,1)
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda q:self.b)):
            await s.check_activity(2.7)
        self.assertFalse(s.activity_scores[self.christoph.id]);self.assertEqual(len(s.memory.users),1)
    async def test_stale_candidate_finishing_during_tts_never_latches_decoding(self):
        s=self.s;s.closed=False;s.c=SimpleNamespace(identity_enabled=True,speaking=True)
        s.transcribe=AsyncMock(return_value='Background words.')
        t=self.track(self.b,None);t.decoding=True
        await s.candidate(t,np.full(16000,.02,np.float32))
        self.assertFalse(t.decoding)
    def test_native_timestamp_preserves_real_continuation_but_not_quiet_room(self):
        s=self.s;t=self.track(self.a,self.christoph.id);t.start=0.;t.last_speech=2.;t.last_direct_speech=2.;t.levels.append(.08)
        s.turns.tracks[t.key]=t
        s.native_continuity({'raw_embedding':self.a},np.full(5120,.05,np.float32),2.,2.32)
        self.assertEqual(t.last_speech,2.32);self.assertEqual(t.last_direct_speech,2.)
        s.native_continuity({'raw_embedding':self.a},np.full(5120,.008,np.float32),2.32,2.64)
        self.assertEqual(t.last_speech,2.32)
        s.native_continuity({'raw_embedding':self.b},np.full(5120,.08,np.float32),2.32,2.64)
        self.assertEqual(t.last_speech,2.32)
        s.native_continuity({'raw_embedding':self.a},np.full(5120,.05,np.float32),4.,4.32)
        self.assertEqual(t.last_speech,2.32,'Native-only continuity must expire without direct activity')
    def test_background_relative_floor_does_not_assume_absolute_mic_gain(self):
        s=self.s;s.background_channels={0};s.channel_levels={0:deque([.008,.01,.012,.01])}
        self.assertAlmostEqual(s.foreground_floor(),.02)
        s.channel_levels[0]=deque([.0008,.001,.0012,.001])
        self.assertAlmostEqual(s.foreground_floor(),.002)
    def test_verified_reference_installs_native_time_aligned_fast_cue(self):
        t=self.track(self.a,self.christoph.id);t.enrollment_fast=self.b;t.enrollment_long=self.b;t.enrollment_templates=(self.a,)
        self.s.install_reference(self.christoph,self.a,t)
        np.testing.assert_array_equal(self.christoph.fast,self.b)
        np.testing.assert_array_equal(self.christoph.teacher,self.a)
    def test_strong_recent_known_voice_bridges_new_channel_without_reenrollment(self):
        s=self.s;t=self.track(self.a,self.christoph.id)
        t.start=0.;t.channel=1;t.last_direct_speech=2.;t.levels.append(.08)
        s.turns.tracks[t.key]=t;s.turns.channels[1]=t.sid
        cue={'kind':'speech','confidence':.95,'raw_embedding':self.a}
        result=s.bridge_gap(3,cue,np.full(5120,.05,np.float32),2.,2.32,np.zeros(8))
        self.assertEqual(result,1);self.assertNotIn(3,s.turns.channels)
        self.assertEqual(len(s.memory.users),1)
        self.assertIsNone(s.bridge_gap(3,{**cue,'raw_embedding':self.b},np.full(5120,.05,np.float32),2.,2.32,np.zeros(8)))
        self.assertIsNone(s.bridge_gap(3,cue,np.full(5120,.05,np.float32),4.,4.32,np.zeros(8)))
    async def test_sparse_diarizer_hops_still_reach_known_voice_verification(self):
        s=self.s;s.closed=False;s.c=SimpleNamespace(identity_enabled=True,speaking=False,ws=SimpleNamespace(send_json=AsyncMock()))
        s.transcribe=AsyncMock(return_value='Kannst du mir eine kleine Geschichte erzählen?')
        s.recover_existing_channel=AsyncMock(return_value=None)
        t=self.track(self.b,None);t.speech_s=.87;t.channel=1
        s.turns.tracks[t.key]=t
        await s.candidate(t,np.full(42720,.03,np.float32),coverage=.65)
        s.recover_existing_channel.assert_awaited_once_with(t,False,context=None)
        self.assertNotIn(1,s.background_channels,'Missing wake/context does not make dominant foreground background')
    async def test_sparse_loud_returning_voice_can_verify_raw_context_without_asr_words(self):
        s=self.s;s.closed=False;s.c=SimpleNamespace(identity_enabled=True,speaking=False,ws=SimpleNamespace(send_json=AsyncMock()))
        s.transcribe=AsyncMock(return_value='Erzählen')
        s.recover_existing_channel=AsyncMock(return_value=None)
        s.records=deque([(i*.32,(i+1)*.32,{},np.full(5120,.01,np.float32))for i in range(8)])
        t=self.track(self.b,None);t.speech_s=.88;t.channel=1;t.levels.extend([.06,.05,.07])
        s.turns.tracks[t.key]=t
        await s.candidate(t,np.full(42720,.03,np.float32),coverage=.23)
        self.assertIsNotNone(s.recover_existing_channel.await_args.kwargs['context'])
        self.assertNotIn(1,s.background_channels)
    async def test_known_prefix_crop_requires_other_voice_and_preserves_full_request(self):
        s=self.s;t=self.track(self.a,self.christoph.id);t.start=0.;t.last_speech=4.6
        wave=np.r_[np.full(25600,.008,np.float32),np.full(48000,.06,np.float32)]
        t.verification_audio=wave.copy();offsets=[0.,.64,1.28,1.6]
        views=[wave[round(o*16000):] for o in offsets]
        texts=['Background words. Please tell me a story.']*3+['Please tell me a story.']
        def embed(x):return self.b if np.mean(abs(x))<.02 else self.a
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=embed)):
            result=await s.known_prefix_view(t,wave,views,texts,'tell me a story.',offsets)
        self.assertEqual(result,('Please tell me a story.',3))
        self.assertLess(t.known_prefix_proof['prefix_cosine'],.35)
        self.assertEqual(t.precomputed_identity[0],'Please tell me a story.')
        # The same crop may not discard a genuine quiet sentence merely to
        # improve similarity or select fewer words.
        wave[:25600]=.06;views=[wave[round(o*16000):] for o in offsets]
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=embed)):
            self.assertIsNone(await s.known_prefix_view(t,wave,views,texts,'tell me a story.',offsets))
    async def test_acoustic_prefix_repair_never_overrides_clean_asr_with_room_words(self):
        s=self.s;t=self.track(self.a,self.christoph.id);t.start=0.;t.last_speech=4.6
        wave=np.r_[np.full(25600,.008,np.float32),np.full(48000,.06,np.float32)]
        t.verification_audio=wave.copy();offsets=[0.,.96,1.6]
        views=[wave[round(o*16000):] for o in offsets]
        texts=['Room words. Tell me a story.','Room words. Tell me a story.','Tell me a story.']
        def embed(x):return self.b if np.mean(abs(x))<.02 else self.a
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=embed)):
            value=await s.known_prefix_view(t,wave,views,texts,'Tell me a story.',offsets)
        self.assertTrue(value is None or value==('Tell me a story.',2))
    async def test_prefix_repair_cannot_use_old_person_reference_for_new_introduction(self):
        s=self.s;t=self.track(self.b,self.christoph.id)
        wave=np.full(48000,.04,np.float32);teacher=Mock(return_value=self.a)
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=teacher)):
            result=await s.known_prefix_view(t,wave,[wave,wave[10240:]],['Robert, wie geht es dir?']*2,
                'Hey Robby, ich bin Robert.',[0.,.64])
        self.assertIsNone(result);teacher.assert_not_called()
        self.assertFalse(hasattr(t,'precomputed_identity'))
    async def test_pause_prefetch_reused_once_and_invalidated_on_resumed_speech(self):
        s=self.s;s.prepared={};s.prepare_quiet_s=.48;s.tasks=set();s.epoch=0;s.closed=False
        s.c=SimpleNamespace(speaking=False)
        t=self.track(self.a,self.christoph.id);t.speech_s=1.28;s.turns.tracks[t.key]=t;s.turns.now=t.last_speech+.64
        s.transcribe_turn=AsyncMock(return_value='Kannst du mir eine Geschichte erzählen?')
        s.identity_embedding=AsyncMock(return_value=(self.a,.99,2))
        s.schedule_preparation();s.schedule_preparation();await asyncio.gather(*s.tasks)
        self.assertEqual(s.transcribe_turn.await_count,1)
        self.assertIsNotNone(await s.use_preparation(t,0));self.assertEqual(t.precomputed_identity[1][2],2)
        s.schedule_preparation();await asyncio.gather(*s.tasks);t.version+=1
        self.assertIsNone(await s.use_preparation(t,0),'A resumed syllable must invalidate speculative words/identity')

    async def test_pause_prefetch_never_starts_for_unverified_identity_or_before_half_second(self):
        s=self.s;s.prepared={};s.prepare_quiet_s=.48;s.tasks=set();s.epoch=0;s.closed=False;s.c=SimpleNamespace(speaking=False)
        t=self.track(self.a,self.christoph.id);t.speech_s=1.28;s.turns.tracks[t.key]=t
        s.turns.now=t.last_speech+.32;s.schedule_preparation();self.assertFalse(s.tasks)
        s.turns.now=t.last_speech+.64;self.christoph.teacher=None;s.schedule_preparation();self.assertFalse(s.tasks)
    async def test_native_interruption_needs_full_known_voice_not_bot_echo(self):
        s=self.s;s.interrupt_checked_at=-100.;s.interrupt_reference=None;s.epoch=0;s.closed=False
        s.c=SimpleNamespace(current_speaker={'speaker_id':self.christoph.id},ears=SimpleNamespace(sr=16000),speaking=True,
            playback=SimpleNamespace(rid=7,n16=64000,audio=np.full(64000,.1,np.float32)))
        s.audio_between=lambda a,b:np.full(30720,.05,np.float32)
        s.c.visual=SimpleNamespace(gate=Mock(side_effect=AssertionError('Visual reply filtering must not block acoustic cancellation')))
        def teacher(query):return self.b if len(query)==64000 else self.a
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=teacher)):
            sid=await s.confirm_interrupt({'kind':'BARGE'},32000)
        self.assertEqual(sid,self.christoph.id);self.assertEqual(len(s.memory.users),1)
        s.interrupt_checked_at=-100.;s.interrupt_reference=None
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda q:self.a)):
            sid=await s.confirm_interrupt({'kind':'BARGE'},32000)
        self.assertIsNone(sid,'Bot and user can be similar: own playback must remain a competing reference')
    async def test_weak_or_unknown_interruption_is_rejected(self):
        s=self.s;s.interrupt_checked_at=-100.;s.interrupt_reference=None;s.epoch=0;s.closed=False
        s.c=SimpleNamespace(current_speaker={'speaker_id':self.christoph.id},ears=SimpleNamespace(sr=16000),speaking=True,
            playback=SimpleNamespace(rid=7,n16=64000,audio=np.full(64000,.1,np.float32)))
        s.audio_between=lambda a,b:np.full(30720,.05,np.float32)
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda q:self.b)):
            self.assertIsNone(await s.confirm_interrupt({'kind':'BARGE'},32000))
        self.assertEqual(len(s.memory.users),1)
    async def test_medium_cosine_on_unowned_background_cannot_interrupt(self):
        s=self.s;s.interrupt_checked_at=-100.;s.interrupt_reference=None;s.epoch=0;s.closed=False
        s.c=SimpleNamespace(current_speaker={'speaker_id':self.christoph.id},ears=SimpleNamespace(sr=16000),speaking=True,
            playback=SimpleNamespace(rid=7,n16=64000,audio=np.full(64000,.1,np.float32)))
        s.audio_between=lambda a,b:np.full(30720,.05,np.float32)
        query=.6*self.a+.8*np.eye(192,dtype=np.float32)[2]
        def teacher(q):return self.b if len(q)==64000 else query
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=teacher)):
            self.assertIsNone(await s.confirm_interrupt({'yield_prob':.9},32000))
    async def test_known_voice_can_interrupt_before_any_tts_audio_arrives(self):
        s=self.s;s.interrupt_checked_at=-100.;s.interrupt_reference=None;s.epoch=0;s.closed=False
        s.c=SimpleNamespace(current_speaker={'speaker_id':self.christoph.id},ears=SimpleNamespace(sr=16000),speaking=True,
            playback=SimpleNamespace(rid=7,n16=0,audio=np.zeros(24000,np.float32)))
        s.audio_between=lambda a,b:np.full(30720,.05,np.float32)
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda q:self.a)):
            self.assertEqual(await s.confirm_interrupt({'yield_prob':.9},32000),self.christoph.id)
    async def test_other_enrolled_person_cannot_interrupt_current_addressee(self):
        s=self.s;s.interrupt_checked_at=-100.;s.interrupt_reference=None;s.epoch=0;s.closed=False
        other,_=s.memory.enroll(self.b,name='Robert');other.teacher=self.b
        bot=np.eye(192,dtype=np.float32)[2]
        s.c=SimpleNamespace(current_speaker={'speaker_id':self.christoph.id},ears=SimpleNamespace(sr=16000),speaking=True,
            playback=SimpleNamespace(rid=7,n16=64000,audio=np.full(64000,.1,np.float32)))
        s.audio_between=lambda a,b:np.full(30720,.05,np.float32)
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda q:bot if len(q)==64000 else self.b)):
            self.assertIsNone(await s.confirm_interrupt({'yield_prob':1.},32000))
            s.c.current_speaker={'speaker_id':other.id};s.interrupt_checked_at=-100.
            self.assertEqual(await s.confirm_interrupt({'yield_prob':1.},32000),other.id)
    async def test_no_verified_addressee_never_accepts_voice_interrupt(self):
        self.s.c=SimpleNamespace(current_speaker=None,ears=SimpleNamespace(sr=16000))
        with patch('speaker_teacher.get_cached_teacher') as teacher:
            self.assertIsNone(await self.s.confirm_interrupt({'yield_prob':1.},32000))
            teacher.assert_not_called()
    async def test_literal_short_stop_requires_fast_identity_and_teacher_bot_separation(self):
        s=self.s;s.interrupt_checked_at=-100.;s.interrupt_reference=None;s.epoch=0;s.closed=False
        wave=np.full(5120,.05,np.float32)
        s.c=SimpleNamespace(current_speaker={'speaker_id':self.christoph.id},ears=SimpleNamespace(sr=16000),speaking=True,
            playback=SimpleNamespace(rid=7,n16=64000,audio=np.full(64000,.1,np.float32)))
        s.audio_between=lambda a,b:wave
        s.records=deque([(1.68,2.,{'raw_embedding':self.a},wave)])
        query=.36*self.a+np.sqrt(1-.36**2)*np.eye(192,dtype=np.float32)[2]
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda q:self.b if len(q)==64000 else query)):
            self.assertIsNone(await s.confirm_interrupt({'yield_prob':1.},32000))
            s.interrupt_checked_at=-100.
            self.assertEqual(await s.confirm_interrupt({'yield_prob':1.,'source':'verified_stop_command'},32000),self.christoph.id)
            s.interrupt_checked_at=-100.;s.records=deque([(1.68,2.,{'raw_embedding':self.b},wave)])
            self.assertIsNone(await s.confirm_interrupt({'yield_prob':1.,'source':'verified_stop_command'},32000))
        s.interrupt_checked_at=-100.;s.interrupt_reference=None;s.records=deque([(1.68,2.,{'raw_embedding':self.a},wave)])
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda q:self.a)):
            self.assertIsNone(await s.confirm_interrupt({'yield_prob':1.,'source':'verified_stop_command'},32000),'Literal stop is not an echo bypass')
    async def test_stop_probe_is_bounded_to_current_fast_winner_and_has_no_llm_reply(self):
        s=self.s;s.tasks=set();s.stop_task=None;s.stop_checked_at=-100.;s.epoch=0;s.closed=False
        s.c=SimpleNamespace(current_speaker={'speaker_id':self.christoph.id},ears=SimpleNamespace(sr=16000),speaking=True,
            playback=SimpleNamespace(rid=7),_on_barge=AsyncMock(),ws=SimpleNamespace(send_json=AsyncMock()))
        s.records=deque([(0.,2.,{'raw_embedding':self.a},np.full(32000,.05,np.float32))])
        s.transcribe_stop=AsyncMock(return_value='Stopp.');s.confirm_interrupt=AsyncMock(return_value=self.christoph.id)
        s.schedule_stop_probe(32000);task=s.stop_task;s.schedule_stop_probe(32000);self.assertIs(s.stop_task,task)
        await task;s.c._on_barge.assert_awaited_once();s.c.ws.send_json.assert_awaited_once()
        self.assertEqual(s.c.ws.send_json.call_args.args[0]['type'],'stop_command')
        s.stop_task=None;s.stop_checked_at=-100.
        s.records=deque([(0.,2.,{'raw_embedding':self.b},np.full(32000,.05,np.float32))])
        s.schedule_stop_probe(32000);self.assertIsNone(s.stop_task)
    async def test_weak_short_stop_is_not_admitted_by_lowering_guards_until_it_passes(self):
        s=self.s;s.interrupt_checked_at=-100.;s.interrupt_reference=None;s.epoch=0;s.closed=False
        s.utterance_threshold=.32
        wave=np.full(5120,.05,np.float32)
        s.c=SimpleNamespace(current_speaker={'speaker_id':self.christoph.id},ears=SimpleNamespace(sr=16000),speaking=True,
            playback=SimpleNamespace(rid=7,n16=64000,audio=np.full(64000,.1,np.float32)))
        s.audio_between=lambda a,b:wave
        fast=.396*self.a+np.sqrt(1-.396**2)*np.eye(192,dtype=np.float32)[3]
        s.records=deque([(1.68,2.,{'raw_embedding':fast},wave)])
        query=.29*self.a+np.sqrt(1-.29**2)*np.eye(192,dtype=np.float32)[2]
        proposal={'yield_prob':1.,'source':'verified_stop_command','stop_text':'Stopp.'}
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda q:self.b if len(q)==64000 else query)):
            self.assertIsNone(await s.confirm_interrupt(proposal,32000))
            s.interrupt_checked_at=-100.;s.records=deque([(1.68,2.,{'raw_embedding':self.b},wave)])
            self.assertIsNone(await s.confirm_interrupt(proposal,32000))
    async def test_long_unknown_stop_cannot_use_short_word_fast_floor(self):
        s=self.s;s.interrupt_checked_at=-100.;s.interrupt_reference=None;s.epoch=0;s.closed=False
        wave=np.full(16000,.05,np.float32)
        s.c=SimpleNamespace(current_speaker={'speaker_id':self.christoph.id},ears=SimpleNamespace(sr=16000),speaking=True,
            playback=SimpleNamespace(rid=7,n16=64000,audio=np.full(64000,.1,np.float32)))
        s.audio_between=lambda a,b:wave
        q=.32*self.a+np.sqrt(1-.32**2)*np.eye(192,dtype=np.float32)[2]
        fast=.53*self.a+np.sqrt(1-.53**2)*np.eye(192,dtype=np.float32)[3]
        s.records=deque([(1.,2.,{'raw_embedding':fast},wave)])
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda w:self.b if len(w)==64000 else q)):
            self.assertIsNone(await s.confirm_interrupt({'yield_prob':1.,'source':'verified_stop_command'},32000))
    async def test_brief_crop_consensus_requires_stronger_fast_and_joint_score(self):
        s=self.s;s.interrupt_reference=None;s.epoch=0;s.closed=False;s.utterance_threshold=.32
        wave=np.full(5120,.05,np.float32)
        s.c=SimpleNamespace(current_speaker={'speaker_id':self.christoph.id},ears=SimpleNamespace(sr=16000),speaking=True,
            playback=SimpleNamespace(rid=7,n16=64000,audio=np.full(64000,.1,np.float32)))
        s.audio_between=lambda a,b:wave
        for teacher,fast,agree,expected in ((.24,.55,True,self.christoph.id),(.24,.55,False,None),(.24,.49,True,None),(.20,.55,True,None)):
            s.interrupt_checked_at=-100.
            f=fast*self.a+np.sqrt(1-fast**2)*np.eye(192,dtype=np.float32)[3]
            q=teacher*self.a+np.sqrt(1-teacher**2)*np.eye(192,dtype=np.float32)[2]
            s.records=deque([(1.68,2.,{'raw_embedding':f},wave)])
            with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda w:self.b if len(w)==64000 else q)):
                result=await s.confirm_interrupt({'yield_prob':1.,'source':'verified_stop_command','stop_text':'Stopp','stop_views_agree':agree},32000)
            self.assertEqual(result,expected)
    async def test_compact_stop_needs_crop_consensus_teacher_fast_and_current_owner(self):
        s=self.s;s.interrupt_reference=None;s.epoch=0;s.closed=False;s.utterance_threshold=.32
        wave=np.full(12800,.05,np.float32)
        s.c=SimpleNamespace(current_speaker={'speaker_id':self.christoph.id},ears=SimpleNamespace(sr=16000),speaking=True,
            playback=SimpleNamespace(rid=7,n16=64000,audio=np.full(64000,.1,np.float32)))
        s.audio_between=lambda a,b:wave
        fast=.42*self.a+np.sqrt(1-.42**2)*np.eye(192,dtype=np.float32)[3]
        s.records=deque([(1.2,2.,{'raw_embedding':fast},wave)])
        for cosine,consensus,expected in ((.37,True,self.christoph.id),(.37,False,None),(.32,True,None)):
            s.interrupt_checked_at=-100.
            q=cosine*self.a+np.sqrt(1-cosine**2)*np.eye(192,dtype=np.float32)[2]
            with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda w:self.b if len(w)==64000 else q)):
                result=await s.confirm_interrupt({'yield_prob':1.,'source':'verified_stop_command','stop_text':'Warte bitte','stop_views_agree':consensus},32000)
            self.assertEqual(result,expected)
    async def test_late_redecode_cannot_rescue_an_expired_short_stop(self):
        s=self.s;s.interrupt_checked_at=-100.;s.epoch=0;s.closed=False
        s.c=SimpleNamespace(current_speaker={'speaker_id':self.christoph.id},ears=SimpleNamespace(sr=16000),speaking=True)
        s.audio_between=lambda a,b:np.r_[np.full(5120,.05,np.float32),np.zeros(25600,np.float32)]
        with patch('speaker_teacher.get_cached_teacher') as teacher:
            self.assertIsNone(await s.confirm_interrupt({'yield_prob':1.,'source':'verified_stop_command','stop_text':'Stopp.'},32000))
            teacher.assert_not_called()
    def test_verified_stop_tail_is_consumed_but_new_requests_are_not(self):
        s=self.s;s.remember_stop(self.christoph.id,'Bitte sei mal leise.',10.)
        track=SimpleNamespace(sid=self.christoph.id,voiced_start=9.8,start=9.,last_speech=10.2)
        self.assertTrue(s.stopped_fragment(track,'Bitte'))
        self.assertFalse(s.stopped_fragment(track,'Bitte erzähle etwas Neues.'))
        track.sid='another-person';self.assertFalse(s.stopped_fragment(track,'Bitte'))
        track.sid=self.christoph.id;track.voiced_start=12.;track.last_speech=13.
        self.assertFalse(s.stopped_fragment(track,'Bitte'))
    async def test_queued_stop_cannot_bypass_stronger_interruption_verification(self):
        s=self.s;t=self.track(self.a,self.christoph.id)
        context={'speaker_id':self.christoph.id,'display_name':'Christoph'}
        s.c=SimpleNamespace(speaking=True,current_speaker={'speaker_id':self.christoph.id},hop_pos=48000,
            ears=SimpleNamespace(sr=16000),_on_barge=AsyncMock(),session=SimpleNamespace(reset=Mock()),ws=SimpleNamespace(send_json=AsyncMock()))
        s.confirm_interrupt=AsyncMock(return_value=None)
        self.assertFalse(await s.stop_turn(t,'Stop bitte hör auf.',context))
        s.c._on_barge.assert_not_awaited();s.c.ws.send_json.assert_not_awaited()
        s.confirm_interrupt.return_value=self.christoph.id
        self.assertTrue(await s.stop_turn(t,'Stop bitte hör auf.',context))
        s.c._on_barge.assert_awaited_once();self.assertEqual(s.c.ws.send_json.call_args.args[0]['type'],'stop_command')
        s.confirm_interrupt.reset_mock();s.c.speaking=False
        self.assertFalse(await s.stop_turn(t,'Stopp.',context));s.confirm_interrupt.assert_not_awaited()
        s.c.speaking=True;context['speaker_id']='another-enrolled-person'
        self.assertFalse(await s.stop_turn(t,'Stopp.',context));s.confirm_interrupt.assert_not_awaited()
    async def test_stop_asr_batch_keeps_full_audio_and_silence_pads_short_views(self):
        s=self.s;s.asr_lock=asyncio.Lock();s.decode_count=0
        decoder=Mock(return_value=['unrelated words','Stop.',''])
        s.c=SimpleNamespace(ears=SimpleNamespace(service=SimpleNamespace(engine=SimpleNamespace(asr=SimpleNamespace(transcribe=decoder)))))
        wave=np.r_[np.zeros(8000,np.float32),np.full(4800,.05,np.float32),np.zeros(8000,np.float32)]
        self.assertEqual(await s.transcribe_stop(wave),'Stop.')
        batch=decoder.call_args.args[0];self.assertEqual(len(batch),3)
        self.assertTrue(all(len(v)>=40960 for v in batch))
        np.testing.assert_array_equal(batch[0][3200:3200+len(wave)],wave)
        self.assertEqual(s.decode_count,1)
        decoder.return_value=['What does stop mean?','Das ist halt so.','']
        self.assertEqual(await s.transcribe_stop(wave),'')
    async def test_interrupt_identity_uses_available_long_context_without_collecting_delay(self):
        s=self.s;s.interrupt_checked_at=-100.;s.interrupt_reference=None;s.epoch=0;s.closed=False;s.identity_context_s=3.04
        s.c=SimpleNamespace(current_speaker={'speaker_id':self.christoph.id},ears=SimpleNamespace(sr=16000),speaking=True,speaker_reply_start_s=0.,
            playback=SimpleNamespace(rid=7,n16=0,audio=np.zeros(24000,np.float32)))
        s.audio_between=Mock(side_effect=lambda a,b:np.full(round((b-a)*16000),.05,np.float32))
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda q:self.a)):
            self.assertEqual(await s.confirm_interrupt({'yield_prob':.9},80000),self.christoph.id)
        self.assertAlmostEqual(s.audio_between.call_args.args[1]-s.audio_between.call_args.args[0],3.04)
    async def test_playback_content_disagreement_distinguishes_similar_voice_from_exact_echo(self):
        s=self.s;s.epoch=0;s.closed=False;s.identity_context_s=3.04
        rng=np.random.default_rng(718)
        mic=rng.normal(0,.04,80000).astype(np.float32)
        other=rng.normal(0,.04,80000).astype(np.float32)
        vector=.55*self.a+np.sqrt(1-.55**2)*self.b
        for echo,expected in ((False,self.christoph.id),(True,None)):
            s.interrupt_checked_at=-100.;s.interrupt_reference=None
            s.c=SimpleNamespace(current_speaker={'speaker_id':self.christoph.id},ears=SimpleNamespace(sr=16000),speaking=True,speaker_reply_start_s=0.,
                playback=SimpleNamespace(rid=7,n16=64000,audio=other[:64000]))
            s.records=deque([(0.,5.,{'playback_audio16':mic if echo else other},mic)])
            # Same speaker-embedding direction for mic and bot: timbre alone
            # cannot decide. The correctly aligned waveform can.
            with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda q:vector)):
                result=await s.confirm_interrupt({'yield_prob':.3},80000)
            self.assertEqual(result,expected)
    async def test_previous_utterance_is_not_reused_to_interrupt_its_own_reply(self):
        s=self.s;s.interrupt_checked_at=-100.;s.interrupt_reference=None;s.epoch=0;s.closed=False
        s.c=SimpleNamespace(current_speaker={'speaker_id':self.christoph.id},ears=SimpleNamespace(sr=16000),speaking=True,speaker_reply_start_s=2.,
            playback=SimpleNamespace(rid=7,n16=0,audio=np.zeros(24000,np.float32)))
        s.audio_between=lambda a,b:np.full(max(0,round((b-a)*16000)),.05,np.float32)
        with patch('speaker_teacher.get_cached_teacher') as teacher:
            self.assertIsNone(await s.confirm_interrupt({'yield_prob':.9},32000))
            teacher.assert_not_called()
    async def test_strong_native_continuity_needs_fast_and_acoustic_agreement(self):
        s=self.s;s.interrupt_checked_at=-100.;s.interrupt_reference=None;s.epoch=0;s.closed=False
        s.c=SimpleNamespace(current_speaker={'speaker_id':self.christoph.id},ears=SimpleNamespace(sr=16000),speaking=True,
            playback=SimpleNamespace(rid=7,n16=64000,audio=np.full(64000,.1,np.float32)))
        s.audio_between=lambda a,b:np.full(30720,.05,np.float32)
        other=np.eye(192,dtype=np.float32)[2];query=.45*self.a+np.sqrt(1-.45**2)*other
        def teacher(q):return self.b if len(q)==64000 else query
        t=self.track(self.a,self.christoph.id);t.last_speech=2.;s.turns.tracks[t.sid]=t
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=teacher)):
            self.assertEqual(await s.confirm_interrupt({'yield_prob':.9},32000),self.christoph.id)
            s.interrupt_checked_at=-100.
            self.assertIsNone(await s.confirm_interrupt({'yield_prob':.3},32000))
    def track(self,vector,sid):
        t=Track(sid,0,1.28,sid=sid,channel=1);t.fast.extend([vector]*4)
        t.chunks.append((0,1.28,np.ones(20480,np.float32)*.05));return t
    def test_contiguous_known_voice_tail_merges_before_prefix_submission(self):
        s=self.s;old=self.track(self.a,self.christoph.id)
        newer=Track(old.sid,1.3,2.6,sid=old.sid,voiced_start=1.3)
        newer.chunks.append((1.3,2.6,np.full(20800,.05,np.float32)))
        newer.submission_audio=np.zeros(1,np.float32)
        newer.precomputed_identity=('stale crop',(self.a,None,1))
        s.turns.tracks[newer.key]=newer
        self.assertTrue(s.merge_continuation(old))
        self.assertEqual(newer.start,0);self.assertEqual(newer.last_speech,2.6)
        self.assertIsNone(newer.submission_audio);self.assertFalse(hasattr(newer,'precomputed_identity'))
        self.assertGreater(len(newer.waveform()),40000)
    def test_continuation_never_merges_other_user_or_real_long_pause(self):
        s=self.s;old=self.track(self.a,self.christoph.id)
        other,_=s.memory.enroll(self.b,name='Robert');other.teacher=self.b
        newer=Track(other.id,1.3,2.6,sid=other.id,voiced_start=1.3)
        s.turns.tracks[newer.key]=newer
        self.assertFalse(s.merge_continuation(old))
        newer.sid=old.sid;newer.voiced_start=2.5
        self.assertFalse(s.merge_continuation(old))
    def test_queued_continuation_remains_owned_by_successor_deadline(self):
        s=self.s;old=self.track(self.a,self.christoph.id)
        newer=Track(old.sid,1.3,2.6,sid=old.sid,voiced_start=1.3,closed=True)
        s.pending_turns={newer.token:newer}
        self.assertTrue(s.merge_continuation(old))
        self.assertEqual(newer.last_speech,2.6);self.assertTrue(newer.closed)
    async def refine(self,t,text,embedding):
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda wave:embedding)):
            await self.s.refine_identity(t,text)
    async def test_slider_relaxes_single_known_speaker_without_enrolling_someone(self):
        query=.46*self.a+np.sqrt(1-.46**2)*self.b
        t=self.track(self.a,self.christoph.id)
        await self.refine(t,'Tell me a story.',query)
        self.assertIsNone(t.sid)
        self.assertIn('0.46 / required 0.50',t.identity_rejection)
        self.s.utterance_threshold=.45
        t=self.track(self.a,self.christoph.id)
        await self.refine(t,'Tell me a story.',query)
        self.assertEqual(t.sid,self.christoph.id)
        self.assertEqual(len(self.s.memory.users),1)
        np.testing.assert_array_equal(self.christoph.teacher,self.a)
    async def test_relaxed_threshold_still_rejects_unrelated_background_and_close_rival(self):
        self.s.utterance_threshold=.45
        t=self.track(self.a,self.christoph.id)
        await self.refine(t,'Tell me a story.',self.b)
        self.assertIsNone(t.sid)
        other,_=self.s.memory.enroll(self.b,name='Robert');other.teacher=self.b
        query=.47*self.a+.46*self.b+np.sqrt(1-.47**2-.46**2)*np.eye(192,dtype=np.float32)[2]
        t=self.track(self.a,self.christoph.id)
        await self.refine(t,'Tell me a story.',query)
        self.assertIsNone(t.sid)
        self.assertIn('gap to other users 0.01',t.identity_rejection)
    async def test_new_diarization_channel_honors_user_threshold(self):
        s=self.s;s.c=SimpleNamespace(speaking=False);s.epoch=0;s.closed=False
        query=.46*self.a+np.sqrt(1-.46**2)*self.b
        t=self.track(self.a,None);t.key='new-channel';t.channel=3;s.turns.tracks[t.key]=t
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda q:query)):
            self.assertIsNone(await s.recover_existing_channel(t))
            s.utterance_threshold=.45
            match=await s.recover_existing_channel(t)
        self.assertEqual(match['id'],self.christoph.id)
        self.assertEqual(t.sid,self.christoph.id)
        self.assertEqual(len(s.memory.users),1)
    async def test_weak_unknown_fast_cue_cannot_steal_channel_at_relaxed_teacher_threshold(self):
        s=self.s;s.c=SimpleNamespace(speaking=False);s.epoch=0;s.closed=False;s.utterance_threshold=.45
        query=.46*self.a+np.sqrt(1-.46**2)*self.b
        t=self.track(self.b,None);t.key='background-channel';t.channel=3;s.turns.tracks[t.key]=t
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda q:query)):
            self.assertIsNone(await s.recover_existing_channel(t))
        self.assertIsNone(t.sid);self.assertNotIn(3,s.turns.channels)
    async def test_merged_channel_new_name_does_not_overwrite_old_profile(self):
        t=self.track(self.b,self.christoph.id)
        await self.refine(t,'He Robby, ich bin Robert.',self.b)
        self.assertNotEqual(t.sid,self.christoph.id);self.assertEqual(self.christoph.name,'Christoph')
        self.assertEqual(self.s.memory.users[t.sid].name,'Robert');self.assertEqual(len(self.s.memory.users),2)
        self.assertTrue(np.allclose(self.s.memory.users[t.sid].teacher,self.b))
    async def test_returning_voice_reassigned_to_original_id(self):
        t=self.track(self.b,self.christoph.id);await self.refine(t,'Hey Robbie, ich bin Robert.',self.b)
        robert=t.sid
        returning=self.track(self.a,robert);await self.refine(returning,'Kannst du mir eine Geschichte erzählen?',self.a)
        self.assertEqual(returning.sid,self.christoph.id);self.assertEqual(self.s.memory.users[robert].name,'Robert')
    async def test_unmatched_voice_cannot_rename_without_wake(self):
        t=self.track(self.b,self.christoph.id);await self.refine(t,'Ich bin Robert.',self.b)
        self.assertIsNone(t.sid);self.assertEqual(self.christoph.name,'Christoph')

    async def test_matching_embedding_cannot_overwrite_different_name_without_wake(self):
        t=self.track(self.a,self.christoph.id);await self.refine(t,'Ich bin Robert.',self.a)
        self.assertIsNone(t.sid);self.assertEqual(self.christoph.name,'Christoph')

    async def test_llm_metadata_cannot_bypass_acoustic_name_binding(self):
        sid=self.christoph.id;self.s.pending_names={sid:'Ich bin Robert.'}
        await self.s.metadata({'speaker_update':{'speaker_id':sid,'display_name':'Robert'}})
        self.assertEqual(self.christoph.name,'Christoph')

    def test_submission_preserves_internal_phonemes_but_not_other_known_user(self):
        t=self.track(self.a,self.christoph.id);t.start=0;t.last_speech=1.28
        wave=np.ones(20480,np.float32)*.05
        self.s.records.append((0,1.28,{},wave))
        # Unknown detector gaps may contain this speaker's syllables.
        self.s.owners.extend([(0,.32,t.sid),(.32,.64,None),(.64,.96,'different-user'),(.96,1.28,t.sid)])
        result=self.s.submission_wave(t)
        self.assertTrue(np.allclose(result[:10240],.05));self.assertTrue(np.allclose(result[10240:15360],0))
        t.submission_audio=result.copy();self.s.records.clear()
        np.testing.assert_array_equal(self.s.submission_wave(t),result)
    def test_postroll_keeps_quiet_word_ending_without_extending_turn_clock(self):
        t=self.track(self.a,self.christoph.id);last=t.last_speech
        self.s.records.append((0,2.56,{},np.full(40960,.05,np.float32)))
        self.s.owners.append((1.9,2.2,'another-user',2))
        wave=self.s.submission_wave(t)
        self.assertAlmostEqual(len(wave)/16000,2.08)
        self.assertEqual(t.last_speech,last)
        self.assertTrue(np.all(wave[30400:]==0))

    async def test_prefix_views_recover_wake_without_inventing_names(self):
        t=self.track(self.b,self.christoph.id);t.submission_audio=np.ones(48000,np.float32)*.02;t.voiced_start=1.
        transcribe=lambda views:['How are you?', 'Background. Hey Robbie, ich bin Robert.', 'Ich bin Robert.', 'How are you?'] if len(views)==4 else ['Ich bin Robert.']
        self.s.c=SimpleNamespace(ears=SimpleNamespace(service=SimpleNamespace(engine=SimpleNamespace(asr=SimpleNamespace(transcribe=transcribe)))))
        import asyncio
        self.s.asr_lock=asyncio.Lock();self.s.decode_count=0
        text=await self.s.transcribe_turn(t)
        self.assertEqual(text,'Hey Robbie, ich bin Robert.');self.assertEqual(self.s.decode_count,5)
        self.assertEqual(len(t.submission_audio),48000-2560)

    def test_only_energy_dominant_candidate_can_repair_diarizer_holes(self):
        wave=np.full(20480,.05,np.float32);self.s.records.append((0,1.28,{},wave))
        foreground=self.track(self.a,'foreground');foreground.chunks.clear()
        foreground.chunks.extend([(0,.96,wave[:15360]),(1.2,1.28,wave[-1280:])])
        _,coverage,continuous=self.s.candidate_wave(foreground)
        self.assertGreater(coverage,.65);self.assertTrue(continuous)
        background=self.track(self.b,'background');background.chunks.clear()
        background.chunks.extend([(0,.24,wave[:3840]),(1.2,1.28,wave[-1280:])])
        _,coverage,continuous=self.s.candidate_wave(background)
        self.assertLess(coverage,.65);self.assertFalse(continuous)

    def test_incremental_wake_is_bound_to_same_name_and_current_audio_window(self):
        t=self.track(self.a,self.christoph.id)
        self.s.note_asr('Hey Robbie, ich bin Robert.',1.)
        self.assertTrue(self.s.observed_wake(t,'Ich bin Robert.'))
        self.assertFalse(self.s.observed_wake(t,'Ich bin Andreas.'))
        t.start=10.;t.last_speech=12.
        self.assertFalse(self.s.observed_wake(t,'Ich bin Robert.'))

    async def test_low_energy_background_cannot_enroll_even_with_hallucinated_full_wake(self):
        s=self.s;s.closed=False;s.c=SimpleNamespace(identity_enabled=True,ws=SimpleNamespace(send_json=AsyncMock()))
        t=self.track(self.b,None);t.key='channel-0';t.channel=0;s.turns.tracks[t.key]=t
        s.background_channels.add(0);s.transcribe=AsyncMock(return_value='Hey Robbie, ich bin Selma.')
        await s.candidate(t,np.full(20480,.01,np.float32),coverage=.27)
        self.assertIsNone(t.sid);self.assertEqual(len(s.memory.users),1)

    async def test_background_activation_requires_independent_raw_name_agreement(self):
        s=self.s;s.closed=False;s.c=SimpleNamespace(identity_enabled=True,ws=SimpleNamespace(send_json=AsyncMock()))
        t=self.track(self.b,None);t.key='channel-0';t.channel=0;s.turns.tracks[t.key]=t
        s.background_channels.add(0);s.records.append((0,1.28,{},np.full(20480,.04,np.float32)))
        s.transcribe=AsyncMock(side_effect=['Hey Robbie, ich bin Selma.','Das sind die Nachrichten.'])
        await s.candidate(t,np.full(20480,.01,np.float32),coverage=.55)
        self.assertIsNone(t.sid);self.assertEqual(len(s.memory.users),1)

    def test_coherent_energy_winning_foreground_can_recover_channel_transition(self):
        wave=np.full(20480,.05,np.float32);self.s.records.append((0,1.28,{},wave))
        t=self.track(self.a,None);t.channel=2;t.chunks.clear();t.chunks.append((.64,1.28,wave[10240:]))
        self.s.turns.tracks['channel-2']=t
        _,coverage,continuous=self.s.candidate_wave(t)
        self.assertAlmostEqual(coverage,.5);self.assertTrue(continuous)
        other=self.track(self.b,None);other.channel=0;self.s.turns.tracks['channel-0']=other
        _,_,continuous=self.s.candidate_wave(t)
        self.assertFalse(continuous,'Two competing voices must not borrow a shared greeting')

    def test_provisional_introduction_recovers_old_channel_prefix_but_not_overlap(self):
        user,_=self.s.memory.enroll(self.b,name='Selma',force_new=True)
        t=self.track(self.b,user.id);t.wake=True;t.partial='Hey Robbie, ich bin Selma.'
        self.s.records.append((0,1.28,{},np.full(20480,.05,np.float32)))
        self.s.owners.append((0,.32,self.christoph.id,1));self.s.overlaps.append((.64,.96))
        wave=self.s.submission_wave(t)
        self.assertTrue(np.all(wave[:5120]>.04));self.assertTrue(np.all(wave[10240:15360]==0))

    async def test_split_greeting_requires_original_audio_and_same_explicit_name(self):
        import asyncio
        s=self.s;s.asr_lock=asyncio.Lock();s.decode_count=0
        s.records.append((0,3.,{},np.full(48000,.05,np.float32)))
        t=self.track(self.b,None);t.start=1.6;t.last_speech=3.
        decoder=SimpleNamespace(transcribe=lambda waves:['Nachrichten. Hey Robbie, ich bin Selma.','Hey Robbie, ich bin Selma.','Ich bin Selma.','Ich bin Selma.'])
        s.c=SimpleNamespace(ears=SimpleNamespace(service=SimpleNamespace(engine=SimpleNamespace(asr=decoder))))
        value,start=await s.recover_intro_prefix(t,'Ich bin Selma.')
        self.assertIn('Hey Robbie',value);self.assertAlmostEqual(start,.32)
        self.assertIsNone(await s.recover_intro_prefix(t,'Ich bin Robert.'))
        s.overlaps.append((.5,1.))
        self.assertIsNone(await s.recover_intro_prefix(t,'Ich bin Selma.'))
    def test_room_wide_name_cannot_end_another_channels_wake_grace(self):
        guest,_=self.s.memory.enroll(self.b,force_new=True)
        t=self.track(self.b,guest.id);t.wake=True;t.partial='Hey Robbie.'
        self.s.turns.tracks[t.sid]=t
        self.s.note_asr('Hey Robbie, ich bin Selma.',1.)
        self.assertFalse(t.name_heard)
    def test_truncated_owned_name_requires_unique_timestamped_full_intro(self):
        t=self.track(self.a,None)
        self.s.asr_wakes.append((t.last_speech,'christoph'))
        self.assertTrue(self.s.observed_wake(t,'Ich bin Christ.'))
        self.assertFalse(self.s.observed_wake(t,'Ich bin Robert.'))
        self.s.asr_wakes.append((t.last_speech,'christian'))
        self.assertFalse(self.s.observed_wake(t,'Ich bin Christ.'))
    def test_new_address_only_quarantines_reused_channel_with_acoustic_difference(self):
        s=self.s;s.c=SimpleNamespace(speaking=False);s.turns.channels[1]=self.christoph.id
        s.asr_wakes.append((1.,'robert'))
        s.consider_new_address(1,{'raw_embedding':self.a},1.5)
        self.assertEqual(s.turns.channels[1],self.christoph.id)
        s.consider_new_address(1,{'raw_embedding':self.b},1.5)
        self.assertNotIn(1,s.turns.channels);self.assertEqual(len(s.memory.users),1)
    def test_room_asr_activation_uses_last_addressed_person_not_old_buffer_prefix(self):
        s=self.s;s.note_asr('Hey Robbie, ich bin Christoph. Hey Robbie, ich bin Robert.',2.)
        self.assertEqual(s.asr_wakes[-1],(2.,'robert'))
    async def test_known_person_can_recover_new_diarization_channel_without_wake(self):
        s=self.s;s.c=SimpleNamespace(speaking=False);s.epoch=0;s.closed=False
        t=self.track(self.a,None);t.key='channel-3';t.channel=3;s.turns.tracks[t.key]=t
        s.turns.channels[0]=self.christoph.id;s.background_channels.add(0)
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda q:self.a)):
            decision=await s.recover_existing_channel(t)
        self.assertEqual(decision['id'],self.christoph.id);self.assertFalse(decision['new_user'])
        self.assertEqual(s.turns.channels[3],self.christoph.id);self.assertNotIn(0,s.turns.channels)
        self.assertEqual(len(s.memory.users),1)
    async def test_continuous_known_recovery_uses_captured_context_not_zero_masked_holes(self):
        s=self.s;s.c=SimpleNamespace(speaking=False);s.epoch=0;s.closed=False
        t=self.track(self.a,None);t.key='channel-3';t.channel=3;t.start=0.;t.voiced_start=1.
        t.chunks=deque([(1.,1.64,np.full(10240,.05,np.float32))]);s.turns.tracks[t.key]=t
        embed=Mock(return_value=self.a)
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=embed)):
            result=await s.recover_existing_channel(t,context=np.full(35200,.05,np.float32))
        self.assertEqual(result['id'],self.christoph.id)
        self.assertGreater(len(embed.call_args.args[0]),20480)
    async def test_recovery_accepts_configured_threshold_with_separated_existing_identity(self):
        s=self.s;s.c=SimpleNamespace(speaking=False);s.epoch=0;s.closed=False
        other,_=s.memory.enroll(self.b,name='Robert');other.teacher=self.b
        t=self.track(self.a,None);t.key='channel-3';t.channel=3;s.turns.tracks[t.key]=t
        vector=.52*self.a+.28*self.b+np.sqrt(1-.52**2-.28**2)*np.eye(192,dtype=np.float32)[2]
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda q:vector)):
            decision=await s.recover_existing_channel(t)
        self.assertEqual(decision['id'],self.christoph.id)
        self.assertEqual(len(s.memory.users),2)
    async def test_known_recovery_consensus_requires_same_teacher_and_fast_winner(self):
        s=self.s;s.c=SimpleNamespace(speaking=False);s.epoch=0;s.closed=False
        other,_=s.memory.enroll(self.b,name='Robert');other.teacher=self.b
        vector=.58*self.a+.51*self.b+np.sqrt(1-.58**2-.51**2)*np.eye(192,dtype=np.float32)[2]
        for fast,expected in ((self.a,self.christoph.id),(self.b,None)):
            t=self.track(fast,None);t.key='channel-3';t.channel=3;s.turns.tracks[t.key]=t
            with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda q:vector)):
                result=await s.recover_existing_channel(t)
            self.assertEqual(result.get('id') if result else None,expected)
    def test_recovery_keeps_sentence_on_both_sides_of_800ms_pause(self):
        s=self.s;t=self.track(self.a,None);t.start=0.;t.last_speech=8.
        wave=np.full(8*16000,.008,np.float32)
        wave[4*16000:5*16000]=.06;wave[round(5.8*16000):round(7.5*16000)]=.06
        s.records.append((0.,8.,{},wave))
        start=s.recovery_start(t)
        self.assertGreater(start,2.5);self.assertLess(start,4.)
    def test_foreground_crop_keeps_internal_pause_but_not_quiet_tv_edges(self):
        wave=np.full(8*16000,.008,np.float32)
        wave[2*16000:3*16000]=.06;wave[round(3.8*16000):5*16000]=.06
        cropped,start=self.s.foreground_crop(wave)
        self.assertEqual(start,0.)
        self.assertGreater(len(cropped)/16000,3.);self.assertLess(start+len(cropped)/16000,6.)
    def test_equal_level_mixture_cannot_be_separated_by_energy(self):
        wave=np.full(64000,.02,np.float32);cropped,start=self.s.foreground_crop(wave)
        self.assertEqual(start,0.);self.assertEqual(len(cropped),len(wave))
    async def test_unrelated_voice_without_wake_never_creates_identity(self):
        s=self.s;s.c=SimpleNamespace(speaking=False);s.epoch=0;s.closed=False
        t=self.track(self.b,None);t.key='channel-3';t.channel=3;s.turns.tracks[t.key]=t
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda q:self.b)):
            self.assertIsNone(await s.recover_existing_channel(t))
        self.assertEqual(len(s.memory.users),1);self.assertNotIn(3,s.turns.channels)
    async def test_agreeing_verified_followup_can_improve_anchored_reference(self):
        s=self.s;other,_=s.memory.enroll(self.b,name='Robert');other.teacher=self.b
        t=self.track(self.a,self.christoph.id);t.speech_s=1.6;t.verification_audio=np.full(25600,.05,np.float32)
        s.turns.channels[1]=t.sid
        query=.46*self.a+np.sqrt(1-.46**2)*np.eye(192,dtype=np.float32)[2]
        await self.refine(t,'Kannst du mir eine Geschichte erzählen?',query)
        self.assertEqual(t.sid,self.christoph.id);self.assertEqual(len(self.christoph.teacher_examples),1)
        np.testing.assert_array_equal(self.christoph.teacher_anchor,self.a)
        self.assertGreater(float(self.christoph.teacher@query),.46)
    async def test_single_acoustic_view_does_not_adapt_voice_reference(self):
        s=self.s;other,_=s.memory.enroll(self.b,name='Robert');other.teacher=self.b
        t=self.track(self.a,self.christoph.id);t.speech_s=1.6;s.turns.channels[1]=t.sid
        query=.46*self.a+np.sqrt(1-.46**2)*np.eye(192,dtype=np.float32)[2]
        await self.refine(t,'Kannst du mir eine Geschichte erzählen?',query)
        self.assertEqual(t.sid,self.christoph.id);self.assertFalse(self.christoph.teacher_examples)
        np.testing.assert_array_equal(self.christoph.teacher,self.a)
    async def test_strong_full_reference_match_allows_moderate_crop_variation(self):
        other,_=self.s.memory.enroll(self.b,name='Robert');other.teacher=self.b
        t=self.track(self.a,self.christoph.id);t.speech_s=1.6;t.verification_audio=np.full(25600,.05,np.float32)
        self.s.turns.channels[1]=t.sid
        direction=.60*self.a+.80*np.eye(192,dtype=np.float32)[2]
        def teacher(q):return np.sqrt(.93)*direction+(1 if len(q)==20480 else -1)*np.sqrt(.07)*self.b
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=teacher)):
            await self.s.refine_identity(t,'Kannst du mir eine Geschichte erzählen?')
        self.assertEqual(len(self.christoph.teacher_examples),1)
    async def test_weak_match_with_moderate_crop_variation_never_adapts(self):
        other,_=self.s.memory.enroll(self.b,name='Robert');other.teacher=self.b
        t=self.track(self.a,self.christoph.id);t.speech_s=1.6;t.verification_audio=np.full(25600,.05,np.float32)
        self.s.turns.channels[1]=t.sid
        direction=.46*self.a+np.sqrt(1-.46**2)*np.eye(192,dtype=np.float32)[2]
        def teacher(q):return np.sqrt(.93)*direction+(1 if len(q)==20480 else -1)*np.sqrt(.07)*self.b
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=teacher)):
            await self.s.refine_identity(t,'Kannst du mir eine Geschichte erzählen?')
        self.assertFalse(self.christoph.teacher_examples)
    async def test_strong_full_and_crop_consensus_can_recover_close_known_voices(self):
        person,_=self.s.memory.enroll(self.b,name='Robert');person.teacher=self.b
        t=self.track(self.b,person.id);t.speech_s=1.6;t.verification_audio=np.full(25600,.05,np.float32)
        query=.58*self.a+.52*self.b+np.sqrt(1-.58**2-.52**2)*np.eye(192,dtype=np.float32)[2]
        await self.refine(t,'Kannst du mir eine Geschichte erzählen?',query)
        self.assertEqual(t.sid,self.christoph.id)
        self.assertFalse(self.christoph.teacher_examples,'Close-score decisions must not adapt the reference')
    async def test_single_view_cannot_relax_close_known_voice_margin(self):
        person,_=self.s.memory.enroll(self.b,name='Robert');person.teacher=self.b
        t=self.track(self.b,person.id);t.speech_s=1.6
        query=.58*self.a+.52*self.b+np.sqrt(1-.58**2-.52**2)*np.eye(192,dtype=np.float32)[2]
        await self.refine(t,'Kannst du mir eine Geschichte erzählen?',query)
        self.assertIsNone(t.sid)
    async def test_equal_length_shifted_acoustic_views_are_not_deduplicated(self):
        t=self.track(self.a,self.christoph.id);t.verification_audio=np.full(20480,.07,np.float32)
        from unittest.mock import Mock
        teacher=SimpleNamespace(embed=Mock(return_value=self.a))
        with patch('speaker_teacher.get_cached_teacher',return_value=teacher):await self.s.refine_identity(t,'Please tell me a story.')
        self.assertEqual(teacher.embed.call_count,2)
    async def test_new_self_introduction_on_reused_channel_uses_whole_reference(self):
        t=self.track(self.b,self.christoph.id);t.verification_audio=np.full(30720,.07,np.float32)
        from unittest.mock import Mock
        teacher=SimpleNamespace(embed=Mock(return_value=self.b))
        with patch('speaker_teacher.get_cached_teacher',return_value=teacher):await self.s.refine_identity(t,'Hey Robbie, ich bin Robert.')
        self.assertEqual(teacher.embed.call_count,1)
        self.assertEqual(len(teacher.embed.call_args.args[0]),30720)
        self.assertEqual(self.s.memory.users[t.sid].name,'Robert')
    async def test_full_turn_rejects_actual_bot_voice_even_on_bound_user_channel(self):
        s=self.s;s.epoch=0;s.closed=False;s.interrupt_reference=None
        s.c=SimpleNamespace(playback=SimpleNamespace(rid=4,n16=64000,audio=np.full(64000,.05,np.float32)))
        t=self.track(self.a,self.christoph.id)
        await self.refine(t,'What a lovely day.',self.b)
        self.assertIsNone(t.sid);self.assertFalse(self.christoph.teacher_examples)
    async def test_unverified_partial_name_can_be_corrected_without_orphan_user(self):
        person,_=self.s.memory.enroll(self.b,name='Selber',force_new=True)
        t=self.track(self.b,person.id)
        await self.refine(t,'Hey Robbie, ich bin Selma.',self.b)
        self.assertEqual(t.sid,person.id);self.assertEqual(person.name,'Selma')
        self.assertEqual(len(self.s.memory.users),2);self.assertIsNotNone(person.teacher)
    async def test_incomplete_split_introduction_does_not_create_duplicate_guest(self):
        t=self.track(self.b,self.christoph.id)
        await self.refine(t,'Hey Robbie, ich bin',self.b)
        self.assertIsNone(t.sid);self.assertEqual(len(self.s.memory.users),1)
    async def test_wake_fragment_overlapping_full_introduction_is_not_second_user(self):
        person,_=self.s.memory.enroll(self.b,name='Selma',force_new=True)
        other=self.track(self.b,person.id);other.wake=True;other.name_heard=True;other.channel=2
        self.s.turns.tracks[person.id]=other
        t=self.track(self.b,self.christoph.id)
        await self.refine(t,'Hey Robbie.',self.b)
        self.assertIsNone(t.sid);self.assertEqual(len(self.s.memory.users),2)

    def test_gap_bridge_requires_recent_same_speaker_and_foreground_energy(self):
        self.s.background_channels.add(0);self.s.channel_levels[0]=deque([.005]*10)
        t=self.track(self.a,self.christoph.id);t.last_direct_speech=1.;t.levels.append(.05)
        self.s.turns.tracks[t.key]=t
        cue={'kind':'speech','confidence':.95,'raw_embedding':self.a};p=np.zeros(8);p[1]=.3
        self.assertEqual(self.s.bridge_gap(0,cue,np.full(5120,.04,np.float32),1.,1.32,p),1)
        self.assertIsNone(self.s.bridge_gap(0,cue,np.full(5120,.004,np.float32),1.32,1.64,p))
        self.assertIsNone(self.s.bridge_gap(0,{**cue,'raw_embedding':self.b},np.full(5120,.04,np.float32),1.64,1.96,p))
        self.assertIsNone(self.s.bridge_gap(0,cue,np.full(5120,.04,np.float32),3.,3.32,p))

    async def test_provisional_wake_guest_upgrades_from_full_followup_sentence(self):
        guest,_=self.s.memory.enroll(self.b,force_new=True);guest.teacher=self.a
        t=self.track(self.b,guest.id)
        await self.refine(t,'Ja, ich bin Robert.',self.b)
        self.assertEqual(t.sid,guest.id);self.assertEqual(guest.name,'Robert')
        np.testing.assert_allclose(guest.teacher,self.b)

    async def test_channel_continuity_relaxation_requires_a_separated_existing_gallery(self):
        other,_=self.s.memory.enroll(self.b,name='Robert');other.teacher=self.b
        self.s.turns.channels[1]=self.christoph.id
        vector=self.a*.46+np.eye(192,dtype=np.float32)[2]*np.sqrt(1-.46**2)
        t=self.track(self.a,self.christoph.id);await self.refine(t,'Tell me a story.',vector)
        self.assertEqual(t.sid,self.christoph.id)
        self.s.memory.users.pop(other.id)
        t=self.track(self.a,self.christoph.id);await self.refine(t,'Tell me a story.',vector)
        self.assertIsNone(t.sid)
    async def test_weak_full_voice_cannot_override_strong_different_student_on_disagreeing_views(self):
        s=self.s;other,_=s.memory.enroll(self.b,name='Robert');other.teacher=self.b
        t=self.track(self.b,self.christoph.id)
        vector=.54*self.a+.40*self.b+np.sqrt(1-.54**2-.40**2)*np.eye(192,dtype=np.float32)[2]
        s.identity_embedding=AsyncMock(return_value=(vector,.67,2))
        await s.refine_identity(t,'Erzähl mir bitte einen.')
        self.assertIsNone(t.sid)
    async def test_stale_provisional_wake_cannot_enroll_background_guest(self):
        t=self.track(self.b,self.christoph.id);t.wake=True
        await self.refine(t,'Im Hintergrund laufen die Nachrichten.',self.b)
        self.assertIsNone(t.sid)
        self.assertEqual(len(self.s.memory.users),1)
    async def test_single_user_relaxation_requires_independent_activity_and_playback_competitor(self):
        s=self.s;s.playback_voice=AsyncMock(return_value=self.b)
        vector=.48*self.a+np.sqrt(1-.48**2)*np.eye(192,dtype=np.float32)[2]
        for independent,expected in ((False,None),(True,self.christoph.id)):
            t=self.track(self.a,self.christoph.id);t.speech_s=1.5;t.independent_activity_confirmed=independent
            s.turns.channels[t.channel]=self.christoph.id
            s.identity_embedding=AsyncMock(return_value=(vector,.87,2))
            await s.refine_identity(t,'Tell me a story.')
            self.assertEqual(t.sid,expected)

    def test_async_identity_rebind_keeps_inflight_continuation_and_its_owners(self):
        other,_=self.s.memory.enroll(self.b,name='Robert',force_new=True)
        old=self.christoph.id
        closed=self.track(self.b,old);closed.start=2.;closed.voiced_start=2.;closed.last_speech=2.64;closed.closed=True
        continuation=self.track(self.b,old);continuation.start=2.64;continuation.last_speech=3.6
        self.s.turns.tracks[old]=continuation;self.s.turns.channels[1]=old
        self.s.owners=deque([(0.,1.,old,1),(2.,2.64,old,1),(2.64,3.6,old,1),(3.,3.2,old,0)])
        self.s.rebind(closed,other.id)
        self.assertEqual(closed.sid,other.id);self.assertNotIn(old,self.s.turns.tracks)
        self.assertIs(self.s.turns.tracks[other.id],continuation);self.assertEqual(continuation.sid,other.id)
        self.assertEqual([r[2] for r in self.s.owners],[old,other.id,other.id,old])
        self.assertEqual(self.christoph.name,'Christoph')
    def test_late_final_verification_cannot_bind_a_new_background_track(self):
        old=self.christoph.id
        closed=self.track(self.a,old);closed.closed=True;closed.last_speech=2.
        background=self.track(self.b,None);background.start=1.2;background.voiced_start=3.2;background.channel=closed.channel
        self.s.turns.tracks['new-background']=background
        self.s.rebind(closed,old)
        self.assertIsNone(background.sid)
        self.assertNotIn(closed.channel,self.s.turns.channels)

    async def test_asr_crop_does_not_discard_reliable_full_sentence_speaker_cues(self):
        t=self.track(self.a,self.christoph.id);t.submission_audio=np.full(16000,.05,np.float32);t.verification_audio=np.full(32000,.05,np.float32)
        noise=np.eye(192,dtype=np.float32)[2]
        def embed(wave):
            cosine=.64 if len(wave)>20000 else .47
            return self.a*cosine+noise*np.sqrt(1-cosine**2)
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=embed)):
            await self.s.refine_identity(t,'Tell me a story.')
        self.assertEqual(t.sid,self.christoph.id)
    async def test_short_context_does_not_hide_long_background_prefix(self):
        t=self.track(self.a,self.christoph.id)
        t.submission_audio=np.full(16000,.05,np.float32)
        t.verification_audio=np.full(96000,.05,np.float32)
        from unittest.mock import Mock
        teacher=Mock(return_value=self.a)
        self.s.identity_context_s=3.04
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=teacher)):
            await self.s.identity_embedding(t,'Tell me a story.')
        self.assertIn(96000,[len(call.args[0]) for call in teacher.call_args_list])
    async def test_enrollment_anchor_uses_verified_intro_crop_not_background_preroll(self):
        t=self.track(self.b,self.christoph.id)
        t.verification_audio=np.ones(48000,np.float32)*.008
        t.enrollment_audio=np.ones(32000,np.float32)*.06
        teacher=Mock(side_effect=lambda x:self.b if len(x)==32000 else self.a)
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=teacher)):
            vector,agreement,count=await self.s.identity_embedding(t,'Hey Robbie, ich bin Robert.')
        np.testing.assert_allclose(vector,self.b)
        self.assertEqual(count,1);self.assertEqual(len(teacher.call_args.args[0]),32000)

    async def test_independently_verified_crop_outweighs_background_preroll(self):
        person,_=self.s.memory.enroll(self.b,name='Robert');person.teacher=self.b
        crop=.4*self.a+.7*self.b+np.sqrt(.35)*np.eye(192,dtype=np.float32)[2]
        whole=.6*self.a+.4*self.b+np.sqrt(.48)*np.eye(192,dtype=np.float32)[3]
        def teacher(wave):return crop if len(wave)==20480 else whole
        for verified in (False,True):
            t=self.track(self.a,self.christoph.id);t.asr_crop_verified=verified
            t.verification_audio=np.full(30720,.07,np.float32)
            with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=teacher)):
                await self.s.refine_identity(t,'Erzähl mir bitte einen kurzen Witz.')
            self.assertEqual(t.sid,person.id if verified else None)
        self.assertFalse(person.teacher_examples,'Divergent acoustic views must not adapt the identity anchor')

    async def test_strong_student_can_corroborate_but_not_override_close_teacher_winner(self):
        person,_=self.s.memory.enroll(self.b,name='Robert');person.teacher=self.b
        query=.53*self.a+.52*self.b+np.sqrt(1-.53**2-.52**2)*np.eye(192,dtype=np.float32)[2]
        for fast,expected in ((self.a,self.christoph.id),(self.b,None)):
            t=self.track(fast,person.id);t.speech_s=1.6;t.verification_audio=np.full(25600,.05,np.float32)
            await self.refine(t,'Kannst du mir eine Geschichte erzählen?',query)
            self.assertEqual(t.sid,expected)
        self.assertFalse(self.christoph.teacher_examples)

if __name__=='__main__':unittest.main()
