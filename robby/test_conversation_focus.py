import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
import numpy as np
from conversation_focus import ConversationFocus
from sentence_session import OpenVoiceMemory, SentenceSession, unit


class FocusPolicyTests(unittest.TestCase):
    def setUp(self):
        self.v=np.eye(192,dtype=np.float32);self.memory=OpenVoiceMemory()
        self.a=self.memory.assign(self.v[0],'Ich bin Christoph.')['speaker_id']
        self.b=self.memory.assign(self.v[1],'My name is Bob.')['speaker_id']
        self.clock=[0.];self.f=ConversationFocus(clock=lambda:self.clock[0]);self.f.accept(self.a)

    def test_same_voice_and_changed_prosody_keep_owner(self):
        for score in [.33,.45,.6,.9,1.]:
            q=score*self.v[0]+np.sqrt(1-score**2)*self.v[2]
            d=self.f.check(q,self.memory.users)
            self.assertTrue(d['accepted']);self.assertEqual(d['speaker_id'],self.a)

    def test_background_and_competing_known_speaker_do_not_take_over(self):
        for q in [self.v[1],self.v[2],unit(.5*self.v[0]+.6*self.v[1])]:
            self.assertFalse(self.f.check(q,self.memory.users,True)['accepted'])
        self.assertEqual(self.f.owner,self.a)

    def test_close_rival_and_missing_audio_are_ambiguous(self):
        self.assertFalse(self.f.check(unit(self.v[0]+self.v[1]),self.memory.users)['accepted'])
        self.assertFalse(self.f.check(None,self.memory.users)['accepted'])

    def test_bot_echo_cannot_cancel_just_because_owner_is_similar(self):
        bot=unit(.8*self.v[0]+.6*self.v[2])
        self.assertFalse(self.f.check(bot,self.memory.users,True,bot)['accepted'])
        self.assertTrue(self.f.check(self.v[0],self.memory.users,True,bot)['accepted'])

    def test_expiry_release_and_disabled_allow_onboarding(self):
        self.clock[0]=181
        self.assertTrue(self.f.check(self.v[1],self.memory.users)['accepted']);self.assertIsNone(self.f.owner)
        self.f.accept(self.a);self.f.release();self.assertIsNone(self.f.owner)
        self.f.accept(self.a);self.f.enabled=False
        self.assertTrue(self.f.check(self.v[1],self.memory.users,True)['accepted'])

    def test_focus_does_not_expire_mid_reply(self):
        self.clock[0]=181;self.assertEqual(self.f.active(self.memory.users,True),self.a)

    def test_moderate_match_preserves_label_and_learns_safe_variant(self):
        variant=unit(.5*self.v[0]+np.sqrt(.75)*self.v[2])
        result=self.memory.assign(variant,'Heute klinge ich aufgeregt.',preferred=self.a,adapt=True)
        self.assertEqual(result['speaker_id'],self.a);self.assertEqual(result['voice_variants'],2)
        np.testing.assert_array_equal(self.memory.users[self.a]['vectors'][0],self.v[0])

    def test_weak_owner_match_does_not_train_anchor(self):
        variant=unit(.34*self.v[0]+np.sqrt(1-.34**2)*self.v[2])
        result=self.memory.assign(variant,'Eine andere Betonung.',preferred=self.a,adapt=True)
        self.assertEqual(result['voice_variants'],1)


class FocusIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def make(self):
        c=SimpleNamespace(ws=SimpleNamespace(send_json=AsyncMock()),speaking=True,halting=False,
            session=SimpleNamespace(reset=Mock()),ears=SimpleNamespace(sr=16000),
            playback=SimpleNamespace(rid=7,audio=np.zeros(0),n16=0),_on_barge=AsyncMock(),_send_turn=AsyncMock())
        s=SentenceSession(c);s.focus.enabled=True;s.now=s.mic_now=s.last_speech=2.;s.reply_started=0.
        s.records.append((0.,2.,np.ones(32000,np.float32)*.02))
        s.memory.assign(np.eye(192,dtype=np.float32)[0],'Ich bin Christoph.');s.focus.accept('speaker-1')
        return s

    async def test_native_barge_waits_for_owner_and_rejects_background(self):
        s=self.make();s.verify_barge=AsyncMock(return_value={'accepted':False})
        s.propose_barge({'yield_prob':.99},32000);await s.barge_task
        s.c._on_barge.assert_not_awaited();s.verify_barge.assert_awaited_once()

    async def test_literal_stop_still_uses_the_owner_guard(self):
        s=self.make();s.verify_barge=AsyncMock(return_value={'accepted':True,'similarity':.8})
        await s.preview_text('Stopp!',2.);await s.barge_task
        s.c._on_barge.assert_awaited_once();s.c.session.reset.assert_called_once()

    async def test_background_stop_never_cancels_or_discards_pending_words(self):
        s=self.make();s.verify_barge=AsyncMock(return_value={'accepted':False})
        await s.preview_text('Warte bitte!',2.);await s.barge_task
        s.c._on_barge.assert_not_awaited();s.c.session.reset.assert_not_called()

    async def test_stale_confirmation_does_not_cancel_next_reply(self):
        s=self.make();release=asyncio.Event()
        async def delayed(*args,**kwargs):await release.wait();return {'accepted':True,'similarity':.8}
        s.verify_barge=delayed;s.propose_barge({'yield_prob':1.},32000)
        await asyncio.sleep(0);s.c.playback.rid=8;release.set();await s.barge_task
        s.c._on_barge.assert_not_awaited()

    async def test_concurrent_proposals_are_single_flight(self):
        s=self.make();s.verify_barge=AsyncMock(return_value={'accepted':True,'similarity':.8})
        for _ in range(10):s.propose_barge({'yield_prob':1.},32000)
        await s.barge_task;s.c._on_barge.assert_awaited_once()

    async def test_silence_never_authorizes_a_barge(self):
        s=self.make();s.last_speech=0.
        self.assertFalse((await s.verify_barge(2.,7))['accepted'])

    async def test_moderate_voice_match_without_stop_does_not_cancel(self):
        s=self.make();s.verify_barge=AsyncMock(return_value={'accepted':True,'similarity':.40})
        s.propose_barge({'yield_prob':.99},32000);await s.barge_task
        s.c._on_barge.assert_not_awaited()

    async def test_independent_recent_features_can_veto_stop(self):
        s=self.make();s.verify_barge=AsyncMock(return_value={'accepted':True,'similarity':.8})
        s.fast_match=Mock(return_value=.05)
        await s.preview_text('Stopp!',2.);await s.barge_task
        s.c._on_barge.assert_not_awaited()

    async def test_real_teacher_path_matches_current_not_background(self):
        s=self.make();v=np.eye(192,dtype=np.float32)
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=Mock(return_value=v[0]))):
            self.assertTrue((await s.verify_barge(2.,7))['accepted'])
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=Mock(return_value=v[1]))):
            self.assertFalse((await s.verify_barge(2.,7))['accepted'])

    async def test_background_turn_is_not_enrolled_or_sent_to_llm(self):
        s=self.make();s.c.speaking=False;v=np.eye(192,dtype=np.float32)
        worker=asyncio.create_task(s.deliver())
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=Mock(return_value=v[1]))):
            await s.queue.put(('Ich erzähle etwas im Hintergrund.',np.ones(16000,np.float32),2.,0.,None))
            for _ in range(100):
                await asyncio.sleep(.002)
                if s.c.ws.send_json.await_count:break
        worker.cancel();await worker
        s.c._send_turn.assert_not_awaited();self.assertEqual(len(s.memory.users),1)

    async def test_explicit_wake_hands_over_while_idle(self):
        s=self.make();s.c.speaking=False;v=np.eye(192,dtype=np.float32)
        worker=asyncio.create_task(s.deliver())
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=Mock(return_value=v[1]))):
            await s.queue.put(('Hey Robbie, ich bin Bob.',np.ones(16000,np.float32),2.,0.,None))
            for _ in range(100):
                await asyncio.sleep(.002)
                if s.c._send_turn.await_count:break
        worker.cancel();await worker
        s.c._send_turn.assert_awaited_once();self.assertEqual(s.focus.owner,'speaker-2')

    async def test_short_speaker_rejection_cannot_fake_a_pause_or_cut_words(self):
        s=self.make();s.last_speech=1.;s.focus_activity={'at':1.9,'accepted':False,'last_owned':1.}
        s.vad=SimpleNamespace(active=True,probability=.99)
        self.assertFalse(s.quiet_ready())


if __name__=='__main__':unittest.main()
