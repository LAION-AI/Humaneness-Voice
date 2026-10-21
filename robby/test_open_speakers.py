import unittest
from collections import deque
from unittest.mock import AsyncMock,Mock,patch
from types import SimpleNamespace
import numpy as np
from speaker_turns import SpeakerMemory,SpeakerTurns,Track
from open_speakers import claimed_name,eligible,greeting_context
from speaker_session import SpeakerSession

class OpenSpeakersTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.v=np.eye(192,dtype=np.float32);self.memory=SpeakerMemory()
    def track(self,v,sid=None):
        t=Track('candidate',0.,3.,sid=sid,speech_s=2.,channel=2)
        t.fast.extend([v]*8);t.long.extend([v]*3);return t
    def test_claims_are_explicit_not_name_mentions(self):
        for t,n in [("Ich bin's doch, Bob",'Bob'),('Ich bin es doch, Alice.','Alice'),("It's me, Bob",'Bob'),("Ich bin’s doch Bob",'Bob')]:self.assertEqual(claimed_name(t),n)
        for t in ['Bob ist hier.','My name is Alice','Ich bin müde','Es ist doch Bob.']:self.assertIsNone(claimed_name(t))
        from speaker_turns import self_name
        self.assertIsNone(self_name('Ich bin es doch, Bob.'))
    def test_open_enrollment_still_requires_clean_owned_speech(self):
        t=self.track(self.v[0]);args=[t,'Erzähle mir eine Geschichte',.9,False,False,[],self.memory]
        self.assertTrue(eligible(*args))
        for at,value in [(2,.4),(3,True),(4,True),(5,[(1.,2.)])]:
            changed=args.copy();changed[at]=value;self.assertFalse(eligible(*changed))
        t.speech_s=.4;self.assertFalse(eligible(*args))
    def test_disabled_keeps_wake_requirement(self):
        turns=SpeakerTurns(self.memory);t=self.track(self.v[0]);turns.tracks[t.key]=t
        self.assertIsNone(turns.candidate_text(t.key,'Erzähle mir etwas.'))
        decision=turns.candidate_text(t.key,'Erzähle mir etwas.',allow_unprompted=True)
        self.assertTrue(decision['new_user']);self.assertFalse(self.memory.users[decision['id']].named)
    def test_variants_preserve_anchor_and_match_each_mode(self):
        u,_=self.memory.enroll(self.v[0],self.v[0],name='Bob');u.teacher=self.v[0]
        for i in range(1,4):self.assertTrue(self.memory.confirm_variant(u.id,self.v[i],self.v[i],self.v[i]))
        self.assertFalse(self.memory.confirm_variant(u.id,self.v[4],self.v[4],self.v[4]))
        for i in range(4):
            self.assertEqual(self.memory.match(self.v[i],self.v[i])['id'],u.id)
            self.assertAlmostEqual(u.teacher_score(self.v[i]),1.)
        np.testing.assert_array_equal(u.teacher,self.v[0]);self.memory.clear();self.assertFalse(self.memory.users)
    def test_name_request_once(self):
        u,_=self.memory.enroll(self.v[0]);self.assertTrue(greeting_context({},u)['needs_name'])
        self.assertNotIn('needs_name',greeting_context({},u))
    async def test_explicit_claim_merges_guest_variant_but_not_two_named_people(self):
        s=SpeakerSession.__new__(SpeakerSession);s.memory=self.memory;s.turns=SpeakerTurns(self.memory)
        s.open_enrollment=True;s.utterance_threshold=.32;s.utterance_margin=.08;s.notice=AsyncMock()
        s.identity_embedding=AsyncMock(return_value=(self.v[1],.99,2));s.playback_voice=AsyncMock(return_value=None)
        s.owners=deque();s.asr_wakes=deque();s.records=deque()
        bob,_=self.memory.enroll(self.v[0],name='Bob');bob.teacher=self.v[0]
        guest,_=self.memory.enroll(self.v[1],force_new=True);t=self.track(self.v[1],guest.id)
        await s.refine_identity(t,"Ich bin's doch, Bob, ich spreche heute ein bisschen anders.")
        self.assertEqual(t.sid,bob.id);self.assertNotIn(guest.id,self.memory.users)
        self.assertEqual(len(bob.confirmed_variants),1)
        self.assertEqual(s.turns.speaker_context(t,'Ich bin es doch, Bob.')['display_name'],'Bob')
        alice,_=self.memory.enroll(self.v[2],name='Alice',force_new=True);t=self.track(self.v[2],alice.id)
        await s.refine_identity(t,"It's me, Bob, I sound different today.")
        self.assertIsNone(t.sid);self.assertIn(alice.id,self.memory.users)
    async def test_second_unknown_guest_is_not_compared_as_an_old_person(self):
        s=SpeakerSession.__new__(SpeakerSession);s.memory=self.memory;s.turns=SpeakerTurns(self.memory)
        s.open_enrollment=True;s.utterance_threshold=.32;s.utterance_margin=.08;s.notice=AsyncMock()
        s.identity_embedding=AsyncMock(return_value=(self.v[1],.99,2));s.playback_voice=AsyncMock(return_value=None)
        old,_=self.memory.enroll(self.v[0],name='Bob');old.teacher=self.v[0]
        guest,_=self.memory.enroll(self.v[1],force_new=True);t=self.track(self.v[1],guest.id);t.open_enrollment=True
        await s.refine_identity(t,'Was können wir heute zusammen machen?')
        self.assertEqual(t.sid,guest.id);self.assertIsNotNone(guest.teacher)
        self.assertFalse(guest.named);np.testing.assert_array_equal(old.teacher,self.v[0])
    async def test_channel_split_claim_requires_independent_words_and_acoustic_agreement(self):
        for agree in (True,False):
            s=SpeakerSession.__new__(SpeakerSession);s.memory=self.memory
            s.closed=False;s.c=SimpleNamespace(speaking=False);s.overlaps=deque();s.owners=deque()
            wave=np.full(64000,.04,np.float32);s.records=deque([(0.,4.,{},wave)])
            s.claim_observations=deque([(2.,'bob')]);s.transcribe=AsyncMock(return_value='Ich bin es doch Bob. Heute klinge ich anders.')
            t=self.track(self.v[1]);t.start=2.;t.voiced_start=2.5;t.last_speech=4.
            teacher=SimpleNamespace(embed=Mock(side_effect=[self.v[1],self.v[1] if agree else self.v[2]]))
            with patch('speaker_teacher.get_cached_teacher',return_value=teacher):result=await s.recover_voice_claim(t)
            self.assertEqual(result is not None,agree)
            self.assertEqual(hasattr(t,'verified_claim_prefix'),agree)
        s.overlaps.append((1.,2.));s.transcribe.reset_mock()
        self.assertIsNone(await s.recover_voice_claim(t));s.transcribe.assert_not_awaited()

if __name__=='__main__':unittest.main()
