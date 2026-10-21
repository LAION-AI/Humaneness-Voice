import unittest
import numpy as np
from identity import IdentityManager
from audio_sync import estimate_echo_delay


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.a=np.eye(192,dtype=np.float32)[0];self.b=np.eye(192,dtype=np.float32)[1]
        self.r=np.eye(192,dtype=np.float32)[2];self.now=100
        self.manager=IdentityManager(robo=self.r,clock=lambda:self.now)

    def say(self,vector,text):
        for _ in range(4):self.manager.observe({'embedding':vector,'kind':'speech','confidence':.9})
        return self.manager.finish(text,enabled=True)

    def test_multiuser_and_unknown(self):
        self.assertFalse(self.say(self.a,'How are you?')['accepted'])
        self.assertTrue(self.say(self.a,'Hey Robo, I would like to talk to you.')['new_user'])
        self.assertTrue(self.say(self.a,'Tell me something.')['accepted'])
        self.assertTrue(self.say(self.b,'Hallo Robo, ich bin auch hier.')['new_user'])
        self.assertEqual(len(self.manager.users),2)
        self.assertFalse(self.say(self.r,'Hey Robo, this is your own voice.')['accepted'])

    def test_uncertainty_expiry_and_clear(self):
        self.manager.observe({'embedding':None,'kind':'overlap','confidence':.9})
        self.assertFalse(self.manager.finish('Hey Robo',enabled=True)['accepted'])
        self.say(self.a,'Hey Robo, please enroll me.');self.now+=301
        self.assertFalse(self.say(self.a,'Are you there?')['accepted'])
        self.manager.clear();self.assertEqual(self.manager.status()['users'],[])

    def test_calibration(self):
        rng=np.random.default_rng(7);reference=rng.normal(0,.03,32000)
        for ms in [0,13.5,80,180,440]:
            delay=round(ms*16);mic=np.r_[np.zeros(delay),reference][:len(reference)]*.1
            result=estimate_echo_delay(mic,reference)
            self.assertTrue(result['available']);self.assertAlmostEqual(result['delay_ms'],ms,places=1)
        self.assertFalse(estimate_echo_delay(np.zeros_like(reference),reference)['available'])
        self.assertFalse(estimate_echo_delay(rng.normal(size=len(reference)),reference)['available'])
    def test_content_check_catches_advanced_delayed_and_inverted_playback(self):
        reference=np.random.default_rng(19).normal(0,.03,32000)
        for shift in (-1280,0,1920):
            for sign in (-1,1):
                mic=np.roll(reference,shift)*sign*.05
                value=estimate_echo_delay(mic,reference,symmetric=True,absolute=True)
                self.assertGreater(value['confidence'],.99)
                self.assertAlmostEqual(value['delay_ms'],shift/16,places=2)


if __name__=='__main__':unittest.main()
