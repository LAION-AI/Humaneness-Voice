import unittest
import numpy as np
from addressed_speech import AddressedSpeechGate

class Addressing(unittest.TestCase):
    def setUp(self):
        self.g=AddressedSpeechGate();self.v=np.eye(192,dtype=np.float32)
        self.users={'a':{'level':.06,'vectors':[self.v[0]]}}
    def test_unknown_foreground_can_join_without_wake_word(self):
        self.assertTrue(self.g.evaluate('Was können wir heute machen?',self.v[1],self.users,.05)['accepted'])
    def test_quiet_unknown_broadcast_statement_is_withheld(self):
        row=self.g.evaluate('Morgen gibt es einen Wetterbericht.',self.v[1],self.users,.006,-55)
        self.assertFalse(row['accepted']);self.assertEqual(row['relative_level'],.1)
    def test_quiet_known_voice_is_not_rejected_for_volume_alone(self):
        self.assertTrue(self.g.evaluate('Noch einen bitte.',self.v[0],self.users,.006)['accepted'])
    def test_quiet_direct_address_and_self_introduction_can_join(self):
        for text in ('Robby, hör mal.','Kannst du mir helfen?','Ich bin Alice.','Could you tell me a story?'):
            self.assertTrue(self.g.evaluate(text,self.v[1],self.users,.006)['accepted'],text)
    def test_no_anchor_cannot_invent_background_identity(self):
        self.assertTrue(self.g.evaluate('Guten Morgen.',self.v[1],{},.006)['accepted'])
    def test_disabling_restores_open_audio_logic(self):
        self.g.enabled=False
        self.assertTrue(self.g.evaluate('Morgen Wetterbericht.',self.v[1],self.users,.006)['accepted'])
    def test_low_snr_alone_is_never_an_identity_veto(self):
        self.assertTrue(self.g.evaluate('Guten Morgen.',self.v[1],self.users,.05,-25)['accepted'])
    def test_name_answer_labels_recent_unknown_and_keeps_both_variants(self):
        from sentence_session import OpenVoiceMemory
        memory=OpenVoiceMemory()
        guest=memory.assign(self.v[0],'Was können wir heute machen?')
        changed=.31*self.v[0]+np.sqrt(1-.31**2)*self.v[1]
        named=memory.assign(changed,'Ich bin Selma.')
        self.assertEqual(named['speaker_id'],guest['speaker_id']);self.assertEqual(named['voice_variants'],2)
        self.assertEqual(memory.assign(self.v[0],'Erzähl mir etwas.')['display_name'],'Selma')
    def test_very_different_name_answer_does_not_force_guest_merge(self):
        from sentence_session import OpenVoiceMemory
        memory=OpenVoiceMemory();guest=memory.assign(self.v[0],'Hallo zusammen.')
        other=memory.assign(self.v[1],'Ich bin Bob.')
        self.assertNotEqual(guest['speaker_id'],other['speaker_id'])
if __name__=='__main__':unittest.main()
