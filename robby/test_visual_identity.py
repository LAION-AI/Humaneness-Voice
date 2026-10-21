import unittest
from visual_identity import VisualIdentity

class VisualIdentityTests(unittest.TestCase):
    def setUp(self):
        self.now=100.;self.v=VisualIdentity(lambda:self.now)
        self.v.configure({'ignoreOffscreenSpeakers':True,'faceIdEnabled':True})
    def frames(self,faces,start=0.,count=15):
        for i in range(count):
            at=round(start+i*.1,3);self.v.observe({'camera':'on','audio_s':at,'faces':faces},at)
    def test_defaults_independent_typed(self):
        defaults=VisualIdentity().settings
        self.assertFalse(defaults['ignoreOffscreenSpeakers']);self.assertFalse(defaults['faceIdEnabled'])
        self.assertTrue(defaults['lookAtActiveSpeaker']);self.v.configure({'ignoreOffscreenSpeakers':False,'lookAtActiveSpeaker':'false'})
        self.assertFalse(self.v.settings['ignoreOffscreenSpeakers']);self.assertTrue(self.v.settings['lookAtActiveSpeaker'])
    def test_camera_fallback(self):
        self.assertTrue(self.v.gate(0,2)['allowed']);self.v.observe({'camera':'unavailable'},2);self.assertTrue(self.v.gate(0,2)['fallback'])
    def test_static_or_offscreen(self):
        self.frames([{'id':'face-1','confidence':0,'speaking':False}]);self.assertFalse(self.v.gate(0,1.4)['allowed'])
    def test_aligned_speaker(self):
        self.frames([{'id':'face-1','confidence':.9,'speaking':True},{'id':'face-2','confidence':0,'speaking':False}])
        self.assertEqual(self.v.gate(0,1.4)['track_id'],'face-1');self.assertTrue(self.v.gate(0,1.4)['allowed'])
    def test_overlap(self):
        self.frames([{'id':'face-1','confidence':.9,'speaking':True},{'id':'face-2','confidence':.8,'speaking':True}]);self.assertFalse(self.v.gate(0,1.4)['allowed'])
    def test_equal_speaker_switch_ambiguous(self):
        self.frames([{'id':'face-1','confidence':.9,'speaking':True}],count=7);self.frames([{'id':'face-2','confidence':.9,'speaking':True}],.7,7)
        self.assertFalse(self.v.gate(0,1.4)['allowed'])
    def test_disabled_filter(self):
        self.frames([]);self.v.configure({'ignoreOffscreenSpeakers':False});self.assertTrue(self.v.gate(0,1.4)['allowed'])
    def test_disabled_face_id_cannot_corroborate_or_bind(self):
        self.v.bind('face-1',{'speaker_id':'a','display_name':'Alice'});self.v.face_verified['face-1']=self.now
        self.v.configure({'faceIdEnabled':False})
        self.assertFalse(self.v.bindings);self.assertFalse(self.v.face_verified)
        self.assertFalse(self.v.corroborates('a',0,2));self.assertIsNone(self.v.bind('face-1',{'speaker_id':'a','display_name':'Alice'}))
        self.assertTrue(self.v.people)
    def test_default_gate_does_not_depend_on_camera(self):
        self.v=VisualIdentity(lambda:self.now)
        self.frames([]);self.now+=30
        result=self.v.gate(0,1.4);self.assertTrue(result['allowed']);self.assertIsNone(result['track_id'])
    def test_stale_and_other_utterance(self):
        self.frames([{'id':'face-1','confidence':.9,'speaking':True}]);self.assertFalse(self.v.gate(10,12)['allowed']);self.now+=3;self.assertFalse(self.v.gate(0,1.4)['allowed'])
    def test_bad_clocks(self):
        for value in [float('nan'),float('inf'),-1,100,'1',True]:
            with self.assertRaises(ValueError):self.v.observe({'camera':'on','audio_s':value,'faces':[]},0)
    def test_bad_clocks_do_not_keep_camera_fresh(self):
        self.frames([{'id':'face-1','confidence':.9,'speaking':True}]);self.now+=3
        with self.assertRaises(ValueError):self.v.observe({'camera':'on','audio_s':100,'faces':[]},2)
        self.assertFalse(self.v.gate(0,1.4)['allowed'])
        self.assertIn('stale',self.v.gate(0,1.4)['reason'])
    def test_rejection_diagnostics(self):
        self.frames([]);self.assertEqual(self.v.gate(0,1.4)['code'],'visual_no_face')
        self.assertEqual(self.v.gate(10,12)['code'],'visual_timing')
        self.v=VisualIdentity(lambda:self.now);self.v.configure({'ignoreOffscreenSpeakers':True});self.frames([{'id':'face-1','confidence':.2,'speaking':False}])
        result=self.v.gate(0,1.4);self.assertEqual(result['code'],'visual_no_motion')
        self.assertEqual(result['diagnostics']['visible_frames'],15)
    def test_voice_face_conflict(self):
        a=self.v.bind('face-1',{'speaker_id':'a','display_name':'Alice'});self.assertIsNone(self.v.bind('face-1',{'speaker_id':'b','display_name':'Bob'}))
        self.assertEqual(a['person_id'],self.v.bind('face-2',{'speaker_id':'a','display_name':'Alice'})['person_id'])
    def test_greeting_cooldowns_language(self):
        self.assertEqual(self.v.greeting('face-1','Hallo','de'),'Hallo, wer bist du?');self.assertIsNone(self.v.greeting('face-1','Hallo','de'))
        self.assertIsNone(self.v.greeting('face-2','Hello','en'));self.now+=16;self.assertEqual(self.v.greeting('face-2','Hello','en'),'Hi, who are you?')
    def test_question_or_introduction_not_replaced(self):
        for text in ['Hey Robby, ich bin Bob','Hi Robby, my name is Alice','Wie geht es dir','Can you help','Erzähl mir etwas?']:
            self.assertIsNone(self.v.greeting('face-1',text,'de'))
    def test_greeting_disabled_or_known(self):
        self.v.configure({'greetUnknownSpeakers':False});self.assertIsNone(self.v.greeting('face-1','Hello','en'))
        self.v.configure({'greetUnknownSpeakers':True});self.v.bind('face-1',{'speaker_id':'a','display_name':'Alice'});self.assertIsNone(self.v.greeting('face-1','Hello','en'))
    def test_ambiguity_forget(self):
        self.v.bind('face-1',{'speaker_id':'a','display_name':'Alice'});self.frames([{'id':'face-1','confidence':0,'speaking':False,'ambiguous':True}]);self.assertNotIn('face-1',self.v.bindings)
        self.v.forget();self.assertFalse(self.v.people);self.assertFalse(self.v.frames)
    def test_face_corroboration_needs_fresh_identity_and_matching_moving_track(self):
        person=self.v.bind('face-1',{'speaker_id':'a','display_name':'Alice'})
        self.frames([{'id':'face-1','confidence':.9,'speaking':True}])
        self.assertFalse(self.v.corroborates('a',0,1.4))
        self.v.face_verified['face-1']=self.now
        self.assertTrue(self.v.corroborates('a',0,1.4));self.assertFalse(self.v.corroborates('b',0,1.4))
        self.now+=6.1;self.assertFalse(self.v.corroborates('a',0,1.4))

if __name__=='__main__':unittest.main()
