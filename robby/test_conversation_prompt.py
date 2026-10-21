import unittest
from conversation_prompt import message,speaker_update,POLICY


class PromptTests(unittest.TestCase):
    def test_information_reading_is_calm_without_a_slow_preamble(self):
        from conversation_director import RECIPE
        self.assertIn('unhurried',POLICY);self.assertIn('Wikipedia',RECIPE)
        self.assertIn('clear articulation',RECIPE);self.assertIn('takes precedence',RECIPE)
    def test_local_metadata_requires_bound_name_and_full_introduction(self):
        identity={'enrolled':True,'self_introduction':True,'speaker_id':'opaque1','display_name':'Christoph','named':True}
        self.assertEqual(speaker_update('Hey Robbie, ich bin Christoph.',identity)['display_name'],'Christoph')
        self.assertIsNone(speaker_update('Hey Robbie, ich bin Robert.',identity))
        self.assertIsNone(speaker_update('Christoph sagt hallo.',identity))
        self.assertIsNone(speaker_update('Ich bin Christoph.',{**identity,'enrolled':False}))
    def test_prompt_separates_task_context_from_identity(self):
        value=message('Bitte genau drei Ideen.',{'enrolled':True,'named':True,'display_name':'Selma','speaker_id':'opaque'})
        self.assertIn(POLICY,value);self.assertIn('Current user utterance:\nBitte genau drei Ideen.',value)
        self.assertIn('exact words',value);self.assertIn('no device-control',value)
