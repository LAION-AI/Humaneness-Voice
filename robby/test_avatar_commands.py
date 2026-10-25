import asyncio,unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock,patch
from avatar_actions import EMOTIONS
from avatar_commands import explicit_actions

class CommandTests(unittest.TestCase):
    def test_common_german_english_commands_and_order(self):
        cases=[('Kannst du bitte eine lustige Grimasse schneiden?',['grimace']),
            ('Mach mal eine verrückte Grimasse!',['grimace']),('Make some funny faces.',['grimace']),
            ('Schiele bitte und rolle dann mit den Augen.',['grimace','eye_roll']),
            ('Guck bitte wütend und mach dann eine lustige Grimasse.',['emotion','grimace']),
            ('Schüttel bitte den Kopf und nicke danach.',['shake','nod']),
            ('Please wink.',['wink']),('Mach ein Fischgesicht.',['grimace'])]
        for text,kinds in cases:self.assertEqual([s['kind'] for s in explicit_actions(text)],kinds,text)
        self.assertEqual(explicit_actions(cases[3][0])[0]['variant'],'crossed')
        self.assertEqual(explicit_actions(cases[4][0])[0]['emotion'],'anger')
        self.assertEqual(explicit_actions(cases[-1][0])[0]['variant'],'fish')
        self.assertEqual(explicit_actions('Mach ein wütendes Gesicht.')[0]['emotion'],'anger')
        self.assertEqual(explicit_actions('Guck bitte lustig.')[0]['emotion'],'amusement')
        self.assertEqual(explicit_actions('Show me how you make a goofy face.')[0]['kind'],'grimace')
        for text in ['Mach eine Grimasse und sag etwas Lustiges.','Mach eine Grimasse und sag "Hallo".','Mach bitte grimmassen.','Sag etwas Lustiges und mach eine Grimasse.']:
            self.assertTrue(explicit_actions(text),text)
    def test_negation_quotes_explanations_conditions_and_descriptions_never_trigger(self):
        for text in ['Mach keine Grimasse.','Do not make a funny face.','Sag "Mach eine Grimasse".',
            'Wenn ich Banane sage, mach eine Grimasse.','Erkläre mir, wie man Grimassen schneidet.',
            'Alice looks angry.','Ich bin Nicky.','Mach ein Gedicht über einen traurigen Roboter.',
            'Bitte sag mach eine Grimasse.','Kannst du bitte sagen mach eine Grimasse?']:
            self.assertEqual(explicit_actions(text),[],text)
    def test_all_catalogue_emotions_and_bounded_varied_faces(self):
        for emotion in EMOTIONS:
            self.assertEqual(explicit_actions('Zeige den Gesichtsausdruck '+emotion)[0]['emotion'],emotion)
        a=explicit_actions('Mach Grimassen.',0);b=explicit_actions('Mach Grimassen.',1)
        self.assertEqual(len(a),2);self.assertNotEqual(a[0]['variant'],b[0]['variant'])
        self.assertLessEqual(len(explicit_actions('Mach Grimassen und Grimassen und Grimassen und Grimassen.')),6)

class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_animation_dispatched_before_even_starting_llm_and_survives_failure(self):
        from bolt_pipeline import pipeline
        async def fail(*args,**kwargs):
            raise RuntimeError('LLM unavailable')
            yield
        with patch('bolt_pipeline.BoltDirector.stream',fail):
            stream=pipeline(None,'Mach eine lustige Grimasse.',{},[],tts=False)
            kind,value=await anext(stream)
            self.assertEqual(kind,'avatar_action');self.assertEqual(value['steps'][0]['kind'],'grimace')
            with self.assertRaisesRegex(RuntimeError,'unavailable'):await anext(stream)
            await stream.aclose()
    async def test_direct_action_does_not_use_fragile_tool_json_and_still_produces_speech(self):
        from bolt_pipeline import pipeline
        from bolt_text import Sentence
        async def words(*args,**kwargs):yield Sentence('Schau mal, ein lustiges Fischgesicht!','amusement',language='de')
        with patch('bolt_pipeline.BoltDirector.stream',words),patch('conversation_director.CompactDirector.generate',side_effect=AssertionError('No JSON tool loop')):
            events=[(k,v) async for k,v in pipeline(None,'Mach eine Grimasse.',{},[],interaction_state={'rules':[]},tts=False)]
        self.assertEqual(events[0][0],'avatar_action')
        speech=next(v for k,v in events if k=='sentence')
        self.assertTrue(speech['text']);self.assertEqual(speech['avatar_action_id'],events[0][1]['action_id'])
    async def test_pure_spin_keeps_its_playback_anchor(self):
        from bolt_pipeline import pipeline
        from bolt_text import Sentence
        async def words(*args,**kwargs):yield Sentence('Hui, ich drehe mich!','elation',language='de')
        with patch('bolt_pipeline.BoltDirector.stream',words):
            events=[(k,v) async for k,v in pipeline(None,'Dreh dich bitte.',{},[],tts=False)]
        self.assertNotIn('avatar_action',[k for k,v in events])
        self.assertEqual(next(v for k,v in events if k=='sentence')['avatar_actions'][0]['kind'],'spin')
    async def test_empty_or_malformed_visual_speech_has_one_llm_retry(self):
        from bolt_pipeline import pipeline
        from bolt_text import Sentence
        for malformed in [False,True]:
            calls=[]
            async def words(*args,**kwargs):
                calls.append(args)
                if len(calls)==1:
                    if malformed:raise ValueError('Invalid acting cue')
                    return
                yield Sentence('Jetzt kommt mein lustiges Fischgesicht!','amusement',language='de')
            with patch('bolt_pipeline.BoltDirector.stream',words):
                events=[(k,v) async for k,v in pipeline(None,'Mach eine Grimasse.',{},[],tts=False)]
            self.assertEqual(len(calls),2);self.assertEqual(sum(k=='avatar_action' for k,v in events),1)
            self.assertTrue(any(k=='sentence' for k,v in events))
    async def test_explicitly_silent_request_needs_neither_llm_nor_tts(self):
        from bolt_pipeline import pipeline
        with patch('bolt_pipeline.BoltDirector.stream',side_effect=AssertionError('Silent action')):
            events=[(k,v) async for k,v in pipeline(None,'Mach eine Grimasse ohne etwas zu sagen.',{},[])]
        self.assertEqual([k for k,v in events],['avatar_action','done']);self.assertTrue(events[-1][1]['silent'])

if __name__=='__main__':unittest.main()
