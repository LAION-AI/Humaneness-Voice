"""Race/cancellation/word-limit contracts; no model or network calls."""
import asyncio,json,unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch,AsyncMock
import numpy as np
from quick_reaction import validate,eligible,duplicate_acknowledgment
from bolt_text import Sentence

class Words(unittest.TestCase):
    def test_emergency_and_trivial_arithmetic_skip_only_the_preface(self):
        for s in ['Hilfe, mein Kind bekommt keine Luft!','He is not breathing.','Wie viel ist sieben mal acht?']:
            self.assertFalse(eligible(s,[]),s)
        self.assertTrue(eligible('Erklär mir, wie Multiplikation funktioniert.',[]))
    def test_recent_reaction_is_not_repeated_and_contractions_are_valid(self):
        from quick_reaction import repeated_opening
        h=[{'role':'assistant','content':'Das kitzelt meine Lachschraube! Warum sitzt die Katze am PC?'}]
        self.assertTrue(repeated_opening('Das kitzelt meine Lachschraube.',h))
        self.assertFalse(repeated_opening('Noch eine Runde Lachmuskeltraining.',h))
        self.assertEqual(validate("That's really sweet of you."),"That's really sweet of you.")
        for s in ['Das notiere ich mir.','Ich schaue nach.','Ein weiterer Witz kommt.']:
            self.assertIsNone(validate(s),s)
    def test_tone_preserves_seriousness_and_compound_joke_thread(self):
        from quick_reaction import reaction_tone
        from conversation_tasks import joke_task
        for subject in ['Katzenwitz','Pinguinwitz','Roboterwitz']:
            h=[{'role':'user','content':'Erzähl mir einen '+subject+'.'}]
            self.assertIsNotNone(joke_task('Und noch einen!',h))
            self.assertEqual(reaction_tone('Und noch einen!',h),'joke')
            self.assertEqual(reaction_tone('Der war lahm. Noch einen, aber lustiger.',h),'joke')
        self.assertIsNone(joke_task('Ich schwitze.',[]))
        self.assertIsNone(joke_task('Erzähl mir etwas über Auschwitz.',[]))
        self.assertEqual(reaction_tone('Mein Hund ist gestorben. Keine Witze bitte.',[]),'serious')
        self.assertEqual(reaction_tone('Ich streichle dir den Kopf.',[]),'affection')
        self.assertEqual(reaction_tone('Ich habe Angst vor morgen.',[]),'support')
    def test_explicit_english_command_applies_to_the_reaction(self):
        from speech_language import reply_language
        self.assertEqual(reply_language('Please speak English. My cat died yesterday.'),'English')
        self.assertEqual(reply_language('Can you speak English for this answer?'),'English')
        self.assertEqual(reply_language('I speak English and German.'),'German')
    def test_malformed_optional_cue_does_not_drop_the_spoken_answer(self):
        from bolt_text import SentenceFramer
        for raw in ('fröhlich!', 'warm / soft', 'pause: 2 seconds'):
            frames=SentenceFramer().feed('('+raw+') Hier ist die ganze Antwort.',final=True)
            self.assertEqual(frames[0].text,'Hier ist die ganze Antwort.')
            self.assertEqual(frames[0].direction,'contentment, soft')
    def test_short_reaction_envelope_preserves_samples_and_soft_edges(self):
        from quick_reaction import AudioEnvelope
        for size in (90,240,5000):
            env=AudioEnvelope();source=np.ones(size,np.float32)
            parts=[env.feed(source[i:i+70]) for i in range(0,size,70)]+[env.finish()]
            out=np.concatenate(parts)
            self.assertEqual(len(out),size);self.assertEqual(out[0],0);self.assertEqual(out[-1],0)
            self.assertLessEqual(float(out.max()),1.)
    def test_six_words_is_a_limit_not_a_truncation(self):
        self.assertEqual(validate('Das klingt nach einem langen Tag.'),'Das klingt nach einem langen Tag.')
        for s in ['Das klingt nach einem wirklich langen Tag.','(warm) Ich höre dir zu.','Ja.','NONE',
                  'Ich höre dir zu.\nEtwas anderes.','Na klar! 😊','Ein Witz kommt gleich.']:
            self.assertIsNone(validate(s),s)
    def test_exact_output_and_stop_remain_exact(self):
        for s in ['Sag nur Hallo.','Ein Gedicht mit acht Zeilen.','Danke, das reicht.','Be quiet.','Antworte in genau zwei Sätzen.']:
            self.assertFalse(eligible(s,[],remote=True),s)
        self.assertTrue(eligible('Heute war mein Tag anstrengend.',[],remote=True))
        self.assertFalse(eligible('Heute war mein Tag anstrengend.',[],remote=False))
    def test_only_duplicate_short_acknowledgment_is_dropped(self):
        self.assertTrue(duplicate_acknowledgment('Das klingt nach viel.','Das klingt nach viel!'))
        self.assertFalse(duplicate_acknowledgment('Das klingt nach viel.','Das klingt nach viel, aber morgen gilt ein anderer Plan.'))
        self.assertTrue(duplicate_acknowledgment('Das tut mir sehr leid.','Oh, das tut mir so, so sehr leid.'))
        self.assertTrue(duplicate_acknowledgment("I'm so sorry.",'I am so deeply sorry to hear that.'))
        self.assertFalse(duplicate_acknowledgment('Das tut mir leid.','Das tut mir leid, und ich kann dir beim Planen helfen.'))

class Race(unittest.IsolatedAsyncioTestCase):
    async def collect(self,main_delay,lead_delay,enabled=True,tool=False,close_early=False,invalid=False,pcm=False):
        from bolt_pipeline import pipeline
        closed=[]
        async def stream(*args,**kwargs):
            try:
                await asyncio.sleep(main_delay);yield Sentence('Hier steht die eigentliche vollständige Antwort.',language='de')
            finally:closed.append('main')
        async def lead(*args,**kwargs):
            try:
                await asyncio.sleep(lead_delay)
                return {'text':None if invalid else 'Ich höre dir zu.','ms':1.,'status':'invalid_or_abstained' if invalid else 'ready','model':'local','word_count':4}
            finally:closed.append('local')
        async def directed(*args,**kwargs):
            await asyncio.sleep(main_delay)
            return {'script':'(interest) Die Regel ist eingerichtet.','language':'de','director_ms':10.,'interaction_actions':[{'name':'real_action'}]}
        class Response:
            async def aiter_lines(self):
                if pcm:
                    import base64
                    yield 'data: '+json.dumps({'event':'audio_chunk','pcm_f32':base64.b64encode(np.ones(48000,dtype='<f4').tobytes()).decode(),'sample_rate':48000,'channels':1})
                else:yield 'data: '+json.dumps({'event':'audio_chunk','url':'/synthetic.mp3'})
        @asynccontextmanager
        async def admission(*args,**kwargs):yield Response()
        class Client:
            async def get(self,*args,**kwargs):return SimpleNamespace(content=b'synthetic',raise_for_status=lambda:None)
        output=[]
        with patch('bolt_pipeline.BoltDirector.stream',stream),patch('quick_reaction.generate',lead),\
             patch('bolt_pipeline.admitted_turn',admission),patch('tts_bridge.decode_mp3',AsyncMock(return_value=np.ones(48000,np.float32)*.01)),\
             patch('bolt_pipeline.needs_tools',return_value=tool),patch('conversation_director.CompactDirector.generate',directed):
            it=pipeline(Client(),'Heute war mein Tag anstrengend.',{},[],llm_selection={'model':'qwen','provider':'cerebras/fp16'},quick_replies=enabled)
            async for kind,value in it:
                if kind!='audio':output.append((kind,json.loads(json.dumps(value))))
                if close_early and kind=='audio':await it.aclose();break
        return output,closed
    async def test_main_first_skips_and_cancels_local(self):
        output,closed=await self.collect(.001,.2)
        self.assertEqual([v['sentence_role'] for k,v in output if k=='sentence'],['answer'])
        self.assertEqual(output[-1][1]['quick_reply_status'],'main_answer_won');self.assertIn('local',closed)
    async def test_local_first_keeps_full_answer(self):
        output,_=await self.collect(.02,.001)
        self.assertEqual([v['sentence_role'] for k,v in output if k=='sentence'],['reaction','answer'])
        self.assertIn('eigentliche vollständige Antwort',next(v['text'] for k,v in reversed(output) if k=='sentence'))
        self.assertIn('answer_first_audio_ms',output[-1][1]);self.assertEqual(output[-1][1]['quick_reply_words'],4)
    async def test_disable_and_invalid_reaction_still_answer(self):
        for enabled,invalid in [(False,False),(True,True)]:
            output,_=await self.collect(.01,0,enabled=enabled,invalid=invalid)
            self.assertEqual([v['sentence_role'] for k,v in output if k=='sentence'],['answer'])
    async def test_tool_actions_are_not_lost_after_reaction(self):
        output,_=await self.collect(.02,.001,tool=True)
        actions=[v['interaction_actions'] for k,v in output if k=='sentence']
        self.assertEqual(actions,[[],[{'name':'real_action'}]])
    async def test_cancel_closes_main_generation_and_queue(self):
        _,closed=await self.collect(.4,.001,close_early=True)
        self.assertIn('main',closed);self.assertIn('local',closed)
    async def test_direct_pcm_keeps_the_same_reply_contract(self):
        output,_=await self.collect(.02,.001,pcm=True)
        self.assertEqual([v['sentence_role'] for k,v in output if k=='sentence'],['reaction','answer'])
        self.assertGreater(output[-1][1]['audio_s'],1.)

if __name__=='__main__':unittest.main()
