"""Two-stage ordering, lookahead, failure and cancellation contracts."""
import asyncio,base64,time,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from bolt_text import Sentence
from informed_bridge import validate

class Validation(unittest.TestCase):
    def test_memory_hint_is_bounded_and_prefers_explicit_correction(self):
        from quick_reaction import memory_hint
        rows=[{'title':'Bob','detail':'Jazz'},{'title':'Alice','detail':'Wandern'}]
        self.assertEqual(memory_hint({'display_name':'Bob'},[],{'entries':rows})['current_person_facts'][0]['detail'],'Jazz')
        hint=memory_hint({'display_name':'Bob'},[],{'entries':rows},'Ich heiße Alice, nicht Bob.')
        self.assertEqual(hint['current_person'],'Alice')
        self.assertEqual(hint['current_person_facts'][0]['detail'],'Wandern')
        self.assertFalse(memory_hint({},[],{'entries':rows})['current_person_facts'])
        self.assertFalse(memory_hint({},[],None)['memory_available'])
        self.assertLess(len(str(memory_hint({},[],{'entries':[{'title':str(i),'detail':'x'*2000} for i in range(500)]}))),4600)
    def test_waiting_announcements_are_never_spoken(self):
        from quick_reaction import validate as reaction
        for value in ['Lass mich mal nachdenken.','Einen kleinen Moment bitte.','Ich muss kurz überlegen.','Let me think.','Give me a moment.']:
            self.assertIsNone(reaction(value),value)
    def test_accepts_short_link_but_not_repeated_or_malformed_speech(self):
        self.assertEqual(validate('Dabei hilft ein Blick aufs Licht.','Das ist interessant.','Licht besteht aus Farben.'),'Dabei hilft ein Blick aufs Licht.')
        for text in ('NONE','Das ist interessant.','Warum ist das so?','(warm) Ein Gedanke.','Ich werde nachschauen.','a '*13):
            self.assertIsNone(validate(text,'Das ist interessant.','Licht besteht aus Farben.'))

class FakeLive:
    def __init__(self,blocked=False):
        self.submissions=[];self.closed=False;self.finished=False;self.performance={'weight_instances':1};self.blocked=blocked
        self.receiver=asyncio.get_running_loop().create_future()
    async def submit(self,sentence,role,order,chunk_frames=5):
        job=SimpleNamespace(id=str(order),started=time.perf_counter(),first_pcm=time.perf_counter())
        self.submissions.append((role,sentence.text,job));return job
    async def events(self,job):
        if self.blocked:await asyncio.Future()
        yield {'event':'audio_chunk','pcm_f32':base64.b64encode(np.ones(4800,dtype='<f4').tobytes()).decode()}
        yield {'event':'job_done'}
    async def finish(self):
        self.finished=True;self.receiver.set_result(None)
    async def close(self):self.closed=True

class Flow(unittest.IsolatedAsyncioTestCase):
    async def test_retrieved_memory_reaches_first_reaction_without_second_lookup(self):
        from bolt_pipeline import pipeline
        remembered={'entries':[{'title':'Bob','summary':'Bob mag Jazz.'}],'metrics':{'retrieval_ms':1}}
        class Memory:
            async def prepare(self,*args):return [],remembered
        async def main(*args,**kwargs):yield Sentence('Du magst Jazz.',language='de')
        async def reaction(*args,**kwargs):
            self.assertIs(kwargs['memory'],remembered)
            return {'text':None,'word_count':0,'status':'invalid_or_abstained','ms':1}
        live=FakeLive()
        async def open_live(*args):return live
        with patch('bolt_pipeline.BoltDirector.stream',main),patch('quick_reaction.generate',reaction),patch('live_speech.LiveSpeech.open',open_live):
            events=[e async for e in pipeline(None,'Erinnerst du dich an mich?',{'display_name':'Bob'},[],memory=Memory(),llm_selection={'model':'qwen'},quick_replies=True)]
        self.assertTrue(live.closed)
        self.assertEqual([e[1]['sentence_text'] for e in events if e[0]=='sentence'],['Du magst Jazz.'])
    def setup_flow(self,live,*,intro=True,main=True,count=2,text='Warum ist der Himmel blau?'):
        from two_stage_pipeline import render
        queue=asyncio.Queue()
        for i in range(count):queue.put_nowait(Sentence(f'Die weitere Antwort Nummer {i}.',language='de'))
        queue.put_nowait(None)
        async def ready(value):return value
        self.first=asyncio.create_task(ready(Sentence('Licht wird in der Luft gestreut.',language='de') if main else None))
        self.reaction=asyncio.create_task(ready({'text':'Das ist interessant.' if intro else None,'word_count':3,'status':'ready','ms':1}))
        return render(None,live,text=text,identity={},history=[],queue=queue,first_main=self.first,
            reaction_task=self.reaction,metrics={},start=time.perf_counter(),context={},avatar_steps=[],action_id=None,
            actions=[{'action':'one'}],sources=[],lookups=[],chunk_frames=20)
    async def test_answer_prefetch_and_no_second_filler_model(self):
        from unittest.mock import AsyncMock
        live=FakeLive()
        with patch('informed_bridge.generate',new_callable=AsyncMock) as bridge:
            events=[e async for e in self.setup_flow(live)]
            bridge.assert_not_awaited()
        sentences=[e[1] for e in events if e[0]=='sentence']
        self.assertEqual([s['sentence_role'] for s in sentences],['reaction','answer','answer','answer'])
        self.assertEqual(sum(bool(s['interaction_actions']) for s in sentences),1)
        self.assertEqual(sentences[1]['sentence_text'],'Licht wird in der Luft gestreut.')
        self.assertTrue(live.finished and live.closed)
    async def test_abstention_keeps_the_whole_main_answer(self):
        live=FakeLive()
        async def no_link(*a):return {'text':None,'ms':2,'status':'timeout'}
        with patch('informed_bridge.generate',no_link):events=[e async for e in self.setup_flow(live)]
        self.assertEqual([e[1]['sentence_role'] for e in events if e[0]=='sentence'],['reaction','answer','answer','answer'])
    async def test_joke_bridge_does_not_receive_second_sentence_punchline(self):
        live=FakeLive()
        async def bridge(client,text,reaction,opening,*args):
            self.assertEqual(opening,'Licht wird in der Luft gestreut.')
            return {'text':None,'ms':1,'status':'invalid_or_abstained'}
        with patch('informed_bridge.generate',bridge):
            events=await self.collect(self.setup_flow(live,text='Erzähl mir einen Katzenwitz.'))
        self.assertEqual(len([e for e in events if e[0]=='sentence']),4)
    async def test_no_main_seals_session_without_waiting_ten_seconds(self):
        live=FakeLive();await asyncio.wait_for(self.collect(self.setup_flow(live,main=False)),.5)
        self.assertTrue(live.closed and live.finished)
    async def test_no_reaction_still_delivers_full_answer(self):
        live=FakeLive();events=await self.collect(self.setup_flow(live,intro=False))
        self.assertEqual([e[1]['sentence_role'] for e in events if e[0]=='sentence'],['answer']*3)
    async def test_cancel_with_full_prefetch_queue_does_not_deadlock(self):
        live=FakeLive(blocked=True)
        async def bridge(*a):return {'text':'Ein Blick aufs Licht hilft.','ms':1,'status':'ready','word_count':5}
        with patch('informed_bridge.generate',bridge):
            task=asyncio.create_task(self.collect(self.setup_flow(live,count=15)))
            await asyncio.sleep(.04);task.cancel()
            with self.assertRaises(asyncio.CancelledError):await asyncio.wait_for(task,.5)
        self.assertTrue(live.closed)
    async def collect(self,flow):return [e async for e in flow]

if __name__=='__main__':unittest.main()
