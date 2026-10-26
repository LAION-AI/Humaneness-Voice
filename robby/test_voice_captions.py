"""Offline caption safety, deadline and prompt contracts (no model inference)."""
import asyncio
import unittest
from unittest.mock import patch
import numpy as np
from voice_captioning import describe,prompt_hint,INSTRUCTION,completed_hint

class Captions(unittest.IsolatedAsyncioTestCase):
    async def test_late_parallel_result_never_blocks_or_leaks(self):
        pending=asyncio.create_task(asyncio.sleep(1,result={'status':'available','caption':'old'}))
        self.assertEqual(completed_hint(pending)['status'],'not_ready')
        await asyncio.gather(pending,return_exceptions=True)
        self.assertTrue(pending.cancelled())
        done=asyncio.create_task(asyncio.sleep(0,result={'status':'available','caption':'current'}));await done
        self.assertEqual(completed_hint(done)['caption'],'current')
        self.assertEqual(completed_hint(done,False)['status'],'disabled')

    async def test_caption_attaches_only_to_current_accepted_turn(self):
        from test_sentence_reliability import ReliabilityTests
        from unittest.mock import AsyncMock
        s,c,v=ReliabilityTests().session()
        ready=asyncio.create_task(asyncio.sleep(0));await ready
        caption=asyncio.create_task(asyncio.sleep(0,result={'status':'available','caption':'Possibly calm','request_ms':20}));await caption
        cached={'task':ready,'embedding':v[0],'compute_ms':20.}
        await s.queue.put(('Wie geht es dir?',np.ones(32000,np.float32)*.03,2.,0.,cached,{},None,s.admission_epoch,caption))
        task=asyncio.create_task(s.deliver())
        for _ in range(50):
            if c._send_turn.await_count:break
            await asyncio.sleep(.002)
        task.cancel();await task
        self.assertEqual(c.current_speaker['voice_delivery']['caption'],'Possibly calm')
        self.assertEqual(c.pending_input_timing['voice_caption_status'],'available')
        c._send_turn.assert_awaited_once_with('Wie geht es dir?')

    async def test_no_network_for_disabled_short_silent_invalid(self):
        cases=[(np.ones(16000),False,'disabled'),(np.ones(1000),True,'too_short'),
            (np.zeros(16000),True,'quiet'),(np.full(16000,np.nan),True,'invalid_audio')]
        with patch('httpx.AsyncClient',side_effect=AssertionError('Network must not be used')):
            for wave,enabled,status in cases:self.assertEqual((await describe(wave,enabled))['status'],status)

    async def test_explicit_deadline_and_no_late_result(self):
        class Client:
            def __init__(self,*a,**k):pass
            async def __aenter__(self):return self
            async def __aexit__(self,*a):pass
            async def post(self,*a,**k):await asyncio.sleep(1);raise AssertionError('Late result used')
        with patch('httpx.AsyncClient',Client):
            result=await describe(np.ones(16000,np.float32),timeout=.01)
        self.assertEqual(result['status'],'timeout');self.assertLess(result['request_ms'],200)

    async def test_current_turn_scope_no_unbounded_audio(self):
        captured={}
        class Response:
            def raise_for_status(self):pass
            def json(self):return {'status':'available','caption':'Possibly quiet','window_s':8}
        class Client:
            def __init__(self,*a,**k):pass
            async def __aenter__(self):return self
            async def __aexit__(self,*a):pass
            async def post(self,*a,**k):captured.update(k);return Response()
        with patch('httpx.AsyncClient',Client):result=await describe(np.arange(160000,dtype=np.float32))
        self.assertEqual(result['scope'],'last eight seconds of this turn')
        self.assertEqual(len(captured['content']),128000*4)
        self.assertEqual(np.frombuffer(captured['content'],'<f4')[0],32000)

    def test_no_hint_for_typed_missing_unavailable(self):
        for identity in [None,{}, {'voice_delivery':{'status':'timeout'}},
                         {'voice_delivery':{'status':'available','caption':None}}]:
            self.assertEqual(prompt_hint(identity),'')

    def test_hints_are_not_personal_facts_and_raw_scores_not_sent(self):
        hint=prompt_hint({'voice_delivery':{'status':'available','caption':'Possibly breathy',
            'window_s':2,'scope':'this turn','selected':{'raw_secret_scores':123}}})
        self.assertIn(INSTRUCTION,hint);self.assertIn('Possibly breathy',hint)
        self.assertNotIn('raw_secret_scores',hint);self.assertIn('not facts',hint)
        self.assertIn("user's own account takes",hint)

if __name__=='__main__':unittest.main()
