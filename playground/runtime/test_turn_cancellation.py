import asyncio,unittest
from turn_cancellation import DisconnectCancellation,TurnCancelled

class CancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_is_opt_in_and_scoped_to_one_worker(self):
        async def events():
            yield 'first'
            yield 'second'
        for enabled in (False,True):
            a=DisconnectCancellation(enabled);b=DisconnectCancellation(True)
            stream=a.watch(events());self.assertEqual(await anext(stream),'first')
            await stream.aclose();b.check()
            if enabled:
                with self.assertRaises(TurnCancelled):a.check()
            else:a.check()
    async def test_worker_keeps_ownership_until_cooperative_unwind(self):
        cancel=DisconnectCancellation(True);started=asyncio.Event();release=asyncio.Event()
        state={'busy':True}
        async def worker():
            try:
                started.set();await release.wait();cancel.check()
            except TurnCancelled:pass
            finally:state['busy']=False
        task=asyncio.create_task(worker());await started.wait()
        cancel.event.set();self.assertTrue(state['busy']);self.assertFalse(task.done())
        release.set();await task;self.assertFalse(state['busy'])

if __name__=='__main__':unittest.main()
