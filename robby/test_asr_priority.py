import asyncio
import unittest
from asr_priority import PriorityGate


class PriorityTests(unittest.IsolatedAsyncioTestCase):
    async def test_foreground_before_queued_background_and_fifo_ties(self):
        gate=PriorityGate();order=[]
        async def run(name,p):
            async with gate.slot(p):order.append(name);await asyncio.sleep(0)
        async with gate.slot():
            tasks=[asyncio.create_task(run(n,p)) for n,p in [('bg',20),('fg1',0),('fg2',0)]]
            await asyncio.sleep(0)
        await asyncio.gather(*tasks)
        self.assertEqual(order,['fg1','fg2','bg']);self.assertFalse(gate.busy)

    async def test_cancelled_waiter_does_not_lose_admission(self):
        gate=PriorityGate()
        async def pending():
            async with gate.slot(0):return True
        async with gate.slot():
            cancelled=asyncio.create_task(pending());await asyncio.sleep(0);cancelled.cancel()
            with self.assertRaises(asyncio.CancelledError):await cancelled
            following=asyncio.create_task(pending());await asyncio.sleep(0)
        self.assertTrue(await following);self.assertFalse(gate.busy)
