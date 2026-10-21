import asyncio,unittest
from unittest.mock import AsyncMock
from contextlib import asynccontextmanager
from shared_turn import admitted_turn

class Client:
    def __init__(self,busy=0,statuses=(200,)):
        self.busy=busy;self.statuses=list(statuses);self.posts=0
    async def get(self,path,**kwargs):
        busy=self.busy>0;self.busy-=1
        class State:
            status_code=200
            def json(self):return {'busy':busy,'ready':True}
        return State()
    @asynccontextmanager
    async def stream(self,*args,**kwargs):
        self.posts+=1;code=self.statuses.pop(0)
        class Response:
            status_code=code
            def raise_for_status(self):
                if code>=400:raise ValueError('Non-retryable status')
        yield Response()

class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_busy_and_admission_race_retry_only_before_success(self):
        c=Client(busy=3,statuses=(409,503,200));notice=AsyncMock()
        async with admitted_turn(c,{},notice,timeout=.2,poll=.001) as response:self.assertEqual(response.status_code,200)
        self.assertEqual(c.posts,3);notice.assert_awaited()
    async def test_accepted_stream_failure_is_never_retried(self):
        c=Client(statuses=(200,200))
        with self.assertRaises(RuntimeError):
            async with admitted_turn(c,{},AsyncMock(),poll=.001):raise RuntimeError('Accepted stream disconnected')
        self.assertEqual(c.posts,1)
    async def test_wait_is_bounded_and_does_not_send_busy_generation(self):
        c=Client(busy=100)
        with self.assertRaises(TimeoutError):
            async with admitted_turn(c,{},AsyncMock(),timeout=.01,poll=.001):pass
        self.assertEqual(c.posts,0)
    async def test_wait_is_immediately_cancellable(self):
        c=Client(busy=100)
        async def wait():
            async with admitted_turn(c,{},AsyncMock(),poll=.001):pass
        task=asyncio.create_task(wait());await asyncio.sleep(.005);task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(c.posts,0)
