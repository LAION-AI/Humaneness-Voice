"""Cancellation-safe admission for one shared decoder; lower number goes first.

Never interrupts a running GPU call. Foreground turn finalization goes before
queued background/wake searches, not before a decoder call already admitted.
"""
import asyncio
import heapq
import itertools
from contextlib import asynccontextmanager


class PriorityGate:
    def __init__(self):
        self.busy=False;self.waiters=[];self.order=itertools.count()

    def release(self):
        while self.waiters:
            _,_,future=heapq.heappop(self.waiters)
            if not future.done():future.set_result(True);return
        self.busy=False

    @asynccontextmanager
    async def slot(self,priority=10):
        admitted=False
        if not self.busy:self.busy=True;admitted=True
        else:
            future=asyncio.get_running_loop().create_future()
            heapq.heappush(self.waiters,(priority,next(self.order),future))
            try:
                await asyncio.shield(future);admitted=True
            except BaseException:
                # Cancellation can race the handoff to this waiter.
                if future.done() and not future.cancelled():self.release()
                else:future.cancel()
                raise
        try:yield
        finally:
            if admitted:self.release()
