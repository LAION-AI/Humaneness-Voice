"""Opt-in, per-request cooperative cancellation; never cancel another job.

Do not cancel an asyncio.to_thread task: its GPU thread would keep running
after the global busy flag was released. Raise at the next real audio callback
instead, and let the worker unwind normally before releasing model ownership.
"""
import threading

class TurnCancelled(Exception):
    pass

class DisconnectCancellation:
    def __init__(self,enabled=False):
        self.enabled=enabled;self.event=threading.Event()
    def check(self):
        if self.enabled and self.event.is_set():raise TurnCancelled()
    async def watch(self,events):
        try:
            async for item in events:yield item
        finally:
            if self.enabled:self.event.set()
            await events.aclose()
