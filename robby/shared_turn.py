"""Cancellable pre-generation arbitration for the single shared TTS worker.

Retry only an explicit 409/503 before SSE admission, never a partially accepted
generation/network failure. No other session is cancelled or evicted.
"""
import asyncio,time
from contextlib import asynccontextmanager

@asynccontextmanager
async def admitted_turn(client,body,on_wait,timeout=45.,poll=.2):
    started=time.monotonic();last_notice=-100.
    while time.monotonic()-started<timeout:
        state=await client.get('/api/state',timeout=3.)
        ready=state.status_code==200 and not state.json().get('busy') and state.json().get('ready',True)
        if ready:
            async with client.stream('POST','/api/turn',json=body) as response:
                if response.status_code not in (409,503):
                    response.raise_for_status()
                    await on_wait('Voice model available. Preparing your spoken reply…','preparing')
                    yield response
                    return
        now=time.monotonic()
        if now-last_notice>=2.:
            last_notice=now
            await on_wait(f'Waiting for the shared voice model ({now-started:.0f} s). You can stop this reply.','waiting')
        await asyncio.sleep(poll)
    raise TimeoutError('Shared voice model did not become available within the bounded wait')
