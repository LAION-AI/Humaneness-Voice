"""Optional parallel CPU AudioSet hints with a hard latency budget."""
import asyncio,os,time
import httpx

async def detect(wave,enabled=True,timeout=.18):
    if not enabled:return {'status':'disabled'}
    start=time.perf_counter()
    try:
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(timeout=timeout,trust_env=False) as client:
                response=await client.post(os.environ.get('ROBO_EVENT_URL','http://127.0.0.1:8804')+'/events',content=wave.astype('<f4').tobytes())
                response.raise_for_status();result=response.json()
                if result.get('status')!='available' or not isinstance(result.get('candidates'),list):raise ValueError('Invalid event hints')
                return {**result,'request_ms':round((time.perf_counter()-start)*1000,2)}
    except asyncio.CancelledError:raise
    except (httpx.HTTPError,ValueError,TimeoutError):return {'status':'unavailable','request_ms':round((time.perf_counter()-start)*1000,2)}
