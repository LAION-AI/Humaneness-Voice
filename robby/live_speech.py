"""Bounded multiplexed TTS client: submit early, consume in spoken order."""
import asyncio,json,time
from dataclasses import dataclass,field
from urllib.parse import urlparse
import aiohttp

@dataclass
class Job:
    id:str
    started:float
    events:asyncio.Queue=field(default_factory=lambda:asyncio.Queue(maxsize=192))
    first_pcm:float|None=None
    done:bool=False

class LiveSpeech:
    def __init__(self,session,ws):
        self.session=session;self.ws=ws;self.jobs={};self.counter=0;self.closed=False
        self.performance={};self.error=None;self.receiver=asyncio.create_task(self.receive())
    @classmethod
    async def open(cls,client,voice):
        base=str(getattr(client,'base_url','')).rstrip('/')
        if urlparse(base).hostname not in ('127.0.0.1','localhost','::1'):return None
        try:
            state=await client.get('/api/state',timeout=3.)
            if not state.json().get('continuous_tts'):return None
        except Exception:return None
        started=time.monotonic()
        while time.monotonic()-started<45:
            session=aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
            try:
                ws=await session.ws_connect(base+'/api/live-speech',max_msg_size=8*1024*1024,timeout=10)
                await ws.send_json({'type':'start','model':'small_rl','voice_profile':voice,
                    'guidance':2.,'temperature':.8,'seed':777,'max_active':2})
                message=await ws.receive_json(timeout=45)
                if message.get('event')=='ready':return cls(session,ws)
                await ws.close();await session.close()
                if message.get('event')!='unavailable':return None
                await asyncio.sleep(.1)
            except asyncio.CancelledError:await session.close();raise
            except Exception:await session.close();return None
        raise TimeoutError('Shared TTS did not become available')
    async def submit(self,sentence,role,order,chunk_frames=5):
        if self.closed or self.error:raise RuntimeError(self.error or 'TTS session closed')
        spec=sentence.spec();self.counter+=1;id_='s'+str(self.counter)
        job=Job(id_,time.perf_counter());self.jobs[id_]=job
        from quick_reaction import word_count
        short=role in ('reaction','informed_bridge')
        body={'type':'submit','id':id_,'order':order,'message':spec['manual_prompt'],
              'language':spec['language'],'first':2,'chunk':5 if short else min(chunk_frames,5),
              'align':not short}
        if short:body['seconds']=max(2.,word_count(sentence.text)*.4)
        await self.ws.send_json(body);return job
    async def receive(self):
        try:
            async for msg in self.ws:
                if msg.type!=aiohttp.WSMsgType.TEXT:raise RuntimeError('TTS socket closed unexpectedly')
                event=json.loads(msg.data);kind=event.get('event')
                if kind=='error':raise RuntimeError('Continuous TTS: '+str(event.get('message')))
                if kind=='session_done':self.performance=event.get('performance',{});return
                job=self.jobs.get(event.get('id'))
                if job is None:continue
                if kind=='audio_chunk':
                    if job.first_pcm is None:job.first_pcm=time.perf_counter()
                    await job.events.put(event)
                elif kind=='job_done':
                    job.done=True;await job.events.put(event)
            if not self.closed:raise RuntimeError('TTS stream ended before session_done')
        except asyncio.CancelledError:raise
        except Exception as exc:
            self.error=str(exc)
            for job in self.jobs.values():
                if not job.done:
                    # Never deadlock cancellation on a slow consumer.
                    while job.events.full():job.events.get_nowait()
                    job.events.put_nowait({'event':'error','message':self.error})
    async def events(self,job):
        while True:
            event=await asyncio.wait_for(job.events.get(),90)
            if event.get('event')=='error':raise RuntimeError(event['message'])
            yield event
            if event.get('event')=='job_done':break
        self.jobs.pop(job.id,None)
    async def finish(self):
        if not self.closed:await self.ws.send_json({'type':'finish'})
    async def close(self):
        if self.closed:return
        self.closed=True;self.receiver.cancel();await asyncio.gather(self.receiver,return_exceptions=True)
        await self.ws.close();await self.session.close();self.jobs.clear()
