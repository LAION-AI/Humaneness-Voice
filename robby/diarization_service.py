"""Isolated Nemotron-3-Diarization service; no private audio on disk.

Run with the pinned newer Transformers in an isolated PYTHONPATH. Never upgrade
the shared ASR/TTS environment. Streaming cache and a sub-second audio buffer
are per session, in RAM; idle sessions expire, and disconnect explicitly clears.
"""
import asyncio,json,os,re,time
from pathlib import Path
import numpy as np
import torch
from aiohttp import web
from transformers import AutoModelForAudioFrameClassification,AutoProcessor

ROOT=Path(__file__).resolve().parent
MODEL=Path(os.environ.get('ROBO_DIARIZATION_MODEL','/mnt/nvme/robo-speaker-studies/diarization-model'))
DEVICE=os.environ.get('ROBO_DIARIZATION_DEVICE','cuda')
STATE={'ready':False,'sessions':{},'model':None,'processor':None,'times':[]}
LOCK=asyncio.Lock()

def local(request):
    if request.remote!='127.0.0.1' or request.host!='127.0.0.1:8796' or any(k in request.headers for k in ('CF-Ray','Forwarded','X-Forwarded-For','CF-Connecting-IP')):raise web.HTTPForbidden()

class Stream:
    def __init__(self):
        self.audio=np.zeros(0,np.float32);self.base=0;self.received=0;self.mel=0;self.cache=None;self.frames=0;self.first=True;self.seen=time.monotonic()
    @torch.inference_mode()
    def feed(self,audio):
        p=STATE['processor'];m=STATE['model'];self.audio=np.r_[self.audio,audio];self.received+=len(audio);self.seen=time.monotonic()
        start_frame=self.frames;probs=[];started=time.perf_counter()
        while True:
            start=0 if self.first else p.audio_chunk_start(self.mel)
            n=p.num_samples_first_audio_chunk if self.first else p.num_samples_per_audio_chunk
            if start+n>self.received:break
            wave=self.audio[start-self.base:start-self.base+n]
            if len(wave)!=n:raise RuntimeError('Streaming input-buffer alignment changed')
            inputs=p(wave,sampling_rate=16000,is_streaming=True,is_first_audio_chunk=self.first)
            inputs=inputs.to(m.device,dtype=m.dtype)
            output=m(**inputs,speaker_cache=self.cache);self.cache=output.speaker_cache
            values=output.logits.sigmoid()[0].float().cpu().numpy();probs.append(values);self.frames+=len(values)
            self.first=False;self.mel+=p.num_mel_frames_per_step
        next_start=0 if self.first else p.audio_chunk_start(self.mel)
        cut=max(0,next_start-self.base)
        if cut:self.audio=self.audio[cut:].copy();self.base=next_start
        elapsed=time.perf_counter()-started;STATE['times'].append(elapsed);STATE['times']=STATE['times'][-500:]
        values=np.concatenate(probs) if probs else np.zeros((0,8),np.float32)
        return {'start_frame':start_frame,'frame_seconds':.01,'probabilities':values.tolist(),
            'emitted_until_s':self.frames*.01,'received_seconds':self.received/16000,'compute_seconds':elapsed,
            'buffer_samples':len(self.audio),'input_latency_mode':'ultra_low_latency','speaker_cache_is_session_local':True}

async def load(app):
    def work():
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if DEVICE.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.08,0)
        p=AutoProcessor.from_pretrained(MODEL,local_files_only=True);p.set_streaming_mode('ultra_low_latency')
        m=AutoModelForAudioFrameClassification.from_pretrained(MODEL,local_files_only=True,dtype=torch.float16 if DEVICE.startswith('cuda') else torch.float32).to(DEVICE).eval()
        return p,m
    STATE['processor'],STATE['model']=await asyncio.to_thread(work);STATE['ready']=True
    async def prune():
        while True:
            await asyncio.sleep(10)
            async with LOCK:
                now=time.monotonic()
                for sid in [s for s,v in STATE['sessions'].items() if now-v.seen>60]:del STATE['sessions'][sid]
    app['pruner']=asyncio.create_task(prune())

async def cleanup(app):
    app['pruner'].cancel();await asyncio.gather(app['pruner'],return_exceptions=True);STATE['sessions'].clear()

async def state(request):
    local(request);times=STATE['times']
    return web.json_response({'ready':STATE['ready'],'model':'nvidia/Nemotron-3-Diarization','revision':'f667ed73aee57d40cc39428eb768b4fd87a0a29e',
        'device':DEVICE,'visible_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'sessions':len(STATE['sessions']),'mode':'ultra_low_latency','audio_saved':False,
        'gpu_allocated_MiB':torch.cuda.memory_allocated()/1024**2 if DEVICE.startswith('cuda') else 0.,
        'mean_request_compute_ms':1000*float(np.mean(times)) if times else None})

async def feed(request):
    local(request)
    if not STATE['ready']:raise web.HTTPServiceUnavailable()
    sid=request.match_info['sid']
    if not re.fullmatch(r'[a-zA-Z0-9_-]{12,64}',sid):raise web.HTTPBadRequest()
    raw=await request.read()
    if not raw or len(raw)%4 or len(raw)>40960:raise web.HTTPBadRequest()
    audio=np.frombuffer(raw,'<f4')
    if not np.isfinite(audio).all():raise web.HTTPBadRequest()
    async with LOCK:
        now=time.monotonic()
        for key in [k for k,s in STATE['sessions'].items() if now-s.seen>60]:del STATE['sessions'][key]
        if sid not in STATE['sessions']:
            if len(STATE['sessions'])>=4:raise web.HTTPTooManyRequests()
            STATE['sessions'][sid]=Stream()
        result=await asyncio.to_thread(STATE['sessions'][sid].feed,audio.copy())
    return web.json_response(result)

async def close(request):
    local(request)
    async with LOCK:STATE['sessions'].pop(request.match_info['sid'],None)
    return web.json_response({'cleared':True})

def main():
    app=web.Application(client_max_size=40960);app.on_startup.append(load);app.on_cleanup.append(cleanup)
    app.router.add_get('/state',state);app.router.add_post('/session/{sid}',feed);app.router.add_delete('/session/{sid}',close)
    runtime=ROOT/'.runtime';runtime.mkdir(exist_ok=True)
    (runtime/'diarization.json').write_text(json.dumps({'pid':os.getpid(),'started':time.time(),'device':DEVICE}))
    web.run_app(app,host='127.0.0.1',port=8796,access_log=None)

if __name__=='__main__':main()
