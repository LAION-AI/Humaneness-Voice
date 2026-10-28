"""Private local LFM3 audio/video analysis service. No recordings are retained."""
import asyncio,base64,contextlib,json,os,tempfile,time,sys
from pathlib import Path
from aiohttp import web
ROOT=Path(__file__).resolve().parent
LIMIT=24*1024*1024;MAX_DURATION=30
lock=asyncio.Lock();workers={};status={'ready':False,'error':None}
async def start_worker(mode,gpu):
 env={**os.environ,'PYTHONPATH':os.environ.get('LFM3_PYTHONPATH','/mnt/nvme/moss-15-v2/lfm3-env'),'HF_HOME':os.environ.get('HF_HOME','/mnt/nvme/moss-15-v2/.cache/lfm3'),'HF_HUB_OFFLINE':'1','PYTHONUNBUFFERED':'1'}
 p=await asyncio.create_subprocess_exec(sys.executable,str(ROOT/'lfm3_worker.py'),mode,str(gpu),stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=None,env=env,limit=1024*1024)
 ready=await asyncio.wait_for(p.stdout.readline(),90);info=json.loads(ready);assert info.get('ready'),info
 workers[mode]=p;return info
async def startup(app):
 try:
  info=await asyncio.gather(start_worker('audio',int(os.getenv('LFM3_AUDIO_GPU','0'))),start_worker('video',int(os.getenv('LFM3_VIDEO_GPU','1'))));status.update(ready=True,models=info)
 except Exception as e:status.update(ready=False,error=type(e).__name__+': '+str(e)[:160]);print('Worker startup failed',status['error'],flush=True)
async def cleanup(app):
 for p in workers.values():
  if p.returncode is None:p.terminate()
 for p in workers.values():
  with contextlib.suppress(Exception):await asyncio.wait_for(p.wait(),5)
async def cmd(*args):
 p=await asyncio.create_subprocess_exec(*args,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
 try:out,err=await asyncio.wait_for(p.communicate(),20)
 except asyncio.TimeoutError:p.kill();await p.wait();raise ValueError('Media decoding timed out')
 if p.returncode:raise ValueError('Unsupported or invalid media')
 return out
async def worker(mode,path,previous,duration,context):
 p=workers[mode]
 if p.returncode is not None:raise RuntimeError(mode+' model stopped')
 p.stdin.write((json.dumps({'path':str(path),'previous':previous,'duration_s':duration,'context':context},ensure_ascii=False)+'\n').encode());await p.stdin.drain()
 reading=asyncio.create_task(p.stdout.readline())
 try:response=await asyncio.wait_for(asyncio.shield(reading),120)
 except asyncio.CancelledError:
  # Drain the current response before the lock is released, otherwise the
  # next recording could receive a stale response from the previous one.
  with contextlib.suppress(Exception):await asyncio.wait_for(reading,120)
  raise
 if not response:raise RuntimeError(mode+' model disconnected')
 d=json.loads(response)
 if not d.get('ok'):raise RuntimeError(mode+' model error: '+d.get('error','unknown'))
 return d['result']
async def get_status(request):
 import httpx
 try:
  from lfm3_omni import local_base
  async with httpx.AsyncClient(timeout=1.) as c:
   response=await c.get(local_base()+'/health')
  omni_ready=response.status_code==200
 except (httpx.HTTPError,ValueError):omni_ready=False
 return web.json_response({**status,'omni_ready':omni_ready,'busy':lock.locked(),'max_duration_s':MAX_DURATION,'max_upload_mb':LIMIT//1024//1024})
async def analyze(request):
 if not status['ready']:raise web.HTTPServiceUnavailable(text='Models are not ready')
 if lock.locked():raise web.HTTPConflict(text='Another analysis is running; please retry')
 async with lock:
  with tempfile.TemporaryDirectory(prefix='lfm3-',dir=ROOT/'.runtime') as tmp:
   tmp=Path(tmp);upload=tmp/'input.webm';size=0;window=3.;stride=1.;capability=None;recent=[];use_context=False
   try:
    reader=await request.multipart()
    async for part in reader:
     if part.name=='media':
      with upload.open('wb') as f:
       while True:
        chunk=await part.read_chunk(65536)
        if not chunk:break
        size+=len(chunk)
        if size>LIMIT:raise web.HTTPRequestEntityTooLarge(max_size=LIMIT,actual_size=size)
        f.write(chunk)
     elif part.name in ('window','stride'):
      raw=await part.text()
      if len(raw)>20:raise ValueError('Invalid window setting')
      if part.name=='window':window=float(raw)
      else:stride=float(raw)
     elif part.name in ('context_enabled','memory_capability','recent_context'):
      raw=await part.read_chunk(8192)
      if await part.read_chunk(1):raise ValueError('Context field is too large')
      value=raw.decode('utf-8')
      if part.name=='context_enabled':use_context=value=='true'
      elif part.name=='memory_capability':capability=value
      else:recent=json.loads(value)
    if not upload.exists() or size==0:raise ValueError('A recording is required')
    if not 2<=window<=5 or not .5<=stride<=3 or stride>window:raise ValueError('Invalid window or stride')
    # MediaRecorder WebM commonly omits duration; remuxing writes a finite
    # container duration without transcoding or retaining the recording.
    normalized=tmp/'normalized.mkv'
    await cmd('ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(upload),'-map','0:v:0','-map','0:a:0','-c','copy',str(normalized))
    out=await cmd('ffprobe','-v','error','-show_entries','format=duration:stream=codec_type','-of','json',str(normalized));meta=json.loads(out)
    duration=float(meta['format']['duration']);types={s['codec_type'] for s in meta['streams']}
    if not 0<duration<=MAX_DURATION+.25:raise ValueError('Recording must be between 0 and 30 seconds')
    if not {'audio','video'}<=types:raise ValueError('The recording must contain audio and video')
    starts=[];t=0.
    while t+window<=duration+.05:
     starts.append(round(t,3));t+=stride
    if not starts:starts=[0.]
    if len(starts)>60:raise ValueError('Too many windows')
    from lfm3_context import build_context
    context,context_meta=await asyncio.to_thread(build_context,capability,recent) if use_context else ({},{'memory_used':False,'recent_turns':0,'conservative_token_bound':0})
    async def process(mode):
     previous=[];results=[]
     for index,start in enumerate(starts):
      length=min(window,duration-start);path=tmp/(mode+str(index)+('.wav' if mode=='audio' else '.mp4'))
      base=['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',str(start),'-i',str(normalized),'-t',str(length)]
      if mode=='audio':args=base+['-vn','-ac','1','-ar','16000',str(path)]
      else:args=base+['-an','-vf','fps=5,scale=320:240:force_original_aspect_ratio=decrease','-c:v','libx264','-preset','ultrafast','-pix_fmt','yuv420p',str(path)]
      await cmd(*args);result=await worker(mode,path,previous,length,context)
      if mode=='video' and result['parse_ok']:
       from lfm3_timeline import normalize_timeline
       result['annotation']=normalize_timeline(result['annotation'],start,length)
      elif mode=='audio' and result['parse_ok']:
       from lfm3_timeline import normalize_scores
       result['annotation']=normalize_scores(result['annotation'])
      if result['parse_ok']:
       from lfm3_context import flag_context_echo
       result['annotation']=flag_context_echo(result['annotation'],context)
      observation=result['annotation'] if result['parse_ok'] else {'unstructured':result['raw_text']}
      # Prior events have their own window-relative clock; label their window
      # explicitly so their times cannot be mistaken for the current window.
      previous.append({'window_start_s':round(start,3),'window_end_s':round(start+length,3),'observations':observation} if mode=='video' else observation)
      results.append(result);path.unlink(missing_ok=True)
     return results
    async def process_omni():
     from lfm3_omni import make_contact_sheet,analyze as analyze_omni
     from lfm3_timeline import normalize_timeline
     previous=[];results=[]
     for index,start in enumerate(starts):
      length=min(window,duration-start);folder=tmp/('omni-'+str(index));audio_path=tmp/('omni-'+str(index)+'.wav')
      item_start=time.perf_counter()
      try:
       await cmd('ffmpeg','-hide_banner','-loglevel','error','-y','-ss',str(start),'-i',str(normalized),'-t',str(length),'-vn','-ac','1','-ar','16000',str(audio_path))
       grid,count=await make_contact_sheet(normalized,folder,start,length,cmd)
       result=await analyze_omni(grid,audio_path,context,previous,length);result['contact_sheet_frames']=count
       result['total_ms']=round((time.perf_counter()-item_start)*1000,1)
       result['preprocess_ms']=round(max(0,result['total_ms']-result['generation_ms']),1)
       if result['parse_ok']:
        from lfm3_context import flag_context_echo
        result['annotation']=flag_context_echo(normalize_timeline(result['annotation'],start,length,fps=3),context)
       observation=result['annotation'] if result['parse_ok'] else {'unstructured':result['raw_text']}
       previous.append({'window_start_s':round(start,3),'window_end_s':round(start+length,3),'observations':observation})
      except Exception as exc:
       result={'annotation':None,'parse_ok':False,'raw_text':'Omni analysis unavailable: '+type(exc).__name__,'preprocess_ms':0,'generation_ms':0,'total_ms':0,'output_tokens':0,'output_tokens_per_s':0,'model_calls':0}
      results.append(result);audio_path.unlink(missing_ok=True)
     return results
    started=time.perf_counter()
    # A seekable MP4 is returned to the same browser for its window players;
    # MediaRecorder WebM often has infinite duration and cannot be sought.
    async def playback():
     out=tmp/'playback.mp4'
     await cmd('ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(normalized),'-vf','scale=320:240:force_original_aspect_ratio=decrease','-c:v','libx264','-preset','ultrafast','-pix_fmt','yuv420p','-c:a','aac','-max_muxing_queue_size','4096','-movflags','+faststart',str(out))
     return base64.b64encode(out.read_bytes()).decode('ascii')
    tasks=[asyncio.create_task(coro) for coro in (process('audio'),process('video'),process_omni(),playback())]
    try:audio,video,omni,playable=await asyncio.gather(*tasks)
    except BaseException:
     for task in tasks:task.cancel()
     await asyncio.gather(*tasks,return_exceptions=True)
     raise
    return web.json_response({'duration_s':round(duration,3),'window_s':window,'stride_s':stride,'wall_ms':round((time.perf_counter()-started)*1000,1),'playback_mp4_base64':playable,'relevance_context':context_meta,
      'segments':[{'start_s':s,'end_s':round(min(s+window,duration),3),'audio':a,'video':v,'omni':o} for s,a,v,o in zip(starts,audio,video,omni)],
      'notes':['Models may hallucinate. Person IDs are approximate, not face recognition.','Visual event times are model estimates, not verified frame timestamps; samples are about 5 fps.','Previous two descriptions are supplied separately to each model.','Token/s includes prefill and decoding. Media is deleted when the request finishes.']})
   except web.HTTPException:raise
   except (ValueError,KeyError,TypeError,json.JSONDecodeError) as e:raise web.HTTPBadRequest(text=str(e)[:200])
   except Exception as e:raise web.HTTPBadGateway(text=type(e).__name__+': '+str(e)[:180])
if __name__=='__main__':
 (ROOT/'.runtime').mkdir(exist_ok=True)
 app=web.Application(client_max_size=LIMIT+1024*1024);app.on_startup.append(startup);app.on_cleanup.append(cleanup)
 app.router.add_get('/status',get_status);app.router.add_post('/analyze',analyze)
 web.run_app(app,host='127.0.0.1',port=8810,access_log=None)
