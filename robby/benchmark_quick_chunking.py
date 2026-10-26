"""Same-seed real TTS: uniform 160-ms vs 160-ms first / 400-ms subsequent blocks."""
import asyncio,base64,datetime,json,time
from pathlib import Path
import httpx,numpy as np
from bolt_text import Sentence
ROOT=Path(__file__).resolve().parent
async def main():
 rows=[];waves=[]
 async with httpx.AsyncClient(base_url='http://127.0.0.1:8792',timeout=60) as c:
  for chunk,first in [(2,None),(5,2)]:
   spec=Sentence('Das klingt nach viel.','interest, warm, natural',language='de').spec()
   body={'model':'small_rl','input_mode':'manual','format':'structured','message':spec['manual_prompt'],'language':'de','generation':'single','n':1,'streaming':True,
    'chunk_frames':chunk,'first_chunk_frames':first,'stream_transport':'pcm_f32','live_only':True,'cancel_on_disconnect':True,
    'temperature':.7,'guidance':2.,'voice_mode':'profile','voice_profile':'emolia_c1699','voice_conversion':False,'speaker_steering':False,'dynamic_reference':False,'align':False,'seed':777,'seconds':2.}
   start=time.perf_counter();chunks=[];audio=[];result={}
   assert not (await c.get('/api/state')).json()['busy']
   async with c.stream('POST','/api/turn',json=body) as r:
    r.raise_for_status()
    async for line in r.aiter_lines():
     if not line.startswith('data: '):continue
     e=json.loads(line[6:])
     if e.get('event')=='error':raise RuntimeError(e.get('message'))
     if e.get('pcm_f32'):
      pcm=np.frombuffer(base64.b64decode(e['pcm_f32']),'<f4');audio.append(pcm);chunks.append({'at_ms':round((time.perf_counter()-start)*1000,2),'samples':len(pcm)})
     if e.get('event')=='result':result=e
   wave=np.concatenate(audio);waves.append(wave)
   rows.append({'chunk_frames':chunk,'first_chunk_frames':first,'chunks':chunks,'audio_s':len(wave)/48000,'total_ms':round((time.perf_counter()-start)*1000,2),'result':result})
   print({k:rows[-1][k] for k in ['chunk_frames','first_chunk_frames','audio_s','total_ms']},flush=True)
 equal=len(waves[0])==len(waves[1]);diff=float(np.max(np.abs(waves[0]-waves[1]))) if equal else None
 report={'scope':__doc__,'rows':rows,'equal_lengths':equal,'max_absolute_sample_difference':diff,'allclose_at_1e_4':bool(equal and np.allclose(waves[0],waves[1],atol=1e-4))}
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');raw=json.dumps(report,indent=2)
 (ROOT/'static/quick-reaction-chunking.json').write_text(raw);(ROOT/'logs'/('quick-reaction-chunking-'+stamp+'.json')).write_text(raw)
 assert rows[1]['chunks'][0]['samples']==2*3840
 assert rows[1]['chunks'][1]['samples']==5*3840
 assert rows[1]['result']['scoring_ms']==0
if __name__=='__main__':asyncio.run(main())
