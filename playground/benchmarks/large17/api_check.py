"""Warm HTTP PCM onset including the real reference and alignment path."""
import base64,json,time
from pathlib import Path
import httpx,numpy as np
ROOT=Path(__file__).resolve().parent

def main():
 rows=[]
 with httpx.Client(base_url='http://127.0.0.1:8792',timeout=900) as client:
  for model in ('large17','small_rl'):
   started=time.perf_counter();response=client.post('/api/models/select',json={'model':model});response.raise_for_status()
   rows.append({'kind':'switch','model':model,'wall_seconds':time.perf_counter()-started})
   for align in (False,True):
    for repeat in range(4):
     body={'model':model,'input_mode':'manual','format':'structured',
       'message':'GENERAL: Use the same speaker identity as the reference. Warm, natural spontaneous conversation.\nSCRIPT:\n(warm, natural, spontaneous) Hallo, schön, dass du da bist. Was möchtest du heute machen?',
       'language':'de','seconds':5.,'generation':'single','n':1,'batch_size':1,
       'streaming':True,'stream_transport':'pcm_f32','live_only':True,'chunk_frames':20,
       'temperature':.8,'guidance':2.,'voice_mode':'profile','voice_profile':'emolia_c1699',
       'voice_conversion':False,'speaker_steering':False,'dynamic_reference':False,
       'align':align,'seed':1700+repeat,'cancel_on_disconnect':True}
     start=time.perf_counter();chunks=[];events=[]
     with client.stream('POST','/api/turn',json=body) as response:
      response.raise_for_status()
      for line in response.iter_lines():
       if not line.startswith('data: '):continue
       event=json.loads(line[6:]);kind=event.get('event')
       if kind=='error':raise RuntimeError(event)
       if kind=='audio_chunk':
        wave=np.frombuffer(base64.b64decode(event['pcm_f32']),dtype='<f4')
        assert np.isfinite(wave).all()
        chunks.append({'at':time.perf_counter()-start,'samples':len(wave),'rms':float(np.sqrt(np.mean(wave*wave)))})
       else:events.append(event)
     result=next(e for e in events if e['event']=='result')
     actual=next(e for e in events if e['event']=='model');assert actual['model_id']==model
     row={'kind':'generation','model':model,'align':align,'warmup':repeat==0,'chunks':chunks,
       'wall_seconds':time.perf_counter()-start,'result':result,'checkpoint':actual['checkpoint']}
     rows.append(row);(ROOT/'api-check.json').write_text(json.dumps(rows,indent=2))
     print(model,'align',align,'repeat',repeat,'first',round(chunks[0]['at'],3),'total',round(row['wall_seconds'],3),flush=True)
 (ROOT/'api-check.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
