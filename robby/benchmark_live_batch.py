"""Synthetic dynamic-arrival GPU benchmark; one shared TTS weight instance."""
import asyncio,base64,datetime,json,time
from pathlib import Path
import aiohttp,httpx,numpy as np
from bolt_text import Sentence
ROOT=Path(__file__).resolve().parent
TEXTS=['Das klingt nach viel.','Dabei hilft ein Blick aufs Sonnenlicht.','Blaues Licht wird in der Luft stärker gestreut.']
async def trial(max_active,batch_decode,temperature=0.,frames=18):
    t=time.perf_counter();rows=[];audio={str(i):[] for i in range(3)};added=False
    async with aiohttp.ClientSession() as client:
        async with client.ws_connect('http://127.0.0.1:8792/api/live-speech',max_msg_size=8*1024*1024) as ws:
            await ws.send_json({'type':'start','model':'small_rl','voice_profile':'emolia_c1699',
                'temperature':temperature,'guidance':2.,'seed':777,'max_active':max_active,'batch_decode':batch_decode})
            async def submit(i):
                spec=Sentence(TEXTS[i],'contentment, soft, warm',language='de').spec()
                await ws.send_json({'type':'submit','id':str(i),'order':i,'message':spec['manual_prompt'],
                    'language':'de','seconds':3.,'first':2,'chunk':5,'max_frames':frames,'align':False})
            async for msg in ws:
                if msg.type!=aiohttp.WSMsgType.TEXT:raise RuntimeError('Unexpected socket closure')
                e=json.loads(msg.data);e['at_ms']=round((time.perf_counter()-t)*1000,2)
                if e.get('event') in ('error','unavailable'):raise RuntimeError(e)
                if e.get('event')=='ready':await submit(0)
                if e.get('pcm_f32'):
                    audio[e['id']].append(np.frombuffer(base64.b64decode(e.pop('pcm_f32')),'<f4').copy())
                    if not added:
                        added=True;await submit(1);await submit(2);await ws.send_json({'type':'finish'})
                rows.append(e)
                if e.get('event')=='session_done':break
    return {'max_active':max_active,'batch_decode':batch_decode,'temperature':temperature,'events':rows,
        'total_ms':round((time.perf_counter()-t)*1000,2)}, {k:np.concatenate(v) if v else np.zeros(0,np.float32) for k,v in audio.items()}
async def main():
    path=ROOT/'static/two-stage-batch-pilot.json'
    if path.exists():raise SystemExit('Existing result; preserve it before rerunning')
    report={'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Truncated 18-frame deterministic cache-isolation/performance pilot, not speech quality. New jobs arrive after first PCM.','trials':[]};waves=[]
    for n,b in [(1,False),(3,True),(3,False),(3,True)]:
        if httpx.get('http://127.0.0.1:8792/api/state').json().get('busy'):raise RuntimeError('Shared TTS occupied')
        row,audio=await trial(n,b);report['trials'].append(row);waves.append(audio)
        path.write_text(json.dumps(report,indent=2,ensure_ascii=False));print({'n':n,'batched':b,'ms':row['total_ms'],'batches':row['events'][-1].get('performance',{}).get('batch_histogram')},flush=True)
        # session_done is sent just before the worker releases admission.
        for _ in range(20):
            if not httpx.get('http://127.0.0.1:8792/api/state').json().get('busy'):break
            await asyncio.sleep(.05)
    report['audio_comparisons']=[]
    for trial_index in range(1,len(waves)):
        for key in waves[0]:
            a,b=waves[0][key],waves[trial_index][key];same=len(a)==len(b)
            report['audio_comparisons'].append({'trial':trial_index,'job':key,'same_samples':same,
                'max_abs_diff':float(np.max(np.abs(a-b))) if same and len(a) else None,
                'allclose_1e_4':bool(same and np.allclose(a,b,atol=1e-4))})
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False));print(report['audio_comparisons'],flush=True)
if __name__=='__main__':asyncio.run(main())
