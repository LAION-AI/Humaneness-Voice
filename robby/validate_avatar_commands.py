"""Real-model visual command checks and one paced synthetic microphone turn."""
import asyncio,json,struct,time
from pathlib import Path
import aiohttp,httpx,numpy as np
from scipy.signal import resample_poly
from bolt_pipeline import pipeline
from benchmark_bolt import measure
ROOT=Path(__file__).resolve().parent

async def main():
    report={'passed':False,'physical_microphone':False,'rows':[],'started':time.time()}
    async with httpx.AsyncClient(base_url='http://127.0.0.1:8792',timeout=60) as c:
        cases=[('Kannst du bitte eine lustige Grimasse schneiden?',['grimace']),
            ('Mach mal eine verrückte Grimasse!',['grimace']),
            ('Schiele bitte und rolle dann mit den Augen.',['grimace','eye_roll']),
            ('Guck bitte wütend und mach dann eine lustige Grimasse.',['emotion','grimace']),
            ('Please make a goofy face.',['grimace'])]
        for repeat in range(2):
            for text,kinds in cases:
                start=time.perf_counter();row={'input':text,'repeat':repeat,'passed':False}
                try:
                    events=[]
                    async for kind,value in pipeline(c,text,{},[],interaction_state={'enabled':True,'rules':[]},tts=False):
                        events.append((kind,value,(time.perf_counter()-start)*1000))
                    assert events[0][0]=='avatar_action',events
                    assert [s['kind'] for s in events[0][1]['steps']]==kinds,events[0]
                    parts=[v for k,v,t in events if k=='sentence'];assert parts and parts[-1]['text']
                    assert all(p['avatar_action_id']==events[0][1]['action_id'] for p in parts)
                    assert 'tool_director_ms' not in events[-1][1]
                    row.update(passed=True,actions=events[0][1]['steps'],reply=parts[-1]['text'],
                        action_dispatch_ms=round(events[0][2],2),first_spoken_text_ms=round(next(t for k,v,t in events if k=='sentence'),2))
                except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)[:300]
                report['rows'].append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
        _,source=await measure(c,'http://127.0.0.1:8798/api/bolt/tts',{'text':'Mach bitte eine lustige Grimasse.','cue':'interest, natural','voice':'anime_088'})
    audio=resample_poly(source,2,3).astype(np.float32)
    audio=np.r_[np.zeros(16000,np.float32),audio,np.zeros(10*16000,np.float32)]
    spoken={'passed':False,'events':[],'audio_chunks':0}
    async with aiohttp.ClientSession() as c,c.ws_connect('http://127.0.0.1:8798/ws') as ws:
        start=time.perf_counter();done=asyncio.Event();received={}
        async def consume():
            async for packet in ws:
                if packet.type==aiohttp.WSMsgType.TEXT:
                    item=json.loads(packet.data)
                    if item.get('type') in ('avatar_action','endpoint','reply','turn_done','error'):
                        spoken['events'].append({k:v for k,v in item.items() if k in ('type','text','transcript','steps','action_id','avatar_action_id','error')})
                    if item.get('type')=='turn_done':done.set()
                elif packet.type==aiohttp.WSMsgType.BINARY:
                    rid=struct.unpack_from('<i',packet.data)[0];received[rid]=received.get(rid,0)+(len(packet.data)-4)//2
                    spoken['audio_chunks']+=1;await ws.send_str(f'played:{rid}:{received[rid]}')
        task=asyncio.create_task(consume())
        try:
            for at in range(0,len(audio),320):
                if done.is_set():break
                await asyncio.sleep(max(0,start+at/16000-time.perf_counter()))
                await ws.send_bytes(struct.pack('<ii',0,-1)+(np.clip(audio[at:at+320],-1,1)*32767).astype('<i2').tobytes())
            await asyncio.wait_for(done.wait(),30)
            assert any(e['type']=='avatar_action' and e['steps'][0]['kind']=='grimace' for e in spoken['events']),spoken
            assert spoken['audio_chunks']>0 and any(e['type']=='reply' for e in spoken['events']),spoken
            assert not any(e['type']=='error' for e in spoken['events']),spoken
            spoken['passed']=True
        except Exception as exc:spoken['error']=str(exc)
        finally:
            task.cancel();await asyncio.gather(task,return_exceptions=True)
    report.update(spoken_command=spoken,passed=all(r['passed'] for r in report['rows']) and spoken['passed'],finished=time.time())
    (ROOT/'static/avatar-command-check.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('PASSED',report['passed'],flush=True)
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':asyncio.run(main())
