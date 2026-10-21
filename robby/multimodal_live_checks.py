"""Real ASR/enrollment/TTS with public synthetic speech and timestamped synthetic
visual activity. Tests the actual WebSocket gate, NOT physical lip-sync accuracy.
No raw audio or biometric vectors are included in the saved receipt.
"""
import argparse,asyncio,json,struct,time
from pathlib import Path
import aiohttp,numpy as np
from speaker_live.live_test import audio
from final_verify import runtime_signature

ROOT=Path(__file__).parent

async def case(mode):
    events=[];sample=0;binary=0;ready=False;camera='off' if mode=='camera_off' else 'on'
    async with aiohttp.ClientSession() as client:
        async with client.ws_connect('http://127.0.0.1:8793/ws') as ws:
            async def reader():
                nonlocal binary,ready
                async for msg in ws:
                    if msg.type==aiohttp.WSMsgType.BINARY:binary+=len(msg.data)
                    elif msg.type==aiohttp.WSMsgType.TEXT:
                        item=json.loads(msg.data)
                        if item.get('type')=='ready':ready=True
                        if item.get('type') in ('ready','endpoint','reply','ignored','error','barge','reply_cancelled','visual_person','visual_greeting','visual_warning'):
                            if item.get('type')=='ready':item={'type':'ready','visual_identity':item.get('visual_identity')}
                            events.append(item)
            task=asyncio.create_task(reader())
            try:
                for _ in range(200):
                    if ready:break
                    await asyncio.sleep(.1)
                assert ready,'No ready message'
                await ws.send_str('visual:'+json.dumps({'type':'settings','ignoreOffscreenSpeakers':mode!='disabled','faceIdEnabled':True}))
                if mode=='greet_disabled':await ws.send_str('visual:'+json.dumps({'type':'settings','greetUnknownSpeakers':False}))
                wave=np.r_[np.zeros(16000,np.float32),audio('background' if mode.startswith('greet_') else 'christoph-intro'),np.zeros(16000*16,np.float32)]
                started=time.monotonic();voice_begin=1.;voice_end=(len(wave)-16000*16)/16000
                for i in range(0,len(wave),320):
                    chunk=wave[i:i+320];await ws.send_bytes(struct.pack('<ii',0,-1)+np.clip(chunk*32767,-32768,32767).astype('<i2').tobytes());sample+=len(chunk)
                    if i%1600==0:
                        speech=voice_begin+.4<=sample/16000<=voice_end+.1
                        faces=[]
                        if mode=='visible' or mode.startswith('greet_'):faces=[{'id':'face-1','speaking':speech,'confidence':.9 if speech else 0},{'id':'face-2','speaking':False,'confidence':0}]
                        if mode=='overlap':faces=[{'id':'face-1','speaking':speech,'confidence':.9 if speech else 0},{'id':'face-2','speaking':speech,'confidence':.8 if speech else 0}]
                        if mode=='static_open':faces=[{'id':'face-1','speaking':False,'confidence':0}]
                        await ws.send_str('visual:'+json.dumps({'type':'frame','camera':camera,'audio_s':sample/16000,'faces':faces}))
                    if sample/16000>voice_end+2 and binary:break
                    await asyncio.sleep(max(0,started+sample/16000-time.monotonic()))
                endpoints=[e for e in events if e['type']=='endpoint'];expected=mode in ('visible','disabled','camera_off')
                assert bool(endpoints)==expected,('Wrong endpoint gating',mode,events)
                if expected or mode=='greet_unknown':assert binary>0,('No real TTS audio',events)
                if mode=='greet_unknown':
                    assert any(e['type']=='visual_greeting' for e in events),events
                    assert any(e['type']=='reply' and e.get('timing',{}).get('director_ms')==0 for e in events),'Greeting used an unnecessary LLM call'
                if mode=='greet_disabled':assert not binary and not any(e['type']=='visual_greeting' for e in events),events
                if mode=='visible':
                    assert any(e['type']=='visual_person' and e.get('name')=='Christoph' and e.get('track_id')=='face-1' for e in events),events
                    await ws.send_str('visual:'+json.dumps({'type':'frame','camera':'on','audio_s':sample/16000,'faces':[]}))
                    await ws.send_str('interrupt')
                    for _ in range(30):
                        if any(e['type']=='barge' for e in events):break
                        await asyncio.sleep(.1)
                    assert any(e['type']=='barge' for e in events),'Manual Stop was blocked by no visible speaker'
                assert not any(e['type'] in ('error','visual_warning') for e in events),events
                return {'mode':mode,'passed':True,'endpoint':bool(endpoints),'streamed_audio_bytes':binary,'events':events}
            finally:
                await ws.close();task.cancel();await asyncio.gather(task,return_exceptions=True)

async def main(modes):
    report={'started':time.time(),'runtime_source_sha256':runtime_signature(),'physical_devices_used':False,'visual_observations':'synthetic','real_asr_and_tts':True,'cases':[],'passed':False}
    target=ROOT/'logs/multimodal-live-checks.json'
    try:
        for mode in modes:
            try:record=await case(mode)
            except Exception as exc:
                report['cases'].append({'mode':mode,'passed':False,'error':str(exc)})
                raise
            report['cases'].append(record);target.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in record.items() if k!='events'}),flush=True)
        report['passed']=report['runtime_source_sha256']==runtime_signature()
    finally:
        report['finished']=time.time();target.write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--modes',nargs='+',default=['visible','offscreen','static_open','overlap','disabled','camera_off','greet_unknown','greet_disabled']);asyncio.run(main(parser.parse_args().modes))
