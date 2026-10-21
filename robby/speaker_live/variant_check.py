"""Real ASR/ECAPA/LLM/TTS test of an EXPLICITLY ASSERTED identity variant.

Two different public synthetic voices intentionally claim one conversational
name. This validates the opt-in session association, NOT biometric accuracy.
"""
import asyncio,json,struct,time,hashlib
from pathlib import Path
import aiohttp,httpx,numpy as np
from .live_test import audio
from .fixtures import ROOT
from .playback import SyntheticPlayback
from final_verify import runtime_signature

def fixture():
    path=ROOT/'selma-claims-christoph.mp3'
    if path.exists():return
    body=dict(model='small_rl',input_mode='manual',format='structured',
        message="GENERAL: Clear natural German speech. Preserve the reference speaker.\nSCRIPT:\n(interest, clearly articulated) Ich bin es doch, Christoph. Heute klingt meine Stimme etwas anders.",
        generation='single',n=1,streaming=False,seconds=12.,temperature=.7,guidance=2.,
        voice_mode='profile',voice_profile='emolia_c0542',voice_conversion=False,align=True,seed=20261021)
    with httpx.Client(base_url='http://127.0.0.1:8792',timeout=120) as c:
        if c.get('/api/state').json().get('busy'):raise RuntimeError('Shared speech service busy; do not compete with a live test')
        candidate=None
        with c.stream('POST','/api/turn',json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith('data: '):
                    event=json.loads(line[6:])
                    if event.get('event')=='result':candidate=event['candidates'][0]
        if candidate is None:raise RuntimeError('No synthetic fixture')
        r=c.get(candidate['url']);r.raise_for_status();path.write_bytes(r.content)
        path.with_suffix('.json').write_text(json.dumps(dict(public_synthetic_only=True,request=body,sha256=hashlib.sha256(r.content).hexdigest()),indent=2))

async def main():
    report=dict(passed=False,synthetic_only=True,security_test=False,
        limitation='Different synthetic voices deliberately assert one name; not biometric recognition accuracy.',
        runtime_source_sha256=runtime_signature(),events=[],turns=[])
    path=Path('logs/speaker-variant-live.json');dac=SyntheticPlayback(lead16=3840);sample=0
    async with aiohttp.ClientSession() as client:
        for _ in range(120):
            async with client.get('http://127.0.0.1:8793/api/state') as response:
                if (await response.json()).get('ready'):break
            await asyncio.sleep(.5)
        else:raise TimeoutError('ASR startup not ready; no speech replay attempted')
        async with client.ws_connect('http://127.0.0.1:8793/ws') as ws:
            t0=time.monotonic()
            async def read():
                async for msg in ws:
                    if msg.type==aiohttp.WSMsgType.BINARY:
                        rid=struct.unpack_from('<i',msg.data)[0]
                        if rid:dac.add(rid,(len(msg.data)-4)//2,sample)
                    elif msg.type==aiohttp.WSMsgType.TEXT:
                        data=json.loads(msg.data)
                        if data.get('type') not in ('state','input_status'):report['events'].append({'audio_s':sample/16000,**data})
                        if data.get('type')=='barge':dac.cancel(sample)
            reader=asyncio.create_task(read())
            async def feed(wave):
                nonlocal sample
                for at in range(0,len(wave),320):
                    x=wave[at:at+320];pcm=np.clip(np.rint(x*32767),-32768,32767).astype('<i2')
                    for lo,hi,rid,pos in dac.pieces(sample,len(x)):await ws.send_bytes(struct.pack('<ii',rid,pos)+pcm[lo:hi].tobytes())
                    sample+=len(x)
                    if sample//320%5==0:
                        for rid,position in dac.acknowledgments(sample).items():await ws.send_str(f'played:{rid}:{position}')
                    await asyncio.sleep(max(0,t0+sample/16000-time.monotonic()))
            try:
                await ws.send_str('identity:open:on');await feed(np.zeros(16000,np.float32))
                for name in ['christoph-intro','selma-claims-christoph','selma-claims-christoph','christoph-followup']:
                    begin=sample/16000;await feed(audio(name));seen=False
                    for _ in range(160):
                        await feed(np.zeros(5120,np.float32))
                        if any(e['type']=='turn_done' and e['audio_s']>begin and not e.get('barged') for e in report['events']):seen=True;break
                    rows=[e for e in report['events'] if e['type']=='endpoint' and e['audio_s']>begin]
                    row=dict(fixture=name,completed=seen,endpoints=rows);report['turns'].append(row);path.write_text(json.dumps(report,indent=2,ensure_ascii=False))
                    print(json.dumps(row,ensure_ascii=False),flush=True)
                    if not seen:raise RuntimeError('No completed turn for '+name)
                    await feed(np.zeros(8000,np.float32))
                variants=[e['voice_variant'] for e in report['events'] if e.get('voice_variant')]
                assert any(v['name']=='Christoph' and v['count']>=2 for v in variants),variants
                for turn in report['turns'][-2:]:assert any(e.get('speaker_name')=='Christoph' for e in turn['endpoints']),turn
                report['passed']=runtime_signature()==report['runtime_source_sha256']
            except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)
            finally:
                await ws.close();await reader;path.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit('See retained variant-test failure.')

if __name__=='__main__':fixture();asyncio.run(main())
