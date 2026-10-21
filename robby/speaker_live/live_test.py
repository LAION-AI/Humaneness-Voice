"""Real WebSocket / ASR / diarization / TTS check with synthetic fixtures only."""
import argparse,asyncio,json,struct,subprocess,time
from pathlib import Path
import aiohttp
import numpy as np
from speaker_v4.common import atomic
from .fixtures import ROOT

def audio(name):
    p=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(ROOT/(name+'.mp3')),'-ar','16000','-ac','1','-f','f32le','pipe:1'],capture_output=True,check=True)
    x=np.frombuffer(p.stdout,'<f4').copy();rms=np.sqrt(np.mean(x*x));return x*(.055/max(float(rms),1e-6))

async def main(url,mode='v4-long'):
    intro=audio('christoph-intro');bg=audio('background');offset=1.28
    duration=offset+len(intro)/16000+5.12;n=round(duration*16000)
    background=np.resize(bg,n)*.20;wave=background.copy();start=round(offset*16000);wave[start:start+len(intro)]+=intro
    events=[];chunks=0;counts={};started=time.monotonic();first_endpoint=None;new_users=[];errors=[]
    async with aiohttp.ClientSession() as client:
        for _ in range(90):
            try:
                async with client.get(url.rstrip('/')+'/api/state') as status:
                    if (await status.json()).get('ready'):break
            except aiohttp.ClientError:pass
            await asyncio.sleep(.5)
        async with client.ws_connect(url.rstrip('/')+'/ws') as ws:
            async def receive():
                nonlocal chunks,first_endpoint
                async for message in ws:
                    if message.type==aiohttp.WSMsgType.BINARY:
                        rid=struct.unpack_from('<i',message.data)[0]
                        if rid:
                            chunks+=1;counts[rid]=counts.get(rid,0)+(len(message.data)-4)//2
                            await ws.send_str(f'played:{rid}:{counts[rid]}')
                    elif message.type==aiohttp.WSMsgType.TEXT:
                        item=json.loads(message.data);kind=item.get('type')
                        if kind in ('warning','error'):errors.append(item)
                        if kind=='identity' and item.get('decision',{}).get('new_user'):new_users.append(item['decision'])
                        if kind in ('endpoint','reply'):events.append({'at_s':time.monotonic()-started,**item})
                        if kind=='endpoint' and first_endpoint is None:first_endpoint=time.monotonic()-started
            task=asyncio.create_task(receive())
            try:
                await ws.send_str('identity:mode:'+mode);await asyncio.sleep(.3)
                started=time.monotonic()
                for at in range(0,len(wave),320):
                    x=wave[at:at+320];pcm=np.clip(np.rint(x*32767),-32768,32767).astype('<i2')
                    if ws.closed:break
                    try:await ws.send_bytes(struct.pack('<ii',0,-1)+pcm.tobytes())
                    except aiohttp.ClientConnectionError:break
                    target=started+(at+len(x))/16000;await asyncio.sleep(max(0,target-time.monotonic()))
                for _ in range(300):
                    if chunks:break
                    await asyncio.sleep(.1)
                report={'synthetic_fixtures_only':True,'mode':mode,'background_continues_after_target':True,
                    'target_end_s':offset+len(intro)/16000,'first_endpoint_s':first_endpoint,
                    'submit_after_target_end_s':None if first_endpoint is None else first_endpoint-(offset+len(intro)/16000),
                    'new_users':new_users,'events':events,'reply_audio_chunks':chunks,'warnings':errors,
                    'playback':'virtual immediate acknowledgments; not a physical microphone/DAC test'}
                folder=Path('/mnt/nvme/robo-speaker-studies/speaker-live-results');folder.mkdir(exist_ok=True)
                atomic(folder/(mode+'-live.json'),report);print(json.dumps(report,indent=2),flush=True)
                assert first_endpoint is not None and chunks>0,'No spoken reply while background continued'
                assert any(e.get('speaker_name')=='Christoph' for e in events if e.get('type')=='endpoint'),'Speaker name not bound after full introduction'
            finally:
                await ws.close();task.cancel();await asyncio.gather(task,return_exceptions=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8793');parser.add_argument('--mode',default='v4-long');a=parser.parse_args()
    asyncio.run(main(a.url,a.mode))
