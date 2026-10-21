"""Actual streamed-TTS loopback with a simulated DAC, delay and playback tags.

All inputs are public synthetic fixtures. This exercises the complete live
speaker gate and second ear, not physical loudspeaker/microphone acoustics.
"""
import argparse,asyncio,json,struct,time
from pathlib import Path
import aiohttp,numpy as np
from scipy.signal import resample_poly
from speaker_v4.common import atomic
from .live_test import audio

async def main(delay_ms=120,echo_gain=.05,user_gain=1.):
    report_path=Path('/mnt/nvme/robo-speaker-studies/speaker-live-results/streamed-echo-live.json')
    atomic(report_path,{'public_synthetic_only':True,'protocol_version':1,'passed':False,'running':True})
    segments=[];events=[];sample=0;done=[];received={};stopped=set();phase='intro'
    first_audio={};reply_end=0;buffer=3840;delay=round(delay_ms*16)
    intro=audio('christoph-intro');follow=audio('christoph-followup')*user_gain
    async with aiohttp.ClientSession() as client:
        async with client.ws_connect('http://127.0.0.1:8793/ws?synthetic_diagnostics=1') as ws:
            t0=time.monotonic()
            async def receive():
                nonlocal reply_end
                async for msg in ws:
                    if msg.type==aiohttp.WSMsgType.BINARY:
                        rid=struct.unpack_from('<i',msg.data)[0]
                        if not rid:continue
                        pcm=np.frombuffer(msg.data,'<i2',offset=4).astype(np.float32)/32768
                        at24=received.get(rid,0);received[rid]=at24+len(pcm)
                        wave=resample_poly(pcm,2,3).astype(np.float32)
                        start=max(sample+buffer,reply_end)
                        segments.append((start,start+len(wave),rid,at24,wave));reply_end=start+len(wave)
                        first_audio.setdefault(rid,start/16000)
                    elif msg.type==aiohttp.WSMsgType.TEXT:
                        item=json.loads(msg.data);kind=item.get('type')
                        if kind in ('reply','endpoint','barge','turn_done','warning','ignored','error'):
                            events.append({'at_s':sample/16000,'phase':phase,**item})
                        if kind in ('synthetic_barge_proposal','synthetic_barge_verification') or (kind=='identity' and phase=='user_interrupt'):
                            events.append({'at_s':sample/16000,'phase':phase,**item})
                        if kind=='state':events.append({'at_s':sample/16000,'phase':phase,'type':'yield_probe','yield':item.get('yield'),'state':item.get('state')})
                        if kind=='barge':stopped.update(received)
                        if kind=='turn_done':done.append(item)
            reader=asyncio.create_task(receive())
            async def feed(wave=None,seconds=None):
                nonlocal sample
                n=len(wave) if wave is not None else round(seconds*16000)
                for offset in range(0,n,320):
                    size=min(320,n-offset);mic=np.zeros(size,np.float32)
                    if wave is not None:mic+=wave[offset:offset+size]
                    left=sample-delay;right=left+size;cuts={0,size};active=[]
                    for a,b,rid,at24,x in segments:
                        if rid in stopped or b<=left or a>=right:continue
                        lo=max(left,a);hi=min(right,b);first=lo-left;last=hi-left
                        mic[first:last]+=x[lo-a:hi-a]*echo_gain
                        active.append((first,last,rid,at24+round((lo-a)*1.5)))
                        cuts.update((first,last))
                    edges=sorted(cuts)
                    for a,b in zip(edges,edges[1:]):
                        match=next(((rid,pos+round((a-lo)*1.5)) for lo,hi,rid,pos in active if lo<=a<hi),None)
                        rid,pos=match if match else (0,-1)
                        pcm=np.clip(np.rint(mic[a:b]*32767),-32768,32767).astype('<i2')
                        await ws.send_bytes(struct.pack('<ii',rid,pos)+pcm.tobytes())
                    sample+=size
                    # DAC acknowledgments refer to actual playback, not when
                    # a streamed chunk first arrived at the client.
                    for rid,total in received.items():
                        if rid in stopped:continue
                        played=max((p+round((min(sample,b)-a)*1.5) for a,b,r,p,x in segments if r==rid and sample>=a),default=0)
                        if played:await ws.send_str(f'played:{rid}:{min(total,played)}')
                    await asyncio.sleep(max(0,t0+sample/16000-time.monotonic()))
            async def until(predicate,limit=45):
                for _ in range(round(limit/.32)):
                    if predicate():return
                    await feed(seconds=.32)
                raise TimeoutError('Synthetic echo conversation did not advance')
            try:
                await feed(seconds=1.28);await feed(intro)
                phase='echo_only'
                await until(lambda:len(done)>=1)
                await feed(seconds=1.6)
                assert not any(e['type']=='barge' for e in events),'Bot echo caused an interruption'
                assert len([e for e in events if e['type']=='endpoint'])==1,'Bot echo caused an extra accepted user turn'
                phase='followup';await feed(follow)
                await until(lambda:len(first_audio)>=2)
                rid=max(first_audio);await until(lambda:sample/16000>=first_audio[rid]+1.0,limit=8)
                phase='user_interrupt';interrupt_at=sample/16000
                await feed(follow);await feed(seconds=2.56)
                barges=[e for e in events if e['type']=='barge']
                report={'public_synthetic_only':True,'protocol_version':1,'echo_gain':echo_gain,'echo_delay_ms':delay_ms,'user_gain':user_gain,
                    'simulated_playback_buffer_ms':buffer/16,'physical_device_test':False,'actual_streamed_tts':True,
                    'echo_only_false_barges':sum(e['phase']=='echo_only' for e in barges),
                    'interruption_start_s':interrupt_at,'events':events,
                    'interruption_latency_s':min((e['at_s']-interrupt_at for e in barges if e['phase']=='user_interrupt'),default=None),
                    'passed':bool(barges) and any(e['phase']=='user_interrupt' for e in barges)}
                folder=Path('/mnt/nvme/robo-speaker-studies/speaker-live-results')
                atomic(folder/'streamed-echo-live.json',report)
                print(json.dumps({k:v for k,v in report.items() if k!='events'}),flush=True)
                assert report['passed'],'Known user did not interrupt while actual streamed TTS leaked into the simulated microphone'
            except Exception as exc:
                atomic(report_path,{'public_synthetic_only':True,'protocol_version':1,'passed':False,
                    'failure_type':type(exc).__name__,'echo_gain':echo_gain,'echo_delay_ms':delay_ms,'user_gain':user_gain,'events':events})
                raise
            finally:
                await ws.close();reader.cancel();await asyncio.gather(reader,return_exceptions=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--delay-ms',type=float,default=120);p.add_argument('--echo-gain',type=float,default=.05)
    p.add_argument('--user-gain',type=float,default=1.)
    a=p.parse_args();asyncio.run(main(a.delay_ms,a.echo_gain,a.user_gain))
