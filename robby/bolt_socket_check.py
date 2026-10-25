"""Real-time paced synthetic ASR -> local LLM -> TTS socket check.

PCM is delivered as a browser would capture it, 20 ms at a time. Playback
acknowledgments here are transport checks, not a physical loudspeaker test.
"""
import argparse,asyncio,json,struct,time
from pathlib import Path
import aiohttp,numpy as np

ROOT=Path(__file__).resolve().parent

async def main(args):
    audio=np.load(args.audio).astype(np.float32)
    audio=np.r_[np.zeros(16000,np.float32),audio,np.zeros(12*16000,np.float32)]
    active=np.flatnonzero(np.abs(audio)>.003);last_speech=active[-1]/16000
    report={'passed':False,'input':'authored synthetic request without a wake phrase','physical_microphone':False,
        'real_time_pacing':True,'source_duration_s':last_speech,'events':[],'audio_chunks':0}
    async with aiohttp.ClientSession() as session,session.ws_connect(args.url+'/ws') as ws:
        start=time.perf_counter();done=asyncio.Event();received=0;first_audio=None
        async def consume():
            nonlocal received,first_audio
            async for packet in ws:
                if packet.type==aiohttp.WSMsgType.TEXT:
                    item=json.loads(packet.data)
                    if item.get('type') not in ('state','input_status'):
                        report['events'].append({**item,'received_ms':round((time.perf_counter()-start)*1000,2)})
                        print(json.dumps(report['events'][-1],ensure_ascii=False),flush=True)
                    if item.get('type')=='turn_done' and received:done.set();return
                elif packet.type==aiohttp.WSMsgType.BINARY:
                    rid=struct.unpack_from('<i',packet.data)[0];received+=(len(packet.data)-4)//2
                    if first_audio is None:first_audio=(time.perf_counter()-start)*1000
                    report['audio_chunks']+=1
                    await ws.send_str(f'played:{rid}:{received}')
        receiver=asyncio.create_task(consume())
        try:
            for at in range(0,len(audio),320):
                if done.is_set():break
                target=start+at/16000
                await asyncio.sleep(max(0,target-time.perf_counter()))
                block=audio[at:at+320]
                raw=struct.pack('<ii',0,-1)+(np.clip(block,-1,1)*32767).astype('<i2').tobytes()
                await ws.send_bytes(raw)
            await asyncio.wait_for(done.wait(),45)
            endpoints=[m for m in report['events'] if m['type']=='endpoint']
            replies=[m for m in report['events'] if m['type']=='reply']
            assert endpoints and replies and received,'Missing ASR, LLM or TTS'
            assert 'gemeinsam machen' in endpoints[0]['transcript'].casefold(),'Final words were clipped'
            report.update(passed=True,transcript=endpoints[0]['transcript'],reply=replies[-1]['text'],
                first_audio_from_capture_start_ms=round(first_audio,2),
                first_audio_after_last_source_speech_ms=round(first_audio-last_speech*1000,2),
                endpoint_after_last_source_speech_ms=round(endpoints[0]['received_ms']-last_speech*1000,2),
                output_audio_s=round(received/24000,3))
        except Exception as exc:report['error']=str(exc)
        finally:
            receiver.cancel();await asyncio.gather(receiver,return_exceptions=True)
    (ROOT/'static/bolt-socket-check.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in report.items() if k!='events'},ensure_ascii=False),flush=True)
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8798');p.add_argument('--audio',type=Path,default=ROOT/'logs/bolt-benchmark/input16.npy')
    asyncio.run(main(p.parse_args()))
