"""Serial real-audio regression checks for pause-only ASR turn submission.

Only authored synthetic TTS fixtures/noise. Acknowledges received bot PCM for
transport completion; this is not physical microphone/room echo validation.
"""
import asyncio,json,struct,time
from pathlib import Path
import aiohttp,httpx,numpy as np
from scipy.signal import resample_poly
from pause_vad import SpeechClock,REVISION,SHA256
from benchmark_bolt import measure

ROOT=Path(__file__).resolve().parent
FOLDER=ROOT/'logs/speech-pause';FOLDER.mkdir(exist_ok=True)

def trim(audio):
    # Retain 80 ms at the edges; remove the synthesis-only long tail so the
    # authored short gap actually tests continuous multi-sentence speech.
    active=np.flatnonzero(np.abs(audio)>.006)
    return audio[max(0,active[0]-1280):min(len(audio),active[-1]+1280)]

async def setup():
    async with httpx.AsyncClient(base_url='http://127.0.0.1:8798',timeout=120) as client:
        _,wave=await measure(client,'/api/bolt/tts',{'text':'Bitte schlage mir ein lustiges Spiel vor.','cue':'interest, natural','voice':'anime_088','chunk_frames':20})
    first=trim(np.load(ROOT/'logs/bolt-benchmark/input16.npy'))
    second=trim(resample_poly(wave,2,3).astype(np.float32))
    np.save(FOLDER/'first.npy',first);np.save(FOLDER/'second.npy',second)
    return first,second

async def case(name,first,second,pause,gap,noise=False,only_noise=False):
    rng=np.random.default_rng(44)
    wave=np.r_[np.zeros(16000),first,np.zeros(round(gap*16000)),second,np.zeros(4*16000)].astype(np.float32)
    if only_noise:wave=np.zeros(6*16000,np.float32)
    if noise or only_noise:
        at=np.arange(len(wave))/16000
        wave+=rng.normal(0,.014,len(wave)).astype(np.float32)+(.012*np.sin(2*np.pi*120*at)).astype(np.float32)
    np.save(FOLDER/(name+'.npy'),wave)
    row={'case':name,'pause_ms':pause,'gap_s':gap,'noise':noise or only_noise,'passed':False,'events':[]}
    async with aiohttp.ClientSession() as client,client.ws_connect('http://127.0.0.1:8798/ws') as ws:
        start=time.perf_counter();received={}
        await ws.send_str('asr:pause_ms:'+str(pause))
        async def consume():
            async for packet in ws:
                if packet.type==aiohttp.WSMsgType.TEXT:
                    item=json.loads(packet.data)
                    if item.get('type') in ('asr_sentence','endpoint','error','warning','asr_settings','barge'):
                        row['events'].append({**item,'received_ms':round((time.perf_counter()-start)*1000,2)})
                elif packet.type==aiohttp.WSMsgType.BINARY:
                    rid=struct.unpack_from('<i',packet.data)[0];received[rid]=received.get(rid,0)+(len(packet.data)-4)//2
                    await ws.send_str(f'played:{rid}:{received[rid]}')
        task=asyncio.create_task(consume())
        try:
            for at in range(0,len(wave),320):
                await asyncio.sleep(max(0,start+at/16000-time.perf_counter()))
                block=wave[at:at+320]
                await ws.send_bytes(struct.pack('<ii',0,-1)+(np.clip(block,-1,1)*32767).astype('<i2').tobytes())
            await asyncio.sleep(.2)
            sent=[e for e in row['events'] if e['type']=='asr_sentence']
            if only_noise:assert not sent,'Noise submitted a turn'
            elif gap<pause/1000:
                assert len(sent)==1,[(e.get('text'),e.get('received_ms')) for e in sent]
                assert 'gemeinsam machen' in sent[0]['text'].casefold() and 'spiel vor' in sent[0]['text'].casefold(),sent
                end=(16000+len(first)+round(gap*16000)+len(second))/16
                row['submission_after_clip_end_ms']=sent[0]['received_ms']-end
            else:
                assert len(sent)==2,sent
                assert 'gemeinsam machen' in sent[0]['text'].casefold() and 'spiel vor' in sent[1]['text'].casefold(),sent
            assert all(e['observed_quiet_ms']>=pause for e in sent),sent
            assert not [e for e in row['events'] if e['type']=='error'],row['events']
            row['passed']=True
        except Exception as exc:row['error']=str(exc)
        finally:
            task.cancel();await asyncio.gather(task,return_exceptions=True)
    print(json.dumps({k:v for k,v in row.items() if k!='events'},ensure_ascii=False),flush=True)
    await asyncio.sleep(.5)
    return row

async def main():
    first,second=await setup()
    report={'model_revision':REVISION,'sha256':SHA256,'physical_microphone':False,'rows':[]}
    target=ROOT/'static/speech-pause-check.json'
    for args in [('two_sentences',400,.12,False,False),('two_sentences_with_noise',400,.12,True,False),
                 ('longer_setting_holds',1000,.50,True,False),('two_separate_turns',300,1.20,False,False),
                 ('short_separate_turns',300,.60,False,False),('noise_only',400,.12,True,True)]:
        report['rows'].append(await case(args[0],first,second,*args[1:]));target.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    audio=np.r_[np.zeros(16000),first,np.zeros(16000)].astype(np.float32)
    clock=SpeechClock();started=time.perf_counter()
    for at in range(0,len(audio),320):clock.feed(audio[at:at+320],min(at+320,len(audio))/16000)
    report.update(vad_ms_per_32ms_frame=clock.compute_ms/max(1,clock.calls),vad_total_wall_ms=(time.perf_counter()-started)*1000,
        passed=all(r['passed'] for r in report['rows']),finished=time.time())
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False));print('PASSED',report['passed'],flush=True)
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':asyncio.run(main())
