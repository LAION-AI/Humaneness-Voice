"""Real-model ASR/caption timing: serial baselines vs simultaneous HTTP jobs.

Uses a pinned public synthetic recording, never a user's microphone. Repeating
its samples creates duration probes, not independent emotion-quality examples.
"""
import asyncio,hashlib,json,time
from pathlib import Path
import numpy as np
import httpx
from generate_adversarial_fixtures import decode,fixture_root

ROOT=Path(__file__).resolve().parent
async def main():
    source=fixture_root()/'christoph-intro.mp3';wave=decode(source);rows=[]
    async with httpx.AsyncClient(timeout=30,trust_env=False) as client:
        state=(await client.get('http://127.0.0.1:8802/state')).json()
        async def request(url,data):
            begin=time.perf_counter();r=await client.post(url,content=data);r.raise_for_status()
            return {'start_s':begin,'finish_s':time.perf_counter(),'http_ms':round((time.perf_counter()-begin)*1000,2),'result':r.json()}
        for seconds in (1,2,4,8):
            data=np.resize(wave,seconds*16000).astype('<f4').tobytes()
            for repeat in range(3):
                for mode in ('asr_only','caption_only','parallel'):
                    asr='http://127.0.0.1:8801/transcribe/parakeet_v3';cap='http://127.0.0.1:8802/caption'
                    urls=[asr] if mode=='asr_only' else [cap] if mode=='caption_only' else [asr,cap]
                    begin=time.perf_counter();results=await asyncio.gather(*(request(url,data) for url in urls))
                    rows.append({'seconds':seconds,'repeat':repeat,'mode':mode,'wall_ms':round((time.perf_counter()-begin)*1000,2),'requests':results})
    report={'passed':True,'worker':state,'fixture_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'scope':'Warm HTTP timing probes; repeated synthetic waveform, not accuracy or physical-room evaluation.',
        'rows':rows}
    report['passed']=all(r['requests'][-1]['result'].get('device','cuda:1').startswith('cuda') for r in rows)
    (ROOT/'static/voice-caption-timing.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'passed':report['passed'],'device':state['device'],'rows':len(rows),
        'median_ms':{f'{seconds}s {mode}':round(float(np.median([r['wall_ms'] for r in rows if r['seconds']==seconds and r['mode']==mode])),1)
            for seconds in (1,2,4,8) for mode in ('asr_only','caption_only','parallel')}},indent=2))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':asyncio.run(main())
