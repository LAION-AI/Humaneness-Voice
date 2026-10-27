"""Real GPU measurements on authored signals; no physical-room accuracy claim."""
import asyncio,json,time
from pathlib import Path
import httpx
import numpy as np
from environment_fixtures import fixtures
from generate_adversarial_fixtures import decode,fixture_root

ROOT=Path(__file__).resolve().parent
async def main():
    rows=[];sounds=fixtures();speech=decode(fixture_root()/'christoph-intro.mp3')[:64000]
    async with httpx.AsyncClient(timeout=20,trust_env=False) as c:
        state=(await c.get('http://127.0.0.1:8803/state')).json()
        async def call(kind,wave):
            url={'sound':'http://127.0.0.1:8803/caption','asr':'http://127.0.0.1:8801/transcribe/parakeet_v3','tone':'http://127.0.0.1:8802/caption'}[kind]
            start=time.perf_counter();r=await c.post(url,content=wave.astype('<f4').tobytes());r.raise_for_status()
            return {'stage':kind,'http_ms':round((time.perf_counter()-start)*1000,2),'result':r.json()}
        for name,wave in sounds.items():
            for repeat in range(3):rows.append({'fixture':name,'repeat':repeat,**await call('sound',wave)})
        for mode in ('asr_only','serial','parallel'):
            for repeat in range(3):
                start=time.perf_counter()
                if mode=='asr_only':jobs=[await call('asr',speech)]
                elif mode=='serial':jobs=[await call('asr',speech),await call('sound',sounds['impacts']),await call('tone',speech)]
                else:jobs=await asyncio.gather(call('asr',speech),call('sound',sounds['impacts']),call('tone',speech))
                rows.append({'mode':mode,'repeat':repeat,'wall_ms':round((time.perf_counter()-start)*1000,2),'jobs':jobs})
    report={'passed':True,'worker':state,'scope':__doc__,'rows':rows,
        'pre_optimization_impacts_ms':[373.54,368.70,368.57],
        'notes':['Synthetic probe captions are hypotheses, not ground-truth accuracy scores.','Timing stages overlap; do not add parallel latencies.','Generated stimuli are CC0; existing speech fixture provenance is in the adversarial test protocol.']}
    (ROOT/'static/environment-benchmark.json').write_text(json.dumps(report,ensure_ascii=False,separators=(',',':')))
    print(json.dumps({'sound_medians_ms':{name:round(float(np.median([r['http_ms'] for r in rows if r.get('fixture')==name])),1) for name in sounds},
        'pipeline_medians_ms':{mode:round(float(np.median([r['wall_ms'] for r in rows if r.get('mode')==mode])),1) for mode in ('asr_only','serial','parallel')}},indent=2))

if __name__=='__main__':asyncio.run(main())
