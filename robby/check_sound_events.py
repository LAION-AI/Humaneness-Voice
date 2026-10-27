"""Paired real CPU event / GPU caption tests, pinned public and authored sounds."""
import asyncio,datetime,json,time
from pathlib import Path
import numpy as np,httpx
from check_environment_diversity import load_fixtures
from environment_fixtures import music_fixture
ROOT=Path(__file__).resolve().parent
async def main():
    waves,manifest=load_fixtures();rows=[];times=[]
    async with httpx.AsyncClient(timeout=15,trust_env=False) as c:
        state=(await c.get('http://127.0.0.1:8804/state')).json()
        async def post(url,wave):
            start=time.perf_counter();r=await c.post(url,content=wave.astype('<f4').tobytes());r.raise_for_status()
            return {**r.json(),'http_ms':round((time.perf_counter()-start)*1000,2)}
        for name,wave in waves.items():
            start=time.perf_counter()
            tags,caption=await asyncio.gather(post('http://127.0.0.1:8804/events',wave),post('http://127.0.0.1:8803/caption',wave))
            rows.append({'fixture':name,'tags':tags,'caption':caption,'parallel_wall_ms':round((time.perf_counter()-start)*1000,2)})
            print(name,tags['candidates'][:3],tags['total_ms'],flush=True)
        # Alternate solo/parallel to reduce simple warm-order bias. Same audio.
        for repeat in range(6):
            for mode in (('caption_only','parallel') if repeat%2==0 else ('parallel','caption_only')):
                wave=music_fixture(3.2);start=time.perf_counter()
                if mode=='parallel':jobs=await asyncio.gather(post('http://127.0.0.1:8804/events',wave),post('http://127.0.0.1:8803/caption',wave))
                else:jobs=[await post('http://127.0.0.1:8803/caption',wave)]
                times.append({'repeat':repeat,'mode':mode,'wall_ms':round((time.perf_counter()-start)*1000,2),'jobs':jobs})
    report={'created':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':__doc__,'worker':state,'fixtures':manifest,'rows':rows,'timings':times,
        'notes':['No spatial localization with a mono microphone.','AudioSet scores are not calibrated probabilities.','The 1.28-second sliding windows are approximate temporal localization, not event boundary ground truth.','These few examples are diagnostic, not a representative accuracy estimate.']}
    (ROOT/'static/environment-event-tagger.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print({m:float(np.median([x['wall_ms'] for x in times if x['mode']==m])) for m in ('caption_only','parallel')},flush=True)
if __name__=='__main__':asyncio.run(main())
