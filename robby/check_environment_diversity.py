"""Pinned, small sound regression; public clips stay in the local test cache.

First two CSV rows in each named ESC-50 category, no success-based selection.
These are diagnostic examples, not a representative accuracy benchmark.
The dataset is CC BY-NC 3.0; per-source credits are recorded verbatim from its
LICENSE. Downloaded audio is not included in the code/data release.
"""
import asyncio,csv,hashlib,io,json,time
from pathlib import Path
import httpx,numpy as np,soundfile as sf
from scipy.signal import resample_poly
from ambient_audio import AmbientDetector,caption_kind
from environment_fixtures import music_fixture,fixtures
ROOT=Path(__file__).resolve().parent
REV='33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6'
BASE=f'https://raw.githubusercontent.com/karolpiczak/ESC-50/{REV}'
CACHE=ROOT/'logs/environment-fixtures'
CATEGORIES=('dog','cat','glass_breaking','car_horn','engine','rain')

def load_fixtures():
    CACHE.mkdir(exist_ok=True,parents=True);waves={};manifest={}
    with httpx.Client(timeout=30,trust_env=False) as c:
        r=c.get(BASE+'/meta/esc50.csv');r.raise_for_status();rows=list(csv.DictReader(io.StringIO(r.text)))
        r=c.get(BASE+'/LICENSE');r.raise_for_status();license_lines=r.text.splitlines()
        for category in CATEGORIES:
            for index,row in enumerate([x for x in rows if x['category']==category][:2]):
                name=f'{category}-{index+1}';path=CACHE/row['filename'];url=BASE+'/audio/'+row['filename']
                if not path.exists():
                    r=c.get(url);r.raise_for_status();path.write_bytes(r.content)
                wave,sr=sf.read(path,dtype='float32');wave=wave.mean(axis=1) if wave.ndim>1 else wave
                wave=resample_poly(wave,160,sr//100).astype(np.float32)
                wave*=min(5.,.07/max(.0001,float(np.sqrt(np.mean(wave*wave)))))
                wave=np.clip(wave,-.95,.95)
                waves[name]=wave
                source='-'.join(row['filename'].split('-')[:3])+'.ogg'
                credit=next((line for line in license_lines if '['+source+']' in line),'See upstream LICENSE')
                manifest[name]={**row,'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'credit':credit,'source_license_url':BASE+'/LICENSE','audio_redistributed':False}
    waves['authored_music']=music_fixture(5.);manifest['authored_music']={'license':'CC0 authored deterministic melody, environment_fixtures.py'}
    waves['authored_impacts']=np.pad(fixtures()['impacts'],(0,48000));manifest['authored_impacts']={'license':'CC0 procedural impacts'}
    return waves,manifest

async def main():
    waves,manifest=load_fixtures();rows=[]
    async with httpx.AsyncClient(timeout=20,trust_env=False) as c:
        for name,wave in waves.items():
            for mode in ('onset','vad_positive','already_playing'):
                detector=AmbientDetector();events=[];now=0.
                full=np.r_[np.zeros(32000,np.float32),wave,np.zeros(32000,np.float32)] if mode!='already_playing' else np.tile(wave,4)
                for i in range(0,len(full),320):
                    now+=len(full[i:i+320])/16000
                    result=detector.feed(full[i:i+320],now,speech=mode=='vad_positive')
                    if result:events.append(result)
                row={'fixture':name,'mode':mode,'events':len(events)}
                if events:
                    start=time.perf_counter();event=events[0]
                    r=await c.post('http://127.0.0.1:8803/caption',content=event['audio'].astype('<f4').tobytes());r.raise_for_status();result=r.json()
                    row.update(caption=result.get('caption'),kind=caption_kind(result.get('caption','')),http_ms=round((time.perf_counter()-start)*1000,2),capture_ms=event['capture_ms'])
                    row['routed']=row['kind'] in ('music','sound')
                rows.append(row)
                print(json.dumps(row),flush=True)
    report={'scope':__doc__,'source_revision':REV,'fixtures':manifest,'rows':rows,'all_routed':all(r.get('routed') for r in rows)}
    (ROOT/'static/environment-diversity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('routing',sum(r.get('routed',False) for r in rows),'/',len(rows))
if __name__=='__main__':asyncio.run(main())
