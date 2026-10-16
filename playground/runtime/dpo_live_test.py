"""Real HTTP DPO/reference smoke, isolated session; not an efficacy study."""
import argparse
import hashlib
import http.cookiejar
import json
from pathlib import Path
import time
import urllib.request

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',default='http://127.0.0.1:8792');args=parser.parse_args()
    opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    def get(path):
        with opener.open(args.base+path,timeout=600) as response:return response.read()
    state=json.loads(get('/api/state'))
    if not state['ready'] or state['busy']:
        raise RuntimeError('Demo not ready or busy; no test started')
    cases=[('base-before',{'id':None},'none',None,'Warm natural delivery.','single'),
           ('previous-dpo-no-ref',{'id':'subset_pitch_quality_p2_rank016'},'none',None,'Warm natural delivery.','single'),
           ('mira-anger',{'id':'subset_pitch_quality_p2_rank016'},'profile','emolia_c1699','Angry delivery, tense and intense.','single'),
           ('mira-sadness',{'id':'subset_pitch_quality_p2_rank016'},'profile','emolia_c1699','Sad delivery, restrained grief, quiet sorrow.','single'),
           ('idris-winner-800',{},'profile','anime_088','Joyful delivery, bright delight.','search_winner'),
           ('nora-top2-1600',{},'profile','k10_age3_bg1','Warm affectionate delivery.','search_beam')]
    for id_ in ('subset_pitch_quality_p2_rank064','rebalanced_rank064','rebalanced_rank128','original_rank064','original_rank128'):
        cases.append((id_,{'id':id_},'none',None,'Warm natural delivery.','single'))
    cases.append(('base-after',{'id':None},'none',None,'Warm natural delivery.','single'))
    rows=[]
    for label,picks,voice_mode,voice,caption,mode in cases:
        if json.loads(get('/api/state'))['busy']:raise RuntimeError('Another user is active; stopping test')
        body={'message':f'CAPTION: {caption}\nTRANSCRIPT: "Ready when you are."','input_mode':'manual',
              'format':'caption','language':'en','generation':mode,'seconds':2.,'search_seconds':3.2,
              'search_n':2,'search_chunk_frames':20 if mode=='search_beam' else 10,'seed':777,
              'align':False,'voice_mode':voice_mode,'voice_profile':voice,'adapters':picks}
        start=time.perf_counter()
        request=urllib.request.Request(args.base+'/api/turn',json.dumps(body).encode(),{'Content-Type':'application/json'})
        with opener.open(request,timeout=600) as response:
            events=[json.loads(line[6:]) for line in response.read().decode().splitlines() if line.startswith('data: ')]
        errors=[x for x in events if x['event']=='error']
        if errors:raise RuntimeError(label+': '+str(errors))
        result=next(x for x in events if x['event']=='result')
        reference=result['reference']
        if voice_mode=='profile':
            assert reference['dynamic'] and reference['gid'].startswith(voice+'|') and reference['language']=='en'
            assert len(get(reference['url']))>1000
        else:assert reference['mode']=='none'
        encoded=get(result['candidates'][0]['url'])
        assert len(encoded)>1000
        (ROOT/'test-results'/f'dpo-live-{label}.mp3').write_bytes(encoded)
        row={'case':label,'mode':mode,'adapters':result['adapters'],'reference':reference,
             'sha256':hashlib.sha256(encoded).hexdigest(),'wall_seconds':time.perf_counter()-start,
             'wer_proxy':result['candidates'][0]['wer'],'search_performance':result.get('search_performance')}
        rows.append(row);print(json.dumps({'passed':label,'reference':reference}),flush=True)
    assert rows[0]['sha256']==rows[-1]['sha256'],'Base restoration changed deterministic audio'
    assert rows[2]['reference']['gid']!=rows[3]['reference']['gid'],'Emotion did not change selected Mira clip'
    report={'verification':'Real HTTP TTS and scorers; no emotion or speaker fidelity quality claim',
            'completed':len(rows),'base_restored_identically':True,'cases':rows}
    (ROOT/'test-results/dpo-live.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'completed':len(rows),'base_restored_identically':True}),flush=True)


if __name__=='__main__':main()
