"""Real live GPU/reference/adapter integration smoke, not a quality study.

Uses an isolated HTTP cookie jar and removes only its own temporary upload.
The source must be a genuinely long, single-speaker clip (no looping).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import http.cookiejar
import json
from pathlib import Path
import time
import urllib.request
import uuid
import settings

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',default='http://127.0.0.1:8792')
    parser.add_argument('--reference',required=True,type=Path)
    args=parser.parse_args()
    opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def request(path,data=None,headers=None,method=None):
        req=urllib.request.Request(args.base+path,data,headers or {},method=method)
        with opener.open(req,timeout=600) as response:return response.read()

    def state():return json.loads(request('/api/state'))
    initial=state()
    if not initial['ready'] or initial['busy']:raise RuntimeError('Demo busy or not ready; no test started')
    assert initial['reference_seconds_max']==15 and initial['adapters']['default']==settings.DEFAULT_ADAPTER
    boundary=uuid.uuid4().hex
    multipart=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="long-selma.mp3"\r\n'
               'Content-Type: audio/mpeg\r\n\r\n').encode()+args.reference.read_bytes()+f'\r\n--{boundary}--\r\n'.encode()
    uploaded=json.loads(request('/api/reference?slot=A',multipart,{'Content-Type':f'multipart/form-data; boundary={boundary}'}))
    assert uploaded['seconds']==15,uploaded
    reference_id=uploaded['id']
    cases=[('base-before',{},'single',20),
           ('dpo',{'id':'subset_pitch_quality_p2_rank016'},'single',20),
           ('ref15',{'ref15':True},'single',20),
           ('both',{'id':'subset_pitch_quality_p2_rank016','ref15':True},'single',20),
           ('base-after',{},'single',20),
           ('ref15-winner-n12-1600',{'ref15':True},'search_winner',20),
           ('both-top2-n4-800',{'id':'subset_pitch_quality_p2_rank016','ref15':True},'search_beam',10)]
    rows=[]
    try:
        for label,pick,mode,frames in cases:
            if state()['busy']:raise RuntimeError('Another session is active; aborting tests')
            body={'message':'CAPTION: Warm, natural conversational delivery, with a little curiosity.\n'
                             'TRANSCRIPT: "I kept the letter. Would you like to read it?"',
                  'input_mode':'manual','format':'caption','language':'en',
                  'voice_mode':'custom','reference_id':reference_id,'adapters':pick,
                  'generation':mode,'seconds':5.,'search_seconds':4.8,
                  'search_n':12 if mode=='search_winner' else 4,'search_chunk_frames':frames,
                  'align':True,'seed':777,'ranking':{'mode':'legacy'}}
            # First isolate TTS/fusion; search cases keep the real ECAPA 60% default.
            if mode=='single':body['speaker']={'method':'off','weight':0}
            tick=time.perf_counter()
            events=[]
            req=urllib.request.Request(args.base+'/api/turn',json.dumps(body).encode(),{'Content-Type':'application/json'})
            with opener.open(req,timeout=600) as response:
                for line in response:
                    if not line.startswith(b'data: '):continue
                    event=json.loads(line[6:]);events.append(event)
                    if event['event'] in ('adapters','reference_context','error'):
                        print(json.dumps({'case':label,**event}),flush=True)
            errors=[event for event in events if event['event']=='error']
            if errors:raise RuntimeError(f'{label}: {errors}')
            result=next(event for event in events if event['event']=='result')
            context=result['reference']['tts_context']
            assert abs(context['used_seconds']-15)<.01 and context['codec_frames']>37,context
            assert context['max_seconds']==15 and context['codec_frames']<=188,context
            active=result['adapters']['active']
            assert len(active)==int(bool(pick.get('id')))+int(bool(pick.get('ref15'))),active
            encoded=request(result['candidates'][0]['url'])
            assert len(encoded)>1000
            target=ROOT/'test-results'/f'ref15-live-{label}.mp3'
            target.write_bytes(encoded)
            row={'case':label,'mode':mode,'adapters':result['adapters'],
                 'context':context,'sha256':hashlib.sha256(encoded).hexdigest(),
                 'wall_seconds':round(time.perf_counter()-tick,3),
                 'wer_proxy':result['candidates'][0]['wer'],
                 'alignment':result['candidates'][0].get('alignment'),
                 'search_performance':result.get('search_performance'),
                 'speaker':result.get('speaker_request'),
                 'live_chunks':sum(e['event']=='search_chunk' for e in events)}
            rows.append(row)
            print(json.dumps({'passed':label,'seconds':context['used_seconds'],'frames':context['codec_frames'],
                              'wall_seconds':row['wall_seconds']}),flush=True)
        assert rows[0]['sha256']==rows[4]['sha256'],'Base restoration changed deterministic audio'
        report={'timestamp':datetime.now(timezone.utc).isoformat(),'base':args.base,
                'verification':'Actual GPU TTS, live HTTP, 15-second codec context, all four adapter settings, '
                               'ECAPA search and alignment. Integration smoke only; no efficacy claim.',
                'completed':len(rows),'base_restored_identically':True,
                'default_state':{key:initial[key] for key in ('reference_seconds_max','adapters','speaker_default',
                    'voice_default','guidance_default','temperature_default','search_n_default','search_chunk_frames_default')},
                'cases':rows}
        (ROOT/'test-results/ref15-live.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'completed':len(rows),'base_restored_identically':True}),flush=True)
    finally:
        request('/api/reference/'+reference_id,method='DELETE')


if __name__=='__main__':main()
