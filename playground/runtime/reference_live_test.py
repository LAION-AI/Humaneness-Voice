"""Custom reference reuse, isolation and live N=12 German profile check."""
import http.cookiejar
import json
from pathlib import Path
import urllib.request
import uuid

ROOT=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8792'


def main():
    opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    def request(path,body=None,headers=None,method=None):
        req=urllib.request.Request(BASE+path,body,headers or {},method=method)
        with opener.open(req,timeout=600) as response:return response.read()
    assert not json.loads(request('/api/state'))['busy']
    boundary=uuid.uuid4().hex
    body=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="test-reference.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
          +(ROOT/'performance/benchmark-reference.wav').read_bytes()+f'\r\n--{boundary}--\r\n'.encode())
    upload=json.loads(request('/api/reference',body,{'Content-Type':f'multipart/form-data; boundary={boundary}'}))
    id_=upload['id'];rows=[]
    try:
        for label,choice in [('custom-first',{'voice_mode':'custom'}),('custom-reused',{'voice_mode':'custom'}),
                             ('no-voice',{'voice_mode':'none'}),('n12-german-mira',{'voice_mode':'profile','voice_profile':'emolia_c1699'})]:
            data={'message':'CAPTION: Sad delivery, restrained grief.\nTRANSCRIPT: "Ich habe nicht vergessen, was du für mich getan hast."',
                  'input_mode':'manual','language':'de','seconds':4.,'align':True,
                  'generation':'search_winner' if label=='n12-german-mira' else 'single',**choice}
            events=[json.loads(line[6:]) for line in request('/api/turn',json.dumps(data).encode(),
                {'Content-Type':'application/json'}).decode().splitlines() if line.startswith('data: ')]
            assert not [x for x in events if x['event']=='error'],events
            result=next(x for x in events if x['event']=='result')
            if choice['voice_mode']=='custom':assert result['reference']['url']=='/api/audio/'+id_
            if choice['voice_mode']=='none':assert result['reference']['mode']=='none'
            if choice['voice_mode']=='profile':
                assert result['reference']['gid'].startswith('emolia_c1699|') and result['reference']['language']=='de'
                assert result['search_performance']['n']==12
            assert result['adapters']['selection']['id']=='subset_pitch_quality_p2_rank016'
            assert len(request(result['candidates'][0]['url']))>1000
            rows.append({'case':label,'reference':result['reference'],'adapters':result['adapters'],
                         'search_performance':result['search_performance']})
            print(json.dumps({'passed':label}),flush=True)
        state=json.loads(request('/api/state'));assert state['references'][0]['expires_at'] is None
        try:urllib.request.urlopen(BASE+'/api/audio/'+id_)
        except urllib.error.HTTPError as exc:assert exc.code==404
        else:raise AssertionError('Reference accessible without owning cookie')
    finally:
        request('/api/reference/'+id_,method='DELETE')
    assert not json.loads(request('/api/state'))['references']
    (ROOT/'test-results/reference-live.json').write_text(json.dumps({'completed':len(rows),'cases':rows},indent=2))


if __name__=='__main__':main()
