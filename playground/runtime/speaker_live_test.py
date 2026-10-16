"""Real HTTP identity scoring, routing and buffered fades; not an efficacy study."""
import http.cookiejar
import json
from pathlib import Path
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8792'


def main():
    opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    def get(path):
        with opener.open(BASE+path,timeout=600) as response:return response.read()
    state=json.loads(get('/api/state'))
    assert state['ready'] and not state['busy'],state
    assert state['voice_default']=='emolia_c0542' and state['speaker_default']=={'method':'ecapa','weight':.6}
    cases=[('defaults-live-n12-1600',{}),
        ('voicenet-best3',{'generation':'best','n':3,'speaker':{'method':'voicenet','weight':.34}}),
        ('voicenet-best3-sad',{'generation':'best','n':3,'_caption':'Sad delivery, restrained grief.',
                             'speaker':{'method':'voicenet','weight':.34}}),
        ('ecapa-best3',{'generation':'best','n':3,'speaker':{'method':'ecapa','weight':.34}}),
        ('ecapa-top2-1600',{'generation':'search_beam','search_n':4,'search_chunk_frames':20,
                          'speaker':{'method':'ecapa','weight':.34}}),
        ('voicenet-live-1600',{'generation':'search_winner','search_n':4,'search_chunk_frames':20,
                              'speaker':{'method':'voicenet','weight':.34}}),
        ('pure-prompt',{'generation':'single','speaker':{'method':'voicenet','weight':0}}),
        ('speaker-heavy',{'generation':'best','n':2,'speaker':{'method':'voicenet','weight':.7}}),
        ('no-reference',{'generation':'single','voice_mode':'none'}),
        ('legacy-off',{'generation':'single','speaker':{'method':'off'}}),
        ('adaptive-plus-identity',{'generation':'best','n':2,'reward_mode':'adaptive_manual',
            'reward_plan':{'terms':[{'dimension':'vn:AROU','objective':'target','target':3}]}})]
    rows=[]
    for name,extra in cases:
        if json.loads(get('/api/state'))['busy']:raise RuntimeError('Another user is active; stopping test')
        caption=extra.pop('_caption','Warm natural delivery, softly amused conversation.')
        data={'message':f'CAPTION: {caption}\nTRANSCRIPT: "Ready when you are."',
              'input_mode':'manual','language':'en','seconds':3.2,'seed':777,**extra}
        request=urllib.request.Request(BASE+'/api/turn',json.dumps(data).encode(),{'Content-Type':'application/json'})
        tick=time.perf_counter()
        with opener.open(request,timeout=600) as response:
            events=[json.loads(line[6:]) for line in response.read().decode().splitlines() if line.startswith('data: ')]
        assert not [e for e in events if e['event']=='error'],(name,events)
        result=next(e for e in events if e['event']=='result')
        if name!='no-reference':
            assert result['reference']['voice']=='emolia_c0542' and result['reference']['dynamic']
        else:assert result['reference']['mode']=='none'
        if name=='defaults-live-n12-1600':
            assert result['search_performance']['n']==12 and result['search_performance']['max_seconds']==30
            assert result['search_performance']['chunk_ms']==1600
        identity=extra.get('speaker',{'method':'ecapa','weight':.6})
        for candidate in result['candidates']:
            assert candidate['alignment']['enabled']
            if identity['method']!='off':
                assert 'speaker' in candidate
                style=candidate.get('style_score',candidate['n_clap'])
                expected=(1-candidate['speaker_weight'])*style+candidate['speaker_weight']*(candidate['speaker_score'] or 0)
                assert abs(candidate['quality']-expected)<1e-5
                assert abs(candidate['reward']-expected*candidate['gate'])<1e-5
            encoded=get(candidate['url']);assert len(encoded)>1000
        chunks=[e for e in events if e['event']=='search_chunk']
        if result['search_performance'] is not None:
            assert chunks and result['search_performance']['live_alignment']['enabled']
            assert all(e['live_alignment']['enabled'] for e in chunks)
        rows.append({'case':name,'wall_seconds':time.perf_counter()-tick,'reference':result['reference'],
            'speaker_request':result['speaker_request'],'candidates':result['candidates'],
            'search_performance':result['search_performance'],'stream_chunks':len(chunks)})
        (ROOT/'test-results/speaker-live.json').write_text(json.dumps({'completed':len(rows),'cases':rows},indent=2))
        print(json.dumps({'passed':name,'speaker_available':[c.get('speaker',{}).get('available') for c in result['candidates']],
            'speaker_cosine':[c.get('speaker_similarity') for c in result['candidates']],
            'live_cutoff':result['search_performance']['live_alignment'] if result['search_performance'] else None}),flush=True)
    assert any(c.get('speaker',{}).get('available') for r in rows if r['case'].startswith('voicenet') for c in r['candidates'])
    assert any(c.get('speaker',{}).get('available') for r in rows if r['case'].startswith('ecapa') for c in r['candidates'])
    print(json.dumps({'completed':len(rows),'verification':'Actual TTS/ASR, 15 VoiceNet heads and ECAPA; functional smoke, not speaker-quality study'}),flush=True)


if __name__=='__main__':main()
