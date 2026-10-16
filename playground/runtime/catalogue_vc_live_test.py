"""HTTPS integration smoke with real post-conversion scoring and MP3 chunks."""
import json,time,subprocess
import httpx
from pathlib import Path

ROOT=Path(__file__).resolve().parent
URL='https://switch-surprised-clock-sparc.trycloudflare.com'


def main():
    cases=[('selma-on','single',True,'emolia_c0542',False),
        ('selma-off','single',False,'emolia_c0542',False),
        ('mira-on','single',True,'emolia_c1699',False),
        ('nora-on','single',True,'k10_age3_bg1',False),
        ('vc-single-live','single',True,'emolia_c0542',True),
        ('vc-single-fp32','single',True,'emolia_c0542',True),
        ('vc-best8','best',True,'emolia_c0542',True),
        ('vc-winner4','search_winner',True,'emolia_c0542',True),
        ('vc-beam4','search_beam',True,'emolia_c0542',True)]
    records=[]
    with httpx.Client(base_url=URL,timeout=240) as client:
        deadline=time.monotonic()+240
        while True:
            state=client.get('/api/state').json()
            if state['ready'] and not state['busy']:break
            if time.monotonic()>deadline:raise RuntimeError(('Server not idle/ready',state['phase']))
            time.sleep(1)
        for name,mode,steering,voice,vc in cases:
            state=client.get('/api/state').json();assert state['ready'] and not state['busy'],state['phase']
            body={'message':'CAPTION: Warm, natural conversation with clear phrasing.\nTRANSCRIPT: "I am glad you came back. We can talk about it now."',
                'input_mode':'manual','language':'en','format':'caption','generation':mode,'n':8 if mode=='best' else 1,
                'voice_mode':'profile','voice_profile':voice,'speaker_steering':steering,'voice_conversion':vc,
                'vc_precision':'fp32' if name=='vc-single-fp32' else 'bf16',
                'streaming':True,'chunk_frames':20,'search_n':4,'search_chunk_frames':20,'search_seconds':6.4,'seed':9217}
            events=[];chunks=[];started=time.perf_counter()
            with client.stream('POST','/api/turn',json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith('data: '):continue
                    event=json.loads(line[6:]);events.append(event)
                    if event['event'] in ('audio_chunk','search_chunk'):chunks.append({'seconds':time.perf_counter()-started,'event':event})
                    if event['event']=='error':raise RuntimeError((name,event))
            result=next(e for e in events if e['event']=='result')
            assert result['speaker_steering']['active']==steering
            assert result['voice_conversion']['enabled']==vc
            if vc:assert result['voice_conversion']['scoring_audio']=='converted output, never raw TTS'
            if vc:assert result['voice_conversion']['precision_mode']==body['vc_precision']
            if vc and mode in ('best','search_winner','search_beam'):
                assert result['voice_conversion']['max_microbatch_size']>1
            if mode in ('single','search_winner','search_beam'):assert chunks,(name,'no live MP3 chunks')
            for suffix,url in [('take',result['candidates'][result['winner']]['url'])]+([('chunk',chunks[0]['event']['url'])] if chunks else []):
                response=client.get(url);response.raise_for_status()
                path=ROOT/'test-results'/f'catalogue-{name}-{suffix}.mp3';path.write_bytes(response.content)
                info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)]))['streams'][0]
                assert info['codec_name']=='mp3' and info['channels']==1 and int(info['bit_rate'])==96000
            records.append({'case':name,'request':body,'chunks':chunks,'result':result})
            (ROOT/'test-results/catalogue-vc-live.json').write_text(json.dumps(records,indent=2))
            print(name,len(chunks),'chunks',result['total_ms'],'ms',flush=True)


if __name__=='__main__':main()
