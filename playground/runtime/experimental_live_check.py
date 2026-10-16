"""Actual HTTPS Alpha load/stream/batch canary; restores the stable default."""
import json
from pathlib import Path
import time

import httpx

ROOT=Path(__file__).resolve().parent


def main():
    url=(ROOT/'logs/tunnel-url.txt').read_text().strip();records=[]
    with httpx.Client(base_url=url,timeout=300) as client:
        def ready(model=None):
            deadline=time.monotonic()+600
            while True:
                state=client.get('/api/state').json()
                if state['error']:raise RuntimeError(state['error'])
                cache=state.get('checkpoint_ram',{}).get('models',{})
                if state['ready'] and not state['busy'] and (model is None or cache.get(model,{}).get('status')=='ready'):
                    return state
                if time.monotonic()>deadline:raise RuntimeError('Readiness timeout')
                time.sleep(2)
        ready()
        cases=[('small_acting','best',3),('small_alpha','single',1),('small_alpha','best',3),
               ('small_alpha','best',8),('large_alpha','single',1),('large_alpha','best',3),
               ('large_acting','best',3),('small_acting','single',1)]
        for model,mode,n in cases:
            if model.endswith('alpha'):ready(model)
            else:ready()
            body={'model':model,'message':'GENERAL: Gentle, natural and spontaneous delivery.\nSCRIPT:\n(gently relieved, natural and spontaneous) [3.5 seconds duration] I am glad you came back. We can talk now.',
                'input_mode':'manual','language':'en','generation':mode,'n':n,'batch_size':1,
                'streaming':mode=='single','seed':43217,'voice_conversion':False}
            started=time.monotonic();events=[]
            with client.stream('POST','/api/turn',json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith('data: '):continue
                    event=json.loads(line[6:]);events.append(event)
                    if event['event']=='error':raise RuntimeError(event)
            result=next(e for e in events if e['event']=='result')
            selected=next(e for e in events if e['event']=='model')
            assert selected['model_id']==model and len(result['candidates'])==n
            assert not result['adapters']['active'] and not result['speaker_steering']['active']
            if mode=='best':
                assert selected['batch_size']==n and selected['requested_batch_size']==1
                assert selected['parallel_candidates']
                assert result['performance']['batches'][0]['batch_size']==n
                assert not result['performance']['oom_batch_retries']
            else:assert any(e['event']=='audio_chunk' for e in events)
            if model.endswith('alpha'):assert selected['checkpoint']['experimental']
            for candidate in result['candidates']:
                audio=client.get(candidate['url']);audio.raise_for_status()
                assert audio.headers['content-type'].startswith('audio/mpeg')
            records.append({'model':model,'mode':mode,'n':n,'wall_seconds':time.monotonic()-started,
                'request':body,'events':events})
            (ROOT/'test-results/experimental-live.json').write_text(json.dumps({'url':url,'cases':records},indent=2))
            print(model,mode,n,'GPU batches',[b['batch_size'] for b in result['performance']['batches']],
                'wall',round(records[-1]['wall_seconds'],2),flush=True)
        state=ready()
        assert state['model_default']=='small_acting' and state['model_id']=='small_acting'
        cache=state['checkpoint_ram']['models']
        assert all(cache[id]['status']=='ready' and cache[id]['device']=='cpu' for id in ['small_alpha','large_alpha'])
        print('Both Alpha CPU caches ready; stable default restored:',url,flush=True)


if __name__=='__main__':main()
