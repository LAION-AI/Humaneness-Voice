"""Real HTTPS RL bundle, streaming and parallel Best-of-N canary."""
import json
from pathlib import Path
import time

import httpx

ROOT=Path(__file__).resolve().parent


def main():
    url=(ROOT/'logs/tunnel-url.txt').read_text().strip()
    records=[]
    with httpx.Client(base_url=url,timeout=900) as client:
        def ready(model=None):
            deadline=time.monotonic()+900
            while True:
                state=client.get('/api/state').json()
                if state['error']:raise RuntimeError(state['error'])
                row=state.get('checkpoint_ram',{}).get('models',{}).get(model,{})
                if state['ready'] and not state['busy'] and (model is None or row.get('status')=='ready'):return state
                if row.get('status')=='error':raise RuntimeError(row['error'])
                if time.monotonic()>deadline:raise RuntimeError('Model readiness timeout')
                time.sleep(3)
        for model,mode,n in [('small_rl','single',1),('small_rl','best',3),
                             ('large_rl','single',1),('large_rl','best',3)]:
            ready(model)
            body={'model':model,'input_mode':'manual','language':'en',
                'message':'GENERAL: Gentle, natural and spontaneous delivery.\nSCRIPT:\n(warmly relieved) [3.5 seconds duration] I am glad you came back. We can talk now.',
                'generation':mode,'n':n,'batch_size':1,'streaming':mode=='single',
                'seed':43217,'voice_conversion':False}
            tick=time.monotonic();events=[]
            with client.stream('POST','/api/turn',json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith('data: '):
                        event=json.loads(line[6:]);events.append(event)
                        if event['event']=='error':raise RuntimeError(event)
            result=next(e for e in events if e['event']=='result')
            selected=next(e for e in events if e['event']=='model')
            checkpoint=selected['checkpoint']
            assert selected['model_id']==model and checkpoint['policy_dtype']=='fp32'
            assert checkpoint['rl_adapter']['enabled'] and not checkpoint['rl_adapter']['merged']
            assert len(result['candidates'])==n and not result['adapters']['active']
            batches=[b['batch_size'] for b in result['performance']['batches']]
            if mode=='single':assert any(e['event']=='audio_chunk' for e in events)
            else:
                assert selected['parallel_candidates']
                assert batches==([3] if model=='small_rl' else [2,1])
            for candidate in result['candidates']:
                audio=client.get(candidate['url']);audio.raise_for_status()
                assert audio.headers['content-type'].startswith('audio/mpeg') and len(audio.content)>1000
                assert candidate['hyp'].strip() and candidate['ears']['available']
                assert candidate['alignment']['available'] and candidate['alignment']['applied']
            records.append({'model':model,'mode':mode,'n':n,'wall_seconds':time.monotonic()-tick,
                'request':body,'events':events})
            (ROOT/'test-results/rl-live.json').write_text(json.dumps({'url':url,'cases':records},indent=2))
            print(model,mode,'actual batches',batches,'wall seconds',round(records[-1]['wall_seconds'],2),flush=True)
        ready()
        client.post('/api/models/select',json={'model':'small_rl'}).raise_for_status()
        state=ready()
        assert state['model_id']==state['model_default']=='small_rl'
        print('Small RL default restored:',url,flush=True)


if __name__=='__main__':main()
