"""Repeatable live canaries: compile, GPU alignment, full VC and model swaps.

Timings separate TTS+codec CUDA events from API wall time, scoring and VC.
Does not touch the research worker, infer human quality, or use remote tokens.
"""
import json
from pathlib import Path
import time

import httpx

ROOT=Path(__file__).resolve().parent


def main():
    baseline=json.loads((ROOT/'test-results/large-speed-before.json').read_text())[0]['request']
    records=[]
    with httpx.Client(base_url='http://127.0.0.1:8792',timeout=300) as client:
        deadline=time.monotonic()+600
        while True:
            state=client.get('/api/state').json()
            if state['error']:raise RuntimeError(state['error'])
            if state['ready'] and not state['busy']:break
            if time.monotonic()>deadline:raise RuntimeError('Startup timeout')
            time.sleep(2)
        assert not state['voice_conversion']['default']
        assert state['voice_conversion']['model_repo']=='laion/Humaneness-Voice-Converter'
        cases=[(f'large-gpu-align-{i}',{}) for i in range(3)]+[
            ('large-batch4',{'streaming':False,'generation':'best','n':4,'batch_size':4}),
            ('vc-stream',{'voice_conversion':True}),
            ('vc-offline-batch2',{'voice_conversion':True,'streaming':False,'generation':'best','n':2,'batch_size':2}),
            ('vc-dynamic',{'voice_conversion':True,'dynamic_reference':True}),
            ('small-compiled',{'model':'small_acting'}),
            ('large-restored',{}),
            ('luna-default',{'input_mode':'chat','message':'Say something reassuring to me.','language':'en'}),
            ('luna-intense',{'input_mode':'chat','message':'Perform exactly these words with intensely angry, shouting delivery: You knew what this meant to me, and you did it anyway.','language':'en'})]
        for label,extra in cases:
            body={**baseline,**extra}
            started=time.monotonic();events=[];chunks=[]
            with client.stream('POST','/api/turn',json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith('data: '):continue
                    event=json.loads(line[6:]);events.append(event)
                    if event['event']=='error':raise RuntimeError((label,event))
                    if event['event']=='audio_chunk':chunks.append(time.monotonic()-started)
            result=next(e for e in events if e['event']=='result')
            if body['streaming']:assert chunks
            if body['voice_conversion']:
                assert result['voice_conversion']['model_repo']=='laion/Humaneness-Voice-Converter'
                assert result['voice_conversion']['step']==40000
                assert result['voice_conversion']['scoring_audio']=='converted output, never raw TTS'
            if label=='vc-dynamic':
                ref=next(e for e in events if e['event']=='reference')
                assert ref['voice']=='emolia_c1699' and ref['dynamic']
            records.append({'case':label,'request':body,'events':events,'result':result,
                'wall':time.monotonic()-started,'first_mp3':chunks[0] if chunks else None})
            (ROOT/'test-results/acting-speed-final.json').write_text(json.dumps(records,indent=2))
            print(label,'wall',round(records[-1]['wall'],3),'alignment_ms',result['alignment_ms'],
                'rtf',result['performance']['aggregate_rtf'],'chunks',len(chunks),flush=True)
        assert client.get('/api/state').json()['model_id']=='large_acting'


if __name__=='__main__':main()
