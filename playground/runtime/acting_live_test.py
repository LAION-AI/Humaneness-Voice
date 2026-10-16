"""Actual HTTPS checkpoint-switch, streaming, dynamic-reference and batch canary."""
import json
from pathlib import Path
import time

import httpx

ROOT=Path(__file__).resolve().parent
URL='https://switch-surprised-clock-sparc.trycloudflare.com'


def main():
    records=[]
    with httpx.Client(base_url=URL,timeout=240) as client:
        deadline=time.monotonic()+240
        while True:
            try:
                response=client.get('/api/state');response.raise_for_status();state=response.json()
            except (httpx.HTTPError,ValueError):
                if time.monotonic()>deadline:raise RuntimeError('Startup unavailable')
                time.sleep(2);continue
            if state.get('error'):raise RuntimeError(state['error'])
            if state['ready'] and not state['busy']:break
            if time.monotonic()>deadline:raise RuntimeError('Startup timeout')
            time.sleep(2)
        assert state['model_id']=='large_acting' and state['checkpoint']['step']==432
        assert state['voice_default']=='emolia_c1699' and not state['speaker_steering_default']
        for label,extra in [
            ('large-default',{}),
            ('large-batch2',{'generation':'best','n':2,'batch_size':2,'streaming':False,'voice_conversion':False}),
            ('large-batch4',{'generation':'best','n':4,'batch_size':4,'streaming':False,'voice_conversion':False}),
            ('large-dynamic',{'dynamic_reference':True,'voice_conversion':False}),
            ('small-stream',{'model':'small_acting','voice_conversion':False}),
            ('small-batch8',{'model':'small_acting','generation':'best','n':8,'batch_size':8,'streaming':False,'voice_conversion':False}),
            ('large-restored',{'model':'large_acting','voice_conversion':False})]:
            body={'message':'GENERAL: Strongly relieved, warm and natural, breathing easier.\nSCRIPT:\n(clearly relieved, gently letting it out) [3.2 seconds duration] I am glad you came back. We can talk now.',
                  'input_mode':'manual','language':'en','seed':43217,**extra}
            started=time.monotonic();events=[];chunks=[]
            with client.stream('POST','/api/turn',json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith('data: '):continue
                    event=json.loads(line[6:]);events.append(event)
                    if event['event']=='error':raise RuntimeError((label,event))
                    if event['event']=='audio_chunk':chunks.append({'seconds':time.monotonic()-started,'event':event})
            result=next(e for e in events if e['event']=='result')
            model=next(e for e in events if e['event']=='model')
            assert model['checkpoint']['step']==(1725 if extra.get('model')=='small_acting' else 432)
            assert not result['speaker_steering']['active'] and not result['adapters']['active']
            if extra.get('streaming',True):assert chunks and events.index(chunks[0]['event'])<events.index(result)
            if label=='large-dynamic':
                ref=next(e for e in events if e['event']=='reference')
                assert ref['voice']=='emolia_c1699' and ref['dynamic'] and ref['policy']=='dynamic',ref
            for candidate in result['candidates']:
                audio=client.get(candidate['url']);audio.raise_for_status()
                (ROOT/'test-results'/f"acting-{label}-{candidate['index']}.mp3").write_bytes(audio.content)
            records.append({'case':label,'request':body,'model':model,'chunks':chunks,'result':result,'wall_seconds':time.monotonic()-started})
            (ROOT/'test-results/acting-live.json').write_text(json.dumps(records,indent=2))
            print(label,'chunks',len(chunks),'wall',round(time.monotonic()-started,2),'performance',result.get('performance'),flush=True)
        client.post('/api/models/select',json={'model':'large_acting'}).raise_for_status()


if __name__=='__main__':main()
