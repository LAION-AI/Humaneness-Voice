"""Small authored German ASR smoke test, swaps, and actual reply languages.

Not a German ASR benchmark: synthetic speech, three prompts, two repeats.
Live audio/browser tests must run serially (one native ASR session slot).
"""
import asyncio,json,re,time
from pathlib import Path
import httpx,numpy as np
from scipy.signal import resample_poly
from asr_catalog import MODELS
from benchmark_bolt import measure

ROOT=Path(__file__).resolve().parent

def wer(reference,hypothesis):
    a=re.findall(r'\w+',reference.casefold());b=re.findall(r'\w+',hypothesis.casefold());d=list(range(len(b)+1))
    for i,x in enumerate(a):
        old=d;d=[i+1]+[0]*len(b)
        for j,y in enumerate(b):d[j+1]=min(d[j]+1,old[j+1]+1,old[j]+(x!=y))
    return d[-1]/max(1,len(a))

async def main():
    folder=ROOT/'logs/asr-models';folder.mkdir(exist_ok=True)
    report={'passed':False,'synthetic_speech':True,'physical_microphone':False,'rows':[],'switches':[],'languages':[],'started':time.time()}
    texts=['Hallo Robby, kannst du bitte eine lustige Grimasse machen?',
        'Heute möchte ich über Äpfel sprechen. Bitte antworte mir ganz ruhig.',
        'Mein Zug fährt um halb sieben nach München.']
    async with httpx.AsyncClient(timeout=90) as client:
        fixtures=[]
        for i,text in enumerate(texts):
            p=folder/f'german-{i}.npy'
            if p.exists():wave=np.load(p)
            else:
                _,audio=await measure(client,'http://127.0.0.1:8798/api/bolt/tts',{'text':text,'cue':'interest, natural','voice':'emolia_c1699'})
                wave=resample_poly(audio,2,3).astype(np.float32);np.save(p,wave)
            fixtures.append(wave)
        for key in MODELS:
            switch=await client.post('http://127.0.0.1:8801/activate/'+key);switch.raise_for_status()
            report['switches'].append(switch.json())
            for repeat in range(2):
                for i,(text,wave) in enumerate(zip(texts,fixtures)):
                    start=time.perf_counter();r=await client.post('http://127.0.0.1:8798/api/bolt/asr?model='+key,content=wave.astype('<f4').tobytes());r.raise_for_status();d=r.json()
                    assert d['model']==key and d['text'].strip(),d
                    if i==0:assert 'machen' in d['text'].casefold(),'Final word was lost: '+str(d)
                    row={'model':key,'case':i,'repeat':repeat,'reference':text,'transcript':d['text'],'audio_s':round(len(wave)/16000,3),
                        'http_ms':round((time.perf_counter()-start)*1000,2),'asr_ms':d['asr_ms'],'wer':round(wer(text,d['text']),4),'passed':True}
                    report['rows'].append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
        for text,expected in [('Tell me what we could play together.','de'),('Bitte schlage mir ein Spiel vor.','de'),('Please answer in English: suggest a game.','en')]:
            response=await client.post('http://127.0.0.1:8798/api/bolt/llm',json={'text':text});response.raise_for_status()
            parts=[]
            for line in response.text.splitlines():
                if line.startswith('data: '):
                    data=json.loads(line[6:])
                    if 'manual_prompt' in data:parts.append(data)
                    if data.get('error'):raise AssertionError(data)
            assert parts and all(p['language']==expected for p in parts),parts
            report['languages'].append({'input':text,'expected':expected,'reply':' '.join(p['text'] for p in parts),'passed':True})
        r=await client.post('http://127.0.0.1:8801/activate/nemotron');r.raise_for_status()
        state=(await client.get('http://127.0.0.1:8801/models')).json()
        assert state['active_gpu_model'] is None and all(m['ready'] for m in state['models']),state
        report['all_cpu_cached_after_offload']=state
    report.update(passed=True,finished=time.time())
    (ROOT/'static/asr-model-check.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print('PASSED',report['passed'],flush=True)

if __name__=='__main__':asyncio.run(main())
