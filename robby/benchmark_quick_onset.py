"""First-PCM diagnostic only; deliberately cancel each request after its first chunk.

This isolates onset, not handoff/audio quality. Run the full HTTPS check too.
"""
import asyncio,datetime,json,time
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parent
async def main():
 rows=[]
 async with httpx.AsyncClient(timeout=40) as c:
  for i in range(8):
   deadline=time.monotonic()+20
   while (await c.get('http://127.0.0.1:8792/api/state')).json().get('busy'):
    if time.monotonic()>deadline:raise RuntimeError('Shared TTS remained busy; no session was displaced')
    await asyncio.sleep(.1)
   start=time.perf_counter();meta={};metrics={}
   text=['Heute war ein wirklich anstrengender Tag. Erzähl mir etwas Aufmunterndes.',
         'Ich freue mich auf einen gemütlichen Abend mit einem guten Buch.'][i%2]
   async with c.stream('POST','http://127.0.0.1:8798/api/bolt/chat',json={'text':text,'llm_model':'qwen','llm_provider':'cerebras/fp16','quick_replies':True}) as r:
    r.raise_for_status()
    async for line in r.aiter_lines():
     if not line.startswith('data: '):continue
     e=json.loads(line[6:])
     if e.get('error'):raise RuntimeError(e['error'])
     if e.get('sentence_role'):meta=e
     if e.get('quick_reply_status'):metrics.update(e)
     if e.get('audio'):
      rows.append({'trial':i,'role':meta.get('sentence_role'),'reaction':meta.get('sentence_text'),
       'first_pcm_ms':round((time.perf_counter()-start)*1000,2),'metrics':metrics});print(json.dumps(rows[-1]),flush=True);break
   raw=json.dumps({'scope':__doc__,'rows':rows},indent=2,ensure_ascii=False)
   (ROOT/'static/quick-reaction-onset.json').write_text(raw)
   (ROOT/'logs/quick-reaction-onset-receipt.json').write_text(raw)
if __name__=='__main__':asyncio.run(main())
