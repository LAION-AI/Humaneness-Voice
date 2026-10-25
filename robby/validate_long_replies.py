"""Real local-LLM regressions: complete poems, stories, detail and follow-ups.

Only authored synthetic prompts. This is structural/task-completion evidence,
not a claim of perfect literary quality or universal instruction following.
"""
import asyncio,json,re,time
from pathlib import Path
import httpx

ROOT=Path(__file__).resolve().parent

def compact_receipt(report):
    """Keep actual words/cues without repeating fixed GENERAL templates."""
    result=json.loads(json.dumps(report))
    for row in result.get('rows',[]):
        row['parts']=[{k:p[k] for k in ('text','script','template_id','separator_before','language') if k in p} for p in row.get('parts',[])]
    return result

async def request(client,text,history=None,speaker=None):
    parts=[];metrics={};start=time.perf_counter()
    async with client.stream('POST','/api/bolt/llm',json={'text':text,'history':history or [],'speaker':speaker or {}}) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith('data: '):continue
            value=json.loads(line[6:])
            if value.get('error'):raise ValueError(value['error'])
            if 'manual_prompt' in value:parts.append(value)
            else:metrics=value
    spoken=''.join(('' if index==0 else p.get('separator_before',' '))+p['text'] for index,p in enumerate(parts))
    return {'input':text,'text':spoken,'parts':parts,'metrics':metrics,'total_ms':round((time.perf_counter()-start)*1000,2)}

async def main():
    report={'passed':False,'rows':[],'started':time.time(),'scope':'Real local model via FastAPI; synthetic prompts; no private conversation log'}
    target=ROOT/'static/long-replies-check.json'
    async with httpx.AsyncClient(base_url='http://127.0.0.1:8798',timeout=45) as client:
        for _ in range(60):
            try:
                if (await client.get('/api/bolt/state')).json()['ready']:break
            except httpx.HTTPError:pass
            await asyncio.sleep(1)
        cases=[('poem_de_'+str(i),'Schreib mir ein lustiges Gedicht mit acht Zeilen über einen kleinen Roboter.',8) for i in range(3)]
        cases += [('poem_default','Schreib mir ein fröhliches Gedicht über einen Roboter mit einer roten Mütze.',8),
                  ('poem_en','Write a twelve-line poem about a cat visiting the moon.',12),
                  ('story_de','Erzähl mir eine Geschichte mit fünf Sätzen über einen Drachen und einen verlorenen Hut.',5),
                  ('detail','Erkläre mir ausführlich in vier Schritten, wie ich mit Robby ein Ratespiel spielen kann.',None),
                  ('warm_chat','Ich hatte einen anstrengenden Tag. Lass uns darüber reden und überlege mit mir, wie wir den Abend schön machen.',None)]
        for name,text,count in cases:
            row={'input':text}
            try:
                row=await request(client,text,speaker={'needs_name':True,'display_name':'Speaker 1','named':False} if name.startswith('poem_de') else {})
                row['case']=name;lines=[line.strip() for line in row['text'].splitlines() if line.strip()]
                if count is not None:assert len(row['parts'])==count,(len(row['parts']),row['text'])
                if name.startswith('poem'):
                    assert len(lines)==count,(len(lines),row['text'])
                    assert not re.search(r'wie hei(?:ß|ss)t|your name|hier ist|here is|freue mich|happy to help',row['text'],re.I),row['text']
                if name=='story_de':assert len(re.findall(r'[.!?]+',row['text']))==5,row['text']
                if name=='detail':assert len(row['text'].split())>=65 and len(row['parts'])>=4,row['text']
                if name=='warm_chat':assert len(row['text'].split())>=25,row['text']
                assert row['metrics'].get('response_complete') is True,row['metrics']
                assert not re.search(r'GENERAL:|SCRIPT:|\([a-z_, ]+\)',row['text']),row['text']
                row['passed']=True
            except Exception as exc:
                row={**row,'case':name,'input':text,'passed':False,'error':str(exc)}
            report['rows'].append(row);target.write_text(json.dumps(report,indent=2,ensure_ascii=False))
            print(json.dumps({k:row.get(k) for k in ['case','passed','error','text']},ensure_ascii=False),flush=True)
        first=report['rows'][0]
        follow=await request(client,'Schreib noch vier zusätzliche Zeilen dazu und bleibe beim selben kleinen Roboter.',
            history=[{'role':'user','content':first['input']},{'role':'assistant','content':first.get('text','')}])
        follow['case']='poem_followup';follow['passed']=len(follow['parts'])==4 and follow['metrics'].get('response_complete') is True
        report['rows'].append(follow)
    report.update(passed=all(r['passed'] for r in report['rows']),finished=time.time())
    target.write_text(json.dumps(compact_receipt(report),indent=2,ensure_ascii=False));print('PASSED',report['passed'])
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':asyncio.run(main())
