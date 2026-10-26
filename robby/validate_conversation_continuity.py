"""Real multi-turn joke/task regressions; no canned answers or hidden retries.

Structural/topic checks are deliberately distinct from human humor judgments.
Full synthetic dialogues are retained so a reviewer can inspect context use.
"""
import argparse, asyncio, json, re, time
from pathlib import Path
import httpx

SCENARIO=[
    ('first_joke','Erzähl mir einen vollständigen Witz über einen kleinen Roboter, mit Situation und Pointe. Erzähl direkt den Witz.',18,'robot'),
    ('another','Noch einen, bitte.',18,'robot'),
    ('longer','Den letzten Witz bitte ausführlicher, mit einem kleinen Dialog und einer richtigen Pointe. Ungefähr sechs Sätze.',45,'robot'),
    ('same_character_new_place','Noch einen über denselben Roboter, diesmal am Bahnhof. Erzähl direkt den Witz.',20,'station'),
    ('change_topic','Jetzt einen Witz über eine Katze, ohne Roboter. Erzähl direkt den Witz.',18,'cat'),
    ('elliptical_new_topic','Und noch einen, bitte.',18,'cat'),
]
TOPICS={'robot':r'robot|blech|schaltkreis|prozessor|akku|robby|batterie',
        'station':r'bahnhof|zug|bahnsteig|schaffner|fahrkarte|gleis',
        'cat':r'katz|kater|miau|schnurr|pfot|miez'}

async def ask(client,url,text,history,model,provider):
    parts=[];metrics={};started=time.perf_counter()
    body={'text':text,'history':history,'llm_model':model,'llm_provider':provider}
    async with client.stream('POST',url.rstrip('/')+'/api/bolt/llm',json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith('data: '):continue
            value=json.loads(line[6:])
            if value.get('error'):raise RuntimeError(str(value['error']))
            if 'manual_prompt' in value:parts.append(value)
            else:metrics=value
    text=''.join(('' if i==0 else p.get('separator_before',' '))+p['text'] for i,p in enumerate(parts))
    return text,parts,metrics,round((time.perf_counter()-started)*1000,2)

async def main(a):
    result={'passed':False,'scope':'Real LLM SSE with actual preceding user/assistant turns; automated structure/topic checks plus inspectable words. Humor quality needs human review.',
        'model':a.model,'provider':a.provider,'repeats':a.repeats,'attempts_per_turn':1,'started':time.time(),'conversations':[]}
    output=Path(a.out);output.parent.mkdir(parents=True,exist_ok=True)
    async with httpx.AsyncClient(timeout=90) as client:
        for repeat in range(a.repeats):
            history=[];dialog={'repeat':repeat,'turns':[]};result['conversations'].append(dialog)
            previous_jokes=[]
            for name,prompt,min_words,topic in SCENARIO:
                row={'case':name,'input':prompt,'history_turns_sent':len(history),'passed':False}
                try:
                    text,parts,metrics,ms=await ask(client,a.url,prompt,history,a.model,a.provider)
                    normalized=re.sub(r'\W+',' ',text.casefold()).strip();count=len(text.split())
                    checks={'minimum_task_content':count>=min_words,'topic_retained':bool(re.search(TOPICS[topic],text,re.I)),
                        'new_words_not_identical_repeat':normalized not in previous_jokes,
                        'finished_stream':metrics.get('response_complete') is True,
                        'not_just_an_offer':not bool(re.fullmatch(r'(?:klar|gerne|natürlich|okay)[,.! ]*(?:ich .*|hier .*|möchtest .*|was .*|soll .*|.*\?)',text.strip(),re.I))}
                    # A retelling deliberately reuses the preceding plot.
                    if name=='longer':checks['new_words_not_identical_repeat']=True
                    row.update(text=text,word_count=count,sentences=len(parts),checks=checks,
                        metrics=metrics,http_ms=ms,passed=all(checks.values()))
                    history.extend([{'role':'user','content':prompt},{'role':'assistant','content':text}])
                    previous_jokes.append(normalized)
                except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)[:400]
                dialog['turns'].append(row);output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
                print(json.dumps({k:row.get(k) for k in ('case','passed','word_count','sentences','text','error')},ensure_ascii=False),flush=True)
    result.update(finished=time.time(),passed=all(t['passed'] for c in result['conversations'] for t in c['turns']))
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://127.0.0.1:8798')
    p.add_argument('--model',choices=['qwen','local'],default='qwen');p.add_argument('--provider',default='cerebras/fp16')
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--out',default='static/conversation-continuity-check.json')
    a=p.parse_args()
    if not 1<=a.repeats<=20:p.error('Use 1–20 independent conversations')
    if a.model=='local':a.provider='local'
    asyncio.run(main(a))
