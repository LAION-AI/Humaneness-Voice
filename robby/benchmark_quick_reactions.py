"""Real local E4B / API Qwen races and an optional longer-transition ablation."""
import asyncio,datetime,json,time
from pathlib import Path
import httpx
from quick_reaction import generate,validate,word_count
from bolt_director import BoltDirector,local_config

ROOT=Path(__file__).resolve().parent
CASES=[
 ('tired','Heute war ein wirklich anstrengender Tag.',[]),
 ('meaning','Was ist der Sinn des Lebens?',[]),
 ('sky','Warum ist der Himmel blau?',[]),
 ('joke','Erzähl mir einen vollständigen kurzen Witz über Katzen.',[]),
 ('followup','Noch einen bitte.',[{'role':'user','content':'Erzähl einen Witz über Katzen.'},{'role':'assistant','content':'Eine Katze bestellt einen Kaffee. Der Kellner fragt: Mit Milch? Die Katze: Nur Milch, bitte.'}]),
 ('correction','Ich bin nicht traurig, nur etwas müde.',[]),
 ('greeting','Wie geht es dir heute?',[]),
 ('explain','Erklär mir einfach, wie ein Regenbogen entsteht.',[]),
]
async def main():
    rows=[];base,model=local_config()
    async with httpx.AsyncClient(base_url='http://127.0.0.1:8792',timeout=45) as c:
        for name,text,history in CASES:
            for repeat in range(2):
                start=time.perf_counter();metric={}
                async def answer():
                    parts=[];first=None
                    async for part in BoltDirector().stream(c,text,{},history,metrics=metric,backend={'model':'qwen','provider':'cerebras/fp16'}):
                        if first is None:first=round((time.perf_counter()-start)*1000,2)
                        parts.append(part.text)
                    return {'first_sentence_ms':first,'sentences':parts,'metrics':metric}
                reaction,main=await asyncio.gather(generate(c,text,{},history),answer())
                row={'case':name,'repeat':repeat,'input':text,'reaction':reaction,'answer':main,
                    'local_ready_first':bool(reaction.get('text') and main['first_sentence_ms'] and reaction['ms']<main['first_sentence_ms'])}
                if repeat==0 and name in ('tired','meaning','joke'):
                    start=time.perf_counter()
                    prompt={'reaction':reaction.get('text'),'main_answer_first_sentences':main['sentences'][:2],'user':text}
                    response=await c.post(base+'/v1/chat/completions',json={'model':model,
                        'messages':[{'role':'system','content':'Write a natural German transition of exactly three short sentences connecting the given first reaction to the following answer. Do not repeat the answer or add new facts. Only spoken text, no labels.'},
                        {'role':'user','content':json.dumps(prompt,ensure_ascii=False)}],
                        'max_tokens':180,'temperature':.35,'chat_template_kwargs':{'enable_thinking':False,'conversational_mode':False}},timeout=15)
                    response.raise_for_status();out=response.json()['choices'][0]['message']['content']
                    row['longer_transition']={'text':out,'ms':round((time.perf_counter()-start)*1000,2),'words':word_count(out)}
                rows.append(row);print(json.dumps({'case':name,'repeat':repeat,'reaction':reaction.get('text'),'local_ms':reaction['ms'],'main_first_ms':main['first_sentence_ms'],'local_ready_first':row['local_ready_first']}),flush=True)
    report={'started_on':datetime.datetime.now(datetime.timezone.utc).isoformat(),'local_model':model,
        'scope':'16 real generation races, three three-sentence transition probes; no TTS in this benchmark.',
        'rows':rows}
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    raw=json.dumps(report,indent=2,ensure_ascii=False);(ROOT/'logs'/('quick-reaction-models-'+stamp+'.json')).write_text(raw)
    (ROOT/'static/quick-reaction-models.json').write_text(raw)
if __name__=='__main__':asyncio.run(main())
