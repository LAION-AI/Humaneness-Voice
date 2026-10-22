"""Actual configured LLM, authored text only. Never records user voices."""
import asyncio,json,time
from pathlib import Path
import httpx
from conversation_director import CompactDirector

async def main():
    cases=[
      ('stroke_de','', [{'role':'user','content':'Hallo Robbie, wir reden Deutsch.'}],{'event':'petted','language':'de','recent_events':[{'event':'petted','duration_ms':1400,'age_s':0}]}),
      ('stroke_en','', [{'role':'user','content':'Hello, little robot!'}],{'event':'petted','language':'en'}),
      ('optional_invite','', [{'role':'user','content':'Hallo, du darfst ruhig ein bisschen spielen.'}],{'event':'ask_pet','language':'de'}),
      ('respect_quiet','',[{'role':'user','content':'Bitte sei jetzt still und bitte mich nicht mehr ums Streicheln.'}],{'event':'ask_pet','language':'de'}),
      ('explain_play','Was machst du denn da?',[],{'recent_events':[{'event':'idle_play','variant':'fish','carousel':True,'age_s':8}]}),
      ('honest_touch','Spürst du das Streicheln wirklich körperlich?',[],{'recent_events':[{'event':'petted','duration_ms':1200,'age_s':5}]})]
    report={'passed':False,'actual_llm':True,'synthetic_only':True,'cases':[]}
    async with httpx.AsyncClient() as client:
        for name,text,history,context in cases:
            start=time.perf_counter()
            r=await CompactDirector().generate(client,text,{},history,avatar_context=context)
            item={'case':name,'silent':r.get('silent',False),'text':r.get('text'),'language':r.get('language'),'seconds':round(time.perf_counter()-start,2)}
            if name=='respect_quiet':assert item['silent'],item
            if name.startswith('stroke') and not item['silent']:
                assert len(item['text'].split())<=25,item
                assert item['language']==('de' if name.endswith('de') else 'en'),item
            report['cases'].append(item);print(json.dumps(item,ensure_ascii=False),flush=True)
    report.update(passed=True,finished=time.time())
    (Path(__file__).parent/'logs/personality-director-checks.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':asyncio.run(main())
