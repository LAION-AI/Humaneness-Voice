"""Real configured LLM/tool calls with authored prompts; no microphone data."""
import asyncio,json,time
from pathlib import Path
import httpx
from conversation_director import CompactDirector

async def main():
    cases=[('Drehe deinen Kopf bitte einmal um 360 Grad gegen den Uhrzeigersinn.',['spin'],{'direction':'counterclockwise'}),
           ('Please rotate your head once clockwise by 360 degrees.',['spin'],{'direction':'clockwise'}),
           ('Kannst du zuerst wütend gucken und danach eine süße Grimasse schneiden?',['emotion','grimace'],{'emotion':'anger'}),
           ('Bitte rolle mit den Augen und nicke danach.',['eye_roll','nod'],{}),
           ('Schau bitte bitter und schüttle dann den Kopf.',['emotion','shake'],{'emotion':'bitterness'}),
           ('Please show a cute crossed-eye face.',['grimace'],{'variant':'crossed'}),
           ('Please do not spin your head. Just say hello.',[],{}),
           ('What is seventeen times four? Answer only the number.',[],{})]
    report={'passed':False,'synthetic_text_only':True,'cases':[]}
    async with httpx.AsyncClient() as client:
        for prompt,kinds,fields in cases:
            start=time.perf_counter();case={'prompt':prompt,'passed':False}
            try:
                r=await CompactDirector().generate(client,prompt,{},[])
                steps=r['avatar_actions'];case.update(steps=steps,text=r['text'],model=r['director_model'],seconds=time.perf_counter()-start)
                assert [s['kind'] for s in steps]==kinds,'Unexpected action sequence'
                assert all(steps[0].get(k)==v for k,v in fields.items()),'Wrong action parameter'
                if 'seventeen' in prompt:assert r['text'].strip(' .')=='68'
                case['passed']=True
            except Exception as exc:case['error']=str(exc)
            report['cases'].append(case);print(json.dumps(case,ensure_ascii=False),flush=True)
    report.update(passed=all(c['passed'] for c in report['cases']),finished=time.time())
    (Path(__file__).parent/'logs/avatar-director-checks.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':asyncio.run(main())
