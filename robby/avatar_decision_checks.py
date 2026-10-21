"""Configured decision API: silent listening actions, never spoken interjections."""
import asyncio,json,time
from pathlib import Path
from live_decisions import DecisionClient

async def main():
    results=[];client=DecisionClient()
    for text,gesture in [('Ich erzähle noch weiter. Bitte nicke jetzt nur kurz mit dem Kopf, ohne mich sprachlich zu unterbrechen.','nod'),
                         ('Bitte schüttle kurz den Kopf, ohne etwas zu sagen. Ich rede danach einfach weiter.','shake'),
                         ('Bitte bewege deinen Kopf jetzt nicht und höre einfach still zu.',None)]:
        case={'prompt':text,'expected':gesture,'passed':False};start=time.perf_counter()
        try:
            result=await client.choose({'agent_speaking':False,'user_partial':text,'history':[],
                'rules':[],'agent_full_reply':'','agent_played_seconds':0,'proactive_enabled':True})
            case.update(result=result,seconds=time.perf_counter()-start)
            assert result['action']=='listen' and result.get('gesture')==gesture
            case['passed']=True
        except Exception as exc:case['error']=str(exc)
        results.append(case);print(json.dumps(case,ensure_ascii=False),flush=True)
    report={'passed':all(c['passed'] for c in results),'synthetic_text_only':True,'cases':results}
    (Path(__file__).parent/'logs/avatar-decision-checks.json').write_text(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':asyncio.run(main())
