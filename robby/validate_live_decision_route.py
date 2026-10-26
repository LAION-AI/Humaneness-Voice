"""Six real CPU Laya requests, including negations. No microphone or cloud LLM."""
import argparse, asyncio, json, os, time
from pathlib import Path
import httpx
from live_decisions import DecisionClient

CASES = [
    ('Stopp!', 'stop_reply'), ('Hör bitte auf.', 'stop_reply'),
    ('Warte bitte, ich möchte etwas fragen.', 'stop_reply'),
    ('Bitte hör nicht auf.', 'listen'),
    ('Was bedeutet das Wort Stopp?', 'listen'),
    ('Es geht um Stop Motion Filme.', 'listen'),
]

async def main(base):
    # Explicit test configuration; exercise the production route in Bolt mode.
    os.environ.update(ROBO_DIRECTOR_MODE='bolt', ROBO_DECISION_BASE=base,
                      ROBO_DECISION_MODEL='convaiinnovations/laya-multilingual')
    async with httpx.AsyncClient(timeout=30) as client:
        before=(await client.get(base+'/health')).json()
        decision=DecisionClient(); rows=[]
        for text, expected in CASES:
            start=time.perf_counter()
            result=await decision.choose({'agent_speaking':True,
                'agent_full_reply':'Ich erzähle eine lange freundliche Geschichte über ein Kätzchen.',
                'agent_played_seconds':2.0, 'user_partial':text,
                'asr_candidates':[text], 'rules':[]})
            rows.append({'input':text,'expected':expected,'choice':result['action'],
                         'ms':round((time.perf_counter()-start)*1000,2)})
        after=(await client.get(base+'/health')).json()
    requests=after.get('requests',0)-before.get('requests',0)
    report={'model':after.get('model'),'revision':after.get('revision'),
        'actual_laya_requests':requests,'cases':rows,
        'passed':requests>=len(CASES) and all(r['expected']==r['choice'] for r in rows),
        'scope':'Text-to-decision service and route only; no acoustic identity or playback claim.'}
    (Path(__file__).parent/'static/rebalance-decision-check.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8797')
    asyncio.run(main(p.parse_args().base.rstrip('/')))
