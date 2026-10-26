"""Real Laya checks for completion/silence requests; no microphone claim."""
import asyncio,json,time
from pathlib import Path
import httpx
from live_decisions import DecisionClient
from stop_commands import has_stop_cue,is_stop_command

CASES=[('Danke, das reicht.','stop_reply'),('Danke, cas reicht.','stop_reply'),
       ('Halt die Klappe!','stop_reply'),('Bitte erzähl weiter.','listen'),
       ('Danke, das reicht nicht.','listen'),('Was bedeutet halt die Klappe?','listen')]

async def main():
    decision=DecisionClient();rows=[]
    if not decision.base:raise RuntimeError('Configure the actual Laya endpoint before this test')
    async with httpx.AsyncClient(timeout=10) as c:before=(await c.get(decision.base+'/health')).json()
    for text,expected in CASES:
        start=time.perf_counter()
        result=await decision.choose({'agent_speaking':True,
            'agent_full_reply':'Ich erzähle eine lange Geschichte über einen kleinen Roboter.',
            'agent_played_seconds':2.,'user_partial':text,'asr_candidates':[text],'rules':[]})
        rows.append({'text':text,'expected':expected,'choice':result['action'],
            'admitted_to_semantic_check':has_stop_cue(text),'direct_command':is_stop_command(text),
            'ms':round((time.perf_counter()-start)*1000,2),'passed':result['action']==expected})
    async with httpx.AsyncClient(timeout=10) as c:after=(await c.get(decision.base+'/health')).json()
    requests=after.get('requests',0)-before.get('requests',0)
    report={'scope':'Actual configured Laya service and caller admission rules; no ASR, speaker or playback assertion.',
        'model':after.get('model'),'revision':after.get('revision'),'actual_requests':requests,
        'cases':rows,'passed':requests>=len(rows) and all(r['passed'] for r in rows)}
    (Path(__file__).parent/'static/completion-stop-decision.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':asyncio.run(main())
