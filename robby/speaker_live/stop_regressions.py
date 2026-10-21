"""Declared phase/identity controls. One pass per case, no hidden retries.

Run only with the live ASR slot free. Fixtures are public synthetic voices.
Use long real streamed answers so a natural playback end cannot count as Stop.
"""
import asyncio,json,time
from pathlib import Path
from final_verify import runtime_signature
from .stop_test import case


async def main():
    target=Path('logs/stop-phase-regressions.json')
    report={'passed':False,'started':time.time(),'synthetic_only':True,
            'runtime_source_sha256':runtime_signature(),'cases':[]}
    conditions=[(name,True,False,offset) for offset in (.08,.16,.24)
                for name in ('christoph-stop','christoph-wait')]
    # Enrolled-rival controls have their own explicit spoken re-addressing
    # setup and receipt. Earlier mixed-run setup failures remain archived.
    for command,authorized,other,offset in conditions:
        result=await case(command,authorized,other,long_reading=True,phase_offset_s=offset)
        report['cases'].append(result)
        target.write_text(json.dumps(report,indent=2,ensure_ascii=False))
        print(json.dumps({k:v for k,v in result.items() if k!='events'},ensure_ascii=False),flush=True)
        await asyncio.sleep(.5)
    report.update(finished=time.time(),passed=all(c['passed'] for c in report['cases'])
                  and runtime_signature()==report['runtime_source_sha256'])
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit('Retained phase/identity regression failures')


if __name__=='__main__':asyncio.run(main())
