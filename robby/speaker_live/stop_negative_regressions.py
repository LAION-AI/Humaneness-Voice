"""Unknown foreground stop voices must not acquire the current speaker's turn."""
import asyncio,json,time
from pathlib import Path
from final_verify import runtime_signature
from .stop_test import case

async def main():
    target=Path('logs/stop-negative-regressions.json')
    report={'passed':False,'started':time.time(),'synthetic_only':True,
            'runtime_source_sha256':runtime_signature(),'cases':[]}
    for name in ('background-stop-short','background-stop'):
        result=await case(name,authorized=False,long_reading=True,phase_offset_s=.16)
        report['cases'].append(result);target.write_text(json.dumps(report,indent=2,ensure_ascii=False))
        print(json.dumps({k:v for k,v in result.items() if k!='events'},ensure_ascii=False),flush=True)
        await asyncio.sleep(.5)
    report.update(finished=time.time(),passed=all(c['passed'] for c in report['cases'])
                  and runtime_signature()==report['runtime_source_sha256'])
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit('Unknown-speaker regression failed; receipt retained')

if __name__=='__main__':asyncio.run(main())
