"""Rival controls with explicit spoken re-addressing before the long request.

The older ambiguous setup failures remain in stop-phase-regressions.json; this
separate receipt must not erase or relabel them as successful Stop checks.
"""
import asyncio,json,time
from pathlib import Path
from final_verify import runtime_signature
from .stop_test import case

async def main():
    target=Path('logs/stop-enrolled-regressions.json')
    report={'passed':False,'started':time.time(),'synthetic_only':True,
            'runtime_source_sha256':runtime_signature(),'cases':[]}
    for name in ('robert-stop-short','robert-stop'):
        result=await case(name,authorized=False,enroll_other=True,long_reading=True,
                          phase_offset_s=.16,request_intro=True)
        report['cases'].append(result);target.write_text(json.dumps(report,indent=2,ensure_ascii=False))
        print(json.dumps({k:v for k,v in result.items() if k!='events'},ensure_ascii=False),flush=True)
        await asyncio.sleep(.5)
    report.update(finished=time.time(),passed=all(c['passed'] for c in report['cases'])
                  and runtime_signature()==report['runtime_source_sha256'])
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit('Enrolled-speaker regression failed; receipt retained')

if __name__=='__main__':asyncio.run(main())
