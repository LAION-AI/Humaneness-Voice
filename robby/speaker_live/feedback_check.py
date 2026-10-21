"""One real-ASR/TTS optional acknowledgment replay; never retries a failed case.

Only public synthetic fixtures; execute after all other live-ASR checks finish.
Cancellation success and an actually observed acknowledgment are separate facts.
"""
import asyncio
import json
from pathlib import Path
from final_verify import runtime_signature
from .stop_test import case


async def main():
    signature = runtime_signature()
    result = await case('christoph-stop-speaking', long_reading=True, feedback=True)
    feedback = result.get('feedback_events', [])
    replies = [e for e in result.get('events', [])
               if e['type'] == 'reply' and e['phase'] == 'stop']
    result['acknowledgment_observed'] = bool(feedback and any(
        r.get('text') == f.get('text') and result.get('reply_audio_samples',{}).get(r.get('reply_id'),0)>0
        for r in replies for f in feedback))
    result['stop_case_passed'] = result['passed']
    result['passed'] = (result['passed'] and result['acknowledgment_observed']
                        and signature == runtime_signature())
    result['runtime_source_sha256'] = signature
    result['synthetic_only'] = True
    Path('logs/reading-feedback-live.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k != 'events'}, ensure_ascii=False))
    if not result['passed']:
        raise SystemExit('Acknowledgment not verified; retained separate stop/feedback results')


if __name__ == '__main__':
    asyncio.run(main())
