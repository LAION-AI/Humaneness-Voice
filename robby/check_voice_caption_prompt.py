"""Real configured LLM smoke check for contradictory, uncertain acoustic cues."""
import json
from pathlib import Path
import httpx

def main():
    body={'text':'Ich bin nicht wütend, ich spreche nur laut. Bitte erzähle mir einen vollständigen kurzen Witz mit Pointe.',
        'speaker':{'display_name':'Christoph','voice_delivery':{'status':'available',
            'caption':'Possibly anger, loud and emphatic','window_s':3,'scope':'this turn',
            'uncertainty':'Uncalibrated acoustic estimate'}},'llm_model':'qwen','llm_provider':'cerebras/fp16'}
    with httpx.Client(timeout=45) as client:
        response=client.post('http://127.0.0.1:8798/api/bolt/llm',json=body);response.raise_for_status()
    events=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    report={'input':body,'events':events,'scope':'Real model response to a deliberately conflicting acoustic hint; text-only, not an audio-caption accuracy test.'}
    Path(__file__).with_name('static').joinpath('voice-caption-model.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('\n'.join(e['text'] for e in events if 'text' in e))
    if any('error' in e for e in events):raise SystemExit(1)
    print('Review wording and task fulfillment manually; no automatic emotion-accuracy claim.')
if __name__=='__main__':main()
