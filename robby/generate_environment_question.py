"""One fixed synthetic spoken sound question for the environment E2E test."""
import hashlib,json
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parent

def main():
    out=ROOT/'logs/environment-question.mp3';meta=out.with_suffix('.json')
    if out.exists() and meta.exists():return
    with httpx.Client(base_url='http://127.0.0.1:8792',timeout=90,trust_env=False) as client:
        state=client.get('/api/state').json()
        if state.get('busy'):raise RuntimeError('TTS is busy; do not interrupt another request')
        text='Robby, was ist denn das hier?'
        body={'model':'small_rl','input_mode':'manual','format':'structured',
            'message':'GENERAL: Clear, natural conversational German. Preserve the reference speaker.\nSCRIPT:\n'+text,
            'generation':'single','n':1,'streaming':False,'seconds':5,'temperature':.8,'guidance':2.,
            'voice_mode':'profile','voice_profile':'anime_088','voice_conversion':False,'align':True,'seed':20261027}
        candidate=None
        with client.stream('POST','/api/turn',json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith('data: '):continue
                event=json.loads(line[6:])
                if event.get('event')=='error':raise RuntimeError('Question fixture generation failed')
                if event.get('event')=='result':candidate=event['candidates'][0]
        if not candidate:raise RuntimeError('No fixture audio')
        response=client.get(candidate['url']);response.raise_for_status();out.write_bytes(response.content)
        receipt={'text':text,'request':body,'source':'Authored public synthetic test speech, no private microphone',
            'sha256':hashlib.sha256(response.content).hexdigest(),'checkpoint':state.get('checkpoint')}
        meta.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
        (ROOT/'static/environment-question-fixture.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
        print('Created authored question',out.name)
if __name__=='__main__':main()
