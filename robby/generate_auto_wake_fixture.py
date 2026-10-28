"""Create one fixed synthetic 'Hi Robbie' test utterance, not a private recording."""
import hashlib,json
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parent
def main():
 audio=ROOT/'logs/auto-wake-hi-robbie.mp3';meta=audio.with_suffix('.json')
 if audio.exists() and meta.exists():return
 if audio.exists() or meta.exists():raise RuntimeError('Incomplete existing test fixture')
 body={'model':'small_rl','input_mode':'manual','format':'structured',
  'message':'GENERAL: Calm, clear natural conversational German. Preserve the reference speaker.\nSCRIPT:\nHi Robbie, wie geht es dir?',
  'generation':'single','n':1,'streaming':False,'seconds':5,'temperature':.8,'guidance':2.,
  'voice_mode':'profile','voice_profile':'anime_088','voice_conversion':False,'align':True,'seed':202610300}
 with httpx.Client(base_url='http://127.0.0.1:8792',timeout=120,trust_env=False) as c:
  if c.get('/api/state').json().get('busy'):raise RuntimeError('TTS is busy; do not interrupt it')
  candidate=None
  with c.stream('POST','/api/turn',json=body) as r:
   r.raise_for_status()
   for line in r.iter_lines():
    if line.startswith('data: '):
     e=json.loads(line[6:])
     if e.get('event')=='error':raise RuntimeError('Fixture generation failed')
     if e.get('event')=='result':candidate=e['candidates'][0]
  if not candidate:raise RuntimeError('No fixture audio')
  response=c.get(candidate['url']);response.raise_for_status();audio.write_bytes(response.content)
  receipt={'text':'Hi Robbie, wie geht es dir?','source':'Authored synthetic speech, one fixed take; no private microphone audio.',
   'request':body,'sha256':hashlib.sha256(response.content).hexdigest()}
  meta.write_text(json.dumps(receipt,ensure_ascii=False,indent=2));(ROOT/'static/auto-wake-fixture.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
