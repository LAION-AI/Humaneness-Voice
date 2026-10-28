"""One authored short wake fixture. Never overwrites an existing test take."""
import hashlib,json
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parent
def main():
 out=ROOT/'logs/wake-short.mp3';meta=out.with_suffix('.json')
 if out.exists() and meta.exists():return
 if out.exists() or meta.exists():raise RuntimeError('Incomplete existing wake fixture')
 with httpx.Client(base_url='http://127.0.0.1:8792',timeout=120,trust_env=False) as c:
  if c.get('/api/state').json().get('busy'):raise RuntimeError('TTS busy; do not interrupt the running request')
  body={'model':'small_rl','input_mode':'manual','format':'structured','message':'GENERAL: Calm, clear natural conversational German. Preserve the reference speaker.\nSCRIPT:\nHey Robby, ich bin Christoph.',
   'generation':'single','n':1,'streaming':False,'seconds':5,'temperature':.8,'guidance':2.,'voice_mode':'profile','voice_profile':'anime_088','voice_conversion':False,'align':True,'seed':202610299}
  candidate=None
  with c.stream('POST','/api/turn',json=body) as r:
   r.raise_for_status()
   for line in r.iter_lines():
    if line.startswith('data: '):
     e=json.loads(line[6:])
     if e.get('event')=='error':raise RuntimeError('TTS generation failed')
     if e.get('event')=='result':candidate=e['candidates'][0]
  if not candidate:raise RuntimeError('No wake audio')
  r=c.get(candidate['url']);r.raise_for_status();out.write_bytes(r.content)
  record={'name':'wake-short','text':'Hey Robby, ich bin Christoph.','request':body,'sha256':hashlib.sha256(r.content).hexdigest(),'source':'Authored synthetic speech, fixed single take; no private audio.'}
  meta.write_text(json.dumps(record,ensure_ascii=False,indent=2));(ROOT/'static/wake-fixture.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
