"""Fixed one-take synthetic identity questions, kept in excluded local logs."""
import json,hashlib
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parent
CASES=[('who-calm','Und wer bin ich?','Calm, clear natural conversational German.',202610280),
       ('who-playful','Weißt du noch, wie ich heiße?','Playful and excited but clear German.',202610281)]
def main():
 folder=ROOT/'logs/social-fixtures';folder.mkdir(exist_ok=True)
 with httpx.Client(base_url='http://127.0.0.1:8792',timeout=120,trust_env=False) as c:
  for name,text,direction,seed in CASES:
   path=folder/(name+'.mp3');meta=folder/(name+'.json')
   if path.exists() and meta.exists():continue
   if c.get('/api/state').json().get('busy'):raise RuntimeError('TTS busy; leave the public request alone')
   body={'model':'small_rl','input_mode':'manual','format':'structured','message':'GENERAL: '+direction+' Preserve the reference speaker.\nSCRIPT:\n'+text,
    'generation':'single','n':1,'streaming':False,'seconds':5,'temperature':.8,'guidance':2.,'voice_mode':'profile','voice_profile':'anime_088','voice_conversion':False,'align':True,'seed':seed}
   candidate=None
   with c.stream('POST','/api/turn',json=body) as r:
    r.raise_for_status()
    for line in r.iter_lines():
     if not line.startswith('data: '):continue
     e=json.loads(line[6:])
     if e.get('event')=='error':raise RuntimeError('TTS fixture failed')
     if e.get('event')=='result':candidate=e['candidates'][0]
   assert candidate
   r=c.get(candidate['url']);r.raise_for_status();path.write_bytes(r.content)
   meta.write_text(json.dumps({'name':name,'text':text,'request':body,'sha256':hashlib.sha256(r.content).hexdigest(),'source':'Authored synthetic speech, first fixed take'},indent=2))
   print('Created',name,flush=True)
 rows=[json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))]
 (ROOT/'static/social-fixtures.json').write_text(json.dumps({'cases':rows,'audio_redistributed':False},indent=2))
if __name__=='__main__':main()
