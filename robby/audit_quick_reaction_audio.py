"""Parakeet audit of the generated short prefixes, not human perceptual grading."""
import base64,json,re,io,math
import soundfile as sf
from pathlib import Path
import httpx,numpy as np
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parent
rows=[]
with httpx.Client(timeout=30) as c:
 for r in json.loads((ROOT/'static/quick-reaction-audio.json').read_text())['rows']:
  if not r['quick_replies']:continue
  intro=next((e['sentence_text'] for e in r['events'] if e.get('sentence_role')=='reaction'),None)
  if not intro:continue
  seconds=r['events'][-1]['quick_reply_audio_s'];wave,rate=sf.read(io.BytesIO(base64.b64decode(r['listen_mp3_b64'])),dtype='float32',always_2d=True);wave=wave[:round(seconds*rate)].mean(axis=1)
  divisor=math.gcd(rate,16000);wav16=resample_poly(wave,16000//divisor,rate//divisor).astype('<f4')
  response=c.post('http://127.0.0.1:8801/transcribe/parakeet_v3',content=wav16.tobytes());response.raise_for_status();data=response.json()
  norm=lambda t:re.findall(r'\w+',t.casefold())
  rows.append({'expected':intro,'recognized':data['text'],'duration_s':seconds,'exact_words':norm(intro)==norm(data['text']),'asr_ms':data['transcribe_ms']})
report={'scope':__doc__,'passed':all(r['exact_words'] for r in rows),'rows':rows}
(ROOT/'static/quick-reaction-prefix-audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps(report,ensure_ascii=False))
