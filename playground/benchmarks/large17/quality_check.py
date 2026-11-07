"""Transcribe synthetic benchmark clips AFTER timing, never concurrently.

This is a transcription canary, not a perceptual or speaker-similarity study.
The local ASR worker must have Parakeet v3 available.
"""
import json, math, re
from pathlib import Path
import httpx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT=Path(__file__).resolve().parent
SHORT='Hallo, schön, dass du da bist. Was möchtest du heute machen?'
LONG='Ich kann dir gerne eine kleine Geschichte erzählen. Heute hat ein neugieriger Roboter zum ersten Mal das Meer gesehen, und er konnte vor Freude kaum stillstehen.'

def words(text):return re.findall(r'\w+',text.casefold())
def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        following=[i]
        for j,y in enumerate(b,1):following.append(min(following[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=following
    return row[-1]

def main():
    rows=[]
    files=sorted(ROOT.glob('sample-fast-*.wav'))+sorted(ROOT.glob('long-*.wav'))
    with httpx.Client(timeout=120) as client:
        for path in files:
            wave,sr=sf.read(path,dtype='float32',always_2d=True);wave=wave.mean(-1)
            factor=math.gcd(sr,16000);pcm=resample_poly(wave,16000//factor,sr//factor).astype('<f4')
            response=client.post('http://127.0.0.1:8801/transcribe/parakeet_v3',content=pcm.tobytes())
            response.raise_for_status();asr=response.json();expected=LONG if path.name.startswith('long-') else SHORT
            row={'file':path.name,'expected':expected,'asr':asr,'finite':bool(np.isfinite(wave).all()),
                'rms':float(np.sqrt(np.mean(wave*wave))),
                'normalized_wer':distance(words(expected),words(asr['text']))/len(words(expected))}
            rows.append(row);(ROOT/'quality.json').write_text(json.dumps(rows,indent=2))
            print(path.name,row['normalized_wer'],asr['text'],flush=True)

if __name__=='__main__':main()
