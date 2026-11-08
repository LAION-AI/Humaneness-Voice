"""Waveform assembly. Internal near-silence compaction is UNVALIDATED."""
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
SR=48000
def read(path,sr=SR):
 wave,rate=sf.read(path,dtype='float32',always_2d=True);wave=wave.mean(1)
 if not np.isfinite(wave).all() or len(wave)==0:raise ValueError('Empty/nonfinite audio')
 if rate!=sr:
  g=math.gcd(rate,sr);wave=resample_poly(wave,sr//g,rate//g).astype(np.float32)
 return wave
def write(path,wave):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);sf.write(path,wave,SR,subtype='PCM_24')
def crossfade(left,right,n):
 n=min(int(n),len(left),len(right))
 if n<=0:return np.concatenate([left,right])
 t=np.linspace(0,np.pi/2,n)
 # Constant gain, not equal power: identical quiet backgrounds do not get +3 dB.
 blend=left[-n:]*np.cos(t)**2+right[:n]*np.sin(t)**2
 return np.concatenate([left[:-n],blend.astype(np.float32),right[n:]])
def crop(wave,start,end,lead_ms=80,tail_ms=150):
 first=round(start*SR);last=round(end*SR)
 if not 0<=first<last<=len(wave):raise ValueError('ASR span outside waveform')
 a=max(0,first-round(lead_ms*SR/1000));b=min(len(wave),last+round(tail_ms*SR/1000))
 out=wave[a:b].copy();fi=min(round(.02*SR),first-a);fo=min(round(.05*SR),b-last)
 if fi:out[:fi]*=np.sin(np.linspace(0,np.pi/2,fi))**2
 if fo:out[-fo:]*=np.cos(np.linspace(0,np.pi/2,fo))**2
 wanted=last+round(tail_ms*SR/1000)-a
 if len(out)<wanted:out=np.pad(out,(0,wanted-len(out)))
 if not np.array_equal(out[first-a:last-a],wave[first:last]):raise RuntimeError('Fade touched a spoken word')
 return out,{'source_start_s':a/SR,'source_end_s':b/SR,'first_word_s':(first-a)/SR,'last_word_s':(last-a)/SR,'duration_s':len(out)/SR}
def compact_pauses(wave,words,*,enabled=False,min_pause_s=.8,keep_s=.18,absolute_dbfs=-45,relative_db=30,crossfade_ms=10):
 """Only eligible gaps BETWEEN timestamped words. No VAD or guessed speech cuts.

 Both the long-gap criterion and a conservative absolute+relative RMS test must
 pass. Cuts run right-to-left and every returned word retains the same samples.
 Quiet nonverbal speech missed by ASR can still be damaged: explicitly unvalidated.
 """
 if not enabled:return wave,words,{'enabled':False,'unvalidated':True,'cuts':[]}
 if not .15<=keep_s<=.2 or min_pause_s<=keep_s:raise ValueError('Use150–200-ms retained pauses and a longer trigger')
 if not words:raise ValueError('Word timestamps are required before pause compaction')
 speech=[]
 for w in words:
  a,b=round(w['start_s']*SR),round(w['end_s']*SR)
  if b>a:speech.append(float(np.sqrt(np.mean(wave[a:b]**2)+1e-20)))
 if not speech:raise ValueError('No speech for relative energy calibration')
 db=20*np.log10(max(np.median(speech),1e-10));threshold_db=min(absolute_dbfs,db-relative_db);threshold=10**(threshold_db/20)
 edits=[];fade=round(crossfade_ms*SR/1000)
 for left,right in zip(words,words[1:]):
  last=round(left['end_s']*SR);first=round(right['start_s']*SR)
  if first-last<min_pause_s*SR:continue
  total=round(keep_s*SR)+fade;leftkeep=total//2;rightkeep=total-leftkeep
  a=last+leftkeep;b=first-rightkeep
  if a-fade<last+round(.03*SR) or b+fade>first-round(.03*SR):continue
  quiet=wave[a-fade:b+fade]
  step=480;rms=[np.sqrt(np.mean(quiet[i:i+step]**2)+1e-20) for i in range(0,len(quiet),step)]
  if not rms or max(rms)>threshold:continue
  edits.append({'a':a,'b':b,'fade':fade,'removed':b-a+fade,'source_word_gap_s':(first-last)/SR,'retained_word_gap_s':keep_s})
 out=wave.copy()
 for e in reversed(edits):out=crossfade(out[:e['a']],out[e['b']:],e['fade'])
 def shift(t):
  sample=round(t*SR);return (sample-sum(e['removed'] for e in edits if sample>=e['b']+e['fade']))/SR
 updated=[dict(w,start_s=shift(w['start_s']),end_s=shift(w['end_s'])) for w in words]
 for old,new in zip(words,updated):
  a,b=round(old['start_s']*SR),round(old['end_s']*SR);x,y=round(new['start_s']*SR),round(new['end_s']*SR)
  if not np.array_equal(wave[a:b],out[x:y]):raise RuntimeError('Pause compaction changed recognized speech')
 return out,updated,{'enabled':True,'unvalidated':True,'threshold_dbfs':float(threshold_db),'min_pause_s':min_pause_s,'keep_s':keep_s,'cuts':edits}
def assemble(units,gap_ms=400):
 chunks=[];maps=[];cursor=0;last=None
 for unit_id,wave,edges in units:
  first=round(edges['first_word_s']*SR);end=round(edges['last_word_s']*SR)
  if last is not None:
   silence=round(gap_ms*SR/1000)-(cursor-last)-first
   if silence<0:raise ValueError('Requested gap is shorter than retained tail+lead. Reduce guards or increase gap.')
   chunks.append(np.zeros(silence,np.float32));cursor+=silence
  maps.append({'unit':unit_id,'start_s':cursor/SR,'first_word_s':(cursor+first)/SR,'last_word_s':(cursor+end)/SR})
  if last is not None and cursor+first-last!=round(gap_ms*SR/1000):raise RuntimeError('Incorrect word-clock gap')
  chunks.append(wave);last=cursor+end;cursor+=len(wave)
 if not chunks:raise ValueError('Cannot assemble an empty episode')
 return np.concatenate(chunks),maps
