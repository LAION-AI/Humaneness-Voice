"""Opt-in repair utility for a documented content failure, using saved candidates.
Requires the same loopback raw-float32 Parakeet service used in the experiment.
This is deliberately separate from the default reward and from speaker_finish.
"""
from pathlib import Path
import os,sys,json,time,copy,hashlib
import argparse
p=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description='Explicit additional content-repaired copy; never modifies strict ranking artifacts')
parser.add_argument('output');parser.add_argument('--language',choices=['en','de'],default='en');parser.add_argument('--turn',type=int,required=True);parser.add_argument('--asr-url',required=True)
a=parser.parse_args()
sys.path.insert(0,str(p))
from run import setup
setup('cpu')
import numpy as np,soundfile as sf,torch,httpx
from recipe import dump,errors
from scoring import audio16
from align_engine import QwenAligner
from generate import crop
from finish import audio_module
out=Path(a.output).resolve();config=json.loads((out/'manifest.json').read_text())['configuration'];n=config['n'];folder=out/f'generation/bestof{n}-speaker5050/{a.language}-{a.turn:02d}';old=json.loads((folder/'audited.json').read_text())
choices=sorted(range(n),key=lambda i:old['performance']['final_scores'][i]['reward'],reverse=True);reports=[];replacement=None
torch.set_num_threads(4);aligner=QwenAligner(device='cpu');assert aligner.ok;aligner.model.float()
with httpx.Client(timeout=120) as c:
 for i in choices:
  if i==old['performance']['winner']:continue
  wave,sr=sf.read(folder/f'candidate-{i}.flac',dtype='float32');assert sr==48000
  resp=c.post(a.asr_url,content=audio16(wave).tobytes());resp.raise_for_status();transcript=resp.json()['text'];err=errors(old['turn']['text'],transcript)
  report=dict(candidate=i,asr=transcript,wer=err['wer'],reward=old['performance']['final_scores'][i]['reward']);reports.append(report)
  if err['wer']>.15:continue
  spans=aligner.align(audio16(wave),old['turn']['text'].split(),language='English' if a.language=='en' else 'German')
  good=bool(spans) and all(x[3]>.5 for x in spans) and 0<=spans[0][1]<spans[-1][2]<=len(wave)/sr+.03
  report['alignment_ok']=good
  if not good:continue
  clip,edit=crop(wave,dict(alignment_ok=True,first_word_s=spans[0][1],last_word_s=min(spans[-1][2],len(wave)/sr)))
  target=out/f'listening-repair-{a.language}{a.turn:02d}.flac';sf.write(target,clip,48000,subtype='PCM_24')
  replacement=dict(old,edited=str(target),edit=edit);report['alignment']=spans;break
assert replacement is not None,reports
rows=json.loads((out/f'meaning-of-life-{a.language}-bestof{n}-speaker5050.audit.json').read_text());rows[a.turn]=replacement
module=audio_module();wave,timeline=module.assemble(rows)
path=out/f'meaning-of-life-{a.language}-spontaneous-listening-repair.mp3';receipt=module.export(wave,path,'Meaning of life — spontaneous; one content repair',a.language)
report=dict(file=receipt['file'],export=receipt,changed_turn_zero_based=a.turn,original_winner=old['performance']['winner'],original_reward=old['performance']['final_scores'][old['performance']['winner']]['reward'],replacement=reports[-1],candidates_checked=reports,policy='Additional content-repaired listening copy only. Highest original 50/50 reward among alternatives passing content and alignment review. This adds a content eligibility check and is NOT the strict reward experiment. Original candidates, argmax, episode and audit are unchanged. No new synthesis or voice conversion.',generation_seconds=0)
dump(out/'listening-repair.json',report);dump(path.with_suffix('.timeline.json'),timeline)
path.with_suffix('.srt').write_text('\n\n'.join(str(i+1)+'\n'+module.stamp(t['speech_start_s'])+' --> '+module.stamp(t['speech_end_s'])+'\n'+t['speaker']+': '+t['text'] for i,t in enumerate(timeline))+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['export','candidates_checked','replacement']},indent=2));print('replacement',reports[-1]['candidate'],'WER',reports[-1]['wer'],'seconds',receipt['seconds'])
