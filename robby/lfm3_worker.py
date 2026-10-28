"""Isolated, pinned LFM3 worker; stdin/stdout JSONL, logs on stderr only."""
import sys,os,json,time,re,traceback
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
mode=sys.argv[1];gpu=int(sys.argv[2]);root=os.path.join(os.environ.get('HF_HOME','/mnt/nvme/moss-15-v2/.cache/lfm3'),'hub')
repos={'audio':('models--LiquidAI--LFM3-Audio-2B','9d49701c372d6011d900f6abb40c6b296fb706dd'), 'video':('models--LiquidAI--LFM3-VL-2B','f83d2b0a2296aaa277f394990a9b8b949ddf9691')}
repo,revision=repos[mode];path=os.path.join(root,repo,'snapshots',revision)
import torch
if mode=='video':
 import av
 _av_open=av.open
 def _compatible_open(*args, **kwargs):
  kwargs.pop('metadata_errors',None)
  return _av_open(*args,**kwargs)
 av.open=_compatible_open
from transformers import AutoProcessor,AutoModelForImageTextToText,AutoModelForMultimodalLM
torch.cuda.set_device(gpu);device='cuda:'+str(gpu)
start=time.perf_counter()
processor=AutoProcessor.from_pretrained(path,trust_remote_code=True,local_files_only=True)
cls=AutoModelForMultimodalLM if mode=='audio' else AutoModelForImageTextToText
model=cls.from_pretrained(path,dtype=torch.bfloat16,device_map=device,trust_remote_code=True,local_files_only=True).eval()
if mode=='audio':processor.audio_tokenizer.to(device=device).eval()
torch.cuda.synchronize();print(json.dumps({'ready':True,'mode':mode,'load_s':round(time.perf_counter()-start,2),'gpu':gpu}),flush=True)
system='Describe the actual recording in detail using only observable evidence. Previous descriptions may be mistaken; use them only for continuity, never to invent present details. Return one valid JSON object, without markdown. Use empty strings or arrays for absent or uncertain details; say when an impression is uncertain.'
questions={'audio':'''Analyze this audio window in detail. Return JSON with string fields foreground, background, speech, music, changes.
Foreground and background: describe distinct audible sounds, their relative prominence and apparent distance when supported. Do not assume every background sound is speech or music.
Speech: if audible, quote only the intelligible words in the original language; mark unclear words as unclear. Describe how it sounds: apparent delivery, emotion or tone as tentative impressions, pace, volume and speaking style. Do not invent speech, a speaker's identity or inner feelings.
Music: only if audible, describe the melody or rhythm, likely instruments or sounds, style, approximate tempo and perceived mood. If a tune, title or artist cannot be recognized with confidence, do not guess it. Music may be faint under speech; if absent, leave music empty.
Changes: describe clear changes in sound within this window or versus the prior window; leave empty if none is evident.''',
'video':'''Analyze this video window in detail. Return JSON with string fields foreground, background, actions, changes and an array people of objects {id, description, action}.
Describe prominent subjects, objects, their appearance, position and surroundings, then visible movement and interactions. Describe only clearly visible changes within this window or versus the prior one; leave changes empty if none is apparent. For each visible person describe clothing, position, posture and visible actions. Use the same person ID as earlier windows only if the visual match is reasonably supported; otherwise use a new or uncertain label. Do not infer gender, age, real identity, intent or feelings as facts. Distinguish observed movement from guesses.'''}
def run(x):
 previous=json.dumps(x.get('previous',[])[-2:],ensure_ascii=False)[:4000]
 def generate(question,system_prompt,limit):
  content=[{'type':mode,mode:x['path']},{'type':'text','text':question}]
  messages=[{'role':'system','content':system_prompt},{'role':'user','content':content}]
  t=time.perf_counter();kw={'add_generation_prompt':True,'tokenize':True,'return_dict':True,'return_tensors':'pt'}
  if mode=='video':kw.update(enable_thinking=False)
  inputs=processor.apply_chat_template(messages,**kw).to(device)
  torch.cuda.synchronize();prep=time.perf_counter()-t;t=time.perf_counter()
  with torch.inference_mode():
   args={'max_new_tokens':limit,'do_sample':False,'return_dict_in_generate':True}
   if mode=='audio':args.update(suppress_mixed=True,suppress_audio_out=True)
   out=model.generate(**inputs,**args)
  torch.cuda.synchronize();elapsed=time.perf_counter()-t
  seq=out.sequences[0,inputs['input_ids'].shape[1]:];raw=processor.decode(seq,skip_special_tokens=True).strip()
  raw=re.sub(r'<think>.*?</think>','',raw,flags=re.S).strip();raw=re.sub(r'^```(?:json)?\s*|\s*```$','',raw).strip()
  return raw,prep,elapsed,len(seq)
 if mode=='audio':
  import soundfile as sf,numpy as np
  samples,_=sf.read(x['path'],dtype='float32');rms=float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.
  if rms<0.0003:
   return {'annotation':{'foreground':'Silence or very quiet audio.','background':'','speech':'','music':'','changes':''},'raw_text':None,'parse_ok':True,'preprocess_ms':0.,'generation_ms':0.,'total_ms':0.,'output_tokens':0,'output_tokens_per_s':0.,'silence_gate':True,'detail_passes':[]}
 raw,prep,elapsed,tokens=generate(questions[mode]+' Previous descriptions (uncertain): '+previous,system,512)
 try:
  a=raw.find('{');b=raw.rfind('}');parsed=json.loads(raw[a:b+1]);assert isinstance(parsed,dict)
 except (ValueError,AssertionError):parsed=None
 passes=[]
 def present(value):
  if not isinstance(value,str):return bool(value)
  value=value.strip().casefold().strip('.! ')
  return bool(value) and not re.match(r'^(?:none|no |not |n/a|silence|silent|absent|inaudible|unknown|unclear)',value)
 if mode=='audio' and parsed:
  prompts={
   'speech':"What exact words are intelligible in this recording? Describe how the speech sounds, including pace, volume, delivery style and any clearly audible emotional tone. Be specific but do not guess unclear words or emotions.",
   'music':"Describe the audible music in detail: melody, rhythm, instruments or sound sources, approximate tempo, style and perceived mood. If there is no music or a detail is unclear, say so; do not guess a song title or artist."}
  for key,question in prompts.items():
   # Ask about faint music independently when speech is present, even when
   # the first pass did not detect it. Keep the model's uncertainty visible.
   if present(parsed.get(key)) or (key=='music' and present(parsed.get('speech'))):
    detail,p,g,n=generate(question,'Answer based only on what is audible, in a few detailed sentences. Express uncertainty; never invent. Do not guess age, gender, identity or song title.',180)
    prep+=p;elapsed+=g;tokens+=n;passes.append(key)
    if detail:
     parsed[key+'_detail']=detail
     if key=='music' and not present(parsed.get('music')) and not re.search(r'\b(?:no|without|not any)\s+(?:audible\s+)?music\b|\b(?:cannot|can.t|do not|don.t)\s+hear\s+(?:any\s+)?music\b',detail,re.I):
      parsed['music']='First pass did not detect music; the follow-up suggests it may be present.'
 return {'annotation':parsed,'raw_text':raw if parsed is None else None,'parse_ok':parsed is not None,'preprocess_ms':round(prep*1000,1),'generation_ms':round(elapsed*1000,1),'total_ms':round((prep+elapsed)*1000,1),'output_tokens':tokens,'output_tokens_per_s':round(tokens/elapsed,2) if elapsed else 0,'detail_passes':passes}

for line in sys.stdin:
 try:
  q=json.loads(line);r=run(q);print(json.dumps({'ok':True,'result':r},ensure_ascii=False),flush=True)
 except Exception as e:
  print(json.dumps({'ok':False,'error':type(e).__name__+': '+str(e)[:250]}),flush=True);traceback.print_exc(file=sys.stderr)
