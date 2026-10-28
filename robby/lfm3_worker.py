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
system='Describe only what is observable, without identifying real people or inventing details. Previous descriptions may be mistaken; use them only for continuity. Return a compact JSON object, with no markdown. If unsure or absent, use an empty string or empty list.'
questions={'audio':'Describe this audio window. JSON keys: foreground, background, music, changes. Use short strings. Listen carefully for melody, instruments or rhythmic music, including quiet background music. Describe it in music only if audible; otherwise use an empty string. Distinguish speech and other foreground sounds from background. For changes only report a clearly audible change within this window or from previous context.',
'video':'Describe this video window. JSON keys: foreground, background, actions, changes, people. foreground/background/actions/changes are short strings, empty if unknown. people is an array of objects {id, description, action}. Use stable person IDs from earlier windows only if reasonably supported; do not guess identities. Describe appearance, visible actions and changes.'}
def run(x):
 previous=json.dumps(x.get('previous',[])[-2:],ensure_ascii=False)[:4000]
 content=[{'type':mode,mode:x['path']}, {'type':'text','text':questions[mode]+' Previous descriptions (uncertain): '+previous}]
 messages=[{'role':'system','content':system},{'role':'user','content':content}]
 t=time.perf_counter();kw={'add_generation_prompt':True,'tokenize':True,'return_dict':True,'return_tensors':'pt'}
 if mode=='video':kw.update(enable_thinking=False)
 inputs=processor.apply_chat_template(messages,**kw).to(device)
 torch.cuda.synchronize();prep=time.perf_counter()-t;t=time.perf_counter()
 with torch.inference_mode():
  args={'max_new_tokens':240,'do_sample':False,'return_dict_in_generate':True}
  if mode=='audio':args.update(suppress_mixed=True,suppress_audio_out=True)
  out=model.generate(**inputs,**args)
 torch.cuda.synchronize();elapsed=time.perf_counter()-t
 seq=out.sequences[0,inputs['input_ids'].shape[1]:];raw=processor.decode(seq,skip_special_tokens=True).strip()
 raw=re.sub(r'<think>.*?</think>','',raw,flags=re.S).strip();raw=re.sub(r'^```(?:json)?\s*|\s*```$','',raw).strip()
 try:
  a=raw.find('{');b=raw.rfind('}');parsed=json.loads(raw[a:b+1]);assert isinstance(parsed,dict)
 except (ValueError,AssertionError):parsed=None
 return {'annotation':parsed,'raw_text':raw if parsed is None else None,'parse_ok':parsed is not None,'preprocess_ms':round(prep*1000,1),'generation_ms':round(elapsed*1000,1),'total_ms':round((prep+elapsed)*1000,1),'output_tokens':len(seq),'output_tokens_per_s':round(len(seq)/elapsed,2) if elapsed else 0}
for line in sys.stdin:
 try:
  q=json.loads(line);r=run(q);print(json.dumps({'ok':True,'result':r},ensure_ascii=False),flush=True)
 except Exception as e:
  print(json.dumps({'ok':False,'error':type(e).__name__+': '+str(e)[:250]}),flush=True);traceback.print_exc(file=sys.stderr)
