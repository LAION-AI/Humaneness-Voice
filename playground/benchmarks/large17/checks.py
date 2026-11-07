"""Teacher-forced semantic cache checks and CUDA-event component accounting."""
import time
import torch
import torch.nn.functional as F
import numpy as np
import performance_core,semantic_graph

@torch.inference_mode()
def run(e,spec,reference):
 result={};old_code=e.last_codes[0].to(e.device)
 with torch.cuda.device(e.device),torch.autocast('cuda',dtype=torch.bfloat16):
  ref=e.reference(reference)
  for nb in (1,2):
   branches=[e.pack(spec['prompt'],spec,ref,1)]
   if nb==2:branches.append(e.pack(spec['unconditional'],spec,ref,1))
   ids,mask=performance_core.combine_branches(branches,e.model.config.pad_token_id,e.model.config.audio_pad_token_id)
   bank=e.fast.semantic.prepare(ids,125)
   native=semantic_graph.raw_forward(e.model.transformer)
   out=native(inputs_embeds=e.model._build_inputs_embeds(ids),input_ids=None,attention_mask=mask,
       past_key_values=None,use_cache=True,return_dict=True)
   past=out.past_key_values;actual=bank.prefill(ids,mask);rows=[]
   def compare(expected,actual):
    a=actual.float();b=expected.float()
    return {'cosine':F.cosine_similarity(a,b).tolist(),'relative_rmse':float(((a-b).square().mean()/b.square().mean()).sqrt()),
        'max_abs':float((a-b).abs().max())}
   rows.append(compare(e.model._global_hidden_to_local(out.last_hidden_state[:,-1]),actual))
   for code in old_code[:8]:
    row=e.model._build_generation_row(1,ids.device,code[None]).repeat(nb,1,1)
    mask=torch.cat([mask,torch.ones((nb,1),dtype=torch.bool,device=ids.device)],dim=1)
    out=native(inputs_embeds=e.model._build_inputs_embeds(row),input_ids=None,attention_mask=mask,
        past_key_values=past,use_cache=True,return_dict=True);past=out.past_key_values
    rows.append(compare(e.model._global_hidden_to_local(out.last_hidden_state[:,-1]),bank.decode(row,mask)))
   result[f'semantic_{nb}_branches']=rows
 # Warm before collecting event spans. No forced synchronizations inside a step.
 e.generate_many([spec],guidance=2.,reference_path=reference,temperature=.8,streaming=True,chunk_frames=20,max_frames=125)
 spans={};originals=[]
 def wrap(cls,name,label):
  original=getattr(cls,name);originals.append((cls,name,original))
  def timed(*a,**kw):
   start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2));start.record()
   value=original(*a,**kw);end.record();spans.setdefault(label,[]).append((start,end));return value
  setattr(cls,name,timed)
 wrap(semantic_graph.SemanticBank,'prefill','semantic_prefill')
 wrap(semantic_graph.SemanticBank,'decode','semantic_decode')
 wrap(performance_core.FastGenerator,'frame','12_acoustic_tokens')
 wrap(performance_core.StreamingCodec,'decode','pcm_codec')
 try:
  e.generate_many([spec],guidance=2.,reference_path=reference,temperature=.8,streaming=True,chunk_frames=20,max_frames=125)
  torch.cuda.synchronize(e.device)
  result['component_profile']={key:{'calls':len(v),'total_ms':sum(a.elapsed_time(b) for a,b in v),
      'median_ms':float(np.median([a.elapsed_time(b) for a,b in v]))} for key,v in spans.items()}
 finally:
  for cls,name,original in originals:setattr(cls,name,original)
 result['note']='Teacher-forced states use identical tokens; BF16 accumulated differences are reported, not assumed zero. Component profile has event instrumentation overhead.'
 if not getattr(e,'layerwise_checked',False):
  e.layerwise_checked=True
  from layerwise_check import main
  main(e)
 return result
