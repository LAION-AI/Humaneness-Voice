"""Minimal Podcast policy loader extracted from the tested workspace runtime."""
import gc,json,time
import torch
import settings
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.initialization import no_init_weights
from safetensors.torch import load_file
from performance_core import FastGenerator
from semantic_layerwise import LayerwiseSemanticBank
def zero_temperature_sampler(sampler):
    """The upstream sampler requires positive T; T=0 means genuine argmax."""
    def sample(*args, **kwargs):
        if float(kwargs.get("temperature", 1)) == 0:
            kwargs["do_sample"] = False
        return sampler(*args, **kwargs)
    return sample

class TokenEngine:
    def pack(self,prompt,spec,reference,n):
        # Keep the generation cap separate from the model's duration condition.
        # Free-timing pilot explicitly uses Tokens=None and no inline durations.
        message=self.proc.build_user_message(text=spec['text'],instruction=prompt,
            tokens=None if spec.get('free_timing',True) else spec['frames'],
            reference=[reference],language='German' if spec['lang']=='de' else 'English')
        ids=self.proc([[message]],mode='generation')['input_ids'][0].long().unsqueeze(0).to(self.device).repeat(n,1,1)
        return ids,torch.ones(ids.shape[:2],dtype=torch.bool,device=self.device)
    def __init__(self,root,proc,refs):
        self.device=settings.ACTING_DEVICE;self.proc=proc;self.refs=refs
        cfg=AutoConfig.from_pretrained(str(root),trust_remote_code=True,local_files_only=True);cfg._attn_implementation='sdpa'
        Model=get_class_from_dynamic_module('modeling_moss_tts.MossTTSLocalModel',str(root))
        index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];state={}
        for name in sorted(set(index.values())):state.update(load_file(str(root/name)))
        with no_init_weights():self.model=Model(cfg)
        missing,unexpected=self.model.load_state_dict(state,strict=False,assign=True)
        allowed={'text_lm_head.weight'}|{f'audio_lm_heads.{i}.weight' for i in range(12)}
        assert not unexpected and not(set(missing)-allowed)
        self.model.tie_weights();self.model=self.model.to(self.device,dtype=torch.bfloat16).requires_grad_(False).eval()
        self.model._sample_next_token=zero_temperature_sampler(self.model._sample_next_token)
        self.fast=FastGenerator(self,graph_limit=1);self.fast.semantic.bank_class=LayerwiseSemanticBank
    @torch.inference_mode()
    def codes(self,spec,speaker,seed,temp):
        with torch.cuda.device(self.device),torch.autocast('cuda',dtype=torch.bfloat16):
            ref=self.refs[speaker]
            branches=[self.pack(spec['prompt'],spec,ref,1),self.pack(spec['unconditional'],spec,ref,1)]
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            budget=min(375,spec['frames']+60)
            p={'max_new_tokens':budget,'audio_temperature':temp,'audio_top_p':.95,'audio_top_k':50,
               'audio_repetition_penalty':1.,'text_temperature':1.,'text_top_p':1.,'text_top_k':50}
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize();wall=time.perf_counter();before=self.fast.capture_seconds;start.record()
            frames=[]
            for frames in self.fast.stream_frames(branches,p,chunk_frames=budget+1,guidance=2.5,graph=True):pass
            end.record();end.synchronize();codes=torch.stack(frames,dim=1).squeeze(0).cpu().contiguous()
            return codes,{'token_wall_seconds':time.perf_counter()-wall,'token_device_seconds':start.elapsed_time(end)/1000,
                'graph_capture_seconds':self.fast.capture_seconds-before,'frames':len(codes),'frame_cap':budget,'capped':len(codes)>=budget}
    def close(self):
        self.fast.clear_graphs();self.fast=None;self.model=None;gc.collect();torch.cuda.empty_cache()
