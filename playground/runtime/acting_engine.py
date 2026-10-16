"""Acting full checkpoints on the existing CFG/streaming codec implementation."""
import gc
import json
from pathlib import Path
import sys
import threading
import time

import torch
from safetensors.torch import load_file

import settings
from acting_models import MODELS, download
from engine import SmallEngine, zero_temperature_sampler


def compile_decode_only(forward):
    """Avoid compiling arbitrary user-prompt prefills on the first request.

    Repeated single-frame continuation is the expensive hot path. Keeping
    variable-length prefills eager avoids new prompt-shape compile stalls.
    """
    compiled=torch.compile(forward,dynamic=True,mode='default')
    def dispatch(*args,**kwargs):
        target=compiled if kwargs.get('past_key_values') is not None else forward
        mask=kwargs.get('attention_mask')
        if mask is not None:
            # Buffer views have prompt/budget-dependent row strides. Use a
            # stable layout for the dynamic compiled continuation graph.
            kwargs={**kwargs,'attention_mask':mask.contiguous()}
        return target(*args,**kwargs)
    return dispatch


def load_acting_state(model, checkpoint, *, format='safetensors',state=None):
    if format=='pytorch_state_dict':
        # Verified hash before deserialization; no optimizer or arbitrary
        # Python objects accepted. This export is a complete, ordinary state.
        if state is None:state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        if not isinstance(state,dict) or not state or not all(isinstance(v,torch.Tensor) for v in state.values()):
            raise ValueError('Expected complete Small tensor state dictionary')
        if any('lora_' in name for name in state):
            raise ValueError('Small checkpoint must be a merged full state, not an adapter')
        model.load_state_dict(state,strict=True,assign=True)
        model.tie_weights()
        return
    if format!='safetensors':raise ValueError('Unsupported checkpoint format')
    if state is None:state = load_file(str(checkpoint), device='cpu')
    if not state or not all(name.startswith('base.') for name in state):
        raise ValueError('Expected complete Acting base.* checkpoint')
    missing, unexpected = model.load_state_dict({name[5:]: value for name, value in state.items()}, strict=False,assign=True)
    allowed = {'text_lm_head.weight'} | {f'audio_lm_heads.{i}.weight' for i in range(12)}
    if unexpected or set(missing) - allowed:
        raise ValueError(f'Acting checkpoint mismatch: missing={missing}; unexpected={unexpected}')
    model.tie_weights()


class ActingEngine(SmallEngine):
    def __init__(self, model_id, processor=None):
        started=time.monotonic();release=MODELS[model_id]
        self.model_id=model_id;self.release=release
        self.device=settings.ACTING_DEVICE;self.lock=threading.Lock()
        self.refs={};self.reference_info={};self.last_codes=[];self.last_performance=None
        self.batch_size=release['batch_size'];self.max_batch_size=release['max_batch_size']
        self.warm_batch_sizes=[1];self.compile_global=settings.ACTING_COMPILE_GLOBAL;self.adapter_bank=None
        self.policy_fp32=bool(release.get('rl_bundle'))
        self.speaker_steering_report={'enabled':False,'active':False,'reason':'acting_checkpoint_no_steering'}
        from checkpoint_ram import RAM
        root,cpu_state=RAM.get(model_id)
        sys.path.insert(0,str(root/'code'))
        # These two pinned releases ship matching helper/processor code. Small
        # alone uses the bridge builder; Large uses its actual 36-layer config.
        import moss_small
        moss_small.SFT3=str(root/'assets/sft3');moss_small.QWEN=str(root/'assets/qwen3')
        MossConfig,MossModel,Processor=moss_small.export_classes()
        schema=json.loads((root/'assets/score_schema.json').read_text())
        print(f"[acting] loading {release['label']} on {self.device}",flush=True)
        from transformers.initialization import no_init_weights
        with no_init_weights():
            if release['kind']=='small':
                from large_talker import build_fresh
                moss_small.load_qwen_backbone=lambda model,log=print:None
                self.model,cfg=build_fresh(schema,log=lambda _:None)
            else:
                cfg=MossConfig.from_pretrained(str(root/'assets/sft3'))
                self.model=MossModel(cfg)
        load_acting_state(self.model,root/release['checkpoint'],format=release.get('format','safetensors'),state=cpu_state)
        # CPU cache already rounded the released SFT5 base to BF16. The RL
        # training/inference contract upcasts that rounded base, not the
        # original checkpoint's unrounded FP32 weights.
        self.model=self.model.to(self.device,dtype=torch.float32 if self.policy_fp32 else torch.bfloat16).eval()
        self.rl_adapter=None
        if self.policy_fp32:
            torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            from rl_adapter import attach_rl
            self.rl_adapter=attach_rl(self.model,root,release)
        self.model._sample_next_token=zero_temperature_sampler(self.model._sample_next_token)
        self.proc=processor or Processor.from_pretrained(str(root/'assets/sft3'),
            codec_path='OpenMOSS-Team/MOSS-Audio-Tokenizer-v2',
            codec_weight_dtype='bf16',codec_compute_dtype='bf16')
        self.proc.audio_tokenizer=self.proc.audio_tokenizer.to(self.device).eval()
        # Acting training separates literal Text from Instruction; the old
        # ScorePacker repeats tagged SCRIPT in Text and must not be reused.
        self.packer=None
        self.sr=int(self.proc.model_config.sampling_rate)
        self.parameters=sum(p.numel() for p in self.model.parameters())
        from performance_core import FastGenerator
        self.fast=FastGenerator(self,graph_limit=2)
        if settings.FAST_INFERENCE and self.compile_global:
            # Fuse pointwise decoder work while retaining dynamic growing KV
            # caches. Warm-up runs before ready; no change to sampling/weights.
            self.model.transformer.forward=compile_decode_only(self.model.transformer.forward)
        self.checkpoint_info={**release,'stage':release.get('stage','Acting'),'steering':False,'adapters':False,
            'rl_adapter':self.rl_adapter,'policy_dtype':'fp32' if self.policy_fp32 else 'bf16'}
        print(f'[acting] ready: {self.parameters:,} parameters, {time.monotonic()-started:.1f}s',flush=True)

    def pack(self,prompt,spec,reference,n):
        user=self.proc.build_user_message(text=spec['text'],instruction=prompt,
            tokens=spec['frames'],reference=[reference] if reference is not None else None,
            language='German' if spec['lang']=='de' else 'English')
        ids=self.proc([[user]],mode='generation')['input_ids'][0].long().unsqueeze(0).to(self.device)
        ids=ids.repeat(n,1,1)
        mask=torch.ones(ids.shape[:2],dtype=torch.bool,device=self.device)
        return ids,mask

    def configure_speaker_steering(self,vector,report):
        if vector is not None:
            raise ValueError('Old checkpoint steering vectors are incompatible with Acting releases')
        self.speaker_steering_report={**report,'enabled':False,'active':False,
                                     'reason':'acting_checkpoint_no_steering'}
        return dict(self.speaker_steering_report)

    def unload(self):
        """Release model and CUDA graphs, preserving the shared codec for a swap."""
        processor=self.proc
        self.fast.clear_graphs();self.fast=None;self.model=None;self.packer=None
        self.refs.clear();self.last_codes=[];self.proc=None
        gc.collect()
        with torch.cuda.device(self.device):torch.cuda.empty_cache()
        return processor
