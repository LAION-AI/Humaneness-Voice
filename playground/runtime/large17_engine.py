"""Pinned Large 1.7 merged release on the shared causal streaming runtime.

No residual adapter is attached. Keep the public processor/configuration and
reuse only the audio codec when switching from older Small/Large checkpoints.
"""
import threading
import time

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.initialization import no_init_weights

import settings
from acting_engine import ActingEngine
from acting_models import MODELS
from checkpoint_ram import RAM
from engine import zero_temperature_sampler


class Large17Engine(ActingEngine):
    def __init__(self, model_id='large17', processor=None):
        started=time.monotonic()
        self.model_id=model_id;self.release=MODELS[model_id]
        self.device=settings.ACTING_DEVICE;self.lock=threading.Lock()
        self.refs={};self.reference_info={};self.last_codes=[];self.last_performance=None
        self.batch_size=1;self.max_batch_size=self.release['max_batch_size']
        self.warm_batch_sizes=[1];self.compile_global=True;self.adapter_bank=None
        self.policy_fp32=False;self.rl_adapter=None;self.packer=None
        self.reference_seconds_cap=14.96;self.reference_min_seconds=5.04
        # Small + Large, CFG branches and batch 1/2/4/8 need more than eight
        # guarded continuation shapes in a long-lived switching worker.
        torch._dynamo.config.cache_size_limit=max(torch._dynamo.config.cache_size_limit,32)
        self.speaker_steering_report={'enabled':False,'active':False,'reason':'large17_no_steering'}
        root,state=RAM.get(model_id)
        cfg=AutoConfig.from_pretrained(str(root),trust_remote_code=True,local_files_only=True)
        cfg._attn_implementation='sdpa'
        Model=get_class_from_dynamic_module('modeling_moss_tts.MossTTSLocalModel',str(root))
        Processor=get_class_from_dynamic_module('processing_moss_tts.MossTTSLocalProcessor',str(root))
        with no_init_weights():self.model=Model(cfg)
        missing,unexpected=self.model.load_state_dict(state,strict=False,assign=True)
        allowed={'text_lm_head.weight'}|{f'audio_lm_heads.{i}.weight' for i in range(12)}
        if unexpected or set(missing)-allowed:
            raise ValueError(f'Large 1.7 state mismatch: {missing}; {unexpected}')
        self.model.tie_weights()
        self.model=self.model.to(self.device,dtype=torch.bfloat16).requires_grad_(False).eval()
        self.model._sample_next_token=zero_temperature_sampler(self.model._sample_next_token)
        if processor is None:
            self.proc=Processor.from_pretrained(str(root),codec_path='OpenMOSS-Team/MOSS-Audio-Tokenizer-v2',
                codec_weight_dtype='bf16',codec_compute_dtype='bf16')
        else:
            self.proc=Processor(tokenizer=AutoTokenizer.from_pretrained(str(root),local_files_only=True),
                audio_tokenizer=processor.audio_tokenizer,model_config=cfg)
        self.proc.audio_tokenizer=self.proc.audio_tokenizer.to(self.device).eval()
        self.sr=int(cfg.sampling_rate);self.parameters=sum(p.numel() for p in self.model.parameters())
        from performance_core import FastGenerator
        self.fast=FastGenerator(self,graph_limit=2)
        import os
        if self.fast.semantic and os.environ.get('HVS_LARGE17_LAYERWISE','0')=='1':
            from semantic_layerwise import LayerwiseSemanticBank
            self.fast.semantic.bank_class=LayerwiseSemanticBank
        self.precision_report={'mode':'merged_bf16','adapter_applied':False}
        self.checkpoint_info={**self.release,'stage':'Large 1.7','steering':False,'adapters':False,
            'policy_dtype':'bf16','compute_precision':self.precision_report}
        print(f'[large17] ready: {self.parameters:,} parameters, {time.monotonic()-started:.1f}s',flush=True)

    def reference(self,path):
        codes=super().reference(path)
        if not 63<=len(codes)<=187:
            raise ValueError('Large 1.7 requires 5.04–14.96 seconds of reference speech (63–187 codec frames).')
        return codes

    def warm_live_stream(self):
        super().warm_live_stream()
        # Prepare the second common KV capacity before declaring a model swap
        # ready. A longer sentence otherwise triggers compilation during speech.
        from prompting import build
        spec=build({'text':'I can tell you a longer story when you are ready. '*8,
                    'caption':'Warm, natural conversation.'},'caption','en',seconds=30.,reference_identity=True)
        self.generate_many([spec],reference_path=settings.HERE/'fixed-references/audio/emolia_c1699.wav',
            temperature=.8,guidance=settings.DEFAULT_GUIDANCE,streaming=True,chunk_frames=20,
            on_audio=lambda chunk:True)
        self.last_codes=[];self.last_performance=None


def load_engine(model_id,processor=None):
    return (Large17Engine if model_id=='large17' else ActingEngine)(model_id,processor=processor)
