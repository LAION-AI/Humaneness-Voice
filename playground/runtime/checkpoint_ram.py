"""Immutable verified CPU copies; model.to(cuda) must not move this cache.

Experimental snapshots are prefetched, never made defaults or hot-updated.
Only tensor state is cached, not live GPU modules, KV caches or user references.
"""
import threading
import time


class CheckpointRAM:
    def __init__(self):
        self.lock=threading.RLock();self.cache={};self.loading={};self.status={}

    def describe(self):
        with self.lock:
            rows={key:dict(value) for key,value in self.status.items()}
            return {'precision':'bf16 floating weights; original integer buffers',
                'device':'cpu','bytes':sum(v.get('bytes',0) for v in rows.values()),
                'models':rows,'policy':'Only the selected model is resident in VRAM; CPU copies remain immutable.'}

    def get(self,model_id):
        from acting_models import MODELS,download
        with self.lock:
            if model_id in self.cache:return self.cache[model_id]
            waiter=self.loading.get(model_id)
            if waiter is None:
                waiter=threading.Event();self.loading[model_id]=waiter;owner=True
                self.status[model_id]={'status':'loading','bytes':0}
            else:owner=False
        if not owner:
            waiter.wait()
            with self.lock:
                if model_id in self.cache:return self.cache[model_id]
                raise RuntimeError(self.status[model_id].get('error','CPU checkpoint preload failed'))
        try:
            import torch
            from safetensors.torch import load_file
            started=time.monotonic();release=MODELS[model_id];root=download(model_id)
            path=root/release['checkpoint']
            state=(torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                   if release.get('format')=='pytorch_state_dict' else load_file(str(path),device='cpu'))
            if not isinstance(state,dict) or not state or not all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()):
                raise ValueError('CPU cache only accepts complete tensor dictionaries')
            # copy=True materializes private physical RAM even for mmap BF16
            # exports. The live module may then move independently to CUDA.
            cpu={k:v.detach().to(device='cpu',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype,
                                 copy=True) for k,v in state.items()}
            size=sum(v.numel()*v.element_size() for v in cpu.values())
            with self.lock:
                self.cache[model_id]=(root,cpu)
                self.status[model_id]={'status':'ready','bytes':size,'load_seconds':time.monotonic()-started,
                    'sha256':release['sha256'],'revision':release['revision'],'device':'cpu'}
            print(f'[checkpoint RAM] {model_id}: {size/1e9:.2f} GB ready',flush=True)
            return root,cpu
        except Exception as exc:
            with self.lock:self.status[model_id]={'status':'error','bytes':0,'error':f'{type(exc).__name__}: {exc}'}
            raise
        finally:
            with self.lock:
                self.loading.pop(model_id,None);waiter.set()

    def preload_experimental(self):
        from acting_models import MODELS
        for model_id,release in MODELS.items():
            if release.get('experimental') or release.get('rl_bundle'):
                try:self.get(model_id)
                except Exception as exc:print(f'[checkpoint RAM] {model_id} preload failed: {exc}',flush=True)


RAM=CheckpointRAM()
