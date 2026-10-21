"""Phase-stable recheck for a recognized, sub-second Stop, not new enrollment.

Reuse the frozen ASR encoder and existing CPU student. Three fixed causal frame
alignments are averaged, never selected by identity score. No weights or audio
are saved; native TYGER's context is untouched. Called only after ordinary
verification of a literal short command is inconclusive.
"""
import threading
import numpy as np
import torch
from identity import unit
from speaker_runtime import LOCK


def isolated_stop_embedding(asr, predictor, wave, speech_end):
    if not hasattr(asr,'enc') or not hasattr(asr,'encode'):return None
    waves=[]
    for offset in (-1280,0,1280):
        end=max(5120,int(speech_end)+offset)
        # Keep the speech's natural onset, with a fixed-length causal context.
        w=np.pad(wave,(0,max(0,end-len(wave))))[:end]
        waves.append(np.pad(w,(max(0,128000-len(w)),0))[-128000:].astype(np.float32))
    owner=threading.get_ident();captured={};hooks=[]
    def tap(layer):
        def hook(module,args,out):
            if threading.get_ident()==owner:
                x=out[0] if isinstance(out,tuple) else out
                captured[layer]=x[:,-4:].detach().clone()
        return hook
    try:
        for layer in (4,8,12):hooks.append(asr.enc.layers[layer-1].register_forward_hook(tap(layer)))
        with torch.inference_mode():
            hidden=asr.encode(waves)  # existing GPU priority lock serializes it
            captured[15]=torch.stack([h[-4:] for h in hidden])
            x=torch.stack([captured[k] for k in (4,8,12,15)],dim=1).float().cpu()
            energy=np.stack([20*np.log10(np.sqrt((w[-5120:].reshape(4,1280)**2).mean(1)+1e-12)+1e-9) for w in waves])
            with LOCK:emb,_=predictor.fast(x,torch.from_numpy(energy))
            return unit(emb.mean(0).numpy())
    finally:
        for hook in hooks:hook.remove()
