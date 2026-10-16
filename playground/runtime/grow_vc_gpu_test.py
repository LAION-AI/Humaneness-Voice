"""Offline GPU canary of the actual 24k adapter, not a quality study."""
import json
from pathlib import Path
import time

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch

import settings
from voice_conversion import ConversionManager


def main():
    settings.VC_DEVICE='cuda:1';torch.set_num_threads(4)
    manager=ConversionManager()
    source=settings.HERE/'test-results/catalogue-mira-on-take.mp3'
    wave,rate=sf.read(source,dtype='float32')
    if wave.ndim>1:wave=wave.mean(1)
    if rate!=48000:wave=resample_poly(wave,48000,rate).astype(np.float32)
    wave=wave[:int(4.8*48000)]
    cases=[]
    for precision,streaming,n in [('bf16',False,2),('bf16',True,1),('fp32',False,1)]:
        session=manager.prepare_optimized(settings.HERE/'fixed-references/audio/emolia_c0542.wav',
            seed=24001,profile=True,precision=precision)
        tick=time.monotonic();output=session.full([wave.copy() for _ in range(n)],streaming=streaming,chunk_frames=20)
        elapsed=time.monotonic()-tick
        assert len(output)==n and all(len(w)==len(wave) and np.isfinite(w).all() for w in output)
        destination=settings.HERE/'test-results'/f'grow24k-{precision}-{"stream" if streaming else "offline"}.wav'
        sf.write(destination,output[0],48000)
        cases.append({'precision':precision,'streaming':streaming,'batch':n,'seconds':elapsed,
            'audio_seconds_per_take':len(wave)/48000,'report':dict(session.report),'audio':str(destination)})
        session.close();print(precision,streaming,n,round(elapsed,3),'seconds',flush=True)
    (settings.HERE/'test-results/grow24k-gpu.json').write_text(json.dumps({'model':manager.describe(),
        'cases':cases,'note':'Functional canary only; not a quality comparison or throughput benchmark.'},indent=2))


if __name__=='__main__':main()
