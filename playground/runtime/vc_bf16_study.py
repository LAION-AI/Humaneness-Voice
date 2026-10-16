"""Benchmark reduced-precision flow only; keep encoder/vocoder in FP32."""
import json
import numpy as np
import soundfile as sf
import torch
import settings
from voice_conversion import ConversionManager
from dynamic_references import fixed_clip
from vc_performance_study import ROOT,measure


def main():
    torch.set_num_threads(4);settings.VC_BATCH_SIZE=8
    manager=ConversionManager();session=manager.prepare(fixed_clip(settings.DEFAULT_VOICE)[0])
    manager.cb._model.s3gen.flow.decoder.estimator.to(dtype=torch.bfloat16)
    paths=sorted((settings.HERE/'catalogue-steering-study/audio').glob('validation-*-baseline-*.wav'))[:16]
    base=[sf.read(p,dtype='float32')[0] for p in paths]
    rows=[]
    for seconds in (1.6,3.2):
        for n in (1,4,8,16):
            size=int(seconds*48000)
            waves=[np.tile(w,1+size//len(w))[:size].copy() for w in base[:n]]
            seeds=[777+104729*i for i in range(n)]
            call=lambda:manager.infer_many(waves,session.conditioning,seeds)
            row={'method':'bf16_flow_only','n':n,'source_seconds':seconds,**measure(call,n*seconds)}
            rows.append(row);print(json.dumps(row),flush=True)
            if n==8 and seconds==1.6:
                for i,w in enumerate(call()):sf.write(ROOT/f'hardware-bf16-n8-{i}.wav',w,48000)
            (ROOT/'bf16.json').write_text(json.dumps({'rows':rows,'policy':'experiment only, not production default'},indent=2))
    print('DONE',flush=True)


if __name__=='__main__':main()
