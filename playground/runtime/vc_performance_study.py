"""Real GPU VC throughput, seeded parity and replica benchmark.

Fixed public study clips only; no live API or user uploads are touched.
"""
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import soundfile as sf
import torch
import settings
from voice_conversion import ConversionManager
from vc_batch import batch_native, tokenize_equal
from dynamic_references import fixed_clip

ROOT=settings.HERE/'vc-performance'


def measure(call,audio_seconds,repeats=3):
    call();torch.cuda.synchronize(settings.VC_DEVICE)
    measurements=[]
    for _ in range(repeats):
        torch.cuda.synchronize(settings.VC_DEVICE);tick=time.perf_counter()
        out=call();torch.cuda.synchronize(settings.VC_DEVICE)
        measurements.append(time.perf_counter()-tick)
        assert len(out) and all(np.isfinite(w).all() and len(w)>0 for w in out)
    seconds=float(np.median(measurements))
    return {'seconds':seconds,'repeats':measurements,'aggregate_audio_seconds':audio_seconds,
        'aggregate_rtf':seconds/audio_seconds,'audio_seconds_per_second':audio_seconds/seconds}


def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(4)
    manager=ConversionManager();session=manager.prepare(fixed_clip(settings.DEFAULT_VOICE)[0])
    model=manager.cb._model
    report={'device':torch.cuda.get_device_name(settings.VC_DEVICE),'model_revision':settings.VC_REVISION,
        'parameters':sum(p.numel() for p in model.s3gen.parameters()),'precision':'FP32, original 10 flow steps',
        'scope':'VC stage wall time: source H2D/resampling/tokenizer/flow/vocoder/D2H/watermark/output resampling; excludes target encoding, TTS, scoring, network',
        'timings':[],'tokenizer_parity':[]}
    sources=sorted((settings.HERE/'catalogue-steering-study/audio').glob('validation-*-baseline-*.wav'))[:16]
    base=[sf.read(p,dtype='float32')[0] for p in sources]
    def waves(n,seconds):
        size=int(seconds*48000)
        return [np.tile(base[i%len(base)],1+size//len(base[i%len(base)]))[:size].copy() for i in range(n)]
    from chatterbox.models.s3gen.s3gen import get_resampler
    with torch.inference_mode():
        for n in (1,8,16):
            source=get_resampler(48000,16000,torch.device(settings.VC_DEVICE))(torch.as_tensor(np.stack(waves(n,1.6)),device=settings.VC_DEVICE))
            old,lengths=model.s3gen.tokenizer(source);new,new_lengths=tokenize_equal(model.s3gen.tokenizer,source)
            item={'n':n,'codes_equal':torch.equal(old,new),'lengths_equal':torch.equal(lengths,new_lengths)}
            report['tokenizer_parity'].append(item);assert item['codes_equal'] and item['lengths_equal'],item
    for seconds in (.8,1.6,3.2):
        for n in (1,4,8,16):
            inputs=waves(n,seconds);seeds=[777+104729*i for i in range(n)]
            legacy=lambda:[manager.infer(w,session.conditioning,s) for w,s in zip(inputs,seeds)]
            row={'method':'legacy_serial','n':n,'source_seconds':seconds,**measure(legacy,n*seconds)}
            report['timings'].append(row);print(json.dumps(row),flush=True)
            for batch_size in sorted({min(n,4),min(n,8),min(n,16)}):
                settings.VC_BATCH_SIZE=batch_size
                batched=lambda:manager.infer_many(inputs,session.conditioning,seeds)
                row={'method':'gpu_batch','microbatch':batch_size,'n':n,'source_seconds':seconds,**measure(batched,n*seconds)}
                report['timings'].append(row);print(json.dumps(row),flush=True)
            if seconds==1.6 and n in (1,8):
                a,b=legacy(),batched()
                diffs=[float(np.sqrt(np.mean((x-y)**2))/max(np.sqrt(np.mean(x*x)),1e-8)) for x,y in zip(a,b)]
                report.setdefault('wave_parity',[]).append({'n':n,'relative_rms_differences':diffs,'lengths_equal':all(len(x)==len(y) for x,y in zip(a,b))})
                for i in range(min(n,2)):
                    sf.write(ROOT/f'parity-n{n}-{i}-serial.wav',a[i],48000)
                    sf.write(ROOT/f'parity-n{n}-{i}-batch.wav',b[i],48000)
            (ROOT/'results.json').write_text(json.dumps(report,indent=2))
    # Independent replicas / streams, benchmark-only. Per-row generators avoid
    # shared torch.manual_seed races; no change to the serving policy here.
    second=ConversionManager();other=second.prepare(fixed_clip(settings.DEFAULT_VOICE)[0])
    streams=[torch.cuda.Stream(device=settings.VC_DEVICE) for _ in range(2)]
    def worker(index,inputs,seeds):
        cb=(manager,second)[index];conditioning=(session,other)[index].conditioning
        with torch.inference_mode(),torch.cuda.device(settings.VC_DEVICE),torch.cuda.stream(streams[index]):
            result=batch_native(cb.cb._model,inputs,conditioning,seeds,settings.VC_DEVICE)
            from scipy.signal import resample_poly
            return [np.asarray(resample_poly(cb.cb._model.watermarker.apply_watermark(w,sample_rate=24000),2,1),np.float32) for w in result]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for n in (8,16):
            inputs=waves(n,1.6);seeds=[777+104729*i for i in range(n)]
            def replicas():
                jobs=[pool.submit(worker,i,inputs[i*n//2:(i+1)*n//2],seeds[i*n//2:(i+1)*n//2]) for i in range(2)]
                return [w for job in jobs for w in job.result()]
            row={'method':'two_replicas_two_cuda_streams','n':n,'source_seconds':1.6,**measure(replicas,n*1.6)}
            report['timings'].append(row);print(json.dumps(row),flush=True)
    (ROOT/'results.json').write_text(json.dumps(report,indent=2))
    print('DONE',flush=True)


if __name__=='__main__':main()
