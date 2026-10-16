"""Isolated TF32/compiler experiment; does not change live defaults."""
import json,time
import numpy as np
import soundfile as sf
import torch
import settings
from voice_conversion import ConversionManager
from dynamic_references import fixed_clip
from vc_performance_study import ROOT,measure


def main():
    torch.set_num_threads(4)
    manager=ConversionManager();session=manager.prepare(fixed_clip(settings.DEFAULT_VOICE)[0])
    paths=sorted((settings.HERE/'catalogue-steering-study/audio').glob('validation-*-baseline-*.wav'))[:8]
    waves=[]
    for path in paths:
        w=sf.read(path,dtype='float32')[0]
        waves.append(np.tile(w,1+76800//len(w))[:76800].copy())
    report={'scope':'Isolated hardware experiment; single VC replica; no serving policy changes',
        'rows':[],'parity':[],'precision_baseline':{'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32':torch.backends.cudnn.allow_tf32}}
    baseline={}
    for n in (1,8):
        settings.VC_BATCH_SIZE=min(n,8)
        call=lambda:manager.infer_many(waves[:n],session.conditioning,[777+104729*i for i in range(n)])
        baseline[n]=call();report['rows'].append({'method':'fp32_batch','n':n,**measure(call,n*1.6)})
        if n==8:
            for i,w in enumerate(baseline[n]):sf.write(ROOT/f'hardware-fp32-n{n}-{i}.wav',w,48000)
            legacy=[manager.infer(w,session.conditioning,777+104729*i) for i,w in enumerate(waves)]
            for i,w in enumerate(legacy):sf.write(ROOT/f'hardware-legacy-n{n}-{i}.wav',w,48000)
    torch.backends.cuda.matmul.allow_tf32=True
    for n in (1,8):
        settings.VC_BATCH_SIZE=min(n,8)
        call=lambda:manager.infer_many(waves[:n],session.conditioning,[777+104729*i for i in range(n)])
        out=call();row={'method':'tf32_batch','n':n,**measure(call,n*1.6)};report['rows'].append(row)
        report['parity'].append({'method':'tf32','n':n,'relative_rms':[float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-8)) for a,b in zip(baseline[n],out)]})
        for i,w in enumerate(out):sf.write(ROOT/f'hardware-tf32-n{n}-{i}.wav',w,48000)
        for i,w in enumerate(baseline[n]):sf.write(ROOT/f'hardware-fp32-n{n}-{i}.wav',w,48000)
        print(json.dumps(row),flush=True)
    torch.backends.cuda.matmul.allow_tf32=False
    decoder=manager.cb._model.s3gen.flow.decoder
    original=decoder.estimator.forward
    try:
        decoder.estimator.forward=torch.compile(original,dynamic=True,mode='default')
        for n in (1,8):
            settings.VC_BATCH_SIZE=min(n,8)
            call=lambda:manager.infer_many(waves[:n],session.conditioning,[777+104729*i for i in range(n)])
            tick=time.perf_counter();out=call();torch.cuda.synchronize(settings.VC_DEVICE)
            cold=time.perf_counter()-tick
            row={'method':'compiled_fp32_estimator','n':n,'cold_seconds':cold,**measure(call,n*1.6)}
            report['rows'].append(row)
            report['parity'].append({'method':'compiled_fp32','n':n,'relative_rms':[float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-8)) for a,b in zip(baseline[n],out)]})
            for i,w in enumerate(out):sf.write(ROOT/f'hardware-compiled-n{n}-{i}.wav',w,48000)
            print(json.dumps(row),flush=True)
            (ROOT/'hardware.json').write_text(json.dumps(report,indent=2))
    except Exception as exc:
        report['compile_error']=str(exc);print('Compile experiment failed',str(exc),flush=True)
    finally:decoder.estimator.forward=original
    (ROOT/'hardware.json').write_text(json.dumps(report,indent=2))
    print('DONE',flush=True)


if __name__=='__main__':main()
