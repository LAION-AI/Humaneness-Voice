"""One public-synthetic failure replay: quiet pre-roll vs clean voice anchor.

This is an engineering diagnostic, not a speaker-accuracy benchmark. It writes
metrics/hashes only, never audio or embedding vectors. The source is explicitly
an opt-in loopback QA artifact, not a physical microphone recording.
"""
import argparse,hashlib,json,os,time
from pathlib import Path
import soundfile as sf
import torch
from speaker_teacher import get_cached_teacher,enrollment_reference,TEACHER_REVISION
from .live_test import audio

def main(source,old_span):
    root=Path(__file__).resolve().parent.parent
    source=Path(source).resolve()
    allowed=Path('/mnt/nvme/robo-speaker-studies/speaker-live-results/synthetic-debug').resolve()
    if not source.is_relative_to(allowed):raise ValueError('Use an explicit public-synthetic loopback QA artifact only')
    config=json.loads((root/'config.local.json').read_text())
    os.environ.setdefault('ROBO_ECAPA_ROOT',config['ROBO_ECAPA_ROOT']);torch.set_num_threads(4)
    wave,sr=sf.read(source,dtype='float32')
    if sr!=16000 or wave.ndim!=1 or not 0<old_span<=len(wave)/sr:raise ValueError('Invalid synthetic clip/span')
    teacher=get_cached_teacher()
    refs={name:teacher.embed(audio(fixture)) for name,fixture in [('intended Robert','robert-intro'),('unrelated background','background')]}
    variants={'old activity-truncated whole':wave[:round(old_span*sr)],
        'three-second energy window, but still missing word endings':enrollment_reference(wave[:round(old_span*sr)],3.04),
        'new full captured ending + three-second voiced anchor':enrollment_reference(wave,3.04)}
    rows=[]
    for name,x in variants.items():
        vector=teacher.embed(x)
        rows.append({'condition':name,'reference_seconds':len(x)/sr,'cosines':{n:round(float(vector@v),5) for n,v in refs.items()}})
    report={'finished':time.time(),'public_synthetic_only':True,'physical_microphone_used':False,
        'teacher_revision':TEACHER_REVISION,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_seconds':len(wave)/sr,'old_reference_seconds':old_span,'speaker':'Robert','rows':rows,
        'limitation':'One replayed failure, correlated views, not held-out accuracy or real-person calibration.'}
    (root/'logs/anchor-reference-diagnostic.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--old-span',required=True,type=float)
    args=p.parse_args();main(args.source,args.old_span)
