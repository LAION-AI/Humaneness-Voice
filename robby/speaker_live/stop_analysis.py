"""CPU-only diagnostic on existing public synthetic stop fixtures, not humans."""
import json,time
from pathlib import Path
import numpy as np
import torch
from identity import unit
from speaker_teacher import get_cached_teacher,enrollment_reference
from speaker_runtime import LivePredictor
from .live_test import audio
from speaker_v4.common import atomic

def main():
    torch.set_num_threads(2)
    teacher=get_cached_teacher();center=LivePredictor('long3').fast.teacher_mean.detach().numpy()
    refs={}
    for name in ('christoph','robert'):
        x=audio(name+'-intro')
        refs[name]=[unit(teacher.embed(enrollment_reference(x,s))) for s in (2.,3.04,4.)]
    rows=[]
    for name in ('christoph-stop','christoph-halt','christoph-quiet','christoph-stop-speaking','robert-stop','background-stop','robert-stop-short','background-stop-short'):
        x=audio(name);n=len(x)//320;level=np.sqrt(np.mean(x[:n*320].reshape(n,320)**2,axis=1))
        active=np.flatnonzero(level>=max(.003,float(level.max())*.15))
        cropped=x[max(0,int(active[0])*320-160):min(len(x),(int(active[-1])+1)*320+160)]
        tiled=unit(teacher.embed(np.pad(cropped,(0,max(0,20480-len(cropped))),mode='wrap')))
        cropped=np.pad(cropped,(0,max(0,10240-len(cropped))))
        combined=unit(np.mean([unit(teacher.embed(np.pad(cropped,(0,max(0,round(s*16000)-len(cropped)))))) for s in (.64,1.28,2.)],axis=0))
        for view,wave in (('whole',x),('active',cropped)):
            q=unit(teacher.embed(wave))
            rows.append({'fixture':name,'view':view,'seconds':len(wave)/16000,
                'raw':{key:max(float(q@v) for v in values) for key,values in refs.items()},
                'centered':{key:max(float(unit(q-center)@unit(v-center)) for v in values) for key,values in refs.items()}})
        rows.append({'fixture':name,'view':'fixed-multiduration-mean',
            'raw':{key:max(float(combined@v) for v in values) for key,values in refs.items()}})
        rows.append({'fixture':name,'view':'short-cycle-context',
            'raw':{key:max(float(tiled@v) for v in values) for key,values in refs.items()}})
    report={'synthetic_only':True,'finished':time.time(),'deployed':False,
        'method':'Fixed full/cropped views; global teacher centering, fixed-duration vector mean and repeated short context. No identity-score-selected crop or real microphone data.',
        'conclusion':'Neither centering nor fixed-duration averaging supplied a convincing general improvement. Retained diagnostics, not a human calibration or production recipe.',
        'teacher_center_norm':float(np.linalg.norm(center)),'rows':rows}
    atomic(Path('/mnt/nvme/robo-speaker-studies/speaker-live-results/stop-teacher-view-diagnostics.json'),report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
