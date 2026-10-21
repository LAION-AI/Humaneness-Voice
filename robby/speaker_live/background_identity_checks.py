"""Correlated public-synthetic background impostor probes; no human calibration."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from speaker_teacher import get_cached_teacher,enrollment_reference
from .live_test import audio

def main(multi_duration=False):
    root=Path(__file__).resolve().parent.parent
    config=json.loads((root/'config.local.json').read_text())
    os.environ.setdefault('ROBO_ECAPA_ROOT',config['ROBO_ECAPA_ROOT']);torch.set_num_threads(4)
    teacher=get_cached_teacher();bg=audio('background')
    refs={n:[teacher.embed(enrollment_reference(audio(n+'-intro'),s)) for s in (2.,3.04,4.)] if multi_duration
        else [teacher.embed(audio(n+'-intro'))] for n in ('christoph','robert','selma')}
    rows=[]
    for seconds in (.96,1.,2.,3.04,6.):
        for offset in (0,36800):
            vector=teacher.embed(np.roll(bg,offset)[-round(seconds*16000):])
            scores={name:round(max(float(vector@v) for v in views),5) for name,views in refs.items()}
            rows.append({'seconds':seconds,'offset_samples':offset,'scores':scores,
                'highest':max(scores.values()),'accepted_at_full_threshold':max(scores.values())>=.5,
                'above_bounded_continuity_floor':max(scores.values())>=.45})
    report={'finished':time.time(),'public_synthetic_only':True,'real_person_calibration':False,
        'multi_duration_enrollment':multi_duration,
        'scope':'Three clean full-introduction references against one unrelated background voice, two phases and five durations. Correlated probes; not a false-acceptance-rate estimate.',
        'rows':rows}
    (root/'logs'/('background-gallery-checks.json' if multi_duration else 'background-identity-checks.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps({'maximum_impostor_cosine':max(r['highest'] for r in rows),
        'accepted':sum(r['accepted_at_full_threshold'] for r in rows),'probes':len(rows)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--multi-duration',action='store_true');main(p.parse_args().multi_duration)
