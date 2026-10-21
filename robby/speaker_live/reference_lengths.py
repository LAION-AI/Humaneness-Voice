"""Separate enrollment-duration effects from turn-boundary errors; synthetic only."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from speaker_teacher import get_cached_teacher,enrollment_reference
from .live_test import audio

def main():
    root=Path(__file__).resolve().parent.parent
    config=json.loads((root/'config.local.json').read_text())
    os.environ.setdefault('ROBO_ECAPA_ROOT',config['ROBO_ECAPA_ROOT']);torch.set_num_threads(4)
    teacher=get_cached_teacher();bg=audio('background');rows=[]
    for name,question in [('christoph','christoph-followup'),('robert','robert-followup'),('selma','selma-followup-clear')]:
        reference=audio(name+'-intro');query=audio(question)
        for duration in (2.,3.04,4.,6.):
            ref=enrollment_reference(reference,duration);anchor=teacher.embed(ref)
            for gain in (0.,.16):
                wave=query+np.resize(bg,len(query))*gain
                v=teacher.embed(wave)
                rows.append({'speaker':name,'max_reference_s':duration,'actual_reference_s':len(ref)/16000,
                    'background_gain':gain,'same_voice_cosine':round(float(anchor@v),5),
                    'background_cosine':round(float(anchor@teacher.embed(bg)),5)})
    report={'finished':time.time(),'public_synthetic_only':True,'rows':rows,
        'limits':'Three repeated synthetic identities; no real-person calibration or held-out statistical conclusion.'}
    (root/'logs/reference-lengths.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
