"""Fixed-window teacher diagnostics on retained synthetic trigger fixtures."""
import json,time
from pathlib import Path
import numpy as np,torch
from identity import unit
from speaker_teacher import get_cached_teacher,enrollment_reference
from .live_test import audio
from speaker_v4.common import atomic
from speaker_runtime import LivePredictor

def main():
    torch.set_num_threads(2);teacher=get_cached_teacher();intro=audio('christoph-intro')
    refs=[unit(teacher.embed(enrollment_reference(intro,s))) for s in (2.,3.04,4.)]
    full=unit(teacher.embed(intro));game=audio('christoph-game')
    game_full=unit(teacher.embed(game));game_refs=[unit(teacher.embed(enrollment_reference(game,s))) for s in (2.,3.04,4.)];rows=[]
    center=LivePredictor('long3').fast.teacher_mean.detach().numpy()
    for name in ('christoph-bananas','background-bananas','christoph-apples'):
        wave=audio(name)
        for end in np.arange(1.6,len(wave)/16000,.96):
            x=wave[max(0,round((end-3.04)*16000)):round(end*16000)]
            if len(x)<10240:continue
            if float(np.sqrt(np.mean(x*x)))<1e-5:continue
            q=unit(teacher.embed(x));scores=[float(q@r) for r in refs]
            rows.append({'fixture':name,'end_s':round(float(end),3),'template_max':max(scores),
                'anchor':scores[1],'full_intro':float(q@full),'template_mean':sum(scores)/len(scores),
                'game_full':float(q@game_full),'game_template_mean':float(np.mean([q@r for r in game_refs]))})
            rows[-1].update(centered_game=float(unit(q-center)@unit(game_full-center)),
                centered_intro=max(float(unit(q-center)@unit(r-center)) for r in refs))
    report={'synthetic_only':True,'deployed':False,'finished':time.time(),'rows':rows}
    atomic(Path('/mnt/nvme/robo-speaker-studies/speaker-live-results/proactive-teacher-diagnostics.json'),report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
