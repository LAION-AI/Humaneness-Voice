"""Bounded CPU ECAPA context ablation on public synthetic QA voices only."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from speaker_teacher import get_cached_teacher
from .live_test import audio

def main():
    torch.set_num_threads(4)
    config=Path(__file__).resolve().parent.parent/'config.local.json'
    if config.exists():
        root=json.loads(config.read_text()).get('ROBO_ECAPA_ROOT')
        if root:os.environ.setdefault('ROBO_ECAPA_ROOT',root)
    teacher=get_cached_teacher();names=['christoph','robert','selma'];bg=audio('background')
    intros={n:audio(n+'-intro') for n in names}
    probes={n:audio(n+('-followup-clear' if n=='selma' else '-followup')) for n in names}
    results=[]
    # Every condition sees the same three voices, two noise phases and gains.
    # No thresholds or source clips are selected using test speaker scores.
    for seconds in (1.,2.,3.04,6.):
        refs={n:teacher.embed(x[-min(len(x),round(seconds*16000)):]) for n,x in intros.items()}
        rows=[]
        for name,clean in probes.items():
            for gain in (0.,.16,.30):
                for offset in (0,round(2.3*16000)):
                    wave=clean+bg[(np.arange(len(clean))+offset)%len(bg)]*gain
                    query=teacher.embed(wave[-min(len(wave),round(seconds*16000)):])
                    scores=sorted(((float(query@v),n) for n,v in refs.items()),reverse=True)
                    accepted=scores[0][0]>=.5 and scores[0][0]-scores[1][0]>=.08
                    rows.append({'target':name,'gain':gain,'offset_s':offset/16000,
                        'winner':scores[0][1],'correct_top1':scores[0][1]==name,'accepted':accepted,
                        'scores':dict((n,round(s,5)) for s,n in scores),
                        'reference_s':min(seconds,len(intros[name])/16000),'query_s':min(seconds,len(wave)/16000)})
        results.append({'max_context_s':seconds,'cases':len(rows),
            'top1_accuracy':sum(r['correct_top1'] for r in rows)/len(rows),
            'correct_accepted':sum(r['correct_top1'] and r['accepted'] for r in rows),
            'wrong_accepted':sum(not r['correct_top1'] and r['accepted'] for r in rows),
            'abstentions':sum(not r['accepted'] for r in rows),'rows':rows})
    report={'finished':time.time(),'public_synthetic_only':True,'physical_microphone_used':False,
        'voices':3,'threshold':.5,'nearest_margin':.08,'results':results,
        'limitations':'Small repeated synthetic QA set, not real-human calibration. Six-second maximum uses only available speech; no padding or waiting to manufacture longer speech. Frozen CPU teacher, not the Nemotron student.'}
    target=Path(__file__).resolve().parent.parent/'logs/speaker-context-lengths.json'
    target.write_text(json.dumps(report,indent=2));print(json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in results]))

if __name__=='__main__':main()
