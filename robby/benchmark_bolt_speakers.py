"""Small public-synthetic check of fast ECAPA preparation and reference length."""
import json,time,statistics
from pathlib import Path
import numpy as np,soundfile as sf,torch
from scipy.signal import resample_poly
from speaker_teacher import get_cached_teacher,enrollment_reference
from sentence_session import OpenVoiceMemory

ROOT=Path(__file__).parent
FIXTURES=Path('/mnt/nvme/robo-speaker-studies/speaker-live-fixtures')

def load(name):
    x,sr=sf.read(FIXTURES/(name+'.mp3'),dtype='float32',always_2d=True);x=x.mean(1)
    if sr!=16000:x=resample_poly(x,16000//np.gcd(sr,16000),sr//np.gcd(sr,16000))
    active=np.flatnonzero(np.abs(x)>.003)
    return x[max(0,active[0]-1600):min(len(x),active[-1]+1601)] if len(active) else x

def main():
    torch.set_num_threads(4);teacher=get_cached_teacher();start=time.perf_counter()
    teacher.embed(np.ones(16000,np.float32)*.01);cold=(time.perf_counter()-start)*1000
    people={'Christoph':('christoph-intro','christoph-followup'),
            'Robert':('robert-intro','robert-followup'),'Selma':('selma-intro','selma-followup-clear')}
    references={name:teacher.embed(enrollment_reference(load(pair[0]))) for name,pair in people.items()}
    rows=[]
    for name,(_,query) in people.items():
        audio=load(query);full=teacher.embed(enrollment_reference(audio))
        for seconds in (.64,1.28,2.,3.04):
            # A causal cached anchor ending 320 ms before the utterance ends.
            end=max(round(.64*16000),len(audio)-5120);window=audio[max(0,end-round(seconds*16000)):end]
            for repeat in range(3):
                start=time.perf_counter();embedding=teacher.embed(window);compute=(time.perf_counter()-start)*1000
                scores={who:float(embedding@reference) for who,reference in references.items()}
                rows.append({'speaker':name,'query':query,'context_s':seconds,'actual_context_s':len(window)/16000,
                    'repeat':repeat,'embedding_ms':round(compute,3),'full_utterance_cosine':round(float(full@embedding),4),
                    'winner':max(scores,key=scores.get),'correct':max(scores,key=scores.get)==name,'scores':scores})
    # Measure NN work at the configured maximum (32 people x 8 prototypes).
    rng=np.random.default_rng(777);memory=OpenVoiceMemory()
    for i in range(32):
        vectors=[v/np.linalg.norm(v) for v in rng.normal(size=(8,192)).astype(np.float32)]
        memory.users[f'speaker-{i}']={'name':f'Speaker {i}','named':True,'vectors':vectors,'asked_turn':0,'turns':0}
    times=[];v=references['Christoph']
    for _ in range(500):
        start=time.perf_counter();max((float(v@r),sid) for sid,u in memory.users.items() for r in u['vectors']);times.append((time.perf_counter()-start)*1000)
    report={'source':'Six authored synthetic fixtures, three voices; distinct intro and follow-up recordings, not a population accuracy benchmark',
        'cold_teacher_load_and_warmup_ms':round(cold,2),'nn_32_people_8_vectors_median_ms':round(statistics.median(times),3),
        'rows':rows,'finished':time.time()}
    (ROOT/'static/bolt-speaker-speed.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    for seconds in (.64,1.28,2.,3.04):
        a=[r for r in rows if r['context_s']==seconds]
        print({'context_s':seconds,'median_ms':round(statistics.median(r['embedding_ms'] for r in a),2),
            'median_full_cosine':round(statistics.median(r['full_utterance_cosine'] for r in a),3),
            'correct_queries':sum(all(r['correct'] for r in a if r['speaker']==s) for s in people),'queries':3},flush=True)
    print({'cold_ms':report['cold_teacher_load_and_warmup_ms'],'NN_max_ms':report['nn_32_people_8_vectors_median_ms']})
if __name__=='__main__':main()
