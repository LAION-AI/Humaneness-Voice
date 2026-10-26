"""Reproducible three-voice stress mixtures; bounded RAM, no private recordings.

Component diagnostics are separate from the real browser conversation tests.
Optional --asr calls the running loopback Nemotron on one free research slot.
Music/foley are procedurally authored here, with no downloaded media.
"""
import argparse,asyncio,json,time,hashlib,re,io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock
import httpx,numpy as np,torch
from scipy.signal import resample_poly
torch.set_num_threads(4)
from speaker_live.live_test import audio
from speaker_teacher import get_cached_teacher,enrollment_reference
from sentence_session import SentenceSession,voice_level,unit
ROOT=Path(__file__).resolve().parent

def normalized(x,rms=.055):
    return x*(rms/max(1e-8,float(np.sqrt(np.mean(x*x)))))
def music(n):
    t=np.arange(n)/16000
    x=sum(np.sin(2*np.pi*f*t+.1*i)*(1+.15*np.sin(2*np.pi*.6*t)) for i,f in enumerate((130.813,164.814,196.,261.626)))
    return normalized((x*(.55+.45*np.sin(np.pi*1.2*t)**2)).astype(np.float32))
def foley(n,seed=42):
    rng=np.random.default_rng(seed);x=np.zeros(n,np.float32)
    for second in (.4,1.9,3.6,4.7):
        at=round(second*16000);size=min(3200,n-at)
        if size<=0:continue
        t=np.arange(size)/16000
        x[at:at+size]+=(rng.normal(0,.15,size)+np.sin(2*np.pi*180*t))*np.exp(-t/0.035)
    return normalized(x) if np.any(x) else x
def repeat(x,n,offset=0):return x[(np.arange(n)+offset)%len(x)]
def words(t):return re.findall(r'\w+',t.casefold())
def wer(reference,hypothesis):
    a=words(reference);b=words(hypothesis);row=list(range(len(b)+1))
    for i,w in enumerate(a,1):
        nxt=[i]
        for j,v in enumerate(b,1):nxt.append(min(row[j]+1,nxt[-1]+1,row[j-1]+(w!=v)))
        row=nxt
    return row[-1]/max(1,len(a))

async def main(with_asr=False,speaker='christoph',seed=42,threshold=.28,pause_ms=700):
    teacher=get_cached_teacher();anchor=audio(speaker+'-intro')
    reference=teacher.embed(enrollment_reference(anchor))
    bg1=audio('background');bg2=audio('selma-followup-clear' if speaker=='christoph' else 'christoph-followup')
    source=audio('christoph-followup' if speaker=='christoph' else 'selma-followup-clear')
    reference_text='Kannst du mir eine kleine Geschichte erzählen?' if speaker=='christoph' else 'Was können wir heute zusammen machen?'
    specs=[('clean',0.,0.,0.,0.,1.),
           ('quiet TV',.06,0.,0.,0.,1.),
           ('TV at 12 percent',.12,0.,0.,0.,1.),
           ('two quiet people',.08,.08,0.,0.,1.),
           ('two people + soft music + foley',.10,.10,.04,.03,1.),
           ('changed delivery + two people',.10,.08,.04,.03,1.04),
           ('louder overlapping people',.20,.16,.05,.04,1.),
           ('hard overlap + music',.35,.25,.10,.06,1.)]
    report={'passed':False,'target_voice':speaker,'augmentation_seed':seed,'scope':'Component ASR/foreground-clock/ECAPA/V4 diagnostics; browser proves full conversational integration separately',
        'private_audio':False,'procedural_music_and_foley':True,'retained_waveforms':0,
        'configuration':{'focus_threshold':threshold,'speech_pause_ms':pause_ms},'cases':[]}
    async with httpx.AsyncClient(timeout=60) as client:
        async def fast_vector(wave):
            response=await client.post('http://127.0.0.1:8793/internal/speaker-predictions',content=wave.astype('<f4').tobytes());response.raise_for_status()
            with np.load(io.BytesIO(response.content)) as z:
                vectors=z['embedding'];vectors=vectors[np.linalg.norm(vectors,axis=1)>.1]
                return unit(vectors.mean(0)) if len(vectors) else None
        fast_anchor=await fast_vector(anchor) if with_asr else None
        for index,(name,b1,b2,m,fx,speed) in enumerate(specs):
            main=resample_poly(source,100,round(100*speed)) if speed!=1 else source.copy()
            energy=np.sqrt(np.convolve(main.astype(np.float64)**2,np.ones(320)/320,mode='same'))
            active=np.flatnonzero(energy>max(1e-4,float(energy.max())*.05))
            source_begin=(int(active[0])/16000) if len(active) else 0.
            source_end=(int(active[-1])/16000) if len(active) else len(main)/16000
            start=10240;n=start+len(main)+24000
            target=np.zeros(n,np.float32);target[start:start+len(main)]=main
            # The second background person has pauses but overlaps the first
            # and the target often; this is not a disguised sequential test.
            gate=(np.sin(np.arange(n)/16000*2*np.pi*.42)>-.25).astype(np.float32)
            distractors=repeat(bg1,n,index*2831+(seed-42)*31)*b1+repeat(bg2,n,index*1937+(seed-42)*17)*b2*gate+music(n)*m+foley(n,seed+index)*fx
            wave=np.clip(target+distractors,-.95,.95)
            c=SimpleNamespace(speaking=False,ws=SimpleNamespace(send_json=AsyncMock()),session=SimpleNamespace(reset=Mock()))
            s=SentenceSession(c);s.memory.assign(reference,'Ich bin '+speaker.title()+'.');s.focus.accept('speaker-1')
            s.focus.threshold=threshold;s.set_pause(pause_ms)
            s.memory.users['speaker-1']['level']=voice_level(anchor);s.begin=0.
            began=time.perf_counter()
            for at in range(0,n,320):await s._speech_audio(wave[at:at+320],min(n,at+320)/16000)
            begin=s.final_audio_start(0.);end=s.final_audio_end(n/16000);crop=wave[round(begin*16000):round(end*16000)]
            fast=None
            if fast_anchor is not None:
                from speaker_runtime import LivePredictor
                s.fast_center=LivePredictor().fast.teacher_mean.detach().numpy().copy();s.fast_anchors['speaker-1']=[fast_anchor]
                fast=await fast_vector(crop)
            vector=teacher.embed(enrollment_reference(crop));decision=s.turn_decision(vector,fast,level=voice_level(crop))
            bg_decision=None
            if np.sqrt(np.mean(distractors*distractors))>1e-5:
                q=teacher.embed(enrollment_reference(distractors))
                bg_decision=s.turn_decision(q,level=voice_level(distractors))
            record={'case':name,'background_gains':[b1,b2],'music_gain':m,'foley_gain':fx,'delivery_speed':speed,
                'target_seconds':len(main)/16000,'mixture_seconds':n/16000,'foreground_start_seconds':round(begin,3),'foreground_crop_seconds':round(end,3),
                'target_audio_retained':end>=start/16000+source_end-.12 and begin<=start/16000+source_begin+.08,
                'source_activity_support_seconds':[round(source_begin,3),round(source_end,3)],
                'retention_definition':'Known source energy support, excluding source padding; not an oracle word aligner. ASR WER is reported separately.',
                'target_similarity':decision.get('similarity'),'current_accepts_main':decision['accepted'],
                'previous_032_cosine_accepts_main':float(vector@reference)>=.32,'fast_similarity':decision.get('fast_similarity'),
                'background_accepted':bg_decision['accepted'] if bg_decision else False,
                'background_reason':bg_decision.get('reason') if bg_decision else None,
                'cpu_component_ms':round((time.perf_counter()-began)*1000,2),
                'wave_sha256':hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest(),
                'working_audio_bytes':sum(x.nbytes for x in (target,distractors,wave,crop))}
            if with_asr:
                t=time.perf_counter();response=await client.post('http://127.0.0.1:8793/internal/transcribe',content=crop.astype('<f4').tobytes())
                response.raise_for_status();transcript=response.json()['text']
                record.update(transcript=transcript,wer=round(wer(reference_text,transcript),3),
                    asr_ms=round((time.perf_counter()-t)*1000,2))
            report['cases'].append(record)
    report['passed']=all(r['target_audio_retained'] and r['current_accepts_main'] and not r['background_accepted'] for r in report['cases'][:5])
    report['all_stress_cases_pass']=all(r['target_audio_retained'] and r['current_accepts_main'] and not r['background_accepted'] for r in report['cases'])
    report['finished']=time.time();(ROOT/'static'/('conversation-stress-check.json' if speaker=='christoph' else 'conversation-stress-selma.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False))
    # Hard overlap results are measured limits, never silently relabelled.
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--asr',action='store_true');p.add_argument('--speaker',choices=['christoph','selma'],default='christoph');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threshold',type=float,default=.28);p.add_argument('--pause-ms',type=int,default=700)
    a=p.parse_args()
    if not .20<=a.threshold<=.65 or not 200<=a.pause_ms<=2000:p.error('Settings outside the supported UI ranges')
    asyncio.run(main(a.asr,a.speaker,a.seed,a.threshold,a.pause_ms))
