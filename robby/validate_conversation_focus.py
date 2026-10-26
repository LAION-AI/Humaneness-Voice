"""Real CPU ECAPA and live SentenceSession guard; public synthetic fixtures only."""
import asyncio,json,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
import numpy as np,soundfile as sf
from scipy.signal import resample_poly
import torch
torch.set_num_threads(4)
from speaker_teacher import get_cached_teacher,enrollment_reference
from sentence_session import SentenceSession,OpenVoiceMemory
from conversation_focus import ConversationFocus

ROOT=Path(__file__).resolve().parent
REF=ROOT.parent/'humaneness-small'
POOL=Path('/mnt/nvme/moss-15-v2-assets/refs3')

def read(path):
    x,sr=sf.read(path,dtype='float32',always_2d=True);x=x.mean(1)
    return resample_poly(x,16000//np.gcd(sr,16000),sr//np.gcd(sr,16000))

async def main():
    report={'passed':False,'physical_microphone_used':False,'source':'Existing public synthetic reference profiles; no private voiceprints',
            'cases':[],'guard_cases':[],'limitations':'Small proxy test, not a room/overlap or human evaluation. Emotion synthesis may itself change identity.'}
    teacher=get_cached_teacher();manifest=json.loads((REF/'fixed-references/selection.json').read_text())
    refs={r['voice']:r for r in manifest['candidates'] if r['voice'] in manifest['selected']}
    anchors={};timings=[]
    for voice,item in refs.items():
        wave=read(REF/item['wav']);anchors[voice]=teacher.embed(enrollment_reference(wave[:48000]))
    for voice in ('emolia_c1699','anime_088'):
        memory=OpenVoiceMemory();owner=memory.assign(anchors[voice],'Ich bin Testperson.')['speaker_id']
        focus=ConversationFocus();focus.accept(owner)
        index=json.loads((POOL/('index_'+voice+'.json')).read_text())
        chosen=[]
        for emotion in ('angry','happy','sad','neutral','excited','calm','confident'):
            matches=[(k,v) for k,v in index.items() if '|E|' in k and emotion in k and k.endswith('|de')]
            for key,paths in matches[:1]:
                path=next((POOL/p for p in paths if (POOL/p).exists()),None)
                if path:chosen.append((key,path))
        if not chosen:
            chosen=[(k,POOL/v[0]) for k,v in index.items() if '|V|' in k and k.endswith('|de') and (POOL/v[0]).exists()][:6]
        for label,path in chosen:
            wave=read(path)
            if len(wave)<20480:continue
            x=enrollment_reference(wave)
            started=time.perf_counter();v=teacher.embed(x);timings.append((time.perf_counter()-started)*1000)
            result=focus.check(v,memory.users)
            report['cases'].append({'voice':refs[voice]['name'],'condition':' | '.join(label.split('|')[2:]),
                'audio_seconds':round(len(x)/16000,3),'legacy_038_accepts':float(v@anchors[voice])>=.38,**result})
        for other,v in anchors.items():
            if other==voice:continue
            result=focus.check(v,memory.users,True)
            report['cases'].append({'voice':refs[voice]['name'],'condition':'background '+refs[other]['name'],
                'expected_background':True,**result})
    # Execute the actual asynchronous guard (including freshness/rid checks)
    # on held-out segments of each identity, rather than only the policy math.
    for name,voice in [('owner','emolia_c1699'),('background','anime_088'),('silence',None)]:
        c=SimpleNamespace(ws=SimpleNamespace(send_json=AsyncMock()),speaking=True,halting=False,
            session=SimpleNamespace(reset=lambda:None),ears=SimpleNamespace(sr=16000),
            playback=SimpleNamespace(rid=7,audio=np.zeros(0),n16=0),_on_barge=AsyncMock())
        s=SentenceSession(c);s.memory.assign(anchors['emolia_c1699'],'Ich bin Testperson.');s.focus.accept('speaker-1')
        wave=read(REF/refs[voice]['wav'])[48000:73600] if voice else np.zeros(25600,np.float32)
        s.records.append((0.,len(wave)/16000,wave));s.now=s.last_speech=len(wave)/16000
        start=time.perf_counter();s.propose_barge({'yield_prob':.99,'stop_text':'Stopp!'},len(wave))
        if s.barge_task:await s.barge_task
        accepted=c._on_barge.await_count>0
        report['guard_cases'].append({'case':name,'cancelled':accepted,'expected':name=='owner','wall_ms':round((time.perf_counter()-start)*1000,2)})
    report['teacher_cpu_ms']={'median':round(float(np.median(timings)),2) if timings else None,'n':len(timings)}
    # Difficult same-sex synthetic probes: execute the current short-window
    # guard, including existing native-hop student features when available.
    from speaker_live.live_test import audio
    cache=ROOT/'logs/focus-student-fixtures.npz'
    student=np.load(cache) if cache.exists() else None
    anchor=teacher.embed(enrollment_reference(audio('christoph-intro')))
    report['short_command_probes']=[]
    for name in ('christoph-stop','christoph-wait','christoph-halt','christoph-pause-clear','robert-stop','background-stop'):
        c=SimpleNamespace(ws=SimpleNamespace(send_json=AsyncMock()),speaking=True,halting=False,
            session=SimpleNamespace(reset=lambda:None),ears=SimpleNamespace(sr=16000),
            playback=SimpleNamespace(rid=8,audio=np.zeros(0),n16=0),_on_barge=AsyncMock())
        s=SentenceSession(c);s.memory.assign(anchor,'Ich bin Christoph.');s.focus.accept('speaker-1');s.reply_seen=8;s.reply_started=0.
        wave=audio(name);at=len(wave)/16000;s.records.append((0.,at,wave));s.last_speech=s.now=at
        if student is not None:
            from speaker_runtime import LivePredictor
            s.fast_center=LivePredictor().fast.teacher_mean.detach().numpy().copy()
            vectors=student['christoph-intro__embedding'];vectors=vectors[np.linalg.norm(vectors,axis=1)>.1]
            from sentence_session import unit
            s.fast_anchors['speaker-1']=[unit(np.mean(vectors,axis=0))]
            for i,v in enumerate(student[name+'__embedding']):
                if np.linalg.norm(v)>.1:s.fast_records.append(((i+1)*.32,v))
        started=time.perf_counter();s.propose_barge({'yield_prob':.99,'stop_text':'Stopp!'},len(wave));await s.barge_task
        receipt=c.ws.send_json.await_args_list[0].args[0]
        report['short_command_probes'].append({'case':name,'expected_owner':name.startswith('christoph'),
            'cancelled':bool(c._on_barge.await_count),'wall_ms':round((time.perf_counter()-started)*1000,2),
            'teacher_similarity':receipt.get('similarity'),'fast_similarity':receipt.get('fast_similarity'),
            'fast_consistency':receipt.get('fast_consistency'),'fast_frames':receipt.get('fast_frames'),
            'reason':receipt.get('reason')})
    background=[r for r in report['cases'] if r.get('expected_background')]
    emotion=[r for r in report['cases'] if not r.get('expected_background')]
    report['summary']={'background_rejected':sum(not r['accepted'] for r in background),'background_total':len(background),
        'emotion_accepted':sum(r['accepted'] for r in emotion),'emotion_total':len(emotion),
        'legacy_038_emotion_accepted':sum(r['legacy_038_accepts'] for r in emotion),
        'short_command_correct':sum(r['cancelled']==r['expected_owner'] for r in report['short_command_probes']),
        'short_command_total':len(report['short_command_probes'])}
    # Keep the difficult acoustic diagnostic separate from deterministic
    # implementation contracts; never relabel an impostor acceptance a pass.
    report['short_commands_all_correct']=all(r['cancelled']==r['expected_owner'] for r in report['short_command_probes'])
    report['known_rival_check']=[]
    for name in ('robert-stop','background-stop'):
        memory=OpenVoiceMemory();memory.assign(anchor,'Ich bin Christoph.')
        memory.assign(teacher.embed(enrollment_reference(audio(name))),'Hey Robbie, ich bin Hintergrund.')
        f=ConversationFocus();f.accept('speaker-1');query=teacher.embed(audio(name)[-25600:]);d=f.check(query,memory.users,True)
        report['known_rival_check'].append({'case':name,**d})
    report['passed']=(all(r['cancelled']==r['expected'] for r in report['guard_cases']) and all(not r['accepted'] for r in background)
        and all(not r['accepted'] for r in report['known_rival_check']))
    report['finished']=time.time();(ROOT/'static/conversation-focus-check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':asyncio.run(main())
