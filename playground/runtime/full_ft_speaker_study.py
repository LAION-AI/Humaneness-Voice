"""Resumable Full-FT-only identity/quality pilot, with a fresh-text validation.

Uses the existing server in a private cookie session. Never loads adapters,
changes defaults, replaces the frozen S3 study or discards bad generations.
"""
import argparse
import base64
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import time
from types import SimpleNamespace
import urllib.error
import uuid

import numpy as np

import settings
from prompting import REFERENCE_IDENTITY
from speaker_study import Client, VOICES, write_json, bootstrap

ROOT=settings.HERE/'full-ft-speaker-study'
CONDITIONS={
    'baseline':{'name':'Identity caption · CFG 1.5 · T 1.0','cfg':1.5,'temperature':1.,'identity':True,'ref_cap':15},
    'cooler':{'name':'Identity caption · CFG 1.5 · T 0.8','cfg':1.5,'temperature':.8,'identity':True,'ref_cap':15},
    'no_cfg':{'name':'Identity caption · CFG 1.0 · T 1.0','cfg':1.,'temperature':1.,'identity':True,'ref_cap':15},
    'strong_cfg':{'name':'Identity caption · CFG 2.0 · T 1.0','cfg':2.,'temperature':1.,'identity':True,'ref_cap':15},
    'affect_only':{'name':'Emotion-only caption · CFG 1.5 · T 1.0','cfg':1.5,'temperature':1.,'identity':False,'ref_cap':15},
    'short_ref':{'name':'Identity caption · reference capped at 5s','cfg':1.5,'temperature':1.,'identity':True,'ref_cap':5}}
DIAGNOSTICS={
    'natural_duration':{**CONDITIONS['baseline'],'name':'Identity caption · automatic duration','seconds':None},
    'offline':{**CONDITIONS['baseline'],'name':'Identity caption · non-streaming codec','streaming':False},
    'transcript_control':{**CONDITIONS['no_cfg'],'name':'Transcript-only control · no requested emotion','format':'transcript','identity':False},
    'script_affect':{**CONDITIONS['baseline'],'name':'GENERAL identity + SCRIPT emotion cues','format':'structured'}}
FOLLOWUP={
    'auto_cfg1':{**CONDITIONS['no_cfg'],'name':'Automatic duration · CFG 1.0 · T 1.0','seconds':None},
    'auto_cfg15':{**CONDITIONS['baseline'],'name':'Automatic duration · CFG 1.5 · T 1.0','seconds':None},
    'auto_cfg2':{**CONDITIONS['strong_cfg'],'name':'Automatic duration · CFG 2.0 · T 1.0','seconds':None}}


def configuration(condition):
    return (CONDITIONS|DIAGNOSTICS|FOLLOWUP)[condition]
EMOTIONS={
    'neutral':'Neutral, matter-of-fact conversation, steady pace and natural phrasing.',
    'joy':'Joyful relief, lively smiling phrasing and spontaneous emphasis; stay intelligible.',
    'anger':'Contained anger, firm clipped emphasis and tense phrasing; do not scream.',
    'sadness':'Sadness, slower phrasing, gentle emphasis and a restrained emotional catch.',
    'fear':'Fear, hesitant starts and restrained urgent phrasing; stay intelligible, do not scream.'}
EMOTION_KEYS={'joy':'eiv:Elation','anger':'eiv:Anger','sadness':'eiv:Sadness','fear':'eiv:Fear'}
TEXTS={
    'en':['You came back. I thought I would never see you again.',
          'The train leaves at seven. Please bring your ticket with you.',
          'I trusted you with this. Tell me what really happened.',
          'I found your letter this morning. I still remember every word.',
          'Wait here for a moment. I need to know that you are safe.'],
    'de':['Du bist wieder da. Ich dachte, ich würde dich nie wiedersehen.',
          'Der Zug fährt um sieben. Bitte bring deine Fahrkarte mit.',
          'Ich habe dir vertraut. Sag mir, was wirklich passiert ist.',
          'Ich habe heute deinen Brief gefunden. Ich erinnere mich an jedes Wort.',
          'Warte bitte einen Moment. Ich muss wissen, dass du in Sicherheit bist.']}
HOLDOUT_TEXTS={
    'en':['We can discuss the details tomorrow. For now, let us take a short break.',
          'You kept your promise. That means more to me than you can imagine.',
          'This was not our agreement. I want an honest answer right now.',
          'The room feels empty without you. I wish we had more time.',
          'There is someone behind the door. Stay close and do not make a sound.'],
    'de':['Wir besprechen die Einzelheiten morgen. Jetzt machen wir erst eine kurze Pause.',
          'Du hast dein Versprechen gehalten. Das bedeutet mir wirklich sehr viel.',
          'Das war nicht unsere Abmachung. Ich will jetzt eine ehrliche Antwort.',
          'Ohne dich fühlt sich das Zimmer leer an. Ich wünschte, wir hätten mehr Zeit.',
          'Hinter der Tür steht jemand. Bleib bei mir und mach kein Geräusch.']}
FOLLOWUP_TEXTS={
    'en':['The meeting starts at nine. I will send you the address tonight.',
          'I cannot believe we made it. Thank you for staying with me.',
          'You knew how important this was. Do not pretend it was a small mistake.',
          'I thought we would grow old together. Now I have to say goodbye.',
          'I heard footsteps in the hallway. Please tell me you locked the door.'],
    'de':['Die Besprechung beginnt um neun. Ich schicke dir heute Abend die Adresse.',
          'Ich kann kaum glauben, dass wir es geschafft haben. Danke, dass du geblieben bist.',
          'Du wusstest, wie wichtig das war. Tu nicht so, als wäre es ein kleiner Fehler.',
          'Ich dachte, wir würden gemeinsam alt werden. Jetzt muss ich Abschied nehmen.',
          'Ich habe Schritte im Flur gehört. Bitte sag mir, dass du die Tür abgeschlossen hast.']}


def blocks(phase):
    texts=TEXTS if phase=='pilot' else FOLLOWUP_TEXTS if phase=='followup' else HOLDOUT_TEXTS
    return [{'block':2*i+j,'seed':810001+1009*(2*i+j)+(900000 if phase=='followup' else 500000 if phase=='validation' else 0),
             'emotion':emotion,'language':lang,'text':texts[lang][i]}
            for i,emotion in enumerate(EMOTIONS) for j,lang in enumerate(('en','de'))]


def rows_for(phase,conditions):
    rows=[]
    for b in blocks(phase):
        for condition in conditions:
            c=configuration(condition)
            caption=(REFERENCE_IDENTITY+' ' if c['identity'] else '')+EMOTIONS[b['emotion']]
            prompt='CAPTION: '+caption+'\nTRANSCRIPT: '+json.dumps(b['text'],ensure_ascii=False)
            if c.get('format')=='transcript':prompt='TRANSCRIPT: '+json.dumps(b['text'],ensure_ascii=False)
            elif c.get('format')=='structured':
                prompt='GENERAL: '+REFERENCE_IDENTITY+'\nSCRIPT:\n('+EMOTIONS[b['emotion']]+') '+b['text']
            for voice in VOICES:
                rows.append({'id':f'{phase}-{voice}-{condition}-{b["block"]}',
                    'phase':phase,'condition':condition,'voice':voice,'prompt':prompt,**b})
    random.Random(101009 if phase=='pilot' else 101010).shuffle(rows)
    return rows


def design():
    value={'version':1,'model':settings.REPO,'revision':settings.REVISION,'stage':settings.FULL_FT_STAGE,
        'checkpoint_sha256':settings.FULL_FT_SHA256,'adapters':'NONE in every request',
        'conditions':CONDITIONS,'emotions':EMOTIONS,'pilot_trials':120,'trials_per_voice_condition':10,
        'rows':rows_for('pilot',CONDITIONS),'validation_blocks':blocks('validation'),
        'sampling':'One independently seeded take per request, incremental codec 1600ms; no live cutoff, fades, best-of-N, beam search, VC, adaptive retrieval or quality retries.',
        'reference_rule':'Fixed catalog clip per voice, uploaded separately per cap. Long condition up to 15s, short condition first 5s of exactly the same normalized clip; actual duration and codec frames recorded. Distinct held-out anchors never condition generation.',
        'selection_rule':{'wer_mean_max':.10,'each_voice_wer_max':.15,'catastrophic_rate_max':.10,
            'style_drop_max':.03,'quality_drop_max':.03,
            'quality':'0.5*CE/normalized + 0.25*PQ/normalized + 0.25*genuineness/6',
            'balanced_score':'harmonic mean of clipped conditioning ECAPA cosine [0,1] and quality [0,1]',
            'validation':'Selected condition and baseline on 20 fresh text/seed blocks (10 per voice); identical reference clips. Selection frozen before validation. If baseline wins, validation is baseline only, 20 trials.'},
        'reliability':'Full untrimmed PCM Parakeet WER, exact-text rate, WER>0.25 rate, stop-budget hits, low-identity tail and generation failures. A failure is retained, never replaced by a good take.',
        'limitations':['Two corpus voices; 5 emotions x 2 languages = 10 mixed-task repetitions per cell, not 10 per emotion.',
            'ASR, ECAPA, Ears and EIV+ are proxies, not human listening scores or calibrated probabilities.',
            'Reference and held-out anchors share corpus provenance; neither train/test nor recording-session independence is guaranteed.',
            'No DPOs, other LoRAs, model-to-model comparison or population-level optimality claim.',
            'Quality/WER use untrimmed raw PCM; offline ECAPA/emotion evaluate actual mono 96k MP3 transport.',
            'No automatic playground default changes. Reference length result is limited to two fixed clips.',
            'Validation has new text/seeds but the same two speakers/references; not a new-speaker generalization test.']}
    path=ROOT/'design.json'
    if path.exists():
        saved=json.loads(path.read_text());assert saved==value,'Frozen study design changed';return saved
    write_json(path,value);return value


def prepare_references():
    """Keep original sources private; do not pad/loop to invent long references."""
    import soundfile as sf
    folder=ROOT/'references';folder.mkdir(parents=True,exist_ok=True,mode=0o700)
    sources={voice:info['source'] for voice,info in VOICES.items()}
    # A different fixed catalog clip gives Idris genuine longer speech context.
    sources['idris']='/mnt/nvme/moss-15-v2-assets/refs3/audio/anime_088/anime_088__E__Interest__A__de.c026.mp3'
    provenance={}
    for voice,source in sources.items():
        long=folder/f'{voice}-15.wav'
        if not long.exists():
            subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',source,'-af',
                'silenceremove=start_periods=1:start_duration=0.02:start_threshold=-45dB',
                '-t','15','-ac','1','-ar','48000','-c:a','pcm_s16le',str(long)],check=True)
        data,sr=sf.read(long,dtype='float32');peak=abs(data).max()
        if peak>.95:sf.write(long,data*.95/peak,sr,subtype='PCM_16');data,sr=sf.read(long,dtype='float32')
        short=folder/f'{voice}-5.wav'
        if not short.exists():sf.write(short,data[:5*sr],sr,subtype='PCM_16')
        provenance[voice]={'source':source,'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            'long_seconds':len(data)/sr,'short_seconds':sf.info(short).duration}
    write_json(folder/'private-provenance.json',provenance)


def upload(client,path):
    boundary=uuid.uuid4().hex
    raw=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="study-reference.wav"\r\n'
         'Content-Type: audio/wav\r\n\r\n').encode()+path.read_bytes()+f'\r\n--{boundary}--\r\n'.encode()
    return json.loads(client.request('/api/reference?slot=A',raw,
        {'Content-Type':f'multipart/form-data; boundary={boundary}'}))


def verify_server(client):
    state=client.state()
    assert state['checkpoint']['sha256']==settings.FULL_FT_SHA256 and state['checkpoint']['step']==26774
    assert state['adapters']['default'] is None,'Unexpected live default; investigate before study'


def generate(base,phase):
    plan=design();prepare_references()
    if phase=='pilot':rows=plan['rows']
    elif phase=='diagnostic':
        rows=rows_for('pilot',DIAGNOSTICS)
        rows=[{**r,'phase':'diagnostic','id':r['id'].replace('pilot-','diagnostic-',1)} for r in rows]
        write_json(ROOT/'diagnostic-design.json',{'rows':rows,'conditions':DIAGNOSTICS,
            'exploratory_after_pilot':True,'reason':'Unexpectedly weak identity prompted codec, duration and prompt-format checks; not part of the original selection grid.'})
    elif phase=='followup':
        rows=rows_for(phase,FOLLOWUP)
        write_json(ROOT/'followup-design.json',{'rows':rows,'conditions':FOLLOWUP,
            'exploratory_after_pilot':True,'baseline':'auto_cfg15','selection':'Same eligibility rule, relative to the actual automatic-duration CFG1.5 baseline; maximize balanced score.',
            'reason':'Test the automatic-duration x CFG combination on another fresh set of texts and seeds; no data from this phase seen before freezing this design.'})
    else:
        selected=json.loads((ROOT/'selection.json').read_text())['selected']
        rows=rows_for('validation',list(dict.fromkeys(['baseline',selected])))
        write_json(ROOT/'validation-design.json',{'rows':rows,'selected':selected})
    client=Client(base);client.wait_idle();verify_server(client)
    current=None;item=None;started=time.monotonic()
    try:
        for i,row in enumerate(rows):
            target=ROOT/'takes'/(row['id']+'.json');audio=ROOT/'audio'/(row['id']+'.mp3')
            if target.exists():continue
            c=configuration(row['condition']);key=(row['voice'],c['ref_cap'])
            client.wait_idle();verify_server(client)
            if key!=current:
                if item:client.request('/api/reference/'+item['id'],method='DELETE');item=None
                item=upload(client,ROOT/'references'/f'{key[0]}-{key[1]}.wav')
                # Exact normalized waveform that actually conditions the server.
                shutil.copyfile(settings.DATA/(item['id']+'.wav'),ROOT/'references'/f'{key[0]}-{key[1]}-actual.wav')
                current=key
            body={'message':row['prompt'],'input_mode':'manual','format':c.get('format','caption'),'language':row['language'],
                'generation':'all','n':1,'streaming':c.get('streaming',True),'chunk_frames':20,'align':False,'identity_cfg':True,
                'voice_conversion':False,'speaker_steering':False,'temperature':c['temperature'],'guidance':c['cfg'],'seconds':c.get('seconds',6.5),
                'seed':row['seed'],'voice_mode':'custom','voice_profile':None,'reference_id':item['id'],
                'adapters':{'id':None,'ref15':False},'speaker':{'method':'off','weight':0},
                'ranking':{'mode':'ears_mix','model':'medium','metric':'ce','weight':.5}}
            tick=time.monotonic()
            try:
                try:events=client.turn(body)
                except urllib.error.HTTPError as exc:
                    if exc.code!=409:raise
                    client.wait_idle();events=client.turn(body)
                errors=[e for e in events if e['event']=='error']
                if errors:raise RuntimeError(str(errors))
                result=next(e for e in events if e['event']=='result')
                assert result['checkpoint']['sha256']==settings.FULL_FT_SHA256
                assert not result['adapters']['active'] and not result['adapters']['selection']['ref15']
                assert not result.get('speaker_steering', {}).get('active', False), 'Frozen study forbids activation steering'
                assert len(result['candidates'])==1
                candidate=result['candidates'][0]
                audio.parent.mkdir(parents=True,exist_ok=True);audio.write_bytes(client.request(candidate['url']))
                perf=result['performance'];frames=round(perf['raw_audio_seconds']*12.5)
                spec_frames=next(e['frames'] for e in events if e['event']=='direction')
                record={**row,'failed':False,'candidate':{k:v for k,v in candidate.items() if k!='url'},
                    'audio':'audio/'+audio.name,'audio_sha256':hashlib.sha256(audio.read_bytes()).hexdigest(),
                    'reference_context':result['reference']['tts_context'],'checkpoint':result['checkpoint'],
                    'adapter_report':result['adapters'],'performance':perf,'frames':frames,
                    'hit_frame_limit':frames>=min(settings.MAX_FRAMES+60,spec_frames+60),
                    'wall_seconds':time.monotonic()-tick}
            except Exception as exc:
                record={**row,'failed':True,'error':str(exc),'wall_seconds':time.monotonic()-tick}
            write_json(target,record)
            if (i+1)%10==0 or record['failed']:
                print(json.dumps({'phase':phase,'completed':i+1,'total':len(rows),'failed':record['failed'],
                    'elapsed_seconds':round(time.monotonic()-started)}),flush=True)
    finally:
        if item:client.request('/api/reference/'+item['id'],method='DELETE')
    write_json(ROOT/f'generation-complete-{phase}.json',{'trials':len(rows)})


def score(phase):
    import soundfile as sf
    import torch
    import torchaudio
    import torch.nn.functional as F
    from speaker_reward import SpeakerModels
    from dimension_predictors import DimensionPredictors
    from adaptive_reward import RewardPlan
    torch.set_num_threads(4)
    speakers=SpeakerModels(SimpleNamespace(retriever=SimpleNamespace(device=settings.SCORE_DEVICE)))
    speakers.prepare('ecapa')
    dimensions=DimensionPredictors(settings.SCORE_DEVICE)
    emotion_plan=RewardPlan(terms=[{'dimension':key} for key in EMOTION_KEYS.values()])
    dimensions.prepare(emotion_plan)

    def read(path):
        data,sr=sf.read(path,dtype='float32');data=data.mean(1) if data.ndim>1 else data
        return torchaudio.functional.resample(torch.as_tensor(data,device=settings.SCORE_DEVICE),sr,16000)

    @torch.inference_mode()
    def embed(path):
        wave=read(path)
        if len(wave)<8000:raise ValueError('Insufficient speech for ECAPA')
        vectors,_=speakers.embed([wave],'ecapa')
        return F.normalize(vectors,dim=-1)[0].cpu().numpy(),wave

    refs={};anchors={};controls={}
    for voice,info in VOICES.items():
        anchors[voice]=embed(Path(info['anchor']))[0]
        for cap in (5,15):refs[(voice,cap)]=embed(ROOT/'references'/f'{voice}-{cap}-actual.wav')[0]
    for voice in VOICES:
        other=next(v for v in VOICES if v!=voice)
        controls[voice]={str(cap):{'seconds':sf.info(ROOT/'references'/f'{voice}-{cap}-actual.wav').duration,
            'reference_to_own_anchor':float(refs[(voice,cap)]@anchors[voice]),
            'reference_to_other_anchor':float(refs[(voice,cap)]@anchors[other])} for cap in (5,15)}
    write_json(ROOT/'reference-controls.json',{'ecapa_repo':settings.SPEAKER_REPO,
        'ecapa_revision':settings.SPEAKER_REVISION,'controls':controls})
    files=sorted((ROOT/'takes').glob(phase+'-*.json'))
    for i,path in enumerate(files):
        row=json.loads(path.read_text());target=ROOT/'scores'/(row['id']+'.json')
        if row['failed']:continue
        if target.exists() and 'canonical_cosine' in json.loads(target.read_text()):continue
        vector,wave=embed(ROOT/row['audio']);voice=row['voice'];cap=configuration(row['condition'])['ref_cap']
        other=next(v for v in VOICES if v!=voice)
        emotions=dimensions.predict([wave],torch.zeros(1,1,device=settings.SCORE_DEVICE),emotion_plan)[0]
        target_key=EMOTION_KEYS.get(row['emotion'])
        write_json(target,{'id':row['id'],'embedding':vector.tolist(),
            'conditioning_cosine':float(vector@refs[(voice,cap)]),'heldout_cosine':float(vector@anchors[voice]),
            'canonical_cosine':float(vector@refs[(voice,15)]),
            'other_cosine':float(vector@anchors[other]),'identity_margin':float(vector@(anchors[voice]-anchors[other])),
            'emotion_scores':emotions,'emotion_target_raw':emotions.get(target_key),
            'emotion_target_wins_four':max(emotions,key=emotions.get)==target_key if target_key else None})
        if (i+1)%20==0:print(json.dumps({'scored_phase':phase,'completed':i+1,'total':len(files)}),flush=True)


def metric(row,key):
    if row['failed']:return {'wer':1.,'exact':0.,'bad_wer':1.,'quality':0.,'style':0.,'conditioning_cosine':0.,'heldout_cosine':0.,'ce':0.,'pq':0.,'genuine':0.}.get(key,0.)
    c=row['candidate'];raw=c['ears']['scores_raw']
    ce=float(np.clip((raw['audiobox:CE']-1)/9,0,1));pq=float(np.clip((raw['audiobox:PQ']-1)/9,0,1))
    special={'wer':c['wer'],'exact':float(c['wer']==0),'bad_wer':float(c['wer']>.25),
        'style':c['clap'],'ce':raw['audiobox:CE'],'pq':raw['audiobox:PQ'],
        'mos':raw['dnsmos:P808_MOS'],'genuine':c['genuineness'],
        'quality':.5*ce+.25*pq+.25*float(np.clip(c['genuineness']/6,0,1))}
    return float(special[key] if key in special else row[key])


def completed_rows(phase):
    rows=[]
    for path in sorted((ROOT/'takes').glob(phase+'-*.json')):
        row=json.loads(path.read_text())
        c=configuration(row['condition']);seconds=c.get('seconds',6.5)
        duration=seconds if seconds is not None else max(2.,len(row['text'].split())*.36)
        if not row['failed']:row['hit_frame_limit']=row['frames']>=min(settings.MAX_FRAMES+60,math.ceil(duration*12.5)+60)
        if not row['failed']:row.update(json.loads((ROOT/'scores'/(row['id']+'.json')).read_text()))
        rows.append(row)
    return rows


def summarize(phase):
    rows=completed_rows(phase);groups=defaultdict(list);cells=[]
    expected=design()['rows'] if phase=='pilot' else json.loads((ROOT/f'{phase}-design.json').read_text())['rows']
    assert len(rows)==len(expected),'Incomplete study; do not report as complete'
    for row in rows:groups[row['condition']].append(row)
    for condition,group in groups.items():
        values={key:[metric(r,key) for r in group] for key in
            ('wer','exact','bad_wer','quality','style','conditioning_cosine','canonical_cosine','heldout_cosine','ce','pq','genuine','mos')}
        cell={'condition':condition,'n':len(group),'failures':sum(r['failed'] for r in group),
            **{key:float(np.mean(v)) for key,v in values.items()},
            'stop_budget_rate':float(np.mean([r.get('hit_frame_limit',True) for r in group])),
            'speaker_p10':float(np.quantile(values['canonical_cosine'],.1)),
            'speaker_sd':float(np.std(values['canonical_cosine'],ddof=1)),
            'emotion_target_raw':float(np.mean([r['emotion_target_raw'] for r in group if not r['failed'] and r['emotion_target_raw'] is not None])),
            'emotion_fourway_rate':float(np.mean([r['emotion_target_wins_four'] for r in group if not r['failed'] and r['emotion_target_wins_four'] is not None])),
            'per_voice':{}}
        for voice in VOICES:
            pervoice=[r for r in group if r['voice']==voice]
            cell['per_voice'][voice]={key:float(np.mean([metric(r,key) for r in pervoice])) for key in values}
            vectors=np.asarray([r['embedding'] for r in pervoice if not r['failed']])
            cell['per_voice'][voice]['pairwise_cosine']=float((vectors@vectors.T)[np.triu_indices(len(vectors),1)].mean()) if len(vectors)>1 else None
        for key in ('conditioning_cosine','canonical_cosine','heldout_cosine','quality','wer'):
            blockmeans=[np.mean([metric(r,key) for r in group if r['block']==i]) for i in range(10)]
            cell[key+'_ci']=bootstrap(blockmeans)
        cells.append(cell)
    baseline_cells=cells if phase!='diagnostic' else json.loads((ROOT/'summary-pilot.json').read_text())['cells']
    baseline_id='auto_cfg15' if phase=='followup' else 'baseline'
    baseline=next(c for c in baseline_cells if c['condition']==baseline_id)
    baseline_rows=rows if phase!='diagnostic' else completed_rows('pilot')
    for c in cells:
        c['eligible']=(c['wer']<=.10 and c['bad_wer']<=.10 and c['failures']==0 and
            all(v['wer']<=.15 for v in c['per_voice'].values()) and
            c['style']>=baseline['style']-.03 and c['quality']>=baseline['quality']-.03)
        if c['condition']=='transcript_control':c['eligible']=False
        s=float(np.clip(c['canonical_cosine'],0,1));q=c['quality']
        c['balanced_score']=2*s*q/max(s+q,1e-9)
        c['paired_vs_baseline']={}
        for key in ('conditioning_cosine','canonical_cosine','heldout_cosine','quality','wer','style'):
            deltas=[np.mean([metric(r,key) for r in rows if r['condition']==c['condition'] and r['block']==i])-
                    np.mean([metric(r,key) for r in baseline_rows if r['condition']==baseline_id and r['block']==i]) for i in range(10)]
            c['paired_vs_baseline'][key]={'mean':float(np.mean(deltas)),'ci':bootstrap(deltas)}
    eligible=[c for c in cells if c['eligible']]
    selected=max(eligible,key=lambda c:c['balanced_score'])['condition'] if eligible else baseline_id
    summary={'phase':phase,'trials':len(rows),'selected':selected,'qualified':bool(eligible),'cells':cells,
        'recommended':selected if eligible else None,
        'created_utc':datetime.now(timezone.utc).isoformat()}
    write_json(ROOT/f'summary-{phase}.json',summary)
    if phase=='pilot':
        write_json(ROOT/'selection.json',{'selected':selected,'qualified':bool(eligible),
            'frozen_before_validation':True,'rule':design()['selection_rule']})
    write_json(ROOT/f'results-{phase}.json',rows)
    print(json.dumps({'summary_phase':phase,'selected':selected,'cells':cells}),flush=True)
    return summary


def report():
    plan=design();pilot=json.loads((ROOT/'summary-pilot.json').read_text())
    validation=json.loads((ROOT/'summary-validation.json').read_text()) if (ROOT/'summary-validation.json').exists() else None
    diagnostic=json.loads((ROOT/'summary-diagnostic.json').read_text()) if (ROOT/'summary-diagnostic.json').exists() else None
    followup=json.loads((ROOT/'summary-followup.json').read_text()) if (ROOT/'summary-followup.json').exists() else None
    rows=completed_rows('pilot')+(completed_rows('validation') if validation else [])+(completed_rows('diagnostic') if diagnostic else [])+(completed_rows('followup') if followup else [])
    esc=html.escape
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Full-FT speaker identity, quality and reliability</title>',
        '<style>body{font:16px system-ui;max-width:1250px;margin:auto;padding:20px;background:#101820;color:#e4edf2}a{color:#82d6ff}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #456;text-align:left}thead{background:#20323f}article,section{padding:14px;background:#182630;border-radius:10px;margin:15px 0}audio{width:100%;max-width:600px}pre{white-space:pre-wrap;overflow-wrap:anywhere}select{max-width:100%;margin:8px;padding:8px}.table{overflow:auto}p{line-height:1.5}[hidden]{display:none!important}</style>',
        '<h1>New Full-FT checkpoint: speaker similarity vs quality</h1>',
        '<p><b>Step 26,774, no DPO, no other LoRAs.</b> Two fixed corpus voices, ten independently seeded mixed-task trials per voice/setting. The 120-take pilot tests six settings with a fixed 6.5-second duration hint, five emotions × English/German, and five bilingual texts. Additional fresh-text validation (40), exploratory prompt/codec/duration diagnostics (80), and fresh automatic-duration follow-up (60) bring the total to 300 takes. The automatic-duration follow-up is shown first. All takes and failures count; no best-of-N selection. Ten trials span mixed tasks, not ten repetitions of each emotion.</p>']
    summaries=([followup] if followup else [])+[pilot]+([validation] if validation else [])+([diagnostic] if diagnostic else [])
    for summary in summaries:
        parts.append(f'<h2>{summary["phase"].title()} — {summary["trials"]} takes</h2><div class="table"><table><thead><tr><th>Setting</th><th>Fixed-reference cosine</th><th>Held-out cosine</th><th>Ref p10</th><th>CE / PQ</th><th>Genuine</th><th>WER</th><th>Exact text</th><th>WER &gt;25%</th><th>CLAP</th><th>Emotion target / four-way</th><th>Eligible</th></tr></thead><tbody>')
        for c in sorted(summary['cells'],key=lambda c:-c['balanced_score']):
            parts.append(f'<tr><td>{esc(configuration(c["condition"])["name"])}</td><td>{c["canonical_cosine"]:.3f}</td><td>{c["heldout_cosine"]:.3f}</td><td>{c["speaker_p10"]:.3f}</td><td>{c["ce"]:.2f} / {c["pq"]:.2f}</td><td>{c["genuine"]:.2f}</td><td>{c["wer"]:.1%}</td><td>{c["exact"]:.0%}</td><td>{c["bad_wer"]:.0%}</td><td>{c["style"]:.3f}</td><td>{c["emotion_target_raw"]:.2f} / {c["emotion_fourway_rate"]:.0%}</td><td>{"Yes" if c["eligible"] else "No"}</td></tr>')
        parts.append('</tbody></table></div>')
    if followup:
        parts.append('<section><h2>Practical conclusion: reliability-first vs identity-first</h2><p><b>For general-purpose emotional speech, keep temperature 1.0, automatic duration, a concise identity Caption and CFG 1.5.</b> In the fresh automatic-duration follow-up, CFG 1.5 has the lowest mean WER (1.25%, 85% exact text), stronger emotion intensity than CFG 1.0, and similar Ears quality. CFG 1.0 improves reference cosine from 0.219 to 0.260, but WER rises to 5.30%; it is an identity-priority alternative, not a clear all-round winner. CFG 2.0 has WER 6.53% and no clear quality advantage. The equal-weight harmonic utility selects CFG 1.0; choosing CFG 1.5 for practical reliability is an explicit different preference, not a relabeling of the declared statistical winner.</p><p><b>High-fidelity cloning was not achieved on these fixed references.</b> At CFG 1.5, mean reference cosine is 0.383 for Idris and 0.055 for Selma; CFG 1.0 reaches 0.449 and 0.072. Low reference-to-anchor controls and the two-voice scope prevent a general verdict on the released model. Output identity is not reliably rescued by a longer reference or by turning codec streaming off. The measured quality scores are learned proxies, not a blind listening assessment. These are individual takes; this is not a validated Best-of-3 ranking study. No live defaults were changed.</p><p><a href="summary-followup.json">Fresh automatic-duration statistics, per-voice results and paired intervals</a> · <a href="followup-design.json">Follow-up design frozen before these generations</a></p></section>')
    parts.append('<h2>Per-voice identity and reliability</h2><div class="table"><table><thead><tr><th>Phase</th><th>Setting</th><th>Voice</th><th>Fixed-ref cosine</th><th>Held-out cosine</th><th>Within-cell pairwise cosine</th><th>WER</th></tr></thead><tbody>')
    for summary in summaries:
        for c in summary['cells']:
            for voice,v in c['per_voice'].items():
                parts.append(f'<tr><td>{summary["phase"]}</td><td>{esc(configuration(c["condition"])["name"])}</td><td>{voice}</td><td>{v["canonical_cosine"]:.3f}</td><td>{v["heldout_cosine"]:.3f}</td><td>{v["pairwise_cosine"]:.3f}</td><td>{v["wer"]:.1%}</td></tr>')
    parts.append('</tbody></table></div>')
    winner=pilot['selected']
    parts.append(f'<section><h2>Declared compromise and independent validation</h2><p>Pilot selection: <b>{esc(CONDITIONS[winner]["name"])}</b>. This is a pilot recommendation, not a universal optimum. Selection requires mean WER ≤10%, each voice WER ≤15%, ≤10% WER&gt;25% outliers, no generation failures, and no meaningful proxy style/quality drop vs baseline. It then maximizes the harmonic mean of fixed-reference cosine and a fixed quality index. All summary statistics, paired bootstrap intervals and per-voice results are downloadable as JSON. Live defaults were not changed.</p>')
    if validation:parts.append(f'<p>Validation uses {validation["trials"]} new text/seed cases with the same speakers. '+('At least one setting passes the declared reliability criteria.' if validation['qualified'] else '<b>No setting passes the declared reliability criteria on validation. The pilot choice is not confirmed as a reliable optimum.</b>')+' Pilot and validation are reported separately; no validation samples were used to choose the pilot winner.</p>')
    parts.append('</section><h2>What the numbers mean</h2><p>ECAPA cosine to a fixed long reference per speaker is primary, so the measurement target does not change in the 5-second conditioning test. Cosine to the actual conditioning WAV is also recorded separately. A separate same-speaker anchor, never used for synthesis, exposes overfitting to one recording; p10 and pairwise output cosine describe identity tails and consistency. Cosine is not a calibrated identity probability. WER is measured on untrimmed generated PCM, so unwanted introductions, repetition and tails remain visible rather than being hidden by forced alignment. CE and PQ are Humaneness Ears Medium regression proxies on a nominal 1–10 scale. Genuineness is the existing commercial head (0–6), not human MOS. VoiceCLAP measures the same emotion-only direction after identity boilerplate removal. EIV+ target intensity and argmax among four heads provide a separate, uncalibrated emotional-performance diagnostic; neutral trials are excluded from those two columns.</p><h2>Reference controls</h2><pre>'+esc(json.dumps(json.loads((ROOT/'reference-controls.json').read_text()),indent=2))+'</pre><h2>Method, uncertainty and limits</h2><ul>')
    for limitation in plan['limitations']:parts.append('<li>'+esc(limitation)+'</li>')
    if (ROOT/'backend-check.json').exists():
        parts.append('<li>Instrumentation controls reproduced two stored ECAPA results to within 3e-8. Two new-checkpoint greedy 40-frame prefix checks gave identical fast eager/graph codes, but not bit-identical legacy/fused codes. Prefix speaker scores were weak in both paths; this small check does not prove full stochastic or long-sequence parity, nor isolate whether the checkpoint, references or inference setup causes weak cloning. <a href="measurement-sanity.json">ECAPA sanity</a> · <a href="backend-check.json">Decoder sanity</a></li>')
    parts.append('</ul><p>Matched bootstrap resamples ten task/seed blocks, keeping both voices together (10,000 resamples). These exploratory intervals are not multiple-comparison-corrected. The fixed scalar blend is a declared preference, not a discovered perceptual law. Higher consistency does not prove better reference cloning. Short references are genuine prefixes of the same normalized clip; neither padding nor looping creates fake context. Idris has only 4.7 seconds of usable reference in both cap conditions; reference-length effects therefore apply primarily to Selma.</p><p>Protocol correction, recorded before any aggregate analysis or selection: use a fixed long-reference identity vector across conditions rather than changing the measurement target with the short-reference test. No generation condition changed. <a href="protocol-amendment.json">Amendment</a> · <a href="design.json">Original frozen design</a> · <a href="summary-pilot.json">Pilot statistics and intervals</a> · <a href="summary-validation.json">Validation statistics</a>.</p><p>Primary sources: <a href="https://huggingface.co/laion/Humaneness-Voice-Small/blob/main/docs/S3_REF15_FULL_FT.md">Full-FT release</a> · <a href="https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb">ECAPA</a> · <a href="https://huggingface.co/laion/humaneness-ears-base-medium">Humaneness Ears</a> · <a href="https://huggingface.co/laion/Empathic-Insight-Voice-Plus">EIV+</a>.</p><h2>Listen to every generated take — embedded Base64 MP3</h2><div id="filters">')
    for key,options in [('phase',{'pilot':'Pilot','validation':'Validation','diagnostic':'Exploratory diagnostics','followup':'Automatic-duration follow-up'}),('voice',{k:v['name'] for k,v in VOICES.items()}),('condition',{k:v['name'] for k,v in (CONDITIONS|DIAGNOSTICS|FOLLOWUP).items()}),('emotion',{k:k.title() for k in EMOTIONS}),('language',{'en':'English','de':'German'})]:
        parts.append(f'<label>{key.title()} <select data-filter="{key}"><option value="">All</option>'+''.join(f'<option value="{k}">{esc(v)}</option>' for k,v in options.items())+'</select></label>')
    parts.append('</div><p id="count"></p>')
    for row in rows:
        attrs=' '.join(f'data-{k}="{row[k]}"' for k in ('phase','voice','condition','emotion','language'))
        parts.append(f'<article {attrs}><h3>{esc(row["id"])}</h3>')
        if row['failed']:parts.append('<p>Generation failed: '+esc(row['error'])+'</p>')
        else:
            encoded=base64.b64encode((ROOT/row['audio']).read_bytes()).decode()
            parts.append(f'<p>Fixed-ref cosine {row["canonical_cosine"]:.3f} · actual-conditioning {row["conditioning_cosine"]:.3f} · held-out {row["heldout_cosine"]:.3f} · WER {row["candidate"]["wer"]:.1%} · CE {metric(row,"ce"):.2f}</p><audio controls preload="none" src="data:audio/mpeg;base64,{encoded}"></audio><details><summary>Exact prompt and recognition</summary><pre>{esc(row["prompt"])}</pre><p>ASR: {esc(row["candidate"]["hyp"])}</p></details>')
        parts.append('</article>')
    parts.append('<script>const filters=[...document.querySelectorAll("[data-filter]")],items=[...document.querySelectorAll("article")];function filter(){let n=0;for(const item of items){item.hidden=filters.some(s=>s.value&&item.dataset[s.dataset.filter]!==s.value);if(!item.hidden)n++;}document.getElementById("count").textContent=n+" of "+items.length+" takes shown";}filters.forEach(s=>s.onchange=filter);filter();</script></html>')
    temporary=ROOT/'index.partial';temporary.write_text('\n'.join(parts));temporary.replace(ROOT/'index.html')
    print(json.dumps({'report':str(ROOT/'index.html'),'embedded_takes':len(rows),'pilot_choice':winner}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['design','generate','score','summarize','report','all'])
    parser.add_argument('--phase',choices=['pilot','validation','diagnostic','followup'],default='pilot')
    parser.add_argument('--base',default='http://127.0.0.1:8792');args=parser.parse_args()
    if args.action=='design':print(json.dumps(design()))
    if args.action in ('generate','all'):generate(args.base,args.phase)
    if args.action in ('score','all'):score(args.phase)
    if args.action in ('summarize','all'):summarize(args.phase)
    if args.action in ('report','all'):report()


if __name__=='__main__':main()
