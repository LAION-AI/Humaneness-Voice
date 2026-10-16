"""Frozen, resumable single-take prompting x DPO speaker-identity pilot.

Reuses the live GPU server without stopping it, in an isolated cookie session.
All takes are retained. Only synthetic MP3s and sanitized metadata are public.
"""
import argparse
import base64
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
import http.cookiejar
import json
from pathlib import Path
import random
import shutil
import time
import urllib.error
import urllib.request
import uuid

import numpy as np

import settings
from metrics import word_errors
from prompting import build, parse_literal

ROOT=settings.HERE/'speaker-study'
DPO='subset_pitch_quality_p2_rank016'
EMOTIONS={
    'neutral':'Neutral emotion, matter-of-fact delivery and steady timing.',
    'joy':'Joyful emotion, excited phrasing and lively emphasis.',
    'anger':'Angry emotion, firm emphasis and clipped phrasing; do not scream.',
    'sadness':'Sad emotion, slower phrasing and gentle emphasis.',
    'fear':'Fearful emotion, hesitant phrasing and urgent emphasis.'}
TEXTS={'en':'You came back. I thought I would never see you again.',
       'de':'Du bist wieder da. Ich dachte, ich würde dich nie wiedersehen.'}
PROMPTS={
    'transcript':'Transcript only (no requested emotion)',
    'caption_affect':'Caption: emotion/prosody only',
    'script_affect':'GENERAL neutral; SCRIPT emotion in parentheses',
    'caption_gender':'Caption: broad perceived gender + emotion',
    'caption_traits':'Caption: perceived gender/age/timbre + emotion',
    'caption_reference':'Caption: explicit same-speaker instruction + emotion'}
VOICES={
    'idris':{'name':'Idris','profile':'anime_088','gender':'Male voice.',
        'traits':'A middle-aged male voice with an even timbre.',
        'source':'/mnt/nvme/moss-15-v2-assets/refs3/audio/anime_088/anime_088__V__RANG__moderately_low__en.c037.mp3',
        'anchor':'/mnt/nvme/moss-15-v2-assets/refs2/pilot/anime_088/reference.wav'},
    'selma':{'name':'Selma','profile':'emolia_c0542','gender':'Female voice.',
        'traits':'A female voice in her twenties with a bright timbre.',
        'source':'/mnt/nvme/moss-15-v2-assets/refs3/audio/emolia_c0542/emolia_c0542__E__Concentration__C__en.c008.mp3',
        'anchor':'/mnt/nvme/moss-15-v2-assets/refs2/pilot/emolia_c0542/reference.wav'}}


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(path)


def prompt_for(condition,voice,emotion,lang):
    text=TEXTS[lang];direction=EMOTIONS[emotion]
    if condition=='transcript':return 'transcript','TRANSCRIPT: '+json.dumps(text,ensure_ascii=False)
    if condition=='script_affect':
        first,rest=text.split('. ',1)
        return 'structured',f'GENERAL: Natural conversational speech.\nSCRIPT:\n({direction}) {first}.\n({direction}) {rest}'
    if condition=='caption_gender':direction=VOICES[voice]['gender']+' '+direction
    elif condition=='caption_traits':direction=VOICES[voice]['traits']+' '+direction
    elif condition=='caption_reference':direction='Use the same speaker as the reference audio. Preserve that speaker\'s voice identity. '+direction
    elif condition!='caption_affect':raise ValueError(condition)
    return 'caption','CAPTION: '+direction+'\nTRANSCRIPT: '+json.dumps(text,ensure_ascii=False)


def make_design():
    blocks=[{'index':i,'seed':73001+97*i,'emotion':emotion,'language':lang}
            for i,(emotion,lang) in enumerate((e,l) for e in EMOTIONS for l in TEXTS)]
    rows=[];rng=random.Random(10408)
    # Counterbalanced adapter order within seed blocks, then shuffle voices and
    # prompt variants; no score is used to choose run order or examples.
    for block in blocks:
        for adapter in (['base','dpo'] if block['index']%2==0 else ['dpo','base']):
            cells=[(voice,condition) for voice in VOICES for condition in PROMPTS];rng.shuffle(cells)
            for voice,condition in cells:
                fmt,prompt=prompt_for(condition,voice,block['emotion'],block['language'])
                parsed=parse_literal(prompt,fmt)
                spec=build(parsed,fmt,block['language'],seconds=6.,literal=prompt)
                rows.append({'id':f'{voice}-{adapter}-{condition}-s{block["seed"]}',
                    'voice':voice,'adapter':adapter,'condition':condition,**block,'format':fmt,'prompt':prompt,'spec':spec})
    return {'created_utc':datetime.now(timezone.utc).isoformat(),'version':1,
        'model':settings.REPO,'revision':settings.REVISION,'stage':'S3',
        'dpo_repo':settings.DPO_REPO,'dpo_revision':settings.DPO_REVISION,'dpo_adapter':DPO,'dpo_strength':1,
        'ref15_enabled':False,'cfg':1.5,'temperature':1.,'duration_hint':6.,
        'generation':'single; n=1; no streaming, search, reward selection, retries for quality or output filtering',
        'alignment':False,'reference_cap':15,'conditions':PROMPTS,'emotions':EMOTIONS,'texts':TEXTS,
        'voices':{k:{field:v[field] for field in ('name','profile','gender','traits')} for k,v in VOICES.items()},
        'trials_per_voice_adapter_condition':10,'total_trials':len(rows),'blocks':blocks,'rows':rows,
        'reference_rule':'Fixed separate catalog clip per speaker, capped at 15s by production pipeline; a second fixed profile anchor is never used for generation. No target recording, looping or adaptive reference retrieval.',
        'speaker_measurement':'Raw-PCM production ECAPA cosine to conditioning clip; independent offline pinned ECAPA on mono-96k MP3, anti-aliased 16k resampling, compared with conditioning clip and held-out same-speaker anchor. Mean pairwise output cosine per cell is a separate consistency proxy.',
        'ranking_rule':'Among emotion-capable conditions with pooled mean conventional raw WER <=0.10 and each speaker mean WER <=0.20, maximize held-out-anchor ECAPA cosine. Transcript-only is a separate no-emotion control. Report failed eligibility explicitly. No joint naturalness/identity optimum claim.',
        'uncertainty':'Matched bootstrap over the ten emotion/language/seed blocks, keeping both speakers together (10000 resamples). Two speakers and one bilingual scene only; exploratory intervals do not establish population generalization. Ten observations vary emotion/language; not ten repeats of every individual emotion.',
        'limitations':['Perceived gender/age/timbre hints come from existing corpus profile-trait predictions, not verified demographic facts.',
            'Standard production CFG neutral branch removes caption voice hints; effects are conditional on CFG=1.5, not a test of identity-preserving CFG.',
            'Transcript-only makes both CFG branches equal: effective guidance=1.0 even though configured CFG=1.5.',
            'Emotion realization is not independently measured; captions vary, so CLAP scores must not be treated as comparable emotion success scores.',
            'Primary held-out metrics use the actual MP3 transport, while production conditioning-reference cosine uses raw PCM; compression is identical across conditions.',
            'Same-speaker anchors are distinct files, but corpus derivation/session independence is not guaranteed. ECAPA is a proxy, not a listener identity verdict.']}


def design():
    path=ROOT/'design.json'
    if path.exists():
        saved=json.loads(path.read_text())
        expected=make_design()
        assert saved['rows']==expected['rows'],'Frozen study differs from current design; use a new directory'
        return saved
    result=make_design();write_json(path,result);return result


class Client:
    def __init__(self,base):
        self.base=base
        self.opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    def request(self,path,data=None,headers=None,method=None):
        req=urllib.request.Request(self.base+path,data,headers or {},method=method)
        with self.opener.open(req,timeout=180) as response:return response.read()
    def state(self):return json.loads(self.request('/api/state'))
    def wait_idle(self):
        while True:
            state=self.state()
            if state['error']:raise RuntimeError(state['error'])
            if state['ready'] and not state['busy']:return
            print(json.dumps({'status':'Waiting for active live-demo user or startup','phase':state['job_phase']}),flush=True)
            time.sleep(5)
    def upload(self,voice,slot):
        source=Path(VOICES[voice]['source']);boundary=uuid.uuid4().hex
        body=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="speaker-study-{voice}.mp3"\r\n'
              'Content-Type: audio/mpeg\r\n\r\n').encode()+source.read_bytes()+f'\r\n--{boundary}--\r\n'.encode()
        item=json.loads(self.request('/api/reference?slot='+slot,body,{'Content-Type':f'multipart/form-data; boundary={boundary}'}))
        private=ROOT/'references';private.mkdir(parents=True,exist_ok=True,mode=0o700)
        # Only this isolated session's newly uploaded WAV; never user uploads.
        shutil.copyfile(settings.DATA/(item['id']+'.wav'),private/(voice+'.wav'))
        return item
    def turn(self,body):
        return [json.loads(line[6:]) for line in self.request('/api/turn',json.dumps(body).encode(),
            {'Content-Type':'application/json'}).decode().splitlines() if line.startswith('data: ')]


def generate(base):
    plan=design();client=Client(base);client.wait_idle()
    if client.state().get('stage') != 'S3':
        raise RuntimeError('Frozen speaker study requires original S3; start a comparison server with SMALL_STAGE=S3.')
    uploads={};started=time.monotonic()
    try:
        for voice,slot in zip(VOICES,('A','B')):uploads[voice]=client.upload(voice,slot)
        for i,row in enumerate(plan['rows']):
            target=ROOT/'takes'/(row['id']+'.json');audio=ROOT/'audio'/(row['id']+'.mp3')
            if target.exists() and audio.exists():continue
            client.wait_idle()
            body={'message':row['prompt'],'format':row['format'],'language':row['language'],'input_mode':'manual',
                'generation':'single','n':1,'streaming':False,'align':False,'speaker_steering':False,'voice_conversion':False,
                'temperature':plan['temperature'],'guidance':plan['cfg'],'seconds':plan['duration_hint'],'seed':row['seed'],
                'voice_mode':'custom','reference_id':uploads[row['voice']]['id'],'identity_cfg':False,
                'ranking':{'mode':'legacy'},
                'adapters':{'id':DPO if row['adapter']=='dpo' else None,'ref15':False},
                'speaker':{'method':'ecapa','weight':1.}}
            tick=time.monotonic()
            try:
                events=client.turn(body)
            except urllib.error.HTTPError as exc:
                if exc.code==409:
                    client.wait_idle();events=client.turn(body)
                else:raise
            errors=[event for event in events if event['event']=='error']
            if errors:
                write_json(ROOT/'failures'/(row['id']+'.json'),{'row':row,'errors':errors})
                raise RuntimeError(str(errors))
            result=next(event for event in events if event['event']=='result');candidate=result['candidates'][0]
            assert len(result['candidates'])==1 and result['generation']=='single'
            assert not result['adapters']['selection']['ref15']
            assert len(result['adapters']['active'])==int(row['adapter']=='dpo')
            assert not result.get('speaker_steering', {}).get('active', False), 'Frozen study forbids activation steering'
            context=result['reference']['tts_context'];assert context['codec_frames']>37 and context['max_seconds']==15
            audio.parent.mkdir(parents=True,exist_ok=True);audio.write_bytes(client.request(candidate['url']))
            clean={k:v for k,v in candidate.items() if k not in ('url',)}
            record={**row,'candidate':clean,'word_errors':word_errors(row['spec']['text'],candidate['hyp']),
                'reference_context':context,'configured_guidance':plan['cfg'],'cfg_effective':result['cfg_effective'],
                'adapter_report':result['adapters'],'seconds':candidate['sec'],
                'wall_seconds':time.monotonic()-tick,'audio':'audio/'+audio.name,
                'audio_sha256':hashlib.sha256(audio.read_bytes()).hexdigest(),'performance':result['performance']}
            write_json(target,record)
            print(json.dumps({'completed':i+1,'total':len(plan['rows']),'id':row['id'],
                'conditioning_cosine_raw':candidate.get('speaker_similarity'),'wer':record['word_errors']['wer'],
                'elapsed_seconds':round(time.monotonic()-started)}),flush=True)
        write_json(ROOT/'generation-complete.json',{'samples':len(plan['rows']),'elapsed_seconds':time.monotonic()-started})
    finally:
        for item in uploads.values():client.request('/api/reference/'+item['id'],method='DELETE')


def score():
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    import torchaudio
    from huggingface_hub import snapshot_download
    from speechbrain.inference.speaker import EncoderClassifier
    plan=design()
    root=snapshot_download(settings.SPEAKER_REPO,revision=settings.SPEAKER_REVISION,
        allow_patterns=['hyperparams.yaml','*.ckpt','label_encoder.txt'],cache_dir=str(Path(settings.HF_CACHE)/'hub'),token=False)
    classifier=EncoderClassifier.from_hparams(source=root,overrides={'pretrained_path':root},
        savedir=str(ROOT/'private-ecapa'),run_opts={'device':settings.SCORE_DEVICE})

    @torch.inference_mode()
    def embed(path):
        wave,sr=sf.read(path,dtype='float32');wave=wave.mean(1) if wave.ndim>1 else wave
        tensor=torchaudio.functional.resample(torch.as_tensor(wave,device=settings.SCORE_DEVICE),sr,16000)
        if len(tensor)<8000 or not torch.isfinite(tensor).all():raise ValueError('Insufficient / invalid audio: '+str(path))
        vector=classifier.encode_batch(tensor[None],wav_lens=torch.ones(1,device=settings.SCORE_DEVICE)).reshape(-1).float()
        if not torch.isfinite(vector).all() or vector.norm()<1e-8:raise ValueError('Undefined embedding')
        return F.normalize(vector,dim=0).cpu().numpy()

    refs={voice:embed(ROOT/'references'/(voice+'.wav')) for voice in VOICES}
    anchors={voice:embed(Path(info['anchor'])) for voice,info in VOICES.items()}
    controls={voice:{'conditioning_to_same_speaker_anchor':float(refs[voice]@anchors[voice]),
        'conditioning_to_other_speaker_anchor':float(refs[voice]@anchors[next(v for v in VOICES if v!=voice)]),
        'conditioning_seconds':sf.info(ROOT/'references'/(voice+'.wav')).duration,
        'conditioning_sha256':hashlib.sha256((ROOT/'references'/(voice+'.wav')).read_bytes()).hexdigest(),
        'heldout_anchor_seconds':sf.info(info['anchor']).duration,
        'heldout_anchor_sha256':hashlib.sha256(Path(info['anchor']).read_bytes()).hexdigest()}
        for voice,info in VOICES.items()}
    write_json(ROOT/'reference-controls.json',{'ecapa_repo':settings.SPEAKER_REPO,
        'ecapa_revision':settings.SPEAKER_REVISION,'torch':torch.__version__,'controls':controls})
    for i,row in enumerate(plan['rows']):
        target=ROOT/'scores'/(row['id']+'.json')
        if target.exists():continue
        record=json.loads((ROOT/'takes'/(row['id']+'.json')).read_text())
        vector=embed(ROOT/record['audio']);other=next(v for v in VOICES if v!=row['voice'])
        heldout=float(vector@anchors[row['voice']]);wrong=float(vector@anchors[other])
        write_json(target,{'id':row['id'],'embedding':vector.tolist(),
            'conditioning_cosine_mp3':float(vector@refs[row['voice']]),'heldout_cosine':heldout,
            'other_speaker_cosine':wrong,'two_anchor_correct':heldout>wrong,
            'identity_margin':heldout-wrong,'speaker_method':'ecapa','sample_rate':16000})
        if (i+1)%20==0:print(json.dumps({'scored':i+1,'total':len(plan['rows'])}),flush=True)


def bootstrap(values):
    values=np.asarray(values,dtype=float);rng=np.random.default_rng(73100)
    means=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1)
    return [float(x) for x in np.quantile(means,[.025,.975])]


def summarize():
    plan=design();groups=defaultdict(list);allrows=[]
    for row in plan['rows']:
        record=json.loads((ROOT/'takes'/(row['id']+'.json')).read_text())
        scored=json.loads((ROOT/'scores'/(row['id']+'.json')).read_text())
        combined={**record,**scored};allrows.append(combined)
        groups[(row['voice'],row['adapter'],row['condition'])].append(combined)
    cells=[]
    for (voice,adapter,condition),rows in groups.items():
        vectors=np.asarray([r['embedding'] for r in rows]);pairwise=vectors@vectors.T
        pairmean=float(pairwise[np.triu_indices(len(rows),1)].mean())
        stats={'voice':voice,'adapter':adapter,'condition':condition,'n':len(rows),
            'heldout_cosine_mean':float(np.mean([r['heldout_cosine'] for r in rows])),
            'heldout_cosine_sd':float(np.std([r['heldout_cosine'] for r in rows],ddof=1)),
            'conditioning_cosine_mean':float(np.mean([r['conditioning_cosine_mp3'] for r in rows])),
            'conditioning_cosine_raw_mean':float(np.mean([r['candidate']['speaker_similarity'] for r in rows if r['candidate']['speaker_similarity'] is not None])),
            'conditioning_cosine_raw_n':sum(r['candidate']['speaker_similarity'] is not None for r in rows),
            'wer_mean':float(np.mean([r['word_errors']['wer'] for r in rows])),
            'pairwise_cosine_mean':pairmean,'two_anchor_accuracy':float(np.mean([r['two_anchor_correct'] for r in rows])),
            'identity_margin_mean':float(np.mean([r['identity_margin'] for r in rows])),
            'genuineness_mean':float(np.mean([r['candidate']['genuineness'] for r in rows]))}
        assert stats['n']==10;cells.append(stats)
    pooled=[]
    for adapter in ('base','dpo'):
        for condition in PROMPTS:
            rows=[r for r in allrows if r['adapter']==adapter and r['condition']==condition]
            vc=[c for c in cells if c['adapter']==adapter and c['condition']==condition]
            blockmeans=[np.mean([r['heldout_cosine'] for r in rows if r['index']==i]) for i in range(10)]
            item={'adapter':adapter,'condition':condition,'n':len(rows),
                'heldout_cosine_mean':float(np.mean(blockmeans)),'heldout_cosine_ci':bootstrap(blockmeans),
                'conditioning_cosine_raw_n':sum(c['conditioning_cosine_raw_n'] for c in vc),
                **{key:float(np.mean([c[key] for c in vc])) for key in ('conditioning_cosine_mean',
                    'conditioning_cosine_raw_mean','wer_mean','pairwise_cosine_mean','two_anchor_accuracy','identity_margin_mean','genuineness_mean')},
                'each_speaker_wer_eligible':all(c['wer_mean']<=.20 for c in vc)}
            item['eligible']=condition!='transcript' and item['wer_mean']<=.10 and item['each_speaker_wer_eligible']
            pooled.append(item)
    eligible=[p for p in pooled if p['eligible']]
    winner=max(eligible,key=lambda p:p['heldout_cosine_mean']) if eligible else None
    paired=[]
    for condition in PROMPTS:
        deltas=[]
        for i in range(10):
            pervoice=[]
            for voice in VOICES:
                pair={r['adapter']:r for r in allrows if r['voice']==voice and r['condition']==condition and r['index']==i}
                pervoice.append(pair['dpo']['heldout_cosine']-pair['base']['heldout_cosine'])
            deltas.append(np.mean(pervoice))
        paired.append({'condition':condition,'dpo_minus_base':float(np.mean(deltas)),'ci':bootstrap(deltas)})
    prompt_deltas=[]
    for adapter in ('base','dpo'):
        for condition in PROMPTS:
            if condition=='caption_affect':continue
            deltas=[]
            for i in range(10):
                pervoice=[]
                for voice in VOICES:
                    pair={r['condition']:r for r in allrows if r['voice']==voice and r['adapter']==adapter and r['index']==i}
                    pervoice.append(pair[condition]['heldout_cosine']-pair['caption_affect']['heldout_cosine'])
                deltas.append(np.mean(pervoice))
            prompt_deltas.append({'adapter':adapter,'condition':condition,'minus_affect_caption':float(np.mean(deltas)),'ci':bootstrap(deltas)})
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'completed':len(allrows),'planned':plan['total_trials'],
        'cells':cells,'pooled':pooled,'dpo_paired_deltas':paired,'prompt_paired_deltas':prompt_deltas,
        'recommended':winner,'all_outputs_retained':True,'reference_controls':json.loads((ROOT/'reference-controls.json').read_text())}
    summary['raw_speaker_unavailable']=[r['id'] for r in allrows if r['candidate']['speaker_similarity'] is None]
    write_json(ROOT/'summary.json',summary)
    # Public results omit private URLs, original source paths and embeddings.
    write_json(ROOT/'results.json',[{k:v for k,v in r.items() if k!='embedding'} for r in allrows])
    return plan,summary,allrows


def report():
    plan,summary,rows=summarize();esc=html.escape
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Speaker identity: prompting × DPO</title><style>body{font:16px system-ui;background:#10151e;color:#e4ebf6;max-width:1300px;margin:auto;padding:24px}a{color:#9ccaff}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #344052;text-align:left}th{background:#1e293b}table caption{text-align:left;font-weight:700;padding:12px 0}.scroll{overflow:auto}.note{color:#b3c2d8;line-height:1.65}.winner{background:#1d3c36}details{border:1px solid #344052;border-radius:8px;padding:12px;margin:12px 0}summary{cursor:pointer;font-weight:600}article{border-top:1px solid #344052;margin-top:14px;padding:14px 0}audio{width:min(100%,440px)}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#172131;padding:12px}select{padding:8px;background:#172131;color:inherit;margin:8px}h1,h2{line-height:1.2}.warning{color:#ffd1a1}button{padding:8px}</style>',
        '<h1>Speaker identity: prompting × DPO</h1><p class="note">240 unselected single-take generations · two reference voices · ten independent seed blocks per voice/condition · English and German · S3 base versus previous rank-16 DPO. All audio below is synthetic.</p>',
        '<div class="scroll"><table><caption>Findings at a glance — means over both speakers (20 takes per row)</caption><thead><tr><th>Prompt</th><th>Model</th><th>Held-out ECAPA ↑ [95% block CI]</th><th>Reference ECAPA ↑</th><th>Across-take cosine ↑</th><th>Raw WER ↓</th><th>Genuine proxy ↑</th><th>Eligible*</th></tr></thead><tbody>']
    ordered=sorted(summary['pooled'],key=lambda p:-p['heldout_cosine_mean'])
    for item in ordered:
        ci=item['heldout_cosine_ci'];winner=item==summary['recommended']
        parts.append(f'<tr class="{"winner" if winner else ""}"><td>{esc(PROMPTS[item["condition"]])}</td><td>{item["adapter"]}</td><td>{item["heldout_cosine_mean"]:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]</td><td>{item["conditioning_cosine_raw_mean"]:.3f} ({item["conditioning_cosine_raw_n"]}/20)</td><td>{item["pairwise_cosine_mean"]:.3f}</td><td>{item["wer_mean"]:.1%}</td><td>{item["genuineness_mean"]:.2f}</td><td>{"Yes — pilot choice" if winner else "Yes" if item["eligible"] else "Control" if item["condition"]=="transcript" else "No"}</td></tr>')
    parts.append('</tbody></table></div><p class="note">*Emotion-capable prompts qualify only if pooled mean WER ≤10% and each voice mean WER ≤20%. The pilot choice maximizes held-out same-speaker cosine among qualifiers. Transcript-only has no emotion request and no effective CFG contrast. “Reference ECAPA” uses raw PCM against the actual conditioning reference; held-out and across-take metrics use the identically encoded MP3s. Cosines are not identity probabilities. Confidence intervals reflect ten matched blocks, not a broad population study.</p>')
    winner=summary['recommended']
    if winner:parts.append(f'<h2>Practical pilot recommendation</h2><p>Start with <strong>{esc(PROMPTS[winner["condition"]])}</strong>, <strong>{winner["adapter"]}</strong>, CFG 1.5 and temperature 1.0. It has the highest held-out speaker cosine among the predeclared intelligibility qualifiers in this small experiment. This is not a universally optimal cloning recipe; inspect the per-voice rows and listen before changing defaults.</p>')
    else:parts.append('<h2>No qualifying optimum</h2><p class="warning">No emotion-capable condition passes the predeclared intelligibility guard. The highest identity score must not be advertised as a reliable recommendation.</p>')
    if winner:
        example=prompt_for(winner['condition'],'idris','joy','en')[1]
        parts.append('<details><summary>Exact starting recipe (replace the transcript and emotion)</summary><pre>'+esc(example)+'</pre><p>Supply your own clean reference through the dedicated audio channel. No gender, age or timbre guesses are required by the selected recipe. Use the previous rank-16 DPO at strength 1; Ref15 is off in this experiment.</p></details>')
        selected=[c for c in summary['cells'] if c['adapter']==winner['adapter'] and c['condition']==winner['condition']]
        details='; '.join(f'{VOICES[c["voice"]]["name"]}: reference cosine {c["conditioning_cosine_raw_mean"]:.3f}, held-out-anchor cosine {c["heldout_cosine_mean"]:.3f}, WER {c["wer_mean"]:.1%}' for c in sorted(selected,key=lambda c:c['voice']))
        parts.append('<p class="warning"><strong>Do not hide the speaker-specific result.</strong> '+esc(details)+'. The pooled recommendation is driven mainly by Idris. Selma is not reliably cloned to her separate profile anchor by any tested recipe. Fixed reference recordings and their lengths differ between the two speakers; this is not evidence that male voices are generally easier to clone.</p>')
        contrast=next((d for d in summary['prompt_paired_deltas'] if d['adapter']==winner['adapter'] and d['condition']==winner['condition']),None)
        if contrast:
            ci=contrast['ci'];parts.append(f'<p class="note">Versus affect-only Caption with the same model, the selected recipe changes held-out cosine by {contrast["minus_affect_caption"]:+.3f}, matched 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]. The observed winner is not proof of a universal prompt advantage, especially when this interval includes zero.</p>')
    parts.append(f'<p class="note">Missing-score handling: {len(summary["raw_speaker_unavailable"])} of 240 takes have an empty ASR transcript and therefore no production raw speaker score. They are retained with full WER, offline MP3 embeddings and listening audio. Raw-reference means average the available scores within each speaker, then average both speakers; the table explicitly reports score coverage. Do not interpret an embedding of non-speech as successful cloning.</p>')
    parts.append('<h2>What DPO changes</h2><div class="scroll"><table><thead><tr><th>Prompt</th><th>DPO − base, held-out cosine</th><th>95% matched-block CI</th></tr></thead><tbody>')
    for delta in summary['dpo_paired_deltas']:
        ci=delta['ci'];parts.append(f'<tr><td>{esc(PROMPTS[delta["condition"]])}</td><td>{delta["dpo_minus_base"]:+.3f}</td><td>[{ci[0]:+.3f}, {ci[1]:+.3f}]</td></tr>')
    parts.append('</tbody></table></div><h2>Do gender/timbre or explicit reference hints beat an affect-only Caption?</h2><p class="note">Matched comparisons within the same model. Positive intervals crossing zero are not decisive evidence of a prompt improvement. These exploratory intervals are not corrected for testing multiple recipes.</p><div class="scroll"><table><thead><tr><th>Model</th><th>Prompt versus affect-only Caption</th><th>Held-out cosine change</th><th>95% paired CI</th></tr></thead><tbody>')
    for delta in summary['prompt_paired_deltas']:
        ci=delta['ci'];parts.append(f'<tr><td>{delta["adapter"]}</td><td>{esc(PROMPTS[delta["condition"]])}</td><td>{delta["minus_affect_caption"]:+.3f}</td><td>[{ci[0]:+.3f}, {ci[1]:+.3f}]</td></tr>')
    parts.append('</tbody></table></div><h2>Per-speaker results</h2><div class="scroll"><table><thead><tr><th>Speaker</th><th>Model</th><th>Prompt</th><th>n</th><th>Held-out cosine ± SD</th><th>Raw WER</th><th>Across-take cosine</th><th>Correct of two anchors*</th></tr></thead><tbody>')
    for item in sorted(summary['cells'],key=lambda p:(p['voice'],p['adapter'],-p['heldout_cosine_mean'])):
        parts.append(f'<tr><td>{VOICES[item["voice"]]["name"]}</td><td>{item["adapter"]}</td><td>{esc(PROMPTS[item["condition"]])}</td><td>{item["n"]}</td><td>{item["heldout_cosine_mean"]:.3f} ± {item["heldout_cosine_sd"]:.3f}</td><td>{item["wer_mean"]:.1%}</td><td>{item["pairwise_cosine_mean"]:.3f}</td><td>{item["two_anchor_accuracy"]:.0%}</td></tr>')
    parts.append('</tbody></table></div><p class="note">*Two-anchor accuracy only chooses between these two profile anchors; it is not a general speaker-identification test.</p><h2>Reference controls</h2><ul>')
    for voice,item in summary['reference_controls']['controls'].items():
        parts.append(f'<li>{VOICES[voice]["name"]}: actual conditioning length {item["conditioning_seconds"]:.2f}s; held-out anchor {item["heldout_anchor_seconds"]:.2f}s; reference-to-own-anchor cosine {item["conditioning_to_same_speaker_anchor"]:.3f}, reference-to-other-anchor {item["conditioning_to_other_speaker_anchor"]:.3f}.</li>')
    parts.append('</ul><p class="note">Source reference audio is intentionally not published. All generated outputs are embedded below, not just selected good examples. The different-speaker control exposes weak reference/anchor separation. The matching Selma reference itself has only 0.260 cosine to her separate anchor; reference and generated scores are not calibrated identity probabilities. The production silence-removal stage shortened the Idris source to 5.48s; actual conditioning length is reported, not its original container duration.</p><h2>Method and limits</h2><p class="note">Reference audio is fixed per voice, not dynamically retrieved per emotion. Ref15 is off. The previous DPO is subset_pitch_quality_p2_rank016 at strength 1, not every DPO release. Temperature 1.0; configured CFG 1.5; one take per request; 6-second duration hint with the production natural-stop/safety budget. No forced alignment, fade, output trimming, streaming, beam search, best-of-N or reward-guided selection. WER is full-transcript Unicode word Levenshtein error using Parakeet v3 hypotheses, including tail insertions. The production speaker score is ECAPA cosine on raw PCM. A separate pinned ECAPA process evaluates the actual mono 96-kbps MP3s against the unused anchor and computes all 45 within-cell output pairs. Higher mutual consistency does not prove the voice matches the reference.</p><ul class="note">')
    for limitation in plan['limitations']:parts.append('<li>'+esc(limitation)+'</li>')
    parts.append('</ul><p class="note">One bilingual scene, two speakers and ten independent seeds per cell; each seed is paired across conditions/models. Five emotions × two languages yields ten blocks, not ten trials per emotion. We bootstrap matched blocks keeping both speakers together; intervals are exploratory and not multiple-comparison-corrected. Genuineness is a learned proxy, not listener MOS. No live-demo defaults were changed by this study.</p><p>Primary sources: <a href="https://huggingface.co/laion/Humaneness-Voice-Small">Small model / trained prompt formats</a> · <a href="https://huggingface.co/laion/Humaneness-Voice-Small-DPO-LoRAs">DPO release</a> · <a href="https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb">ECAPA model and preprocessing</a>.</p><h2>Listen to every take — embedded Base64 MP3</h2><p class="note">Audio is lazy-loaded only when you press Play. Downloading this single HTML preserves all 240 generated examples. Filters do not remove any sample from the study statistics.</p><div id="filters">')
    for field,options in [('voice',VOICES),('adapter',{'base':'Base','dpo':'DPO'}),('condition',PROMPTS),('emotion',EMOTIONS),('language',{'en':'English','de':'German'})]:
        parts.append(f'<label>{field.title()} <select data-filter="{field}"><option value="">All</option>')
        for key,value in options.items():
            label=value['name'] if isinstance(value,dict) else key.title() if field=='emotion' else value
            parts.append(f'<option value="{key}">{esc(label)}</option>')
        parts.append('</select></label>')
    parts.append('</div><style>#filters label{display:inline-block;max-width:100%}#filters select{max-width:100%;box-sizing:border-box;margin-left:0}h3{overflow-wrap:anywhere}</style><p id="count"></p>')
    for item in sorted(rows,key=lambda r:(r['voice'],r['condition'],r['index'],r['adapter'])):
        encoded=base64.b64encode((ROOT/item['audio']).read_bytes()).decode()
        attrs=' '.join(f'data-{key}="{item[key]}"' for key in ('voice','adapter','condition','emotion','language'))
        parts.append(f'<article {attrs}><h3>{VOICES[item["voice"]]["name"]} · {item["adapter"]} · {esc(PROMPTS[item["condition"]])} · {item["emotion"]} · {item["language"]} · seed {item["seed"]}</h3><p>Held-out cosine {item["heldout_cosine"]:.3f} · conditioning cosine {item["conditioning_cosine_mp3"]:.3f} · raw WER {item["word_errors"]["wer"]:.1%} · {item["seconds"]:.2f}s</p><audio controls preload="none" src="data:audio/mpeg;base64,{encoded}"></audio><details><summary>Exact prompt, recognized words and metrics</summary><pre>{esc(item["prompt"])}</pre><p>ASR: {esc(item["candidate"]["hyp"])}</p><p>Effective CFG contrast: {item["cfg_effective"]}. Reference frames: {item["reference_context"]["codec_frames"]}. Genuineness proxy: {item["candidate"]["genuineness"]:.3f}.</p></details></article>')
    parts.append('<script>const filters=[...document.querySelectorAll("[data-filter]")],items=[...document.querySelectorAll("article")];function filter(){let n=0;for(const item of items){item.hidden=filters.some(s=>s.value&&item.dataset[s.dataset.filter]!==s.value);if(!item.hidden)n++;}document.getElementById("count").textContent=n+" of "+items.length+" takes shown";}filters.forEach(s=>s.onchange=filter);filter();</script></html>')
    path=ROOT/'index.html';tmp=ROOT/'index.partial';tmp.write_text('\n'.join(parts));tmp.replace(path)
    print(json.dumps({'report':str(path),'bytes':path.stat().st_size,'completed':summary['completed'],'recommendation':winner}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['design','generate','score','report','all'])
    parser.add_argument('--base',default='http://127.0.0.1:8792');args=parser.parse_args()
    if args.action=='design':print(json.dumps({'trials':len(design()['rows'])}),flush=True)
    if args.action in ('generate','all'):generate(args.base)
    if args.action in ('score','all'):score()
    if args.action in ('report','all'):report()


if __name__=='__main__':main()
