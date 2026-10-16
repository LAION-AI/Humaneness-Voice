"""Prospective follow-up: 50/100/200 seeds, two text sets, 1/2% tails.

Reuses the first study's audited GPU/capture/scoring loop in isolated processes.
No modification of either study's immutable vectors or the live speaker bank.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import numpy as np

import settings
import seed_scaling_study as runner
import seed_scaling_analysis as stats
import dynamic_references as references
from mira_activation_study import write_json

FIRST_ROOT=settings.HERE/'seed-scaling-study'
ROOT=settings.HERE/'strict-tail-study'
VOICES=runner.VOICES
SIZES=(50,100,200)
TAILS=(10,5,2,1)
EMOTIONS=runner.prompts.EMOTIONS

# New held-out text sets: not any first-study collection/screen/confirm text.
TEXTS={
    'screen':{
        'en':[
            ['I left a note beside the telephone. Please read it before you call me back.',
             'We can finally open the doors tomorrow. Everyone has worked so hard for this moment.',
             'You promised to leave those papers alone. I need an explanation for what happened.',
             'I still remember the song we heard together. Sometimes I play it when the house is quiet.',
             'I heard footsteps outside the room. Keep the light on until we know who is there.'],
            ['The train arrives at half past seven. I will wait for you near the ticket office.',
             'They have accepted our proposal at last. I can hardly wait to tell the others.',
             'That was not the agreement we made. Stop changing the rules without asking me.',
             'The photograph was still inside the drawer. I had not looked at it for years.',
             'There is a strange sound behind the wall. Stay close while I find the switch.']],
        'de':[
            ['Neben dem Telefon liegt meine Nachricht. Bitte lies sie, bevor du mich zurückrufst.',
             'Morgen können wir endlich die Türen öffnen. Alle haben so lange auf diesen Moment hingearbeitet.',
             'Du wolltest diese Unterlagen in Ruhe lassen. Erkläre mir bitte, was hier passiert ist.',
             'Ich erinnere mich noch an unser Lied. Manchmal spiele ich es, wenn das Haus ganz still ist.',
             'Vor dem Zimmer waren Schritte zu hören. Lass das Licht an, bis wir wissen, wer dort ist.'],
            ['Der Zug kommt um halb acht an. Ich warte am Fahrkartenschalter auf dich.',
             'Unser Vorschlag wurde endlich angenommen. Ich möchte es den anderen sofort erzählen.',
             'So hatten wir das nicht vereinbart. Ändere die Regeln nicht ständig, ohne mich zu fragen.',
             'Das Foto lag noch in der Schublade. Seit Jahren hatte ich es nicht mehr angesehen.',
             'Hinter der Wand ist ein seltsames Geräusch. Bleib bei mir, während ich den Schalter suche.']]},
    'confirm':{
        'en':[
            ['The spare blanket is on the upper shelf. Take it with you if the evening gets cold.',
             'Your letter brought us wonderful news. We should celebrate before the week is over.',
             'I asked you not to speak for me. Let me make my own decision this time.',
             'Her chair was empty again this morning. I keep expecting to hear her in the hallway.',
             'The gate was moving even though the air was still. Please do not go outside alone.'],
            ['I will collect the tickets after lunch. There is no need to leave work any earlier.',
             'We found exactly what we were looking for. This could be the start of something great.',
             'You cannot pretend that nothing happened. Look at me and tell me the truth.',
             'I put the old scarf back in its box. It was harder than I thought to close the lid.',
             'Someone has turned off the lights downstairs. Wait here while I check the door.']],
        'de':[
            ['Die zusätzliche Decke liegt im oberen Regal. Nimm sie mit, falls der Abend kalt wird.',
             'Dein Brief hat uns wunderbare Nachrichten gebracht. Wir sollten noch diese Woche feiern.',
             'Ich hatte dich gebeten, nicht für mich zu sprechen. Lass mich diesmal selbst entscheiden.',
             'Ihr Stuhl war heute Morgen wieder leer. Ich erwarte noch immer ihre Schritte im Flur.',
             'Das Tor bewegte sich, obwohl kein Wind ging. Bitte geh jetzt nicht allein nach draußen.'],
            ['Nach dem Mittagessen hole ich die Karten ab. Du musst dafür nicht früher von der Arbeit gehen.',
             'Wir haben genau das gefunden, was wir gesucht haben. Vielleicht ist das ein wunderbarer Anfang.',
             'Tu nicht so, als wäre nichts passiert. Schau mich an und sag mir die Wahrheit.',
             'Ich habe den alten Schal zurückgelegt. Es fiel mir schwer, den Deckel der Schachtel zu schließen.',
             'Unten hat jemand das Licht ausgeschaltet. Warte hier, während ich die Tür überprüfe.']]}}


def blocks(phase):
    rows=[]
    repeats=200 if phase=='collection' else 2
    for repeat in range(repeats):
        for textset in range(2):
            for i,emotion in enumerate(EMOTIONS):
                for language in ('en','de'):
                    pos=2*i+(language=='de')
                    if phase=='collection':
                        source='screen' if textset==0 else 'validation'
                        text=runner.prompts.TEXTS[source][language][i]
                        base=(6100000 if repeat<10 else 18100000) if textset==0 and repeat<50 else (
                            24100000 if textset==0 else 25100000)
                    else:
                        text=TEXTS[phase][language][textset][i]
                        base=(27100000 if phase=='screen' else 29100000)+textset*1000000
                    prompt='CAPTION: '+EMOTIONS[emotion]+'\nTRANSCRIPT: '+json.dumps(text,ensure_ascii=False)
                    rows.append({'block':repeat*20+textset*10+pos,'repeat':repeat,
                        'textset':textset,'condition_id':f'{emotion}-{language}-set{textset}',
                        'emotion':emotion,'language':language,'text':text,'prompt':prompt,
                        'seed':base+repeat*1009+(pos//4)*4,'reference_identity':False})
    return rows


def freeze():
    ROOT.mkdir(exist_ok=True)
    refs={}
    for voice in VOICES:
        path,item=references.fixed_clip(voice)
        refs[voice]={'name':VOICES[voice],'path':str(path),'seconds':item['seconds'],
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    return runner.frozen(ROOT/'design.json',{
        'version':1,'voices':refs,'checkpoint_sha256':settings.FULL_FT_SHA256,
        'seed_counts_per_identical_condition':list(SIZES),'top_percentages':list(TAILS),
        'textsets_per_emotion_language':2,'collection_cells':20,
        'collection':blocks('collection'),'screen':blocks('screen'),'confirm':blocks('confirm'),
        'selection':'Per cell top ceil(N*p/100) minus fixed bottom ceil(N*.25), equal sample then equal cell weights. Nested prefixes fixed before outcomes. N50 top1/top2 identical; evaluate only once.',
        'reuse':'Reuse only first-study 50-seed collection cells of textset0 (500/voice) with exact prompt/seed/reference/capture provenance. Add textset1 and extend both sets to 200 seeds: 4000 collection takes/voice, 8000 total; 7000 new collection takes.',
        'settings':{'cfg':2.,'temperature':1.,'max_frames':256,'batch_size':4,'alpha':.75,
                    'layer':16,'normalization':'raw','adapters':[],'voice_conversion':False,
                    'alignment':False,'reference_cap_seconds':15},
        'screening':'11 distinct new vectors plus frozen first-study winner, original UI vector and no-steering control, each 40 matched fresh takes (20 text/emotion/language cells repeated twice).',
        'confirmation':'Freeze best quality-safe ECAPA screen candidate and smallest/loosest candidate within .01 of it, else diagnostic unsafe winner. Confirm both, n50-top10, original UI and no-steering on 40 fresh text/seed blocks. Deduplicate exact names.',
        'plateau':'Paired N50/100/200 comparisons at each tail, and strictness comparisons within each N. ±.01 ECAPA equivalence band; wide intervals do not prove plateau. Final paired confirmation comparison of winner versus cheaper candidate.',
        'limits':['Two chosen speaker profiles and one fixed reference each; not a universal cloning recipe.',
            'Twenty collection cells include two distinct transcripts per emotion/language; held-out textsets and seeds are disjoint.',
            'N100 top1 selects one take/cell; N200 top1 selects two, top2 four. Extremely small tails still have noisy estimators.',
            'Raw alpha fixed, not norm-matched; direction and magnitude both vary with sample count/tail.',
            'Learned ECAPA/Ears/EIV/VoiceNet proxies, not human identity percentages or listening ratings.',
            'Batch rows share RNG; cluster bootstrap by actual batch seed. Exploratory intervals not multiplicity-corrected.',
            'First-stage winner fixed before any follow-up held-out generations; no live defaults changed.']})


def reuse_collection(voice):
    tasks=[t for t in blocks('collection') if t['textset']==0 and t['repeat']<50]
    for task in tasks:
        old_block=task['repeat']*10+(task['block']%20)
        source=FIRST_ROOT/'results'/f'collection-{voice}-baseline-{old_block}.json'
        row=json.loads(source.read_text())
        for key in ('prompt','seed','emotion','language','repeat'):
            if row[key]!=task[key]:raise ValueError('Follow-up reuse mismatch '+key)
        if row.get('failed') or row['checkpoint']['sha256']!=settings.FULL_FT_SHA256:
            raise ValueError('Invalid reuse row')
        activation=Path(row.get('source_activation',FIRST_ROOT/'activations'/(row['id']+'.npz')))
        result={**row,**task,'id':f'collection-{voice}-baseline-{task["block"]}',
                'source_result':str(source),'source_activation':str(activation),
                'reused':True,'reuse_origin':'first seed-scaling study'}
        runner.frozen(ROOT/'results'/(result['id']+'.json'),result)


def actual(n,p):
    return 'n50-top2' if n==50 and p==1 else stats.condition_name(n,p)


def groups_for(voice, phase):
    if phase=='collection':return [('baseline',None)]
    if phase=='confirm':
        selected=list(json.loads((ROOT/'selection.json').read_text())[voice]['conditions'])
        extension=ROOT/'confirmation-extension.json'
        if extension.exists():
            selected=list(dict.fromkeys(selected+json.loads(extension.read_text())['additional_candidates']))
    else:
        selected=[stats.condition_name(n,p) for n in SIZES for p in TAILS if not(n==50 and p==1)]
        selected+=['first-study-winner','original-ui']
    return [('baseline',None)]+[(c,ROOT/'vectors'/(voice+'-'+c+'.npz')) for c in selected]


def derive():
    metadata={};first_selection=json.loads((FIRST_ROOT/'selection.json').read_text())
    options={'expected_cells':20,'allowed_sizes':SIZES,'allowed_percentages':TAILS}
    for voice in VOICES:
        rows=runner.read_rows(voice,'collection');arrays={}
        if len(rows)!=4000:raise ValueError('Need all 4000 collection rows for '+voice)
        for row in rows:
            path=Path(row.get('source_activation',ROOT/'activations'/(row['id']+'.npz')))
            with np.load(path,allow_pickle=False) as data:arrays[row['id']]=data['layer_16'].copy()
        items={};vectors={};(ROOT/'vectors').mkdir(exist_ok=True)
        for n in SIZES:
            for p in TAILS:
                key=stats.condition_name(n,p);vector,item=stats.tail_vector(rows,arrays,n,p,**options)
                dest=ROOT/'vectors'/(voice+'-'+key+'.npz')
                if dest.exists():
                    with np.load(dest,allow_pickle=False) as data:assert np.array_equal(data['layer_16__contrast'],vector)
                else:np.savez_compressed(dest,layer_16__contrast=vector)
                item['total_collection']=20*n
                item['bootstrap_stability']=stats.direction_stability(rows,arrays,n,p,**options)
                items[key]=item;vectors[key]=vector
        for key,path in [('first-study-winner',FIRST_ROOT/'vectors'/(voice+'-'+first_selection[voice]['winner']+'.npz')),
                         ('original-ui',settings.HERE/'speaker-steering'/(voice+'.npz'))]:
            with np.load(path,allow_pickle=False) as data:vector=data['layer_16__contrast'].copy()
            np.savez_compressed(ROOT/'vectors'/(voice+'-'+key+'.npz'),layer_16__contrast=vector)
            items[key]={'source':str(path),'source_condition':first_selection[voice]['winner'] if key=='first-study-winner' else None,
                        'l2':float(np.linalg.norm(vector)),'sha256':hashlib.sha256(vector.tobytes()).hexdigest()}
            vectors[key]=vector
        items['pairwise_cosine']={a:{b:float(x@y/max(float(np.linalg.norm(x)*np.linalg.norm(y)),1e-12))
                  for b,y in vectors.items()} for a,x in vectors.items()}
        metadata[voice]=items
    return runner.frozen(ROOT/'vectors.json',metadata)


def summarize(phase):
    result={}
    for voice in VOICES:
        rows=runner.read_rows(voice,phase)
        items={c:stats.assessment(rows,c) for c in sorted({r['condition'] for r in rows}-{'baseline'})}
        if phase=='screen':items['n50-top1']={**items['n50-top2'],'condition':'n50-top1','alias':'n50-top2'}
        comparisons={}
        if phase=='screen':
            for p in TAILS:
                for a,b in ((100,50),(200,100)):
                    comparisons[f'n{a}-minus-n{b}-top{p}']=stats.compare_conditions(rows,actual(a,p),actual(b,p))
            for n in SIZES:
                for a,b in ((5,10),(2,5),(1,2)):
                    comparisons[f'n{n}-top{a}-minus-top{b}']=stats.compare_conditions(rows,actual(n,a),actual(n,b))
        else:
            selected=json.loads((ROOT/'selection.json').read_text())[voice]
            comparisons['winner-minus-efficient']=stats.compare_conditions(rows,selected['winner'],selected['efficient'])
            comparisons['winner-minus-original-ui']=stats.compare_conditions(rows,selected['winner'],'original-ui')
            for extra in ('n200-top1','n200-top2'):
                if extra in items:
                    comparisons[extra+'-minus-screen-winner']=stats.compare_conditions(rows,extra,selected['winner'])
            if 'n200-top1' in items and 'n200-top2' in items:
                comparisons['n200-top1-minus-top2']=stats.compare_conditions(rows,'n200-top1','n200-top2')
            if 'first-study-winner' in items:
                comparisons['first-study-winner-minus-new-screen-winner']=stats.compare_conditions(rows,'first-study-winner',selected['winner'])
        result[voice]={'conditions':items,'comparisons':comparisons}
    write_json(ROOT/(phase+'-summary.json'),result)
    return result


def select():
    summary=summarize('screen');selected={}
    for voice,item in summary.items():
        candidates=[v for k,v in item['conditions'].items() if k.startswith('n') and not v.get('alias')]
        safe=[v for v in candidates if v['quality_safe']]
        winner=max(safe or candidates,key=lambda c:(c['matched']['ecapa']['mean'],c['matched']['content_enjoyment']['mean']))
        close=[c for c in safe if c['matched']['ecapa']['mean']>=winner['matched']['ecapa']['mean']-.01]
        def cheap(c):
            n,p=c['condition'].split('-');return int(n[1:]),-int(p[3:])
        efficient=min(close,key=cheap) if close else item['conditions']['n50-top10']
        selected[voice]={'winner':winner['condition'],'efficient':efficient['condition'],
            'screen_quality_safe':winner['quality_safe'],
            'conditions':list(dict.fromkeys([winner['condition'],efficient['condition'],'n50-top10','original-ui']))}
    return runner.frozen(ROOT/'selection.json',selected)


def report():
    design=freeze();screen=summarize('screen');confirm=summarize('confirm')
    selection=json.loads((ROOT/'selection.json').read_text());vectors=json.loads((ROOT/'vectors.json').read_text())
    rows=runner.read_rows();public=[{k:v for k,v in r.items() if k not in ('speaker_embedding','traceback')} for r in rows]
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'takes':len(rows),
        'new_takes':sum(not r.get('reused') for r in rows),'reused':sum(r.get('reused',False) for r in rows),
        'failures':sum(bool(r.get('failed')) for r in rows),'screen':screen,'confirm':confirm,
        'selection':selection,'vectors':vectors,'live_defaults_changed':False}
    extension=ROOT/'confirmation-extension.json'
    if extension.exists():summary['confirmation_extension']=json.loads(extension.read_text())
    write_json(ROOT/'results.json',public);write_json(ROOT/'summary.json',summary)
    esc=lambda value:html.escape(str(value),quote=True)
    fmt=lambda value:'—' if value is None else f'{value:+.3f}'
    parts=['<section class="seed-study" id="strict-tails"><h2>Follow-up: 100/200 seeds and Top 1% / 2%, two text sets</h2>',
        '<p>Full-FT base only, fixed reference, CFG 2.0, temperature 1.0, layer 16 raw alpha 0.75. No DPO, VC, trimming or best-of-N. Two text sets × five emotions × two languages, with 200 seeds per identical condition: 4,000 collection takes per voice. The initial study remains unchanged above.</p>',
        '<h3>Practical findings</h3><table><tr><th>Voice</th><th>Practical interpretation</th><th>Fresh confirmation ECAPA</th></tr>'
        '<tr><td>Mira</td><td>50 seeds per cell / Top 2% was the best newly-derived candidate in the screen and stayed strongest in fresh confirmation. It improved over the existing demo vector by +0.041 cosine [95% cluster-bootstrap interval +0.013, +0.073]. CE/PQ/MOS proxies improved vs no steering. Emotion proxy was slightly lower, not unchanged.</td><td>No steering 0.285; existing demo 0.429; N50 Top2 0.470; N200 Top1 0.459; N200 Top2 0.463.</td></tr>'
        '<tr><td>Osku</td><td>The screen-selected N50 Top2 did not retain its ranking in fresh confirmation. The existing demo vector had the highest point estimate among confirmed directions. Moderate N50 Top10 was a stronger new compromise than the most extreme tail. No evidence supports replacing the existing vector with N200 Top1.</td><td>No steering 0.234; existing demo 0.349; N50 Top10 0.332; N50 Top2 0.301; N200 Top1 0.288; N200 Top2 0.327.</td></tr></table>'
        '<p><strong>Bottom line:</strong> More seeds made vector estimates more stable, but did not yield monotonically better generated speech. Practical diminishing returns appeared around 20–50 seeds per cell in this experiment; a universal statistical plateau is <em>not</em> established. Fresh N200-minus-N50 confidence intervals remain wider than the ±0.01 equivalence band. Top 1% is not a reliable general-purpose improvement over Top 2–10%.</p>'
        '<p>At N50, Top2 means one selected take per cell (20 high takes across both text sets); nominal Top1 is exactly the same rounded pool. N200 Top1 selects two per cell. The lower pool stays Bottom25 throughout. Seed count here means <em>per identical text/emotion/language condition</em>, not total takes per voice.</p>',
        '<h3>Fresh confirmation: findings table</h3>']
    for phase,data in [('Fresh confirmation',confirm),('Exploratory screen',screen)]:
        parts.append('<h3>'+phase+'</h3><table><tr><th>Voice / condition</th><th>Pairs</th><th>ECAPA</th><th>Δ ECAPA [95% interval]</th><th>Δ Anchor</th><th>Δ CE</th><th>Δ PQ</th><th>Δ MOS proxy</th><th>Δ WER</th><th>Δ emotion proxy</th><th>Safeguards</th></tr>')
        for voice,item in data.items():
            for key,value in item['conditions'].items():
                ci=value['matched']['ecapa'];d=value['matched']
                label=VOICES[voice]+' / '+key
                if key=='first-study-winner':label+=' ('+vectors[voice][key]['source_condition']+', previous 10-cell collection)'
                parts.append('<tr><td>'+esc(label)+(' (alias)' if value.get('alias') else '')+'</td><td>'+str(value['n'])+'</td><td>'+f'{value["means"]["ecapa"]:.3f}'+'</td><td>'+f'{ci["mean"]:+.3f} [{ci["low"]:+.3f}, {ci["high"]:+.3f}]'+'</td>'+''.join('<td>'+fmt(d[k]['mean'] if d[k] else None)+'</td>' for k in ('anchor_ecapa','content_enjoyment','production_quality','mos','wer','emotion_score'))+'<td>'+esc('passed numerical gates' if value['quality_safe'] else ', '.join(value['quality_gate_reasons']))+'</td></tr>')
        parts.append('</table>')
    parts += ['<h3>Does more data / a smaller tail still help?</h3><p>All differences are paired on identical task and seed. A confidence interval inside ±0.01 indicates a pilot plateau at that comparison; an interval overlapping zero but wider than ±0.01 is inconclusive. N50 Top1 and Top2 both select one take per cell and are mathematically identical. At N100 Top1 selects one; at N200 it selects two (Top2 selects four). No effect beyond 200 seeds has been tested.</p>']
    for phase,data in [('Screening',screen),('Fresh confirmation',confirm)]:
        parts.append('<h3>'+phase+' comparisons</h3><table><tr><th>Voice / comparison</th><th>Δ ECAPA [95% interval]</th><th>Interpretation</th><th>Δ CE</th><th>Δ PQ</th><th>Δ WER</th></tr>')
        for voice,item in data.items():
            for key,d in item['comparisons'].items():
                ci=d['ecapa'];label='Equivalent within ±0.01 (pilot)' if ci['low']>=-.01 and ci['high']<=.01 else 'Positive gain' if ci['low']>0 else 'Negative gain' if ci['high']<0 else 'Inconclusive / no proven plateau'
                if key=='n50-top1-minus-top2':label='Exact rounded-pool alias, not evidence of a plateau'
                if key=='winner-minus-efficient' and selection[voice]['winner']==selection[voice]['efficient']:
                    label='Same preselected candidate, not an independent plateau test'
                parts.append('<tr><td>'+esc(VOICES[voice]+' / '+key)+'</td><td>'+f'{ci["mean"]:+.3f} [{ci["low"]:+.3f}, {ci["high"]:+.3f}]'+'</td><td>'+label+'</td>'+''.join('<td>'+fmt(d[k]['mean'])+'</td>' for k in ('content_enjoyment','production_quality','wer'))+'</tr>')
        parts.append('</table>')
    parts += ['<h3>Selection-pool and vector diagnostics</h3><p>Bottom 25% stays fixed, equal cell weights. Raw alpha stays fixed but norms change: these are combined direction/strength comparisons, not norm-matched identity-only interventions. Bootstrap stability is a collection-only diagnostic, not extra validation audio.</p><table><tr><th>Voice / vector</th><th>Collection</th><th>High/cell</th><th>Low/cell</th><th>High ECAPA</th><th>Low ECAPA</th><th>L2</th><th>Cosine to N50 Top10</th><th>Bootstrap cosine</th></tr>']
    for voice,items in vectors.items():
        for key,item in items.items():
            if not key.startswith('n'):continue
            cell=item['cells'][0];stability=item['bootstrap_stability']
            parts.append('<tr><td>'+esc(VOICES[voice]+' / '+key)+'</td><td>'+str(item['total_collection'])+'</td><td>'+str(cell['high_n'])+'</td><td>'+str(cell['low_n'])+'</td><td>'+f'{item["mean_selected_ecapa"]:.3f}'+'</td><td>'+f'{item["mean_rejected_ecapa"]:.3f}'+'</td><td>'+f'{item["l2"]:.3f}'+'</td><td>'+f'{items["pairwise_cosine"][key]["n50-top10"]:.3f}'+'</td><td>'+f'{stability["cosine_to_estimate_median"]:.3f}'+'</td></tr>')
    parts += ['</table><h3>Listen to every fresh confirmation pair</h3><p>Untrimmed mono 96 kbps MP3 embedded as Base64. No cherry-picking: both textsets, all five emotions and both languages, including failed quality screens.</p>']
    def player(path):return '<audio controls preload="none" src="data:audio/mpeg;base64,'+base64.b64encode(Path(path).read_bytes()).decode()+'"></audio>'
    for voice in VOICES:
        parts.append('<h3>'+esc(VOICES[voice])+' — fixed reference</h3>'+player(references.fixed_clip(voice,playback=True)[0]))
        cases=runner.read_rows(voice,'confirm');controls={r['block']:r for r in cases if r['condition']=='baseline'}
        for block,control in sorted(controls.items()):
            parts.append('<details><summary>'+esc(f'Task {block} / textset {control["textset"]}: {control["emotion"]} / {control["language"]} — '+control['text'])+'</summary><p>No steering</p>'+player(control['audio_path']))
            for row in cases:
                if row['block']==block and row['condition']!='baseline':
                    parts.append('<p>'+esc(row['condition'])+f' — ECAPA {row["ecapa"]:.3f}, CE {row["content_enjoyment"]:.2f}, WER {row["wer"]:.3f}</p>'+player(row['audio_path']))
            parts.append('</details>')
    parts += ['<h3>Audit</h3><p>'+esc(f'{summary["takes"]} measured takes: {summary["new_takes"]} newly generated, {summary["reused"]} reused; {summary["failures"]} technical failures. Live settings and speaker-bank vectors were not changed.')+'</p>'
        '<p>The extended confirmation also included N200 Top1/Top2 and the first-study winners. These candidates were specified from the completed screening before inspecting confirmation scores, while the originally planned confirmation was already running. The same 40 fresh controls were reused exactly; no new winner was chosen from these diagnostic results. The extension protocol is included in the numerical summary below.</p>'
        '<p>Tail cutoffs are relative ranks, not absolute cloning-quality guarantees. All quoted emotion, CE/PQ/MOS and aesthetics values are learned proxies, not human ratings. These results apply to two fixed references and this raw alpha; they do not establish an optimal tail for all speakers, strengths or prompting recipes.</p><p>'+esc(' '.join(design['limits']))+'</p>',
        '<details><summary>Frozen protocol and numerical summary</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'+esc(json.dumps({'design':design,'summary':summary},ensure_ascii=False,indent=2))+'</pre></details></section>']
    section=''.join(parts)
    css='<style>body{background:#0c1422}.seed-study{box-sizing:border-box;font:16px/1.5 system-ui;width:100%;max-width:1200px;margin:30px auto;padding:20px;color:#eef3ff;background:#142033;overflow-wrap:anywhere}.seed-study table{display:block;overflow-x:auto;min-width:0;width:100%;border-collapse:collapse}.seed-study td,.seed-study th{border:1px solid #536277;padding:9px;text-align:left}.seed-study audio{max-width:100%;width:280px}.seed-study summary{cursor:pointer}</style>'
    (ROOT/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Strict-tail speaker study</title>'+css+'<body>'+section+'</body></html>')
    target=settings.HERE/'catalogue-steering-study/index.html';existing=target.read_text()
    merged=stats.insert_report_extension(existing,'strict-tail',section)
    temp=target.with_name('index.strict-build.html');temp.write_text(merged);temp.replace(target)
    print(json.dumps({'takes':len(rows),'new_takes':summary['new_takes'],
        'url':'https://switch-surprised-clock-sparc.trycloudflare.com/catalogue-steering-study/#strict-tails'},indent=2))


# Every process imports its own module state; the live engine is untouched.
runner.ROOT=ROOT
runner.blocks=blocks
runner.freeze=freeze
runner.reuse_collection=reuse_collection
runner.groups_for=groups_for
runner.old.ROOT=ROOT
score_app=runner.score_app


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['design','collection','derive','screen','select','confirm','report'])
    p.add_argument('--voice',choices=list(VOICES));p.add_argument('--device',default='cuda:0')
    p.add_argument('--score-url',default='http://127.0.0.1:8795');args=p.parse_args()
    if args.action=='design':freeze()
    elif args.action=='derive':derive()
    elif args.action=='select':select()
    elif args.action=='report':report()
    else:runner.run(args.voice,args.action,args.device,args.score_url)


if __name__=='__main__':main()
