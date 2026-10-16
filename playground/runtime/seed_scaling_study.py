"""Resumable seed-count / strict-tail study; no changes to the live voice bank."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import subprocess
import time
import traceback
import urllib.request

import numpy as np
import soundfile as sf
import torch

import settings
import dynamic_references as references
import catalogue_steering_study as old
import mira_activation_analysis as prompts
import seed_scaling_analysis as stats
from mira_activation_study import write_json
from activation_steering import ActivationCollector, research_stream_frames
from prompting import build, parse_literal

ROOT = settings.HERE / 'seed-scaling-study'
OLD_ROOT = settings.HERE / 'catalogue-steering-study'
VOICES = {'emolia_c1699':'Mira', 'mediathek_0184':'Osku'}


def frozen(path, data):
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError('Frozen artifact differs: '+str(path))
    else:
        write_json(path, data)
    return data


def blocks(phase):
    source = 'screen' if phase=='collection' else 'refine' if phase=='screen' else 'collection'
    repeats = 50 if phase=='collection' else 2
    output = []
    for repeat in range(repeats):
        for i, emotion in enumerate(prompts.EMOTIONS):
            for language in ('en', 'de'):
                pos = 2*i + (language=='de')
                # Confirm texts differ from both collection and screening.
                text = prompts.TEXTS[source][language][i+5 if phase=='confirm' else i]
                prompt = 'CAPTION: '+prompts.EMOTIONS[emotion]+'\nTRANSCRIPT: '+json.dumps(text,ensure_ascii=False)
                base = 6100000 if phase=='collection' and repeat<10 else {
                    'collection':18100000, 'screen':19100000, 'confirm':20100000}[phase]
                output.append({'block':repeat*10+pos, 'repeat':repeat, 'condition_id':emotion+'-'+language,
                    'emotion':emotion,'language':language,'text':text,'prompt':prompt,
                    'seed':base+repeat*1009+(pos//4)*4, 'reference_identity':False})
    return output


def freeze():
    ROOT.mkdir(exist_ok=True)
    refs = {}
    for voice in VOICES:
        path, item = references.fixed_clip(voice)
        refs[voice] = {'name':VOICES[voice], 'path':str(path), 'seconds':item['seconds'],
                       'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    return frozen(ROOT/'design.json', {'version':1, 'voices':refs,
        'checkpoint_sha256':settings.FULL_FT_SHA256,
        'collection':blocks('collection'), 'screen':blocks('screen'), 'confirm':blocks('confirm'),
        'seed_counts_per_identical_condition':list(stats.SIZES), 'top_percentages':list(stats.PERCENTAGES),
        'selection':'Within each of ten identical text/emotion/language cells: top ceil(N*p/100) minus fixed bottom ceil(N*.25); equal sample and cell weight. Prefix samples determined by repeat number, not outcome. Top5 and top10 at N10 are identical and evaluated once.',
        'reference':'Same fixed reference in high and low pools AND validation. Not reference-minus-unconditional.',
        'reuse':'Ten repetitions/cell per voice reused from completed catalogue collection, exact seeds/captures/provenance retained. Forty new repetitions/cell per voice. No unseen N>50 claim.',
        'settings':{'cfg':2., 'temperature':1., 'max_frames':256, 'batch_size':4,
            'alpha':.75, 'layer':16, 'normalization':'raw', 'adapters':[],
            'voice_conversion':False, 'alignment':False, 'reference_cap_seconds':15},
        'screening':'Each distinct vector plus original UI vector and no-steering control on twenty matched new text/seed blocks. Same 4/4/2 batch layout across arms. Screening chooses best quality-safe ECAPA gain, else best diagnostic explicitly unsafe.',
        'confirmation':'Freeze screen winner and smallest/loosest quality-safe candidate within .01 ECAPA of winner, or n10-top25 if no safe candidate. Twenty fresh matched text/seed blocks each plus no-steering control. No alpha optimization or live-bank replacement.',
        'plateau':'Paired size contrasts at each percentile and paired strictness contrasts at each size; exploratory cluster-bootstrap 95% intervals by actual batch seed. Plateau requires gain below .01 with interval inside [-.01,.01]; a wide interval means inconclusive, not plateau.',
        'limits':['Two selected fixed-reference profiles, not random speaker sample.',
            'At N10, ceil top5/top10 are one sample/cell; at N50 top5 means three samples/cell (6%), not 25 pooled extrema.',
            'Raw-alpha comparison changes both direction and magnitude as tails tighten. Vector norms and angles reported; not a norm-matched direction-only experiment.',
            'Batch rows share RNG; ten mixed cells per repeat use three actual seeds. Intervals cluster by batch; twenty validation takes are not twenty independent clusters.',
            'Selection uses ECAPA; held-out anchor ECAPA is secondary but corpus identity/session labels are not independently verified. Scores are learned proxies, not human ratings.',
            'Many exploratory comparisons; bootstrap intervals not multiplicity-corrected. Fresh confirmation reduces but does not eliminate selection uncertainty.']})


def read_rows(voice=None, phase=None):
    rows=[]
    for path in sorted((ROOT/'results').glob('*.json')):
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            # A progress poll can overlap the final write of a generation row.
            # Completed-stage cardinality checks still reject a corrupt ledger.
            continue
    return [r for r in rows if (voice is None or r['voice']==voice) and (phase is None or r['phase']==phase)]


def reuse_collection(voice):
    for task in blocks('collection')[:100]:
        source = OLD_ROOT/'results'/f'collection-{voice}-baseline-{task["block"]}.json'
        row = json.loads(source.read_text())
        for key in ('prompt','seed','emotion','language','repeat'):
            if row[key] != task[key]: raise ValueError('Existing collection mismatch '+key)
        if row.get('failed') or row['checkpoint']['sha256'] != settings.FULL_FT_SHA256:
            raise ValueError('Invalid old collection')
        fixed_ref=references.fixed_clip(voice)[0]
        if row['reference_sha256']!=hashlib.sha256(fixed_ref.read_bytes()).hexdigest():
            raise ValueError('Existing reference differs')
        if row['adapters'] or row['cfg']!=2. or row['temperature']!=1.:
            raise ValueError('Existing sampling settings differ')
        row['source_result'] = str(source)
        row['source_activation'] = str(OLD_ROOT/'activations'/(row['id']+'.npz'))
        row['audio_path'] = str(OLD_ROOT/row['audio_path'])
        row['reused'] = True
        row['id'] = f'collection-{voice}-baseline-{task["block"]}'
        frozen(ROOT/'results'/(row['id']+'.json'), row)


class RemoteQuality:
    def __init__(self, url): self.url=url.rstrip('/')
    def score(self, waves, rows):
        paths=[]
        for row, wave in zip(rows,waves):
            path=ROOT/'audio'/(row['id']+'.wav')
            path.parent.mkdir(exist_ok=True)
            sf.write(path,wave,48000,subtype='FLOAT');paths.append(str(path))
        request=urllib.request.Request(self.url+'/score',data=json.dumps({'rows':rows,'paths':paths}).encode(),
                                      headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=180) as response: return json.load(response)['scores']


def groups_for(voice, phase):
    if phase=='collection': return [('baseline', None)]
    if phase=='confirm':
        selected=json.loads((ROOT/'selection.json').read_text())[voice]['conditions']
    else:
        selected=[stats.condition_name(n,p) for n in stats.SIZES for p in stats.PERCENTAGES
                  if not (n==10 and p==5)] + ['original-ui']
    return [('baseline',None)] + [(c,ROOT/'vectors'/(voice+'-'+c+'.npz')) for c in selected]


@torch.inference_mode()
def run(voice, phase, device, score_url):
    design=freeze()
    if phase=='collection': reuse_collection(voice)
    settings.DEVICE=device;settings.COMPILE_GLOBAL=False;settings.FAST_INFERENCE=True
    torch.set_num_threads(4);torch.cuda.set_device(device)
    from engine import SmallEngine
    from dpo_adapters import AdapterSelection
    engine=SmallEngine();engine.configure_adapters(AdapterSelection())
    engine.configure_speaker_steering(None, {'active':False,'reason':'offline_seed_scaling_research'})
    reference=Path(design['voices'][voice]['path'])
    quality=RemoteQuality(score_url)
    for condition, vector_path in groups_for(voice, phase):
        vector=None
        if vector_path:
            with np.load(vector_path,allow_pickle=False) as data:
                vector=torch.from_numpy(data['layer_16__contrast'].copy())
        rows=[{**b,'id':f'{phase}-{voice}-{condition}-{b["block"]}', 'voice':voice,
               'phase':phase, 'condition':condition, 'reference_present':True}
              for b in blocks(phase)]
        for offset in range(0,len(rows),10):
            for start in (0,4,8):
                batch=rows[offset+start:offset+min(start+4,10)]
                paths=[ROOT/'results'/(r['id']+'.json') for r in batch]
                if all(p.exists() for p in paths): continue
                if any(p.exists() for p in paths): raise RuntimeError('Partial batch requires explicit recovery')
                collector=ActivationCollector(len(batch)) if phase=='collection' else None
                def stream(branches,p,chunk_frames,guidance=2.,min_frames=0,graph=False):
                    return research_stream_frames(engine,branches,p,chunk_frames,guidance=guidance,
                        collector=collector,collect_layers=[16],
                        layer_vectors={16:vector} if vector is not None else {},factors=.75,
                        min_frames=min_frames,graph=False)
                engine.fast.stream_frames=stream
                specs=[build(parse_literal(r['prompt'],'caption'),'caption',r['language'],
                       literal=r['prompt'],reference_identity=False) for r in batch]
                info={'device':device,'checkpoint':engine.checkpoint_info,'adapters':[],
                    'cfg':2.,'temperature':1.,'batch_seed':batch[0]['seed'],
                    'batch_ids':[r['id'] for r in batch],'batch_size':len(batch),
                    'reference_sha256':design['voices'][voice]['sha256'],
                    'vector_file':str(vector_path) if vector_path else None,
                    'vector_sha256':hashlib.sha256(vector_path.read_bytes()).hexdigest() if vector_path else None,
                    'alpha':.75 if vector is not None else 0}
                tick=time.monotonic()
                try:
                    waves=engine.generate_many(specs,guidance=2.,seed=batch[0]['seed'],reference_path=reference,
                        temperature=1.,streaming=False,batch_size=len(batch),max_frames=256)
                    values=collector.export() if collector else None
                    scores=quality.score(waves,batch)
                    for i,(row,path,wave,score) in enumerate(zip(batch,paths,waves,scores)):
                        count=len(engine.last_codes[i])
                        if values:
                            assert int(values['counts'][i])==count
                            dest=ROOT/'activations'/(row['id']+'.npz');dest.parent.mkdir(exist_ok=True)
                            np.savez_compressed(dest,layer_16=values['means']['layer_16'][i].numpy(),frame_count=np.array(count))
                        wav=ROOT/'audio'/(row['id']+'.wav');mp3=wav.with_suffix('.mp3')
                        subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(wav),
                            '-map_metadata','-1','-ac','1','-ar','48000','-c:a','libmp3lame','-b:a','96k',str(mp3)],check=True,timeout=30)
                        write_json(path,{**row,**info,**score,'audio_path':str(mp3),
                            'generated_frames':count,'acoustic_tokens':count*12,'status':'ok','failed':False,
                            'batch_wall_seconds':time.monotonic()-tick,'reused':False})
                except Exception as exc:
                    for row,path in zip(batch,paths):
                        if not path.exists(): write_json(path,{**row,**info,'failed':True,
                            'status':'failed','error':str(exc),'traceback':traceback.format_exc()})
                    raise
                print(phase, VOICES[voice],condition,offset+start+len(batch),'/',len(rows),
                      'wall',round(time.monotonic()-tick,2),flush=True)
    print('Complete', voice, phase, flush=True)


def derive():
    freeze();meta={}
    for voice in VOICES:
        rows=read_rows(voice,'collection');arrays={}
        for row in rows:
            path=Path(row.get('source_activation', ROOT/'activations'/(row['id']+'.npz')))
            with np.load(path,allow_pickle=False) as data: arrays[row['id']]=data['layer_16'].copy()
        if len(rows)!=500: raise ValueError('Need complete 50 seeds x 10 cells')
        vm={};vectors={}
        for n in stats.SIZES:
            for percentage in stats.PERCENTAGES:
                key=stats.condition_name(n,percentage)
                vector,item=stats.tail_vector(rows,arrays,n,percentage)
                dest=ROOT/'vectors'/(voice+'-'+key+'.npz');dest.parent.mkdir(exist_ok=True)
                if dest.exists():
                    with np.load(dest,allow_pickle=False) as data:
                        assert np.array_equal(data['layer_16__contrast'],vector)
                else: np.savez_compressed(dest,layer_16__contrast=vector)
                vm[key]=item;vectors[key]=vector
                vm[key]['bootstrap_stability']=stats.direction_stability(rows,arrays,n,percentage)
        old_path=settings.HERE/'speaker-steering'/(voice+'.npz')
        with np.load(old_path,allow_pickle=False) as data: vector=data['layer_16__contrast'].copy()
        np.savez_compressed(ROOT/'vectors'/(voice+'-original-ui.npz'),layer_16__contrast=vector)
        vm['original-ui']={'file':str(old_path),'l2':float(np.linalg.norm(vector)),
                          'sha256':hashlib.sha256(vector.tobytes()).hexdigest()}
        vectors['original-ui']=vector
        vm['pairwise_cosine']={a:{b:float(x@y/max(float(np.linalg.norm(x)*np.linalg.norm(y)),1e-12))
             for b,y in vectors.items()} for a,x in vectors.items()}
        meta[voice]=vm
    return frozen(ROOT/'vectors.json',meta)


def summarize(phase):
    out={}
    for voice in VOICES:
        rows=read_rows(voice,phase)
        summaries={c:stats.assessment(rows,c) for c in sorted({r['condition'] for r in rows}-{'baseline'})}
        if phase=='screen': summaries['n10-top5']={**summaries['n10-top10'],
            'condition':'n10-top5','alias':'n10-top10, exact same vector and evaluation; not new takes'}
        comparisons={}
        if phase=='screen':
            def actual(n,p): return 'n10-top10' if n==10 and p==5 else stats.condition_name(n,p)
            for p in stats.PERCENTAGES:
                for a,b in ((20,10),(50,20)):
                    comparisons[f'n{a}-minus-n{b}-top{p}']=stats.compare_conditions(rows,actual(a,p),actual(b,p))
            for n in stats.SIZES:
                for a,b in ((10,25),(5,10)):
                    comparisons[f'n{n}-top{a}-minus-top{b}']=stats.compare_conditions(rows,actual(n,a),actual(n,b))
        out[voice]={'conditions':summaries,'comparisons':comparisons}
    write_json(ROOT/(phase+'-summary.json'),out)
    return out


def select():
    summary=summarize('screen');selection={}
    for voice,item in summary.items():
        candidates=[v for k,v in item['conditions'].items() if k!='original-ui' and not v.get('alias')]
        safe=[v for v in candidates if v['quality_safe']]
        pool=safe or candidates
        winner=max(pool,key=lambda r:(r['matched']['ecapa']['mean'],r['matched']['content_enjoyment']['mean']))
        close=[c for c in safe if c['matched']['ecapa']['mean']>=winner['matched']['ecapa']['mean']-.01]
        def cheap(c):
            n,p=c['condition'].split('-');return int(n[1:]),-int(p[3:])
        efficient=min(close,key=cheap) if close else item['conditions']['n10-top25']
        selection[voice]={'winner':winner['condition'],'efficient':efficient['condition'],
            'screen_quality_safe':winner['quality_safe'],
            'conditions':list(dict.fromkeys([winner['condition'],efficient['condition'],'n10-top25']))}
    return frozen(ROOT/'selection.json',selection)


def report():
    design=freeze();screen=summarize('screen')
    selection=json.loads((ROOT/'selection.json').read_text())
    expected=sum(20*(1+len(s['conditions'])) for s in selection.values())
    confirm=summarize('confirm') if len(read_rows(phase='confirm'))==expected else None
    vm=json.loads((ROOT/'vectors.json').read_text())
    rows=read_rows();public=[{k:v for k,v in r.items() if k not in ('speaker_embedding','traceback')} for r in rows]
    write_json(ROOT/'results.json',public)
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'takes':len(rows),
        'new_takes':sum(not r.get('reused') for r in rows),'reused':sum(r.get('reused',False) for r in rows),
        'failures':sum(bool(r.get('failed')) for r in rows),'screen':screen,'confirm':confirm,
        'selection':selection,'vectors':vm,'live_defaults_changed':False}
    write_json(ROOT/'summary.json',summary)
    esc=lambda x:html.escape(str(x),quote=True)
    fmt=lambda x:'—' if x is None else f'{x:+.3f}'
    parts=['<section class="seed-study" id="seed-scaling"><style>.seed-study{box-sizing:border-box;width:100%;min-width:0;overflow-wrap:anywhere;font:16px/1.55 system-ui;max-width:1200px;margin:40px auto;padding:24px;color:#eaf0f9;background:#142033;border-radius:16px}.seed-study table{border-collapse:collapse;min-width:0;width:100%;display:block;overflow-x:auto}.seed-study td,.seed-study th{padding:9px;border:1px solid #536277;text-align:left}.seed-study audio{width:280px;max-width:100%}.seed-study summary{cursor:pointer}.seed-study h2,.seed-study h3{color:#fff}.seed-study .safe{color:#8fe7b1}.seed-study .warn{color:#ffcf88}</style>',
        '<h2>More seeds and stricter similarity tails: Mira and Osku</h2>',
        '<p>Layer-16 high-minus-low activation steering, full-FT base only. Same reference in both pools; no DPO, conversion, trimming, best-of-N or quality retries. CFG 2.0, temperature 1.0, raw alpha 0.75. Live defaults are unchanged.</p>',
        '<h3>Findings table: fresh confirmation first</h3>']
    for phase,data in [('Fresh confirmation',confirm),('Exploratory screening',screen)]:
        if data is None:continue
        parts += ['<h3>'+phase+'</h3><table><tr><th>Voice / condition</th><th>Pairs</th><th>ECAPA</th><th>Δ ECAPA [95% interval]</th><th>Δ anchor cosine</th><th>Δ Content Enjoyment</th><th>Δ Production Quality</th><th>Δ MOS proxy</th><th>Δ WER</th><th>Δ emotion proxy</th><th>Safeguards</th></tr>']
        for voice,item in data.items():
            for key,value in item['conditions'].items():
                d=value['matched'];ci=d['ecapa'];safe=value['quality_safe']
                parts.append('<tr><td>'+esc(VOICES[voice]+' / '+key)+(' (alias)' if value.get('alias') else '')+'</td><td>'+str(value['n'])+'</td><td>'+f'{value["means"]["ecapa"]:.3f}'+'</td><td>'+f'{ci["mean"]:+.3f} [{ci["low"]:+.3f}, {ci["high"]:+.3f}]'+'</td>'+''.join('<td>'+fmt(d[k]['mean'] if d[k] else None)+'</td>' for k in ('anchor_ecapa','content_enjoyment','production_quality','mos','wer','emotion_score'))+'<td class="'+('safe' if safe else 'warn')+'">'+esc('passed numerical gates' if safe else ', '.join(value['quality_gate_reasons']))+'</td></tr>')
        parts.append('</table>')
    parts += ['<h3>What is being tested?</h3>',
        '<p>For each voice, 10 fixed text/emotion/language cells each have 50 stochastic seeded generations: 500 collection takes per voice. Nested prefixes use 10, 20 or 50 seeds <em>per identical condition</em> (100/200/500 total), not pooled seeds from different emotions. Existing first-ten repetitions were reused with exact provenance; 800 new collection takes were generated.</p>',
        '<p>Within each cell, average the top ceil(N × percentile) layer-16 prediction-state means, subtract the fixed bottom ceil(N × 25%) mean, then average ten cell differences equally. Stricter high-tail selection does not also shrink the low group. Top 5% and top 10% at N=10 are the exact same vector. At N=50, top 5% is rounded up to three takes (6%). Every sample mean includes only positions predicting actual audio frames, not prompt/reference states or EOS/padded tails.</p>',
        '<p>Each distinct candidate plus the existing UI vector was screened on 20 new matched takes (two repetitions of five emotions × two languages). Screen winners were frozen before another 20 fresh text/seed blocks. Bootstrap intervals cluster on actual batch RNG seed; they are exploratory and not corrected for all comparisons. ECAPA cosine is not an identity percentage. CE/PQ/MOS/emotion are learned proxies, not human ratings.</p>',
        '<h3>Size and strictness: where is the plateau?</h3><p>A small point estimate is not evidence of a plateau. We call the ECAPA comparison equivalent only if its interval is entirely inside ±0.01; otherwise the evidence is inconclusive. Comparisons below are screening diagnostics, not extra confirmation tests.</p><table><tr><th>Voice / paired comparison</th><th>Δ ECAPA [95% interval]</th><th>Interpretation</th><th>Δ CE</th><th>Δ PQ</th><th>Δ WER</th></tr>']
    for voice,item in screen.items():
        for key,cmp in item['comparisons'].items():
            ci=cmp['ecapa'];equivalent=ci['low']>=-.01 and ci['high']<=.01
            label='Equivalent within ±0.01 (pilot)' if equivalent else 'Clear positive gain' if ci['low']>0 else 'Clear negative gain' if ci['high']<0 else 'Inconclusive / not a proven plateau'
            if key=='n10-top5-minus-top10':label='Exact rounded-pool alias, not evidence of a plateau'
            parts.append('<tr><td>'+esc(VOICES[voice]+' / '+key)+'</td><td>'+f'{ci["mean"]:+.3f} [{ci["low"]:+.3f}, {ci["high"]:+.3f}]'+'</td><td>'+label+'</td>'+''.join('<td>'+fmt(cmp[k]['mean'])+'</td>' for k in ('content_enjoyment','production_quality','wer'))+'</tr>')
    parts += ['</table><h3>Pool diagnostics and perturbation size</h3><p>Raw alpha was fixed. Tightening a tail changes both direction and magnitude; therefore a gain cannot be attributed solely to a cleaner identity direction. Norms and selected-pool similarity below make that confound visible. Sixty within-cell bootstrap resamples estimate vector uncertainty only; they are not extra generated audio or quality evidence.</p><table><tr><th>Voice / vector</th><th>Collection total</th><th>High/cell</th><th>Low/cell</th><th>High ECAPA</th><th>Low ECAPA</th><th>Vector L2</th><th>Cosine to N10 top25</th><th>Bootstrap cosine stability</th></tr>']
    for voice,items in vm.items():
        for key,item in items.items():
            if key in ('pairwise_cosine','original-ui'):continue
            cell=item['cells'][0]
            stability=item['bootstrap_stability']
            parts.append('<tr><td>'+esc(VOICES[voice]+' / '+key)+'</td><td>'+str(item['total_collection'])+'</td><td>'+str(cell['high_n'])+'</td><td>'+str(cell['low_n'])+'</td><td>'+f'{item["mean_selected_ecapa"]:.3f}'+'</td><td>'+f'{item["mean_rejected_ecapa"]:.3f}'+'</td><td>'+f'{item["l2"]:.3f}'+'</td><td>'+f'{items["pairwise_cosine"][key]["n10-top25"]:.3f}'+'</td><td>'+f'{stability["cosine_to_estimate_median"]:.3f} [{stability["cosine_interval"][0]:.3f}, {stability["cosine_interval"][1]:.3f}]'+'</td></tr>')
    parts += ['</table><h3>Listen: identical held-out task / seed pairs</h3><p>Mono 96 kbps MP3, embedded Base64; full generated output without alignment fades. Files are not cherry-picked: all confirmation pairs are included.</p>']
    def player(path):
        return '<audio controls preload="none" src="data:audio/mpeg;base64,'+base64.b64encode(Path(path).read_bytes()).decode()+'"></audio>'
    if confirm:
        for voice in VOICES:
            parts.append('<h3>'+esc(VOICES[voice])+' — fixed reference</h3>'+player(references.fixed_clip(voice,playback=True)[0]))
            cases=read_rows(voice,'confirm');controls={r['block']:r for r in cases if r['condition']=='baseline'}
            for block,control in sorted(controls.items()):
                parts.append('<details><summary>'+esc(f'Task {block}: {control["emotion"]} / {control["language"]} — '+control['text'])+'</summary><p>No steering</p>'+player(control['audio_path']))
                for row in cases:
                    if row['block']==block and row['condition']!='baseline':
                        parts.append('<p>'+esc(row['condition'])+f' — ECAPA {row["ecapa"]:.3f}, CE {row["content_enjoyment"]:.2f}, WER {row["wer"]:.3f}</p>'+player(row['audio_path']))
                parts.append('</details>')
    parts += ['<h3>Audit and limitations</h3><p>'+esc(f'{summary["takes"]} total measured takes; {summary["new_takes"]} newly generated, {summary["reused"]} reused; {summary["failures"]} technical failures.')+'</p>',
        '<p>'+esc(' '.join(design['limits']))+'</p>',
        '<details><summary>Frozen design, selections and full numerical summary</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'+esc(json.dumps({'design':design,'selection':selection,'summary':summary},ensure_ascii=False,indent=2))+'</pre></details></section>']
    section=''.join(parts)
    (ROOT/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Seed-count and tail-selection speaker study</title><body style="background:#0c1422">'+section+'</body></html>')
    # Publish through the existing HTTPS report route, without restarting TTS.
    target=OLD_ROOT/'index.html';existing=target.read_text()
    merged=stats.insert_report_extension(existing,'seed-scaling',section)
    temp=target.with_name('index.seed-build.html');temp.write_text(merged);temp.replace(target)
    print(json.dumps({'takes':summary['takes'],'new':summary['new_takes'],'confirmation_ready':confirm is not None,
                     'url':'https://switch-surprised-clock-sparc.trycloudflare.com/catalogue-steering-study/#seed-scaling'},indent=2))


# Reuse the audited local scorer implementation, restricted to this new study.
old.ROOT=ROOT
score_app=old.score_app


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['design','collection','derive','screen','select','confirm','report'])
    p.add_argument('--voice',choices=list(VOICES));p.add_argument('--device',default='cuda:0')
    p.add_argument('--score-url',default='http://127.0.0.1:8795');args=p.parse_args()
    if args.action=='design':freeze()
    elif args.action=='derive':derive()
    elif args.action=='select':select()
    elif args.action=='report':report()
    else:run(args.voice,args.action,args.device,args.score_url)


if __name__=='__main__':main()
