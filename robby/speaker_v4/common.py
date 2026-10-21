import hashlib,html,json,os,shutil,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OLD=Path.home()/'.cache/robo-v3'
WORK=Path.home()/'.cache/robo-v4'
FEATURES=ROOT.parent/'.cache/speaker-v4'
SEEDS=(2021,2022,2023)
_previous=json.loads((ROOT/'progress.json').read_text()) if (ROOT/'progress.json').exists() else {}
STATE={'phase':'Preparing','started':_previous.get('started',time.time())}

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def guard():
    if (ROOT/'STOP').exists():raise RuntimeError('STOP requested; completed artifacts retained')
    if FEATURES.is_symlink() and not FEATURES.exists():
        target=FEATURES.readlink()
        if target.parent==Path('/dev/shm') and target.name.startswith('robo-v4-layers-'):
            FEATURES.unlink()  # only our dangling temporary-cache pointer
        else:raise RuntimeError('Unexpected dangling cache symlink; refusing to change it')
    for path in (WORK,FEATURES):
        path.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(path).free<3*1024**3:raise RuntimeError('Storage safety stop: less than 3 GiB free on '+str(path))

def update(**fields):
    STATE.update(fields,updated=time.time());atomic(ROOT/'progress.json',STATE)
    WORK.mkdir(parents=True,exist_ok=True)
    with (WORK/'events.jsonl').open('a') as out:out.write(json.dumps({'time':time.time(),**fields},allow_nan=False)+'\n')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(2**20),b''):h.update(part)
    return h.hexdigest()

def cleanup_unused():
    """Only unreferenced surplus MP3s from our completed selective downloader."""
    sources=OLD/'sources';referenced=set()
    for row in json.loads((OLD/'sources_manifest.json').read_text()):
        referenced.update(Path(c['path']).resolve() for c in row['clips'])
    for path in sources.glob('*/clips.json'):
        referenced.update(Path(c['path']).resolve() for c in json.loads(path.read_text()))
    candidates=[]
    for path in sources.glob('*/*.mp3'):
        if path.is_symlink() or not path.is_file() or path.resolve() in referenced:continue
        if path.resolve().parent.parent!=sources.resolve():raise RuntimeError('Cleanup scope mismatch')
        candidates.append({'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path)})
    atomic(WORK/'cleanup_manifest.json',{'recovery':'Regenerate surplus downloads from pinned V3 dataset ranges; no local trash copy',
        'retained_referenced_clips':len(referenced),'files':candidates})
    for item in candidates:
        path=Path(item['path'])
        if path.resolve() in referenced or digest(path)!=item['sha256']:raise RuntimeError('Cleanup verification failed')
        path.unlink()
    update(cleaned_surplus_files=len(candidates),cleaned_bytes=sum(c['bytes'] for c in candidates))

def report(results=None,finished=False):
    if results is not None:atomic(WORK/'results.json',results)
    data=results if results is not None else json.loads((WORK/'results.json').read_text()) if (WORK/'results.json').exists() else {}
    rows=[];summary=[]
    for stage in ('stage1','factorial'):
        for arm,group in data.get(stage+'_aggregates',{}).items():
            v=group.get('validation',{});t=group.get('test',{})
            if not v:continue
            def value(values,key,percent=False):
                if key not in values:return '-'
                multiplier=100 if percent else 1
                return f"{values[key]['mean']*multiplier:.2f} ± {values[key]['std_across_seeds']*multiplier:.2f}"+('%' if percent else '')
            summary.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [stage,arm,group['seeds'],
                value(v,'macro_speaker_accuracy',True),value(v,'eer',True),value(t,'macro_speaker_accuracy',True),
                value(t,'source_utterance_teacher_cosine')])+'</tr>')
    for stage in ('stage1','factorial'):
        for row in data.get(stage,[]):
            v=row['validation'];test=row.get('test',{})
            rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [stage,row['arm'],row['seed'],
                f"{100*v['macro_speaker_accuracy']:.1f}%",f"{v['source_utterance_teacher_cosine']:.3f}",f"{100*v['eer']:.1f}%",
                '-' if not test else f"{100*test['macro_speaker_accuracy']:.1f}%",'-' if not test else f"{test['source_utterance_teacher_cosine']:.3f}"] )+'</tr>')
    body='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>320-ms speaker ablations</title>
<style>body{font:16px/1.65 system-ui;max-width:1150px;margin:40px auto;padding:20px;background:#faf8f3;color:#403a30}td,th{padding:10px;text-align:left;border-bottom:1px solid #ccc}pre{font-size:12px;white-space:pre-wrap}.scroll{overflow:auto}a{color:#80582e}</style>
<a href="/speaker-training/">← Original study</a><h1>Improving 320-ms speaker predictions</h1><p id="status">'''+html.escape(STATE.get('phase','Preparing'))+'''</p>
<p>Stage 1: specialize the existing Query-GRU for four newest 80-ms states, and compare utterance targets against mixtures with robust training-speaker centroids (initial 50% or 80% centroid objective). If both initial mixtures lose to specialization alone on validation, a documented validation-only extension tests a gentler 20% mixture before test evaluation. Stage 2: matched 2×2 factorial on a fixed subset: baseline (1+2), layer mixing (+3), hard-example learning (+4), and both (+3+4). Three fixed seeds per arm. Layer IDs are one-based 4, 8, 12 and 15; all are frozen, from the same causal eight-second input and four newest states.</p>
<p><strong>320 ms means the state-pooling span, not a 320-ms-only cold-start audio limit.</strong> Encoder history is unchanged. The original 480/10/10 identity split is retained. No private microphone recordings or enrolled user voiceprints. Original V3 artifacts are not overwritten.</p>
<p>All checkpoint, centroid weight, and experiment selection use validation identities only. Test results are deferred until all arms/seeds are frozen. The existing ten test speakers were used in the preceding study; these are repeated benchmark comparisons, not a new independent test population. Unknown-speaker operating thresholds are calibrated on validation.</p>
<p>A higher genuine-speaker cosine alone is not sufficient: a changed target can raise impostor similarities too. Identification, EER and genuine-versus-impostor cosine must be considered together. Utterance-target cosine and enrollment-reference cosine are different metrics and are both retained in the detailed results.</p>
<h2>Summary across seeds</h2><p>Mean ± standard deviation across available seeds; this is not a confidence interval over independent people. Partial groups are provisional. New test metrics remain blank until all arms are frozen.</p>
<div class="scroll"><table><tr><th>Stage</th><th>Arm</th><th>Seeds</th><th>Val macro accuracy</th><th>Val EER</th><th>Test macro accuracy</th><th>Test utterance cosine</th></tr>'''+''.join(summary)+'''</table></div>
<h2>Individual runs</h2><div class="scroll"><table><tr><th>Stage</th><th>Arm</th><th>Seed</th><th>Val macro accuracy</th><th>Val utterance cosine</th><th>Val EER</th><th>Test macro accuracy</th><th>Test utterance cosine</th></tr>'''+''.join(rows)+'''</table></div>
<h2>Resources and interpretation</h2><p>Stage 1 reuses the 500,000-window V3 cache. Factorial training uses the same fixed 200 windows per training anchor voice (96,000 training windows) plus all 10,000 validation and 10,000 test windows; do not attribute differences between full-data and subset stages solely to architecture. Only three additional layers' four-frame states are cached, about 1.43 GB int8 plus float16 scales. No complete audio or hidden-state sequences are duplicated. The derived cache may be migrated, after hash verification, to bounded RAM-backed temporary storage when disk space is tight. It is regenerable from preserved public sources/seeds; a reboot can lose this cache, not the persistent models, original dataset or results. One GPU is used; live TTS on the other GPU is untouched. Browser audio sessions have priority between encoder batches. A three-GiB floor on working filesystems stops further work without deleting results.</p>
<p>Stable centroids use only the training voices' prediction clips, not held-out probe audio. The enrollment gallery remains the original disjoint reference clips. Hard-example arms add cross-emotion positive pairs and nearest-speaker negative pairs, preferring matching full-utterance transcript signatures when available. Matching a full transcript does not guarantee matching phonemes in a 320-ms crop. Additional pair forwards mean higher training compute; measured costs and pair-match rates are recorded. A random-pairs control uses the same extra pair losses/forwards without targeted mining. No guaranteed accuracy gain or real-human authentication claim.</p>
<h2>Interpretation</h2>'''+''.join('<p>'+html.escape(line)+'</p>' for line in data.get('conclusions',[]))+'''
<p>The factorial retains (1+2) as its matched baseline even if stable targets do not beat specialization alone. The full-data specialization-only control is reported separately; selecting a centroid weight does not establish that centroids help.</p>
<h2>Progress / protocol / complete results</h2><pre id="progress"></pre><pre>'''+html.escape(json.dumps(data,indent=2,allow_nan=False))+'''</pre>
<script>async function poll(){try{let s=await fetch('/api/speaker-v4',{cache:'no-store'}).then(r=>r.json());document.getElementById('status').textContent=s.phase;document.getElementById('progress').textContent=JSON.stringify(s,null,2)}catch{}}poll();setInterval(poll,15000)</script></html>'''
    WORK.mkdir(parents=True,exist_ok=True);tmp=WORK/'report.tmp.html';tmp.write_text(body);tmp.replace(WORK/'report.html')
