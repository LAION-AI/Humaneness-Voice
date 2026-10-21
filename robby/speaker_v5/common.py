import html,json,os,shutil,time
from pathlib import Path
from speaker_v4.common import OLD,atomic,digest

ROOT=Path(__file__).resolve().parent
WORK=Path.home()/'.cache/robo-v5'
SEEDS=(2021,2022,2023)
CONTEXTS=(4,25,38)
ARMS={'utterance':(0.,None),'stable20':(.2,None),'hard':(0.,'hard'),
      'stable20_hard':(.2,'hard'),'stable20_random':(.2,'random')}

def guard(cache=None):
    if (ROOT/'STOP').exists():raise RuntimeError('V5 STOP requested; retained epoch checkpoints are resumable')
    WORK.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(WORK).free<3*1024**3:raise RuntimeError('Persistent work volume below 3-GiB reserve')
    if cache is not None and shutil.disk_usage(cache).free<8*1024**3:raise RuntimeError('Temporary feature volume below 8-GiB reserve')

def update(**fields):
    path=ROOT/'progress.json';state=json.loads(path.read_text()) if path.exists() else {'started':time.time()}
    state.update(fields,updated=time.time());atomic(path,state)
    WORK.mkdir(parents=True,exist_ok=True)
    with (WORK/'events.jsonl').open('a') as out:out.write(json.dumps({'time':time.time(),**fields})+'\n')

def report(results=None):
    if results is not None:atomic(WORK/'results.json',results)
    else:results=json.loads((WORK/'results.json').read_text()) if (WORK/'results.json').exists() else {}
    table=[]
    for key,group in results.get('aggregates',{}).items():
        def value(split,metric,scale=1):
            v=group.get(split,{}).get(metric)
            return 'Pending' if v is None else f"{v['mean']*scale:.3f} ± {v['std_across_seeds']*scale:.3f}"
        table.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [key,group['seeds'],
            value('validation','macro_speaker_accuracy',100),value('test','macro_speaker_accuracy',100),
            value('test','eer',100),value('test','source_utterance_teacher_cosine'),value('test','positive_cosine'),
            value('test','impostor_cosine'),group.get('student_gpu_batch1_ms','Pending')])+'</tr>')
    status=json.loads((ROOT/'progress.json').read_text()) if (ROOT/'progress.json').exists() else {}
    body='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Speaker context optimization: 320 ms, 2 s, 3 s</title>
<style>body{font:16px/1.6 system-ui;max-width:1200px;margin:35px auto;padding:20px;background:#faf8f3;color:#403a30}td,th{padding:9px;border-bottom:1px solid #ccc;text-align:left}.scroll{overflow:auto}pre{white-space:pre-wrap;font-size:12px}</style>
<a href="/speaker-training/">Training overview</a> · <a href="/speaker-training/320ms-study/">320-ms layer/hard-negative study</a>
<h1>Context-specific speaker prediction</h1><p id="phase">'''+html.escape(status.get('phase','Queued'))+'''</p>
<p>Dedicated training at 320 ms, 2000 ms and 3040 ms (4, 25, 38 frames). A nominal three-second window rounds up to 3.04 s on the native 80-ms grid. Each prediction still identifies the current/newest 320-ms speaker, NOT the majority speaker over the longer context. Every hidden state already sees causal history from the same original eight-second encoder input; these are not cold-start audio-duration limits.</p>
<p>Matched 480/10/10 synthetic speaker split, 96,000 training and 10,000 validation/test windows each, three paired seeds per condition. Warm start is the same frozen V3 Query-GRU, not a different per-context winner. Compare utterance-only targets, 20% stable training-centroid loss, hard cross-emotion/near-speaker pairs, stable+hard, and a compute-matched stable+random-pairs control. No layer mixing is added here, isolating duration and training recipe. The preceding lower-layer study is retained separately.</p>
<p>Each checkpoint is selected on its own context's validation objective: macro accuracy − 0.5 EER + 0.1 source-utterance cosine. All configurations and context winners are frozen before new test evaluation. These ten synthetic test identities were benchmarked previously; this is a repeated benchmark, not a new independent human population. Genuine and impostor cosine are both reported. No microphone recordings or enrolled-user embeddings are used.</p>
<h2>Summary</h2><p>Mean ± SD across seeds, not a confidence interval across people. Pending test cells are not results. Latency is student-only batch 1, excluding Nemotron.</p><div class="scroll"><table><tr><th>Context / recipe</th><th>Seeds</th><th>Val accuracy %</th><th>Test accuracy %</th><th>Test EER %</th><th>Utterance cosine</th><th>Enrollment cosine</th><th>Impostor cosine</th><th>GPU ms</th></tr>'''+''.join(table)+'''</table></div>
<h2>Findings</h2>'''+''.join('<p>'+html.escape(v)+'</p>' for v in results.get('conclusions',[]))+'''
<h2>Speaker changes and storage</h2><p>Accuracy is also separated by time since the current source utterance began (&lt;320 ms, 320–640 ms, 640–1000 ms, ≥1 s). These are onset/re-entry slices, not proof of clean single-speaker intervals or a measured gate-response delay. Silence and ambiguous overlap retain their own activity labels, without speaker-regression targets. Paired identity bootstrap comparisons average seeds before resampling identities.</p>
<p>The existing 25-frame cache is reused read-only. Only the missing 13 older L15 frames and energy are cached for the exact same windows, in bounded temporary RAM (~1.55 GB). The suffix must match original 25-frame features at cosine ≥0.995. Source audio is replayed from pinned public clips, not downloaded again. The worker waits for the earlier study to release GPU 0; live TTS on GPU 1 is untouched. Models/results persist on disk; the regenerable RAM cache can be lost on reboot. No automatic live identity-filter promotion.</p>
<h2>Live progress</h2><pre id="progress">'''+html.escape(json.dumps(status,indent=2))+'''</pre><h2>Protocol and complete results</h2><pre>'''+html.escape(json.dumps(results,indent=2,allow_nan=False))+'''</pre>
<script>async function poll(){try{let s=await fetch('/api/speaker-v5',{cache:'no-store'}).then(r=>r.json());document.getElementById('phase').textContent=s.phase;document.getElementById('progress').textContent=JSON.stringify(s,null,2)}catch{}}setInterval(poll,15000)</script></html>'''
    WORK.mkdir(parents=True,exist_ok=True);p=WORK/'report.tmp.html';p.write_text(body);p.replace(WORK/'report.html')
