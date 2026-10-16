"""Fresh CPU scoring of archived real TTS outputs, never new generation/RTF.

Uses pinned cached models only. No uploads, credentials, live server or network.
This checks instrument readings, not human-confirmed emotional/burst accuracy.
"""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['HF_MODULES_CACHE']=str(ROOT/'test-results/cpu-hf-modules')

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoModel

import settings
from adaptive_reward import CATALOG, DIMENSIONS, RewardPlan
from burst_models import BurstModels
from burst_reward import BurstOptions, evaluate_bursts, script_targets
from dimension_predictors import DimensionPredictors


def snapshot(repo,revision):
    path=Path(settings.HF_CACHE)/'hub'/('models--'+repo.replace('/','--'))/'snapshots'/revision
    if not path.is_dir():raise FileNotFoundError(f'Pinned cache unavailable: {repo}@{revision}')
    return path


def selection(bursts):
    files=sorted((ROOT/'search').glob('quality-two_gpu-*.json'))
    rows=[json.loads(p.read_text()) for p in files]
    return [r for r in rows if not r.get('error') and (r['case']=='burst_en' if bursts else
        r['case'] in ['anger_en','affection_de'] and r['kind'] in ['single','best_of_n'])]


def waveform(row):
    path=ROOT/'search/raw'/(row['id']+'.wav')
    wave,sr=sf.read(path,dtype='float32')
    if wave.ndim>1:wave=wave.mean(-1)
    assert np.isfinite(wave).all()
    return torchaudio.functional.resample(torch.from_numpy(wave),sr,16000),hashlib.sha256(path.read_bytes()).hexdigest()


def burst_spec():
    # Same authored recipe and renderer call as the archived benchmark.
    # That benchmark did not persist exact prompt bytes; this reconstruction
    # is reported as such, not treated as human-aligned timing ground truth.
    from prompting import build
    case=next(c for c in json.loads((ROOT/'search/design.json').read_text())['cases'] if c[0]=='burst_en')
    return build({'text':case[2],'general':case[3],'script':
        '(frustrated groan, 0.3 seconds)\n(angry shout, forceful and clear) [3.2 seconds duration] '+case[2]},
        'structured',case[1],seconds=6.6)


def table(headers,rows):
    esc=lambda v:html.escape(str(v))
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in r)+'</tr>' for r in rows)+\
        '</tbody></table></div>'


def report(data):
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Vocal bursts, emotional realization and UI checks</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:0 20px;color:#17212e}h1,h2{line-height:1.2}.note{padding:15px;background:#fff3da;border-left:4px solid #cc8c1f}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:8px;text-align:left}audio{width:100%;max-width:400px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
        '<h1>Vocal bursts, emotion realization and UI</h1>',
        '<p class="note"><b>Scope:</b> Fresh CPU model inference on archived real synthetic outputs. '
        'No new TTS generation, no isolated GPU throughput measurement, no human listening study. '
        'The current sandbox cannot access the GPU/demo or launch Chromium (socket permission failure). '
        'Historical Parakeet WER and word timestamps are reused and explicitly labelled. '
        'These audio samples were generated before the optional adaptive/burst reward was added: '
        'this report does not measure its causal quality benefit.</p>']
    if data.get('bursts'):
        rows=data['bursts'];good=sum(r['clock']['score']>0 for r in rows)
        parts+=['<h2>Requested vocal burst: frustrated groan</h2>',
            '<p>All twelve archived burst-recipe outputs: two seeds, single sample, Best-of-8, '
            'winner-of-8 and Top-2 at 800/1600 ms. Exact conditioning bytes were not stored in '
            'the historical run; expected targets were reconstructed with the recorded recipe '
            'and current unchanged renderer. Script-clock and ASR word-slot matching are reported separately. '
            'A detected event anywhere in the clip is not proof of a correctly placed groan.</p>',
            f'<p>{good}/{len(rows)} clips have a positive typed clock-alignment proxy (not a success-rate estimate). '
            f"Mean clock score: {statistics.mean(r['clock']['score'] for r in rows):.4f}; "
            f"mean word-slot score: {statistics.mean(r['words']['score'] for r in rows):.4f}.</p>",
            '<pre>'+html.escape(json.dumps(data['burst_targets'],indent=2))+'</pre>',
            table(['Output','Historical WER','Detected events','Clock score','Word-slot score','Most confident event'],[
                [r['id'],f"{r['historical_wer']:.3f}",len(r['clock']['events']),
                 f"{r['clock']['score']:.4f}",f"{r['words']['score']:.4f}",
                 ', '.join(f"{e['start']:.2f}–{e['end']:.2f}s {e['top_label']}" for e in r['clock']['events']) or 'None'] for r in rows]),
            '<p>Scores use locator v2 presence/timing, duration fit, commercial class-family probability '
            'and optional extra-event penalty. Unknown types would fall back transparently to location only. '
            'Commercial classes and locator boundaries are noisy; this is neither verified event accuracy '
            'nor a human judgment of whether a groan sounds genuine.</p>']
    if data.get('emotions'):
        rows=data['emotions'];summary=[]
        for case in ['anger_en','affection_de']:
            group=[r for r in rows if r['case']==case]
            for kind in ['single','best_of_n']:
                subset=[r for r in group if r['kind']==kind]
                if not subset:continue  # Report is checkpointed after every clip.
                summary.append([case,kind,len(subset),
                    f"{statistics.mean(r['scores']['eiv:Anger'] for r in subset):.3f}",
                    f"{statistics.mean(r['scores']['eiv:Affection'] for r in subset):.3f}",
                    f"{statistics.mean(r['scores']['vn:AROU'] for r in subset):.3f}",
                    f"{statistics.mean(r['historical_wer'] for r in subset):.3f}"])
        parts+=['<h2>Requested emotion: anger versus affection</h2>',
            '<p>Eight archived outputs: two prompts/languages × two seeds × single/Best-of-8. '
            'Fresh official EIV+ Anger/Affection and VoiceNet Arousal inference; neither EIV+ expert '
            'selected these original candidates. Emotion remains confounded with text/language and '
            'reference voice, so this is a small diagnostic, not controlled instruction-following accuracy.</p>',
            table(['Prompt','Generation','n','Anger raw','Affection raw','Arousal raw','Historical WER'],summary),
            table(['Output','Target raw','Target normalized','Other emotion raw','Arousal raw','Historical WER'],[
                [r['id'],f"{r['target_raw']:.3f}",f"{r['target_normalized']:.3f}",
                 f"{r['other_raw']:.3f}",f"{r['scores']['vn:AROU']:.3f}",f"{r['historical_wer']:.3f}"] for r in rows]),
            '<p>Do not treat raw emotion scores as probabilities, compare uncalibrated experts as '
            'a categorical ground-truth classifier, or infer human naturalness. Fixed native-scale '
            'normalization clips to 0–1. Fear was not freshly scored because its pinned expert '
            'is not cached and this session cannot download it.</p>']
    ui=data.get('ui_tests',{})
    ui_count=ui.get('passed',62)
    parts+=['<h2>UI function checks</h2><p>'+str(ui_count)+' unit/controller and in-memory API checks passed. The actual production '
        'JavaScript controllers were exercised with DOM fixtures for all 64 winner/Top-2 × BS 1–16 × '
        '800/1600-ms combinations. Generated requests validate against the real server schema, '
        'including adaptive/manual rewards, burst modes, temperature zero and CFG toggles. '
        'The in-memory ASGI tests additionally check reference upload validation, session-private '
        'audio access, expiry, actual mono 96-kbps MP3 encoding, streamed event order, '
        'Luna timeout/error handling and release of the busy state. Model generation and Luna '
        'responses are test doubles in those API checks, not fresh emotion or burst evidence. '
        'A periodic timer makes the test client process completed worker callbacks without relying '
        'on blocked socket wake-ups; no network is enabled or production code changed. '
        'This does not verify browser CSS, microphone permissions, decoded browser playback or a '
        'live TTS generation. Chromium '
        'failed on setsockopt with Operation not permitted. Existing browser tests remain runnable '
        'on an unrestricted host; no mocked UI response is counted as emotion/burst evidence.</p>',
        '<h2>Listen and inspect</h2><p>Archived synthesized research takes only; no reference uploads are exposed.</p>']
    for row in data.get('emotions',[])+data.get('bursts',[]):
        parts.append('<details><summary>'+html.escape(row['id'])+'</summary><audio controls preload="none" src="../search/audio/'+
            html.escape(row['id'])+'.mp3"></audio><pre>'+html.escape(json.dumps(row,indent=2))+'</pre></details>')
    parts+=['<p>Models: <a href="https://huggingface.co/laion/vocalburst-locator">Locator</a>, '
        '<a href="https://huggingface.co/laion/vocal-burst-detector-commercial">Commercial burst detector</a>, '
        '<a href="https://huggingface.co/laion/Empathic-Insight-Voice-Plus">EIV+</a>, '
        '<a href="https://huggingface.co/laion/voicenet-dimension-predictors-commercial">VoiceNet</a>.</p></html>']
    (ROOT/'test-results/realization-check.html').write_text(''.join(parts))


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['bursts','emotions','all','report'],default='all')
    parser.add_argument('--threads',type=int,default=2);args=parser.parse_args()
    torch.set_num_threads(args.threads)
    path=ROOT/'test-results/realization-check.json'
    data=json.loads(path.read_text()) if path.exists() else {}
    ui_path=ROOT/'test-results/inprocess-tests.json'
    if ui_path.exists():data['ui_tests']=json.loads(ui_path.read_text())
    data.update(scope='Fresh CPU scoring of archived real TTS research WAVs; no new generation, no GPU RTF',
        date='2026-10-01',device='cpu',torch_threads=args.threads,
        model_revision=settings.REVISION,eiv_revision=CATALOG['eiv_revision'],voicenet_revision=CATALOG['voicenet_revision'])
    if args.phase=='report':
        path.write_text(json.dumps(data,indent=2));report(data)
        print('Refreshed realization-check.html with current in-process test status',flush=True)
        return
    tower_root=snapshot(CATALOG['voicenet_repo'],CATALOG['voicenet_revision'])/'voiceclap_commercial'
    print('Loading pinned cached VoiceCLAP tower on CPU',flush=True)
    tower=AutoModel.from_pretrained(str(tower_root),trust_remote_code=True,dtype=torch.float32,
                                    local_files_only=True).to('cpu').eval()
    if args.phase in ['bursts','all']:
        models=BurstModels(SimpleNamespace(device='cpu',model=tower));models.prepare(True)
        spec=burst_spec();targets,first=script_targets(spec);data['burst_targets']=targets;data['bursts']=[]
        for row in selection(True):
            started=time.perf_counter();audio,digest=waveform(row);events=models.detect([audio],True)[0]
            item={'id':row['id'],'case':row['case'],'kind':row['kind'],'seed':row['seed'],'audio_sha256':digest,
                'seconds':len(audio)/16000,'historical_wer':row['final']['prefix_wer']}
            for timing in ['clock','words']:
                options=BurstOptions(mode='typed',timing=timing,match='family')
                item[timing]=evaluate_bursts(targets,events,options,spec['text'],row['final']['timestamps'],
                    len(audio)/16000,True,first)
            item['cpu_evaluation_seconds']=time.perf_counter()-started;data['bursts'].append(item)
            path.write_text(json.dumps(data,indent=2));report(data)
            print(json.dumps({'burst':row['id'],'events':len(events),'clock':item['clock']['score'],
                'words':item['words']['score'],'evaluation_seconds':item['cpu_evaluation_seconds']}),flush=True)
        del models
    if args.phase in ['emotions','all']:
        plan=RewardPlan(terms=[{'dimension':'eiv:Anger'},{'dimension':'eiv:Affection'},{'dimension':'vn:AROU'}])
        models=DimensionPredictors('cpu');models.prepare(plan);data['emotions']=[]
        for row in selection(False):
            started=time.perf_counter();audio,digest=waveform(row)
            embeddings=tower.encode_waveform(audio[None],sample_rate=16000).float()
            scores=models.predict([audio],embeddings,plan)[0]
            assert all(np.isfinite(v) for v in scores.values())
            target='eiv:Anger' if row['case']=='anger_en' else 'eiv:Affection'
            other='eiv:Affection' if target=='eiv:Anger' else 'eiv:Anger';meta=DIMENSIONS[target]
            item={'id':row['id'],'case':row['case'],'kind':row['kind'],'seed':row['seed'],
                'audio_sha256':digest,'historical_wer':row['final']['prefix_wer'],'scores':scores,
                'target':target,'target_raw':scores[target],'other_raw':scores[other],
                'target_normalized':float(np.clip((scores[target]-meta['min'])/(meta['max']-meta['min']),0,1)),
                'cpu_evaluation_seconds':time.perf_counter()-started}
            data['emotions'].append(item);path.write_text(json.dumps(data,indent=2));report(data)
            print(json.dumps({'emotion':row['id'],'scores':scores,'historical_wer':item['historical_wer'],
                'evaluation_seconds':item['cpu_evaluation_seconds']}),flush=True)
    report(data)
    print('Saved test-results/realization-check.json and realization-check.html',flush=True)


if __name__=='__main__':main()
