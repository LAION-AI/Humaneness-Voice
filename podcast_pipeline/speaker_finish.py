"""Independent final audit and two-language 50/50 speaker-reward listening report."""
import argparse
import gc
import html
import json
from pathlib import Path
import shutil
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from recipe import HERE, WORKSPACE, SR, dump, errors, sha
from finish import export_episode  # Resolve local module before setup edits sys.path.
from run import setup

def page(out,receipts,config,times):
    rows=[];cards=[]
    prompting=('The original restrained turn-level direction is used. Repeated natural/authentic tags and extra sentence-by-sentence emotions are disabled.'
        if config.get('prompt_style')=='legacy' else
        'Every sentence starts with the inline direction “Very genuine, very natural, very authentic”, followed by a context-appropriate, restrained emotion: curiosity, wonder, awe, skepticism, amusement, tenderness or sadness. These directions are not intended to be spoken.')
    for r in receipts:
        sec=round(r['seconds']);label=r['language'].upper()
        rows.append(f"<tr><td>{label}</td><td>{sec//60}:{sec%60:02d}</td><td>{r['inference_seconds']:.1f} s</td><td>{r['useful_rtf']:.3f}</td><td>{r['mean_genuineness']:.3f}</td><td>{r['mean_speaker_cosine']:.3f}</td><td>{r['mean_wer']:.1%}</td></tr>")
        stem=Path(r['file']).stem
        cards.append(f"<section><h2>{label} · Best-of-{config['n']}</h2><audio controls preload='none' src='{r['file']}'></audio><p><a href='{r['file']}' download>Download MP3</a> · <a href='{stem}.audit.json'>Scores and transcripts</a> · <a href='{stem}.srt'>Subtitles</a></p><p>Review turns (zero-based): {html.escape(str(r['review_turns']))}</p></section>")
    refs=''.join(f"<p>{s}<audio controls preload='none' src='references/{Path(v['path']).name}'></audio></p>" for s,v in config['references'].items())
    comparison=''
    previous_path=out/'previous-bestof4-speaker-comparison.json'
    if previous_path.exists():
        previous=json.loads(previous_path.read_text())['rows'];lines=[]
        for r in receipts:
            old=[p for p in previous if p['language']==r['language']]
            if old:
                lines.append(f"<tr><td>{r['language'].upper()}</td><td>{statistics.mean(p['ecapa_cosine'] for p in old):.3f} → {r['mean_speaker_cosine']:.3f}</td><td>{statistics.mean(p['genuineness'] for p in old):.3f} → {r['mean_genuineness']:.3f}</td></tr>")
        if lines:comparison=f"<h2>Comparison with the previous Best-of-4</h2><div class='scroll'><table><tr><th>Language</th><th>Speaker cosine · previous → new</th><th>Genuineness / 6 · previous → new</th></tr>{''.join(lines)}</table></div><p>Same reference embeddings and scripted content; both the reward and sentence directions changed. These are model-score means over twenty turns, not a blinded listening result or a claim of statistical significance. <a href='previous-bestof4-speaker-comparison.json'>Per-turn comparison</a>.</p>"
    followup=''
    if (out/'asr-review-en08.json').exists():
        followup="<p><strong>ASR follow-up:</strong> English turn 8 has inconsistent recognizer output: the whole edited clip omits the final sentence in ASR, while ASR of its unchanged tail recognizes that sentence. The raw clip also yields it. The original diagnostic WER is retained; this flag is not proof of a missing spoken sentence. Turn 17 includes a Frankl/name segmentation mismatch. <a href='asr-review-en08.json'>Additional recognition evidence</a>.</p>"
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Podcast — genuineness and speaker identity</title>
<style>body{{font:17px/1.55 system-ui;color:#252333;background:#f7f3eb;max-width:1100px;margin:30px auto;padding:16px;overflow-wrap:anywhere}}.scroll{{overflow:auto}}table{{border-collapse:collapse;background:white;width:100%}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #ddd;overflow-wrap:normal}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(320px,100%),1fr));gap:20px}}section{{background:white;border-radius:16px;padding:24px;min-width:0}}audio{{width:100%;min-width:0}}a{{color:#5a40a0}}pre{{white-space:pre-wrap}}</style>
<h1>Genuineness + speaker consistency</h1><p>Fresh English and German episodes using the actual Podcast checkpoint, the same two published German reference clips, and complete-turn Best-of-{config['n']}. No streaming and no voice conversion.</p>
<p><strong>Reward: 50% genuineness + 50% reference speaker similarity.</strong> Genuineness is clipped after division by 6. Speaker similarity is ECAPA cosine clipped to [0,1]. These are fixed scales, not per-batch min/max values. WER is a final diagnostic and does not influence selection.</p>
<div class="scroll"><table><tr><th>Language</th><th>Duration</th><th>Generation + ranking</th><th>Useful RTF</th><th>Genuineness / 6</th><th>Speaker cosine</th><th>Mean turn WER</th></tr>{''.join(rows)}</table></div>
<p>RTF is generation/ranking time divided by final edited episode duration. Lower is faster; below 1 is faster than real time. Loading, timing probes, final audit and encoding are separate. Each shared batch is counted once, divided equally between its independent segments. This is not request-to-playback latency.</p><main>{''.join(cards)}</main>
<h2>Performance directions</h2><p>{prompting} The transcript is unchanged from the earlier episodes.</p>
<h2>Speed changes</h2><p>Multiple independent segments can share one policy batch, with four distinct takes per segment. Identical candidate prompts reuse prefill only when a real-model KV/hidden-state equality check passes. Acoustic decoding uses larger internal blocks; those blocks are not streamed. ECAPA references are cached once, candidate embeddings and genuineness are batched, and ASR is removed from ranking. The exact measured group-size choice and timing probes are in the manifest.</p>
<p>The previous run shared its RTX 3090 with an unrelated training job. That job was no longer present for this run, so differences from the old table cannot all be attributed to code optimization. The same-run batch-size pilot is the more controlled speed comparison.</p>
{comparison}<h2>Reference voices</h2>{refs}<h2>Quality limits</h2><p>Speaker cosine is an approximate voice-identity cue, not a voice-verification guarantee. Maximizing these two model scores does not ensure an error-free transcript or the most enjoyable performance. The final audit scores the edited winner independently. Review flags are retained, not hidden by reselecting on WER.</p>{followup}
<h2>Reproduce</h2><p><a href='PIPELINE.md'>Recipe and commands</a> · <a href='manifest.json'>Full measurements</a></p><pre>{html.escape(json.dumps(times,indent=2))}</pre></html>''')

def main(a):
    setup(a.device)
    import numpy as np
    import soundfile as sf
    import torch
    from align_engine import QwenAligner
    from generate import crop
    from scoring import audio16
    from speaker_scoring import SpeakerScorer
    torch.set_num_threads(4);torch.cuda.set_device(a.device)
    runs=Path(a.runs).resolve();out=Path(a.output).resolve();out.mkdir(exist_ok=True,parents=True)
    config=json.loads((runs/'configuration.json').read_text());(out/'references').mkdir(exist_ok=True)
    for r in config['references'].values():shutil.copyfile(HERE/r['path'],out/'references'/Path(r['path']).name)
    shutil.copyfile(HERE/'references/PROVENANCE.json',out/'references/PROVENANCE.json')
    shutil.copyfile(HERE/'SPEAKER_RECIPE.md',out/'PIPELINE.md')
    tick=time.perf_counter();scorer=SpeakerScorer(a.device,config['references'],load_asr=True,asr_url=a.asr_url)
    aligner=QwenAligner(device=a.device);assert aligner.ok
    load_s=time.perf_counter()-tick;audit_start=time.perf_counter();rows=[];pending=[]
    for path in sorted(runs.glob('*/*/result.json')):
        folder=path.parent
        if (folder/'audited.json').exists():rows.append(json.loads((folder/'audited.json').read_text()));continue
        row=json.loads(path.read_text());wave,sr=sf.read(folder/'raw.flac',dtype='float32');assert sr==SR
        tick=time.perf_counter();align=aligner.align(audio16(wave),row['turn']['text'].split(),language='German' if row['language']=='de' else 'English')
        good=bool(align) and all(w[3]>.5 for w in align)
        first,last=(float(align[0][1]),float(align[-1][2])) if good else (0.,len(wave)/SR)
        good=good and 0<=first<last<=len(wave)/SR+.03
        report=dict(alignment_ok=good,first_word_s=first,last_word_s=min(last,len(wave)/SR),words=align)
        if good:clip,edit=crop(wave,report)
        else:
            clip=wave;edit=dict(first_word_in_clip_s=0.,last_word_in_clip_s=len(wave)/SR,
                timing_basis='unverified_clip_edges',policy='No trim: alignment uncertain')
        edited=folder/'edited.flac';sf.write(edited,clip,SR,subtype='PCM_24')
        row.update(edited=str(edited),edit=edit,edited_seconds=len(clip)/SR,source_folder=str(folder))
        pending.append((row,clip,report,time.perf_counter()-tick))
    for offset in range(0,len(pending),4):
        group=pending[offset:offset+4];clips=[r[1] for r in group]
        scores,timing=scorer.score_speakers(clips,[r[0]['turn']['speaker'] for r in group])
        transcripts,asr_s=scorer.transcribe([audio16(w) for w in clips])
        for (row,clip,align,align_s),score,transcript in zip(group,scores,transcripts):
            err=errors(row['turn']['text'],transcript)
            row['audit']=dict(**score,**err,asr=transcript,alignment=align,
                review_required=err['wer']>.15 or not align['alignment_ok'],
                wall_seconds=align_s+(timing['scoring_seconds']+asr_s)/len(group))
            dump(Path(row['source_folder'])/'audited.json',row);rows.append(row)
            print('AUDIT',row['language'],row['turn']['index'],'WER',round(err['wer'],3),'ECAPA',round(score['ecapa_cosine'],3),flush=True)
    # Measure the earlier whole-turn winners against the SAME cached targets.
    # This is a paired-content comparison, not an isolated prompt/reward ablation.
    previous=[]
    old_paths=sorted((HERE/'runs/bestof4').glob('*/audited.json'))
    for offset in range(0,len(old_paths),4):
        old=[json.loads(p.read_text()) for p in old_paths[offset:offset+4]]
        waves=[sf.read(r['edited'],dtype='float32')[0] for r in old]
        vectors=scorer.embed([audio16(w) for w in waves])
        target=torch.stack([scorer.targets[r['turn']['speaker']] for r in old])
        similarities=(vectors*target).sum(-1).cpu().tolist()
        previous.extend(dict(language=r['language'],turn=r['turn']['index'],ecapa_cosine=float(s),
            genuineness=r['audit']['ears']['genuineness_0_6'],wer=r['audit']['wer'],
            source=r['edited']) for r,s in zip(old,similarities))
    dump(out/'previous-bestof4-speaker-comparison.json',dict(rows=previous,
        note='Same scripts/references, but both sentence prompting and reward changed. GPU contention also changed. Not an isolated causal ablation.'))
    audit_s=time.perf_counter()-audit_start
    del scorer,aligner;gc.collect();torch.cuda.empty_cache()
    dump(runs/'gpu-finished.json',dict(audit_load_seconds=load_s,audit_seconds=audit_s,finished_unix=time.time()))
    groups={}
    for r in rows:groups.setdefault((r['variant'],r['language']),[]).append(r)
    tick=time.perf_counter()
    jobs=[(name,lang,sorted(rs,key=lambda r:r['turn']['index']),str(out)) for (name,lang),rs in sorted(groups.items())]
    with ProcessPoolExecutor(max_workers=2) as pool:receipts=list(pool.map(export_episode,jobs))
    for r in receipts:
        rs=groups[(r['variant'],r['language'])]
        r['mean_speaker_cosine']=statistics.mean(row['audit']['ecapa_cosine'] for row in rs)
        old=out/r.pop('untrimmed_stream_file');new=old.with_name(old.name.replace('-untrimmed-stream','-untrimmed'))
        old.replace(new);r['untrimmed_file']=new.name
    times=dict(generation=json.loads((runs/'generation-complete.json').read_text()),audit_load_seconds=load_s,
        audit_seconds=audit_s,export_wall_seconds=time.perf_counter()-tick)
    if (runs/'process-start.json').exists():
        created=json.loads((runs/'process-start.json').read_text())['created_unix']
        times['cold_setup_tests_and_pilot_seconds']=times['generation']['finished_unix']-times['generation']['production_seconds']-created
        times['full_experiment_to_exports_seconds']=time.time()-created
        times['cold_timing_note']='Includes this run’s model loading, correctness checks, cold CUDA compilation, timing probes, generation, final audit and MP3 export; excludes TTS service restoration.'
    manifest=dict(configuration=config,episodes=receipts,stage_times=times,
        speed_pilot=json.loads((runs/'speed-pilot.json').read_text()),
        prefill_checks=json.loads((runs/'prefill-checks.json').read_text()),
        previous_speaker_comparison='previous-bestof4-speaker-comparison.json',
        generated_from=str(runs),code_sha256={p.name:sha(p) for p in HERE.glob('*.py')})
    dump(out/'manifest.json',manifest);page(out,receipts,config,times)
    print('EXPORTED',len(receipts),'episodes',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--asr-url')
    main(p.parse_args())
