"""Independent final audit, word-clock editing and listener comparison export.

Streaming commits remain untouched. The polished MP3 is a separate offline
assembly with boundary trims/fades and a 400ms word-to-word turn gap.
"""
import argparse
import gc
import html
import importlib.util
import json
from pathlib import Path
import statistics
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from recipe import HERE, WORKSPACE, SR, GENERAL, dump, sha, errors, reward

def audio_module():
    spec=importlib.util.spec_from_file_location('podcast_export',HERE/'runtime/audio_export.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.ROOT=HERE
    return module

def safe_export(wave,path,title,language):
    """Publish a filename only after encoding and full decode validation succeed."""
    path=Path(path)
    pending=path.with_name('.'+path.stem+'.pending.mp3')
    receipt=audio_module().export(wave,pending,title,language)
    pending.replace(path)
    pending.with_suffix('.source.flac').replace(path.with_suffix('.source.flac'))
    receipt['file']=path.name
    return receipt

def export_episode(job):
    import soundfile as sf
    import numpy as np
    name,lang,rows,out=job
    out=Path(out); audio=audio_module()
    wave,timeline=audio.assemble(rows)
    for timing,row in zip(timeline,rows):
        timing['timing_basis']=row['edit'].get('timing_basis','forced_alignment_estimate')
    stem=f'meaning-of-life-{lang}-{name}'
    tick=time.perf_counter()
    receipt=safe_export(wave,out/(stem+'.mp3'),f'Meaning of life — {name} — authentic/genuine',lang)
    receipt.update(language=lang,variant=name,turns=len(rows),export_seconds=time.perf_counter()-tick,
        mean_genuineness=statistics.mean(r['audit']['ears']['genuineness_0_6'] for r in rows),
        mean_wer=statistics.mean(r['audit']['wer'] for r in rows),
        mean_ce_diagnostic=statistics.mean(r['audit']['ears']['audiobox:CE'] for r in rows),
        mean_mos_diagnostic=statistics.mean(r['audit']['ears']['dnsmos:P808_MOS'] for r in rows),
        inference_seconds=sum(r['performance']['wall_seconds'] for r in rows),
        median_first_commit=statistics.median(r['performance']['first_commit_seconds'] for r in rows),
        review_turns=[r['turn']['index'] for r in rows if r['audit']['review_required']])
    receipt['useful_rtf']=receipt['inference_seconds']/(len(wave)/SR)
    dump(out/(stem+'.timeline.json'),timeline)
    dump(out/(stem+'.audit.json'),rows)
    (out/(stem+'.srt')).write_text('\n\n'.join(str(i+1)+'\n'+audio.stamp(t['speech_start_s'])+
        ' --> '+audio.stamp(t['speech_end_s'])+'\n'+t['speaker']+': '+t['text'] for i,t in enumerate(timeline))+'\n')
    # Exact emitted PCM: preserve every commit and its natural tail, with an
    # explicitly artificial 400ms inter-turn gap for this replay file only.
    chunks=[]
    for row in rows:
        folder=Path(row['source_folder'])
        chunks.append(sf.read(folder/'raw.flac',dtype='float32')[0])
        chunks.append(np.zeros(round(.4*SR),np.float32))
    stream_path=out/(stem+'-untrimmed-stream.flac')
    sf.write(stream_path,np.concatenate(chunks[:-1]),SR,subtype='PCM_24')
    receipt['untrimmed_stream_file']=stream_path.name
    return receipt

def build_page(out,receipts,configuration,stage_times):
    headers='<tr><th>Language / method</th><th>Length</th><th>Median first commit</th><th>Useful RTF</th><th>Genuineness / 6</th><th>WER</th><th>Review turns</th></tr>'
    table=[];cards=[]
    for r in receipts:
        seconds=round(r['seconds']);duration=f'{seconds//60}:{seconds%60:02d}'
        table.append(f"<tr><td>{r['language'].upper()} · {r['variant']}</td><td>{duration}</td><td>{r['median_first_commit']:.2f}s</td><td>{r['useful_rtf']:.2f}</td><td>{r['mean_genuineness']:.2f}</td><td>{r['mean_wer']:.1%}</td><td>{len(r['review_turns'])}</td></tr>")
        stem=Path(r['file']).stem
        review=('Review flagged turns (zero-based): '+', '.join(map(str,r['review_turns']))) if r['review_turns'] else 'No turns exceeded the audit review threshold.'
        cards.append(f"<section><h2>{r['language'].upper()} · {r['variant']}</h2><p>{duration} · MP3, mono, 192 kbps</p><audio controls preload='none' src='{r['file']}'></audio><p><a href='{r['file']}' download>Download edited episode</a> · <a href='{r['untrimmed_stream_file']}'>Untrimmed commit replay</a></p><p>Genuineness {r['mean_genuineness']:.2f}/6 · WER {r['mean_wer']:.1%}</p><p>{review}</p><p><a href='{stem}.audit.json'>Per-turn scores and transcripts</a> · <a href='{stem}.srt'>Subtitles</a></p></section>")
    reference_cards=''.join(f"<p>{html.escape(s)} · <audio controls preload='none' src='references/{html.escape(Path(v['path']).name)}'></audio></p>" for s,v in configuration['references'].items())
    content=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Podcast: single GPU, new references and causal search</title><style>body{{font:17px/1.5 system-ui;background:#f7f3ed;color:#26232b;max-width:1180px;margin:32px auto;padding:16px;overflow-wrap:anywhere}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{text-align:left;padding:12px;border-bottom:1px solid #ddd}}.scroll{{overflow:auto}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(320px,100%),1fr));gap:20px}}section{{padding:22px;background:white;border-radius:16px;min-width:0}}audio{{width:100%;min-width:0}}a{{color:#6243a2}}pre{{white-space:pre-wrap}}</style>
<h1>Podcast listening comparison</h1><p>New generations using the Podcast checkpoint and its two published German reference clips, identical for English and German. General direction: <strong>Very authentic, very genuine.</strong> No pleasantness instruction, no voice conversion and no Content Enjoyment in the reward.</p>
<div class="scroll"><table>{headers}{''.join(table)}</table></div>
<p>RTF = synthesis and online selection time / final edited audio duration; lower is faster. Loading, final audit and MP3 export are measured separately. First commit includes online ranking for search and a complete turn for whole-turn Best-of-N; it is not an end-to-end browser playback measurement. The shared RTX 3090 also ran another user's training job. Means are model estimates, not listening-test results.</p>
<main>{''.join(cards)}</main><h2>What changes between methods?</h2>
<p><strong>Single:</strong> one sampled continuation, no selection. <strong>Best-of-N:</strong> N batched complete turns, ranked by normalized genuineness divided by (1 + WER). <strong>Chunk-N:</strong> N continuations of the same prefix, score cumulative decoded speech every 13 frames (1.04 seconds), commit the winner and fork its semantic CFG caches and causal codec state. The full unspoken script suffix is free in prefix WER. This is a causal search experiment, not conventional token beam search.</p>
<p>The polished files use a separate final forced-alignment pass, small fades outside speech and a 400ms word-to-word gap. If alignment fails, that turn is flagged for review and retained untrimmed; its timeline explicitly uses unverified clip edges instead of claimed word times. This editing cannot retroactively change audio already emitted live. The untrimmed replay links preserve those emitted samples, with a 400ms inter-turn gap; commit timestamps are in the run logs. Very short-window quality scores and ASR are uncertain; chunk search is not guaranteed to improve naturalness.</p>
<h2>Exact shared references</h2>{reference_cards}
<h2>Prompt</h2><pre>{html.escape(GENERAL)}</pre>
<h2>Reproduce and inspect</h2><p><a href='PIPELINE.md'>Pipeline manual</a> · <a href='manifest.json'>Manifest and measurements</a></p><p>The manifest records the local run directory containing code, candidate audio, scores and immutable commit logs.</p>
<pre>{html.escape(json.dumps(stage_times,indent=2))}</pre></html>'''
    (out/'index.html').write_text(content)

def main(args):
    from run import setup
    setup(args.device)
    import numpy as np
    import soundfile as sf
    import torch
    from scoring import Scorer, audio16
    from align_engine import QwenAligner
    from generate import crop
    torch.set_num_threads(4);torch.cuda.set_device(args.device)
    runs=Path(args.runs).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    configuration=json.loads((runs/'configuration.json').read_text())
    (out/'references').mkdir(exist_ok=True)
    for item in configuration['references'].values():
        shutil.copyfile(HERE/item['path'],out/'references'/Path(item['path']).name)
    shutil.copyfile(HERE/'references/PROVENANCE.json',out/'references/PROVENANCE.json')
    shutil.copyfile(HERE/'README.md',out/'PIPELINE.md')
    paths=sorted(runs.glob('*/*/result.json'))
    if not paths:raise ValueError('No generated turns')
    tick=time.perf_counter()
    scorer=Scorer(args.device,args.asr_url)
    aligner=QwenAligner(device=args.device)
    assert aligner.ok
    load_s=time.perf_counter()-tick;audit_start=time.perf_counter();groups={}
    for path in paths:
        row=json.loads(path.read_text());folder=path.parent;checked=folder/'audited.json'
        if checked.exists():
            row=json.loads(checked.read_text())
        else:
            wave,sr=sf.read(folder/'raw.flac',dtype='float32');assert sr==SR
            start=time.perf_counter()
            alignment=aligner.align(audio16(wave),row['turn']['text'].split(),language='German' if row['language']=='de' else 'English')
            good=bool(alignment) and all(w[3]>.5 for w in alignment)
            first,last=(float(alignment[0][1]),float(alignment[-1][2])) if good else (0.,len(wave)/SR)
            good=good and 0<=first<last<=len(wave)/SR+.03
            report=dict(alignment_ok=good,first_word_s=first,last_word_s=last,words=alignment)
            if good:
                # Timestamp quantization can put an estimated final boundary a
                # few milliseconds beyond the physical waveform.
                report['last_word_s']=min(last,len(wave)/SR)
                clip,edit=crop(wave,report)
            else:
                # Preserve a failed first-try/search example rather than invent
                # alignment or silently regenerate it. These are CLIP edges,
                # explicitly not asserted spoken-word timestamps.
                clip=wave.copy()
                edit=dict(first_word_in_clip_s=0.,last_word_in_clip_s=len(wave)/SR,
                    policy='Unverified clip edges; alignment failed; no word-based trim or fade',
                    timing_basis='unverified_clip_edges')
            scores,timing=scorer.score([clip],row['turn']['text'])
            score=scores[0]
            edited=folder/'edited.flac';sf.write(edited,clip,SR,subtype='PCM_24')
            row.update(edited=str(edited),edit=edit,edited_seconds=len(clip)/SR,
                source_folder=str(folder),audit={**score,'review_required':score['wer']>.15 or not good,
                'alignment':report,'wall_seconds':time.perf_counter()-start,**timing})
            dump(checked,row)
            print('AUDIT',row['variant'],row['language'],row['turn']['index'],
                'WER',round(score['wer'],3),'G',round(score['ears']['genuineness_0_6'],2),flush=True)
        groups.setdefault((row['variant'],row['language']),[]).append(row)
    audit_s=time.perf_counter()-audit_start
    del scorer,aligner;gc.collect();torch.cuda.empty_cache()
    dump(runs/'gpu-finished.json',dict(finished_unix=time.time(),audit_load_seconds=load_s,audit_seconds=audit_s))
    jobs=[(name,lang,sorted(rows,key=lambda r:r['turn']['index']),str(out)) for (name,lang),rows in sorted(groups.items())]
    export_start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        receipts=list(pool.map(export_episode,jobs))
    times=dict(final_audit_load_seconds=load_s,final_audit_seconds=audit_s,
               export_wall_seconds=time.perf_counter()-export_start,
               retained_turn_count=len(paths),
               retained_turn_inference_seconds=sum(r['inference_seconds'] for r in receipts),
               retained_turn_audit_seconds=sum(r['audit']['wall_seconds'] for rows in groups.values() for r in rows),
               final_audit_scope_note='Final audit timing describes this invocation; previously audited turns are reused. retained_turn_audit_seconds sums all retained per-turn audits, excluding model loading.')
    if (runs/'generation-complete.json').exists():
        times['generation']=json.loads((runs/'generation-complete.json').read_text())
        times['generation_scope_note']='Generation timing describes the latest invocation only; resumption can retain earlier turns. Episode inference_seconds sums the retained turns. Discarded experiments are not included.'
    dump(out/'manifest.json',dict(configuration=configuration,episodes=receipts,stage_times=times,
        generated_from=str(runs),code_sha256={p.name:sha(p) for p in HERE.glob('*.py')}))
    build_page(out,receipts,configuration,times)
    print('EXPORTED',len(receipts),'episodes',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',default=str(HERE/'runs'))
    p.add_argument('--output',default=str(WORKSPACE/'podcast-episodes-stream-comparison'))
    p.add_argument('--device',default='cuda:1');p.add_argument('--asr-url',default=None)
    main(p.parse_args())
