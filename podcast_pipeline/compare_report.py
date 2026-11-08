"""Build a listening comparison from audited, already-generated N=1/2/4 episodes."""
import argparse,html,json,os
from pathlib import Path
from recipe import dump

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in (1,2,4):p.add_argument(f'--n{n}',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    rows=[];cards=[];configs={};components={};measurements={}
    def link(path):return html.escape(os.path.relpath(path,out))
    for n in (1,2,4):
        root=Path(getattr(a,f'n{n}')).resolve();m=json.loads((root/'manifest.json').read_text());configs[n]=m['configuration']
        assert configs[n]['n']==n and configs[n]['prompt_style']=='spontaneous'
        runs=Path(m['generated_from']);batch=json.loads((runs/'production-batches.json').read_text())
        components[n]={key:sum(x.get(key,0) for x in batch) for key in ('batch_wall_seconds','token_seconds','codec_seconds','scoring_seconds','graph_capture_seconds')}
        measurements[n]=m['stage_times']
        for e in m['episodes']:
            row={key:e[key] for key in ('language','seconds','inference_seconds','useful_rtf','mean_genuineness','mean_speaker_cosine','mean_wer','review_turns')};row['n']=n;rows.append(row)
            label=f"{e['language'].upper()} · {'Single take' if n==1 else f'Best-of-{n}'}"
            cards.append(f"<article><h3>{label}</h3><audio controls preload='none' src='{link(root/e['file'])}'></audio><p><a href='{link(root/e['file'])}' download>Download MP3</a> · <a href='{link(root/Path(e['file']).with_suffix('.audit.json'))}'>Per-turn audit</a></p><p>Flagged turns (zero-based): {e['review_turns'] or 'none'}</p></article>")
    # Compare effective spoken scripts and cues, not machine-specific file paths.
    for n in (1,2):
        assert sorted(configs[n]['scripts'].values())==sorted(configs[4]['scripts'].values())
        if configs[4].get('effective_jobs'):assert configs[n]['effective_jobs']==configs[4]['effective_jobs']
        assert configs[n]['general']==configs[4]['general']
        for speaker in configs[4]['references']:
            assert configs[n]['references'][speaker]['sha256']==configs[4]['references'][speaker]['sha256']
    repaired=Path(a.n4).resolve()/'meaning-of-life-en-spontaneous-listening-repair.mp3'
    if repaired.exists():
        cards.append(f"<article><h3>EN · N4 complete listening copy</h3><audio controls preload='none' src='{link(repaired)}'></audio><p><a href='{link(repaired)}' download>Download MP3</a></p><p>The previously approved listening version. Only incomplete turn 6 was replaced with another saved candidate after a content check. This is not the strict argmax shown in the metrics table.</p></article>")
    table=''.join(f"<tr><td>{r['language'].upper()}</td><td>{r['n']}</td><td>{r['seconds']:.1f} s</td><td>{r['inference_seconds']:.1f} s</td><td>{r['useful_rtf']:.3f}</td><td>{r['mean_genuineness']:.3f}</td><td>{r['mean_speaker_cosine']:.3f}</td><td>{r['mean_wer']:.1%}</td><td>{len(r['review_turns'])}/20</td></tr>" for r in sorted(rows,key=lambda r:(r['language'],r['n'])))
    stages=''.join(f"<tr><td>{n}</td><td>{c['token_seconds']:.1f} s</td><td>{c['codec_seconds']:.1f} s</td><td>{c['scoring_seconds']:.1f} s</td><td>{c['batch_wall_seconds']:.1f} s</td></tr>" for n,c in components.items())
    caveats=['N1 and N2 run with an unrelated training job on the same GPU, using separate token and codec/judge phases. Prior N4 used integrated execution under different GPU load. Timing differences are not an isolated causal effect of N.',
        'Same words, sparse cues, reference pair, temperature 1.0, CFG 2.5 and 50/50 reward. Fresh generation; batch grouping and seeds differ, so the runs are not nested candidate subsets.',
        'N1 skips online ranking and keeps its only take. All selected outputs get the same final diagnostic audit; no silent content repair.',
        'Genuineness, ECAPA cosine and ASR WER are imperfect model estimates. Forty turns per mode do not establish a universal quality ranking.',
        'Prior strict English N4 contains an incomplete turn 6. Its separately labeled repaired listening copy is not used in this table.',
        'Useful RTF divides production time by final edited audio duration. Loading, correctness probes, final audit and MP3 export are excluded. Phased production is a sum of stage work, not request-to-first-audio latency.']
    payload=dict(recipe='approved sparse spontaneous',hardware='RTX 3090 24 GB; shared host',episodes=rows,components=components,caveats=caveats)
    dump(out/'comparison.json',payload)
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Podcast · Single take versus Best-of-2 and Best-of-4</title>
<style>body{{font:17px/1.55 system-ui;background:#f6f2ea;color:#252333;max-width:1200px;margin:25px auto;padding:18px;overflow-wrap:anywhere}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;background:white;width:100%}}td,th{{padding:10px;font-size:15px;text-align:left;border-bottom:1px solid #ddd;white-space:nowrap}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(330px,100%),1fr));gap:18px}}article{{background:white;padding:20px;border-radius:16px;min-width:0}}audio{{width:100%}}a{{color:#593ca4}}.note{{border-left:5px solid #d29e32;padding:12px 18px;background:#fff7e2}}</style>
<h1>Same podcast, fewer candidate takes</h1><p>Two twenty-turn episodes about meaning, responsibility and everyday care. The approved prompt asks for authentic, genuine, relaxed and gentle spontaneous delivery, using the same German reference voices in both languages. No voice conversion.</p>
<div class="note">These are real fresh generations. The shared GPU workload changed after the N4 baseline. Read the timing limits below before comparing speed.</div>
<h2>Measured results</h2><div class="scroll"><table><tr><th>Language</th><th>N</th><th>Final audio</th><th>Generate + rank</th><th>RTF ↓</th><th>Genuine / 6 ↑</th><th>Speaker cosine ↑</th><th>Mean turn WER ↓</th><th>Review flags</th></tr>{table}</table></div>
<p>For N above one: reward = 0.5 × clip(genuineness / 6, 0, 1) + 0.5 × clip(reference ECAPA cosine, 0, 1). WER never selects the winner. N=1 has no choice or retry.</p>
<h2>Listen</h2><div class="cards">{''.join(cards)}</div>
<h2>Where the production time goes · both languages</h2><div class="scroll"><table><tr><th>N</th><th>Policy tokens</th><th>Codec</th><th>Online ranking</th><th>Production sum</th></tr>{stages}</table></div>
<h2>How to interpret this comparison</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in caveats)}</ul>
<p>Each speaker turn is generated independently. N2 groups four turns and N1 groups eight turns per policy batch, keeping eight candidates in flight. N4 grouped two turns × four candidates. Deferred decoding lowers simultaneous VRAM use; it is an offline fallback, not streaming.</p>
<h2>Further speed work</h2><p>The conservative next step is a warmed, dedicated GPU worker and cached reference embeddings. Then test length-aware grouping and safe finished-row compaction: long candidates keep batches running after other rows have ended. Do not shorten the cap without checking that all words are present. Compare N on the same idle GPU and warm-up policy before claiming a percentage speedup.</p>
<p><a href="comparison.json">Machine-readable results</a> · <a href="https://huggingface.co/laion/Humanness-Voice-Podcast">Weights, current defaults and full guides</a></p></html>''')
    print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
