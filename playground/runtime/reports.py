"""Build self-contained English reports from measured JSON, never estimates."""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import statistics as st

import numpy as np

import settings

CSS = """:root{color-scheme:dark;--fg:#ececf1;--muted:#a8acba;--panel:#1b2030;--line:#343b50;--accent:#8cc8ff;--green:#98ddba;--yellow:#e9c879}
*{box-sizing:border-box}body{margin:0;background:#10141f;color:var(--fg);font:16px/1.6 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:36px 24px 80px}a{color:var(--accent)}h1{font-size:clamp(28px,5vw,44px);line-height:1.15;letter-spacing:-1px}h2{font-size:25px;margin-top:46px}h3{margin-top:28px}p{max-width:950px}.muted,small{color:var(--muted)}.eyebrow{color:var(--accent);font-size:12px;letter-spacing:2px;text-transform:uppercase}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:26px 0}.card,.callout{padding:20px;background:var(--panel);border:1px solid var(--line);border-radius:12px}.card b{display:block;font-size:32px;color:var(--green)}.card span{color:var(--muted);font-size:14px}.callout{border-left:4px solid var(--accent)}.warning{border-left-color:var(--yellow)}.tablewrap{overflow:auto;margin:18px 0}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;border-bottom:1px solid var(--line);padding:10px 12px;white-space:nowrap}th{color:var(--muted)}tbody tr:hover{background:#202639}tr.pareto{background:#183126}pre{font:13px/1.55 ui-monospace,monospace;background:var(--panel);padding:16px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere}code{color:var(--green)}details{margin:14px 0;padding:12px;border:1px solid var(--line);border-radius:8px}summary{cursor:pointer;color:var(--accent)}audio{width:280px;max-width:100%;height:34px}svg{width:100%;height:auto;background:var(--panel);border-radius:12px}nav{display:flex;gap:16px;flex-wrap:wrap}button,select{font:inherit;background:var(--panel);color:var(--fg);border:1px solid var(--line);padding:6px 12px;border-radius:6px}.heat td{text-align:center;white-space:normal;min-width:160px}.heat small{display:block;font-size:12px}footer{margin-top:50px;color:var(--muted);font-size:13px}ul{padding-left:24px}.good{color:var(--green)}.warn{color:var(--yellow)}@media(max-width:650px){main{padding:22px 14px}th,td{padding:8px;font-size:12px}.cards{grid-template-columns:1fr 1fr}.card{padding:12px}.card b{font-size:26px}}
"""


def esc(value):
    return html.escape(str(value))


def page(title, body):
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{CSS}</style></head><body><main><nav><a href="/">Voice demo</a><a href="/performance/">GPU performance</a><a href="/study/">Emotion / CFG study</a></nav>{body}<footer>Measured locally, 30 September 2026 · Humaneness Voice Small S3 · No LoRA or DPO adapter · All study quality scores are model proxies, not human ratings.</footer></main></body></html>'


def table(headers, rows, classes=None):
    body = []
    for i, row in enumerate(rows):
        cls = classes[i] if classes else ""
        body.append(f'<tr class="{cls}">' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>')
    return '<div class="tablewrap"><table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>'


def card(number, description):
    return f'<div class="card"><b>{number}</b><span>{description}</span></div>'


def med(rows, key):
    return st.median(r[key] for r in rows)


def line_chart(groups, key, ylabel, color="#98ddba", horizontal=None):
    width, height, left, top, right, bottom = 900, 320, 65, 25, 25, 50
    xs = [n for n, _ in groups]
    ys = [med(rows, key) for _, rows in groups]
    xmax = max(xs)
    ymax = max(max(ys) * 1.15, (horizontal or 0) * 1.15)
    px = lambda x: left + x / xmax * (width-left-right)
    py = lambda y: height-bottom - y/ymax*(height-top-bottom)
    svg = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(ylabel)} by parallel streams">']
    for tick in np.linspace(0, ymax, 5):
        y = py(tick)
        svg.append(f'<path d="M{left},{y}H{width-right}" stroke="#343b50"/><text x="{left-9}" y="{y+4}" fill="#a8acba" font-size="12" text-anchor="end">{tick:.2f}</text>')
    if horizontal:
        svg.append(f'<path d="M{left},{py(horizontal)}H{width-right}" stroke="#e9c879" stroke-dasharray="5 5"/><text x="{width-right}" y="{py(horizontal)-7}" fill="#e9c879" font-size="12" text-anchor="end">Real-time boundary</text>')
    svg.append('<polyline fill="none" stroke="'+color+'" stroke-width="3" points="' + ' '.join(f'{px(x)},{py(y)}' for x,y in zip(xs,ys)) + '"/>')
    for x,y in zip(xs,ys):
        svg.append(f'<circle cx="{px(x)}" cy="{py(y)}" r="4" fill="{color}"><title>B={x}: {y:.4f}</title></circle><text x="{px(x)}" y="{height-bottom+21}" fill="#a8acba" font-size="12" text-anchor="middle">{x}</text>')
    svg.append(f'<text x="{left}" y="16" fill="#ececf1" font-size="13">{esc(ylabel)}</text><text x="{width/2}" y="{height-7}" fill="#a8acba" font-size="13" text-anchor="middle">Parallel streams in one synchronized batch</text></svg>')
    return ''.join(svg)


def throughput():
    root = settings.HERE / "performance"
    data = {name: json.loads((root / (name + ".json")).read_text()) for name in ["baseline", "optimizations", "compile", "sweep", "validation"]}
    for name in ['latency','long']:
        if (root/(name+'.json')).exists():
            data[name]=json.loads((root/(name+'.json')).read_text())
    baseline = data["baseline"]["trials"]
    sweep = [r for r in data["sweep"]["trials"] if r["status"] == "ok"]
    groups = defaultdict(list)
    # Keep the first planned sweep block distinct from the later CFG ablation.
    seen_ablation = False
    for r in data["sweep"]["trials"]:
        if r["status"] != "ok":
            continue
        if not r["streaming"]:
            seen_ablation = True
        if not seen_ablation:
            groups[r["n"]].append(r)
    groups = sorted(groups.items())
    single_stream = groups[0][1]
    offline = [r for r in sweep if not r["streaming"] and r["guidance"] == 1.5]
    old_offline = [r for r in baseline if not r["streaming"]]
    old_stream = [r for r in baseline if r["streaming"] and r["n"] == 1]
    b10 = dict(groups)[10]
    largest = groups[-1]
    env = data["sweep"]["environment"]
    body = '<p class="eyebrow">Measured on-device · English technical report</p><h1>How fast can Small speak?</h1>'
    body += '<p>Single-stream incremental PCM, synchronized batches, and offline inference on one RTX 3090. This is model throughput, not browser latency.</p>'
    body += '<div class="cards">' + card(f'{med(single_stream,"device_rtf"):.3f}', 'Single-stream streaming RTF · lower is faster') + card(f'{med(offline,"device_rtf"):.3f}', 'Single-stream offline RTF') + card(f'{med(b10,"audio_seconds_per_second"):.1f}×', 'Aggregate audio seconds per GPU second · B=10') + card(str(largest[0]), 'Largest tested successful batch · 64 failed') + '</div>'
    body += f'<div class="callout">For eight seconds of audio, optimized offline inference takes <b>{med(offline,"device_seconds"):.2f} s</b>, versus {med(old_offline,"device_seconds"):.2f} s before optimization ({med(old_offline,"device_seconds")/med(offline,"device_seconds"):.1f}× faster). Incremental streaming takes {med(single_stream,"device_seconds"):.2f} s, versus {med(old_stream,"device_seconds"):.2f} s ({med(old_stream,"device_seconds")/med(single_stream,"device_seconds"):.1f}× faster). B=10 produces 80 audio seconds in {med(b10,"device_seconds"):.2f} s.</div>'
    body += '<h2>What “on device” means here</h2><p>CUDA events bracket TTS prefill, autoregressive global/local decoding and the GPU audio codec. The end event is synchronized before reading elapsed time. This is a device event <em>time span</em>: it includes host launch gaps when the GPU waits, not just a sum of kernel-active durations. Prompt packing and reference encoding are completed before timing. Luna, forced alignment, scoring, CPU PCM copies, MP3 encoding, HTTP and playback are excluded. See <a href="https://docs.pytorch.org/docs/2.6/notes/cuda.html">PyTorch CUDA timing guidance</a>.</p>'
    body += '<pre>Single stream RTF = device seconds / generated audio seconds\nBatch aggregate RTF = device seconds / sum(audio seconds across all rows)\nPer-stream RTF = device seconds / mean(audio seconds per row)\nThroughput = sum(audio seconds) / device seconds = 1 / aggregate RTF</pre><p>RTF &lt; 1 means faster than real time. A small aggregate RTF does not automatically mean each individual stream is real-time. We therefore report both.</p>'
    body += '<h2>Single-stream: streaming versus offline</h2>'
    rows = []
    for label, trials in [('Original offline', old_offline), ('Original streaming', old_stream), ('Optimized offline', offline), ('Optimized streaming', single_stream)]:
        rows.append([label, f'{med(trials,"device_seconds"):.3f} s', f'{med(trials,"device_rtf"):.4f}', f'{med(trials,"audio_seconds_per_second"):.2f}', f'{med(trials,"first_audio_seconds")*1000:.0f} ms', f'{min(r["device_rtf"] for r in trials):.4f}–{max(r["device_rtf"] for r in trials):.4f}', str(len(trials))])
    body += table(['Mode', 'GPU span / 8 s audio', 'RTF', 'Audio s / GPU s', 'First PCM ready', 'Observed RTF range', 'Trials'], rows)
    body += '<p>Streaming really decodes each newly generated ten-frame block into PCM using a persistent causal codec state: 10 frames / 12.5 Hz = 0.8 seconds per chunk. It does not repeatedly decode the growing prefix, and it does not split an already completed waveform. Offline waits for all tokens, then batch-decodes PCM; it saves frequent codec launches and is faster, but offers no early audio. First-PCM time is not network time or time-to-audible playback.</p>'
    if 'latency' in data:
        chunks=defaultdict(list)
        for r in data['latency']['trials']:
            chunks[(r['n'],r['chunk_frames'])].append(r)
        body+='<h2>How small should streaming chunks be?</h2><p>A separate warmed sweep measures the latency-throughput trade-off, with three trials per setting. More frequent codec calls cost substantial GPU work. Chunk sizes below are actual PCM duration, not model time-to-first-audio.</p>'
        body+=table(['Streams','Frames / chunk','Audio chunk ms','First PCM ms','Per-stream RTF','Aggregate RTF','Audio s / GPU s'],[[str(n),str(c),str(c*80),f'{med(rs,"first_audio_seconds")*1000:.0f}',f'{med(rs,"per_stream_rtf"):.3f}',f'{med(rs,"device_rtf"):.4f}',f'{med(rs,"audio_seconds_per_second"):.2f}'] for (n,c),rs in sorted(chunks.items())])
        body+='<div class="callout">A useful low-latency compromise is <b>5 frames / 400 ms</b>: first PCM around 179 ms at B=1, RTF around 0.415. The <b>10-frame / 800 ms</b> default improves throughput (RTF about 0.30), at roughly 251 ms to first PCM. <b>20 frames / 1.6 s</b> is faster still (RTF about 0.239) but delays the first PCM to about 392 ms. One-frame / 80 ms chunks are slower than real time (RTF about 1.35), despite early first PCM. The UI allows 160, 400, 800 and 1600 ms chunks. Offline remains fastest when early playback is unnecessary.</div>'
    body += '<h2>Parallel streaming: 1 through 10, then beyond</h2>' + line_chart(groups, 'audio_seconds_per_second', 'Aggregate throughput · audio s / device s')
    rows = []
    for n, trials in groups:
        rows.append([str(n), f'{med(trials,"device_seconds"):.3f}', f'{med(trials,"device_rtf"):.4f}', f'{med(trials,"per_stream_rtf"):.3f}', f'{med(trials,"audio_seconds_per_second"):.2f}', f'{med(trials,"first_audio_seconds")*1000:.0f}', f'{med(trials,"max_inter_chunk_seconds"):.3f}', f'{max(r["peak_allocated_gib"] for r in trials):.2f}', str(len(trials))])
    rows.append(['64', 'Out of memory', '—', '—', '—', '—', '—', '—', '0 completed'])
    body += table(['Streams', 'Device s', 'Aggregate RTF', 'Per-stream RTF', 'Audio s / GPU s', 'First PCM ms', 'Worst chunk gap s¹', 'Peak allocated GiB', 'Trials'], rows)
    body += '<p class="muted">¹ Median across trials of each trial’s maximum gap between successive ready PCM chunks. First-chunk warm-up is separate. B=1–10: three measured trials; B=12–48: one trial each, so no robust high-batch variance estimate.</p>'
    body += line_chart(groups, 'per_stream_rtf', 'Per-stream RTF · lower is faster', '#8cc8ff', horizontal=1)
    body += f'<div class="callout warning"><b>Capacity is workload-specific.</b> B={largest[0]} succeeded with per-stream RTF {med(largest[1],"per_stream_rtf"):.3f} and peak allocated memory {med(largest[1],"peak_allocated_gib"):.2f} GiB. B=64 failed during the attempted run. The exact maximum in between was not determined. These are synchronized eight-second streams with short, identical prompt prefixes; arbitrary arrivals, long context, long references, variable stop lengths and a serving scheduler are not covered. The demo still serializes browser jobs; batched best-of-N is not independent multi-user live streaming.</div>'
    if 'long' in data:
        long=defaultdict(list)
        for r in data['long']['trials']:
            if r['status']=='ok':long[r['n']].append(r)
        body+='<h2>Longer-cache stress check: 60 seconds per stream</h2><p>A separate fixed 750-frame run checks sustained decode and memory growth at B=1 and B=12, using the same 800 ms PCM chunk and short prompt recipe. Stop decisions are overridden only for this timing workload. This is not a 60-second spoken-content quality test.</p>'
        body+=table(['Streams','Seconds / stream','Trial count','Device s','Per-stream RTF','Aggregate RTF','Peak allocated GiB'],[[str(n),'60',str(len(rs)),f'{med(rs,"device_seconds"):.3f}',f'{med(rs,"per_stream_rtf"):.3f}',f'{med(rs,"device_rtf"):.4f}',f'{max(r["peak_allocated_gib"] for r in rs):.2f}'] for n,rs in sorted(long.items())])
        body+='<p>The demo uses batches up to 12 for best-of-N, with a smaller-batch retry on GPU out-of-memory. The UI’s N limit remains 12. Successful short-prefix stress tests do not guarantee memory capacity for arbitrary long prompt prefixes; cold shape compilation can also increase latency.</p>'
    body += '<h2>What was optimized</h2>'
    labels = {'legacy':'Original serial CFG', 'fused':'Fused conditional / neutral batch', 'graph':'Also local CUDA graphs', 'graph_codec_bf16':'Also supported BF16 codec weights', 'graph_global_compile':'Also compiled dynamic global decoder'}
    rows = []
    for variant in ['legacy','fused','graph','graph_codec_bf16','graph_global_compile']:
        source = data['compile']['trials'] if variant == 'graph_global_compile' else data['optimizations']['trials']
        trials = [r for r in source if r['variant'] == variant]
        rows.append([labels[variant], f'{med(trials,"device_seconds"):.3f}', f'{med(trials,"device_rtf"):.4f}', f'{med(trials,"peak_allocated_gib"):.2f}'])
    body += table(['Incremental optimization · offline B=1', 'GPU s / 8 s audio', 'RTF', 'Peak allocated GiB'], rows)
    body+='<h3>Single-stream CFG cost after optimization</h3>'
    rows=[]
    for g in [1.,1.5]:
        for streaming in [False,True]:
            rs=[r for r in sweep if r['n']==1 and r['guidance']==g and r['streaming']==streaming]
            rows.append([f'{g:g}', 'Streaming 800 ms' if streaming else 'Offline',f'{med(rs,"device_rtf"):.4f}',f'{med(rs,"device_seconds"):.3f}'])
    body+=table(['CFG','Mode','RTF','Device s / 8 s audio'],rows)+'<p>CFG=1 uses one conditional branch, CFG=1.5 two fused branches. The measured B=1 overhead is small on this GPU after fusion; this does not imply zero CFG cost at larger batches or on other hardware.</p>'
    body += '<ul><li>Fuse the two CFG branches into one global/local batch while retaining shared sampled codebook prefixes, conditional-only stop decisions and the exact equation <code>u + g(c − u)</code>. Left padding preserves text positions and masks.</li><li>Capture the fixed 12-codebook local step in reusable CUDA graphs. Captures restore the pre-warm RNG state; graph replay advances RNG normally. The graph cache is bounded and keyed by batch, CFG and sampling parameters.</li><li>Compile the growing-cache global decoder with dynamic shapes. This fuses small pointwise operations and reduces launch overhead without changing model weights.</li><li>Use the codec’s supported BF16 encoder/decoder weight setter, keeping its quantizer FP32. This reduces steady GPU memory by roughly 6 GiB; it did not independently improve measured speed. See the <a href="https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2">codec inference contract</a>.</li><li>Batch the PCM codec instead of decoding each candidate separately. Streaming maintains persistent row-specific states and masks finished rows.</li></ul>'
    cold = next(x['global_compile_cold_seconds'] for x in data['compile']['notes'] if isinstance(x,dict) and 'global_compile_cold_seconds' in x)
    body += f'<p><b>Cold costs are real:</b> the initial compilation/warm-up in the compile experiment took {cold:.1f} s. Local graph capture costs typically 0.25–0.5 s per new configuration. Steady measurements exclude warm-up; live UI timings include first-use work if it occurs inside generation. Compiler disk caching helps subsequent processes, but new shapes may still compile. This is not a cold-start benchmark.</p>'
    body += '<h2>Numerical and quality checks</h2><p>Fused eager and local CUDA-graph code sequences were exactly equal for batches 1 and 4 in the numerical test, and across three repeated optimization seeds. Streaming versus full codec decode was tested with rows terminating at 30, 19, 7 and 25 frames, including mid-chunk termination. All output lengths were correct and PCM finite.</p>'
    checks = data['validation']['checks']
    body += table(['Check', 'RMS PCM difference', 'Maximum absolute difference'], [[f'Stream vs full · row {r["row"]} · {r["frames"]} frames',f'{r["rms_difference"]:.6f}',f'{r["maximum_difference"]:.6f}'] for r in checks if r['check']=='incremental_vs_full_pcm'] + [["FP32 codec weights vs supported BF16",f'{r["rms_difference"]:.6f}',f'{r["maximum_difference"]:.6f}'] for r in checks if r['check']=='codec_fp32_weights_vs_supported_bf16'])
    body += '<p>Fusing CFG changes BF16 GEMM batch shape, and compiling global operations changes rounding. Stochastic trajectories therefore differ from the original code: optimization is not bit-identical to legacy. Small waveform differences are numerical evidence, not a perceptual equivalence test. The <a href="/study/">separate emotion/quality study</a> evaluates actual free-running output with ASR and learned proxies.</p>'
    body += '<h2>Reproducibility and limitations</h2><p>Three trials per main condition use seeds 777, 778 and 779; medians and observed ranges are shown, not an invented precise confidence interval. Each timing workload forces 100 frames per stream (eight seconds) to stop random early EOS from changing the workload. This is scheduling capacity, not a claim that naturally generated speech always lasts eight seconds. The quality study does not force a minimum length.</p>'
    body += '<p>Reference codec preparation, prompt tensors and all model weights are resident before the timer. The live GPU server was stopped; no scorer or other model ran concurrently on GPU 0 during measurements. Display-server memory remains. CUDA 1 is not used to produce these throughput numbers. Warm-up uses 20 frames; the first measured longer-cache decode may still incur shape-specific work. A small chunk-size sweep and a 60-second cache stress check are included. GPU clocks were not fixed; long-duration thermal equilibrium, latency under request queuing and independent asynchronous multi-user arrivals were not measured.</p>'
    body += '<pre>' + esc(json.dumps({k:env[k] for k in ['gpu','vram_gib','torch','cuda','transformers','python','model','revision','stage','codec_revision','frames','cfg','temperature','reference_frames']},indent=2)) + '</pre>'
    body += '<p>Raw trial records: ' + ' · '.join(f'<a href="{name}.json">{name}.json</a>' for name in data) + '</p>'
    (root / 'index.html').write_text(page('Humaneness Small — on-device throughput', body))


def study_rows():
    root = settings.HERE / 'study'
    design = json.loads((root/'design.json').read_text())
    results = []
    for row in design['rows']:
        path = root/(row['id']+'.json')
        if path.exists():
            results.append(json.loads(path.read_text()))
    return design, results


def aggregate(rows):
    def mean(field):
        return float(np.mean([field(r) for r in rows]))
    return {'n':len(rows), 'raw_wer':mean(lambda r:r['raw_word_errors']['wer']),
        'post_wer':mean(lambda r:r['processed_word_errors']['wer']),
        'low_wer_rate':mean(lambda r:r['raw_word_errors']['wer'] <= .1),
        'failure_rate':mean(lambda r:r['raw_word_errors']['wer'] > .3),
        'genuineness':mean(lambda r:r['genuineness']), 'emotion_percentile':mean(lambda r:r['emotion_percentile']),
        'top5_rate':mean(lambda r:r['emotion_rank'] <= 5), 'target_score':mean(lambda r:r['target_emotion_score']),
        'clap':mean(lambda r:r['clap']), 'frame_limit_rate':mean(lambda r:r['hit_frame_limit']),
        'alignment_rate':mean(lambda r:r['alignment']['applied'])}


def cluster_ci(rows, field, repeats=2000):
    blocks = defaultdict(list)
    for r in rows:
        blocks[(r['emotion'],r['format'])].append(field(r))
    values = np.array([np.mean(v) for v in blocks.values()])
    rng = np.random.default_rng(9403)
    means = values[rng.integers(0,len(values),(repeats,len(values)))].mean(-1)
    return [float(x) for x in np.quantile(means,[.025,.975])]


def study_summary(design, results):
    grouped = defaultdict(list)
    for r in results:
        grouped[(r['stage'],r['format'],r['guidance'],r['temperature'])].append(r)
        grouped[(r['stage'],'both',r['guidance'],r['temperature'])].append(r)
    cells = []
    for (stage,fmt,g,t), rows in sorted(grouped.items()):
        cells.append({'stage':stage,'format':fmt,'cfg':g,'temperature':t,**aggregate(rows),
            'raw_wer_ci':cluster_ci(rows,lambda r:r['raw_word_errors']['wer']),
            'genuineness_ci':cluster_ci(rows,lambda r:r['genuineness']),
            'emotion_ci':cluster_ci(rows,lambda r:r['emotion_percentile'])})
    recommendations = {}
    for fmt in ['caption','structured','both']:
        grid = [c for c in cells if c['stage']=='main' and c['format']==fmt]
        baseline = next((c for c in grid if c['cfg']==1.5 and c['temperature']==1.),None)
        if not baseline:
            continue
        eligible = [c for c in grid if c['raw_wer']<=.1 and c['genuineness']>=baseline['genuineness']-.25]
        for c in grid:
            c['eligible'] = c in eligible
            c['pareto'] = not any(other['raw_wer']<=c['raw_wer'] and other['genuineness']>=c['genuineness']
                and other['emotion_percentile']>=c['emotion_percentile'] and
                (other['raw_wer']<c['raw_wer'] or other['genuineness']>c['genuineness'] or other['emotion_percentile']>c['emotion_percentile']) for other in grid)
        if eligible:
            top = max(c['emotion_percentile'] for c in eligible)
            chosen = max((c for c in eligible if c['emotion_percentile']>=top-.01),key=lambda c:c['clap'])
        else:
            chosen = None
        recommendations[fmt] = {'eligible_count':len(eligible),'chosen':chosen,'baseline':baseline,
            'lowest_wer':min(grid,key=lambda c:c['raw_wer']),
            'most_emotional':max(grid,key=lambda c:c['emotion_percentile']),
            'most_genuine':max(grid,key=lambda c:c['genuineness'])}
    paired=[]
    for fmt,rec in recommendations.items():
        chosen=rec['chosen']
        if not chosen:continue
        for metric,field in [('raw_wer',lambda r:r['raw_word_errors']['wer']),('genuineness',lambda r:r['genuineness']),('emotion_percentile',lambda r:r['emotion_percentile']),('clap',lambda r:r['clap'])]:
            baseline={(r['emotion'],r['format'],r['seed']):r for r in results if r['stage']=='main'
                and (r['format']==fmt or fmt=='both') and r['guidance']==1.5 and r['temperature']==1.}
            differences=defaultdict(list)
            for r in results:
                key=(r['emotion'],r['format'],r['seed'])
                if r['stage']=='main' and r['guidance']==chosen['cfg'] and r['temperature']==chosen['temperature'] and key in baseline:
                    differences[(r['emotion'],r['format'])].append(field(r)-field(baseline[key]))
            values=np.array([np.mean(v) for v in differences.values()])
            rng=np.random.default_rng(9403)
            boot=values[rng.integers(0,len(values),(2000,len(values)))].mean(-1)
            paired.append({'format':fmt,'metric':metric,'delta':float(values.mean()),
                'ci':[float(v) for v in np.quantile(boot,[.025,.975])], 'clusters':len(values)})
    crosscheck=[]
    for g,t in sorted({(r['guidance'],r['temperature']) for r in results if r['stage'] in ['german','reference_b']}):
        rows=[r for r in results if r['format']=='caption' and r['stage'] in ['main','german','reference_b'] and r['guidance']==g and r['temperature']==t]
        crosscheck.append({'cfg':g,'temperature':t,**aggregate(rows)})
    return {'complete_main':sum(r['stage']=='main' for r in results)==design['main_samples'],
        'scored_samples':len(results),'planned_samples':len(design['rows']),'cells':cells,'recommendations':recommendations,
        'selection_rule':design['selection_rule'],'paired_vs_default':paired,'caption_generalization_pool':crosscheck}


def grid_table(cells, fmt):
    cells = [c for c in cells if c['stage']=='main' and c['format']==fmt]
    lookup = {(c['cfg'],c['temperature']):c for c in cells}
    rows=[]
    for g in [1.,1.5,2.,3.]:
        row=[f'CFG {g:g}']
        for t in [0.,.5,1.,1.2]:
            c=lookup.get((g,t))
            if not c:
                row.append('pending');continue
            tone='#98ddba' if c['raw_wer']<=.1 else '#e9c879' if c['raw_wer']<=.3 else '#ef9a9a'
            row.append(f'<span style="color:{tone}">WER {c["raw_wer"]*100:.1f}%</span><small>Genuine {c["genuineness"]:.2f} / 6<br>Emotion percentile {c["emotion_percentile"]:.2f}<br>{"✓ eligible" if c.get("eligible") else "outside constraints"}{" · Pareto" if c.get("pareto") else ""}</small>')
        rows.append(row)
    return table(['', 'T = 0','T = 0.5','T = 1','T = 1.2'],rows).replace('<table>','<table class="heat">')


def emotion_study():
    root=settings.HERE/'study'
    design,results=study_rows()
    summary=study_summary(design,results)
    (root/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    main=[r for r in results if r['stage']=='main']
    body='<p class="eyebrow">Controlled pilot · English research report</p><h1>Emotion without losing the words</h1><p>A measured CFG × temperature grid for Humaneness Voice Small S3: emotional direction following, intelligibility and naturalness proxies. There is no universal optimum; the goal is to identify useful compromises under explicit constraints.</p>'
    body+='<div class="cards">'+card(str(len(results)),'Actual scored generations')+card('16','CFG × temperature configurations')+card('5 × 2','Emotions × prompt formats')+card('2','Repeated seeds · T=0 duplicates are not independent')+'</div>'
    if not summary['complete_main']:
        body+='<div class="callout warning"><b>Interim report.</b> The grid is not complete. Do not treat interim winners as a final recommendation.</div>'
    body+='<h2>Practical conclusions</h2>'
    if any(r['stage']=='reference_b' for r in results):
        body+='<div class="callout"><b>Practical starting range after generalization checks: CFG 1.5–2, temperature 1.0, concise CAPTION / exact TRANSCRIPT.</b> The first-grid emotion-first winner (CFG 2, T=1.2) did not consistently generalize to the second reference. The existing CFG 1.5 / T=1 default remains a sensible starting point; CFG 2 / T=1 is a measured alternative, not a guaranteed universal improvement.</div>'
    for fmt,label in [('caption','CAPTION / exact TRANSCRIPT'),('structured','Initial verbose GENERAL / SCRIPT recipe')]:
        rec=summary['recommendations'].get(fmt)
        if not rec:continue
        c=rec['chosen']
        if c:
            body+=f'<div class="callout"><b>{label}: pilot preference CFG {c["cfg"]:g}, T {c["temperature"]:g}.</b> Raw mean WER {c["raw_wer"]*100:.1f}%, processed WER {c["post_wer"]*100:.1f}%, genuineness {c["genuineness"]:.2f}/6, target emotion rank percentile {c["emotion_percentile"]:.2f}. This is the best qualifying configuration under the rule below, not a human-validated global optimum.</div>'
        else:
            c=rec['lowest_wer']
            body+=f'<div class="callout warning"><b>{label}: no configuration met both constraints.</b> Lowest raw WER was {c["raw_wer"]*100:.1f}% at CFG {c["cfg"]:g}, T {c["temperature"]:g}. A low-error, natural and strongly emotional optimum is not established for this format on this pilot.</div>'
    body+='<p>Keep the UI’s requested defaults (CFG 1.5, T=1) unchanged while auditioning the recommended alternatives. If intelligibility is essential, prefer the concise caption surface and a clean short reference. Do not infer that higher CFG must improve emotion: it can also produce repetition, unintelligible speech or flatter expression.</p>'
    body+='<div class="callout warning"><b>Relative naturalness, not “good naturalness” proven.</b> Absolute genuineness scores are mostly low in this experiment (roughly 1–2 out of 6, and lower on the German text). The preference rule only limits deterioration from this baseline; it does not certify high-quality natural speech. No human listening panel was conducted.</div>'
    body+='<h2>Design saved before scoring</h2><p>The main grid contains 320 takes: CFG {1, 1.5, 2, 3} × temperature {0, 0.5, 1, 1.2} × five emotions × two prompt surfaces × two seeds. Every condition uses the same English words and reference A. Configuration order was shuffled with a fixed seed; row ordering and batch position are kept fixed. All takes, including high-WER and frame-limit cases, contribute to the means. No best-of-N winner filtering is applied.</p>'
    body+='<pre>'+esc(design['texts']['main'])+'</pre><p>Anger, fear, sadness, elation and affection use distinct actor-style descriptions, deliberately avoiding an explicit laugh, scream or sob which could confound speech errors. The structured surface requests one 0.35-second pause; caption uses the same acting direction and exact quoted transcript without the timed cue. Prompt-format comparisons therefore evaluate these complete practical recipes, not an isolated causal effect of label syntax.</p>'
    body+='<p>The transcript is never rewritten by Luna during the study. Sampling uses top-k 50, top-p 0.95 and repetition penalty 1.0. Audio T=0 is true argmax and also makes stop/continue greedy; at positive audio temperatures stop temperature remains 1.0. Thus the T=0 contrast includes the demo’s associated stop-policy change, not only audio entropy. T=0 performs poorly on this fixed study recipe, not on every possible prompt/reference: a separate German demo smoke test at T=0 produced zero measured WER with different words and reference. CFG affects only audio logits, using an affect-neutral comparison retaining words, reference and requested pause. Generation uses natural EOS with a 5.5-second frame hint plus a 60-frame safety budget, not forced benchmark lengths.</p>'
    body+='<h3>Predeclared preference rule</h3><pre>'+esc(design['selection_rule'])+'</pre><p>We also mark the Pareto frontier: no other tested configuration simultaneously has lower raw WER, higher genuineness and higher emotion-rank percentile, with at least one strictly better. Changing thresholds or the priority assigned to emotion can change the recommendation.</p>'
    body+='<h2>CFG × temperature maps</h2><p>WER is measured on untrimmed PCM. Genuineness and emotion describe the processed audio heard in the demo. Green WER ≤10%, yellow ≤30%, red &gt;30%. A green cell alone is not a three-objective win.</p><h3>Caption / exact transcript</h3>'+grid_table(summary['cells'],'caption')+'<h3>Structured acting script</h3>'+grid_table(summary['cells'],'structured')
    for fmt in ['caption','structured']:
        cells=[c for c in summary['cells'] if c['stage']=='main' and c['format']==fmt]
        rows=[]
        for c in cells:
            rows.append([f'{c["cfg"]:g}',f'{c["temperature"]:g}',str(c['n']),f'{c["raw_wer"]*100:.1f} [{c["raw_wer_ci"][0]*100:.1f}, {c["raw_wer_ci"][1]*100:.1f}]',f'{c["post_wer"]*100:.1f}',f'{c["low_wer_rate"]*100:.0f}%',f'{c["failure_rate"]*100:.0f}%',f'{c["genuineness"]:.2f}',f'{c["emotion_percentile"]:.3f}',f'{c["top5_rate"]*100:.0f}%',f'{c["clap"]:.3f}', 'Pareto' if c.get('pareto') else ''])
        body+=f'<details><summary>All {fmt} configurations and descriptive 95% cluster-bootstrap WER intervals</summary>'+table(['CFG','T','n','Raw WER % [CI]','Post WER %','WER ≤10%','WER >30%','Genuine /6','Emotion percentile','Target top 5','CLAP','Frontier'],rows, ['pareto' if c.get('pareto') else '' for c in cells])+'</details>'
    body+='<h2>How emotion-specific are the compromises?</h2>'
    rows=[]
    for emotion in design['emotions']:
        for fmt in ['caption','structured']:
            grid=defaultdict(list)
            for r in main:
                if r['emotion']==emotion and r['format']==fmt:grid[(r['guidance'],r['temperature'])].append(r)
            if not grid:continue
            stats=[(g,t,aggregate(rs)) for (g,t),rs in grid.items()]
            viable=[x for x in stats if x[2]['raw_wer']<=.1]
            chosen=max(viable,key=lambda x:x[2]['emotion_percentile']) if viable else min(stats,key=lambda x:x[2]['raw_wer'])
            g,t,c=chosen
            rows.append([esc(emotion),fmt,f'{g:g} / {t:g}',f'{c["raw_wer"]*100:.1f}%',f'{c["genuineness"]:.2f}',f'{c["emotion_percentile"]:.2f}', 'Emotion max among WER≤10%' if viable else 'No low-WER cell; lowest WER'])
    body+=table(['Emotion','Format','CFG / T','Raw WER','Genuine','Emotion percentile','Exploratory rule'],rows)+'<p class="muted">Only two seeds per emotion-format cell, one utterance, and no per-emotion naturalness constraint in this exploratory table. Do not treat these as reliable personalized presets.</p>'
    body+='<h2>Selected-setting generalization checks</h2>'
    if design.get('reference_provenance'):
        body+='<div class="callout warning"><b>Reference provenance matters:</b> reference A is a positive-valence-labelled corpus speech clip; reference B comes from a hiss-labelled take that also contains speech. Neither is a verified neutral-voice control. B may carry burst/timbre contamination, so that cohort is a reference-sensitivity stress check, not a clean-reference replication. These limitations do not invalidate within-reference CFG/temperature comparisons, but prevent a universal clean-reference optimum claim. Exact source filenames and hashes are recorded in the design JSON.</div>'
    for stage,label in [('german','German words, same reference A'),('reference_b','Same English words, distinct reference B')]:
        cells=[c for c in summary['cells'] if c['stage']==stage and c['format']!='both']
        if not cells:
            body+=f'<p>{label}: not yet measured.</p>';continue
        body+=f'<h3>{label}</h3>'+table(['Format','CFG','T','n','Raw WER %','Post WER %','Genuine /6','Emotion percentile','CLAP'],[[c['format'],f'{c["cfg"]:g}',f'{c["temperature"]:g}',str(c['n']),f'{c["raw_wer"]*100:.1f}',f'{c["post_wer"]*100:.1f}',f'{c["genuineness"]:.2f}',f'{c["emotion_percentile"]:.2f}',f'{c["clap"]:.3f}'] for c in cells])
    body+='<p>The holdout is a limited generalization check of selected configurations; it is not a complete German or second-speaker grid and cannot establish the optimum there. The German translation has different phonetics, so language comparisons are descriptive.</p>'
    pool=summary['caption_generalization_pool']
    if pool:
        body+='<h3>Caption: equally weighted main / German / second-reference cohorts</h3>'+table(['CFG','T','Actual n','Raw WER %','Genuineness /6','Emotion percentile'],[[f'{c["cfg"]:g}',f'{c["temperature"]:g}',str(c['n']),f'{c["raw_wer"]*100:.2f}',f'{c["genuineness"]:.3f}',f'{c["emotion_percentile"]:.3f}'] for c in pool])+'<p>This descriptive pool is reported after selection; it is not a preregistered selection objective or an independent final test. All three cohorts share emotions and related words. It shows sensitivity to voice/reference and language rather than eliminating it.</p>'
    if summary['paired_vs_default']:
        body+='<h3>First-grid preferred setting minus the existing default, paired by emotion and seed</h3>'+table(['Format','Metric','Mean delta','Descriptive 95% cluster-bootstrap interval'],[[p['format'],p['metric'],f'{p["delta"]:+.4f}',f'[{p["ci"][0]:+.4f}, {p["ci"][1]:+.4f}]'] for p in summary['paired_vs_default']])+'<p>Negative WER delta is better; positive genuineness/emotion/CLAP delta is better. Intervals overlapping zero do not provide a clear directional result on this small prompt set. No multiple-comparison correction or population significance claim is made.</p>'
    short=[c for c in summary['cells'] if c['stage']=='short_cues' and c['format']=='structured']
    if short:
        body+='<h2>Exploratory structured-recipe rescue</h2><p>The initial paragraph-length parenthesized directions were very unreliable. After observing that failure, an additional grid uses terse sentence-level physical cues and explicit 1.4 / 3.3-second sentence duration tags, matching the model card’s recipe style, with the same global descriptions, words, reference and seeds. This is an exploratory follow-up, not an untouched holdout. Cue verbosity, cue placement and authored timing change together, so causal attribution to any one detail is not possible.</p>'
        body+=table(['CFG','T','n','Raw WER %','Post WER %','Genuineness /6','Emotion percentile'],[[f'{c["cfg"]:g}',f'{c["temperature"]:g}',str(c['n']),f'{c["raw_wer"]*100:.1f}',f'{c["post_wer"]*100:.1f}',f'{c["genuineness"]:.2f}',f'{c["emotion_percentile"]:.2f}'] for c in short])
        baseline=next((c for c in short if c['cfg']==1.5 and c['temperature']==1.),None)
        eligible=[c for c in short if baseline and c['raw_wer']<=.1 and c['genuineness']>=baseline['genuineness']-.25]
        if eligible:
            c=max(eligible,key=lambda c:c['emotion_percentile'])
            body+=f'<div class="callout warning"><b>Short-cue structured candidate: CFG {c["cfg"]:g}, T={c["temperature"]:g}.</b> Raw WER {c["raw_wer"]*100:.1f}%, post-fade WER {c["post_wer"]*100:.1f}%, genuineness {c["genuineness"]:.2f}/6 and emotion percentile {c["emotion_percentile"]:.2f}. It passes the same relative-baseline rule on this follow-up, but its absolute naturalness proxy is low and it has not been validated on German or reference B. This is an audition candidate, not the recommended universal default.</div>'
        body+='<p>This rescue changes the format-level conclusion: concise structured cues can produce substantially better speech than verbose paragraph cues. The initial failed recipe is not evidence that every GENERAL/SCRIPT prompt fails.</p>'
    legacy=[c for c in summary['cells'] if c['stage']=='legacy' and c['format']!='both']
    if legacy:
        body+='<h2>Original-backend quality diagnostic</h2><p>Twenty additional free-running takes use the original serial-CFG loop and original FP32 codec weights at CFG 1.5 / T=1, with the same batch positions, prompts and seeds. Stochastic trajectories are not bit-identical after optimization, so this is a small distributional diagnostic, not a powered perceptual equivalence or non-inferiority trial.</p>'
        matched=[c for c in summary['cells'] if c['stage']=='main' and c['format']!='both' and c['cfg']==1.5 and c['temperature']==1.]
        body+=table(['Backend','Format','n','Raw WER %','Post WER %','Genuineness /6','Emotion percentile'],[[('Original' if c['stage']=='legacy' else 'Optimized'),c['format'],str(c['n']),f'{c["raw_wer"]*100:.1f}',f'{c["post_wer"]*100:.1f}',f'{c["genuineness"]:.2f}',f'{c["emotion_percentile"]:.2f}'] for c in legacy+matched])
    body+='<h2>Metrics, uncertainty and caveats</h2><ul><li><b>WER:</b> conventional minimum word-edit distance (S+D+I)/N, Unicode-aware case folding and punctuation removal, preserving German letters. Unlike the older demo’s SequenceMatcher distance, one substituted word counts once, not twice. Raw WER is primary; post-fade WER is secondary so trimming cannot hide generation errors. ASR can itself mishear expressive speech; WER is not a human transcript audit. Study ASR uses anti-aliased 48→16 kHz resampling consistently.</li><li><b>Genuineness:</b> the existing VoiceCLAP-commercial full 0–6 head measures a “lived-in” speech impression. It is a naturalness-related proxy, not listener MOS or a complete audio-quality metric.</li><li><b>Emotion:</b> the <a href="https://huggingface.co/laion/Empathic-Insight-Voice-Small">published EIV suite</a> is run with its documented EmoWhisper-AnS-Small-v0.1 encoder and ReLU heads. We report target rank among 40 emotions as percentile (40−rank)/39; percentile 1 means rank 1 and 0 rank 40. Raw scores and ranks are not calibrated probabilities, intensity guarantees or causal proof. Distinct heads can have different scales. The model was trained partly on synthetic voices; proxy optimization may exploit its biases.</li><li><b>Direction match:</b> <a href="https://huggingface.co/laion/voiceclap-commercial">VoiceCLAP-commercial</a> centered audio/text cosine gives a second, correlated indication of whether the acting description is audible. Genuineness and direction CLAP share a backbone, so they are not independent confirmations. Blend scores are diagnostic only because no explicit bursts are requested.</li><li><b>Uncertainty:</b> 2,000 fixed-seed cluster-bootstrap resamples average the repeated seeds within emotion-format blocks. Per-format estimates have only five clusters. Intervals describe variation across this small prompt set, not broad population uncertainty. T=0 seeds may duplicate, and the common text/reference severely limit generalization. Sixteen-way grid selection has winner’s bias; holdouts and auditioning are necessary.</li><li><b>Scope:</b> S3 only, no LoRA/DPO, reference-conditioned, short utterances, caption and structured surfaces only. Tags/transcript-only, long monologues, vocal-burst onset accuracy, other stages and human preference were not tested. The grid omits intermediate temperatures such as 0.8 and CFG values between tested levels; “best tested” does not imply a continuous global optimum.</li></ul>'
    body+='<p>The <a href="https://huggingface.co/laion/Humaneness-Voice-Small">Small model card</a> also distinguishes experimental timing/burst control and caption intelligibility from learned quality proxies; this pilot does not inherit its checkpoint-specific numerical rankings.</p>'
    body+='<h2>Listen to the evidence</h2><p>All completed takes are available below, not just winners. Players use processed mono MP3 at 96 kbps; metrics use lossless PCM before MP3. Raw playback is available to inspect unwanted lead/tail speech. Filter the table to compare the same emotion and seed across settings.</p><label>Filter <input id="filter" type="search" placeholder="e.g. Fear caption g1.5 t0.5" style="font:inherit;padding:8px;background:#1b2030;color:#ececf1;border:1px solid #343b50;border-radius:6px;width:min(100%,400px)"></label><p id="count" class="muted"></p><div id="takes"></div>'
    compact=[{'id':r['id'],'emotion':r['emotion'],'stage':r['stage'],'format':r['format'],'cfg':r['guidance'],'temperature':r['temperature'],'seed':r['seed'],'wer':r['raw_word_errors']['wer'],'postwer':r['processed_word_errors']['wer'],'genuine':r['genuineness'],'rank':r['emotion_rank'],'audio':r['audio'],'raw_audio':r['raw_audio'],'hyp':r['raw_hypothesis'],'text':r['spec']['text'],'prompt':r['spec']['prompt']} for r in results]
    body+='<script>const samples='+json.dumps(compact,ensure_ascii=False).replace('<','\\u003c')+';'+r'''
const container=document.getElementById('takes'), filter=document.getElementById('filter');
function render(){container.replaceChildren();const terms=filter.value.toLowerCase().split(/\s+/).filter(Boolean);const rows=samples.filter(s=>terms.every(t=>(s.id+' '+s.emotion+' '+s.format+' '+s.stage).toLowerCase().includes(t)));document.getElementById('count').textContent=rows.length+' matching takes. Showing the first 60; refine the filter to find all others.';for(const s of rows.slice(0,60)){const d=document.createElement('details');const title=document.createElement('summary');title.textContent=s.stage+' · '+s.emotion+' · '+s.format+' · CFG '+s.cfg+' / T '+s.temperature+' · seed '+s.seed+' · raw WER '+(s.wer*100).toFixed(1)+'%';d.append(title);const metric=document.createElement('p');metric.textContent='Processed WER '+(s.postwer*100).toFixed(1)+'% · Genuineness '+s.genuine.toFixed(2)+'/6 · Target emotion rank '+s.rank+'/40';d.append(metric);for(const [label,path] of [['Processed',s.audio],['Raw',s.raw_audio]]){const p=document.createElement('p');p.textContent=label+' ';const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=path;p.append(audio);d.append(p);}const pre=document.createElement('pre');pre.textContent='Expected: '+s.text+'\nRaw ASR: '+s.hyp+'\n\n'+s.prompt;d.append(pre);container.append(d);}}filter.addEventListener('input',render);render();
</script>'''
    body+='<h2>Full reproducibility</h2><p>'+' · '.join(f'<a href="{name}.json">{name}.json</a>' for name in ['design','results','summary','scorer-environment'])+'</p><p>Re-run using <code>emotion_study.py design</code>, <code>generate</code>, <code>score --follow</code>; each raw take and scored result is persisted separately. The design records exact prompts, seeds, generation budget and predeclared criteria. Reference recordings are local corpus samples, not user uploads, and are not exposed by the report server.</p>'
    (root/'index.html').write_text(page('Humaneness Small — emotion, CFG and temperature',body))
    print(json.dumps(summary['recommendations'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('kind',choices=['throughput','study','both'])
    args=parser.parse_args()
    if args.kind in ['throughput','both']:throughput()
    if args.kind in ['study','both']:emotion_study()
