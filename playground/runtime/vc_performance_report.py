"""English benchmark report with honest streaming latency and inline samples."""
import base64,html,json,subprocess
import settings


def main():
    root=settings.HERE/'vc-performance'
    report=json.loads((root/'results.json').read_text())
    hardware=json.loads((root/'hardware.json').read_text()) if (root/'hardware.json').exists() else {}
    quality=json.loads((root/'quality.json').read_text()) if (root/'quality.json').exists() else {}
    bf16=json.loads((root/'bf16.json').read_text()) if (root/'bf16.json').exists() else {}
    esc=lambda x:html.escape(str(x))
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Voice conversion throughput</title>',
        '<style>body{font:16px system-ui;max-width:1200px;margin:auto;padding:20px;background:#fafafa}p{line-height:1.5}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}audio{width:100%;max-width:600px}.warning{color:#92381e}</style>',
        '<h1>Chatterbox voice conversion: GPU batching and streaming</h1>',
        '<p>Optional stage in the Humaneness Small Playground. Same target reference, 10 original flow steps, original watermark. No DPO or speaker LoRA. This is a throughput and integration test, not a broad emotion/identity quality study.</p>',
        '<h2>Conclusions and serving policy</h2><p>Single-take conversion already works incrementally using source-context windows and a held 40 ms crossfade. Live chunk search converts every candidate before ranking, then sends only the selected committed converted PCM. Top-2 search can delay delivery until the retained beams share exactly identical converted PCM; stochastic VC output is not assumed to be identical just because TTS tokens match.</p>',
        '<p>The implementation now batches equal-length candidate windows on the GPU, groups unlike lengths rather than silently padding contexts, vectorizes per-wave-normalized log-mel extraction and reuses resamplers. Fixed catalogue target conditionings are cached (maximum four); custom uploads are not put into this shared target cache. Speaker vectors remain optional and separate from VC.</p>',
        '<p class="warning">Aggregate RTF divides by ALL candidate audio processed. For real-time winner streaming, the entire candidate batch must finish within the NEW chunk interval, not within N × that interval. A 3.2-second steady conversion window contains 1.6 seconds of old context and only 1.6 seconds of new audio. TTS, reward scoring, alignment, MP3 and network add further latency.</p>',
        f'<p>GPU: {esc(report["device"])}; S3Gen total: {report["parameters"]:,} parameters. Three warmed repetitions per case; median wall time. {esc(report["scope"])}. No concurrent GPU work was intentionally submitted during timings. First-load/compilation costs are excluded from warmed rows and reported separately where measured.</p>',
        '<h2>FP32 batch size / replicas</h2><div class="scroll"><table><tr><th>Method</th><th>N</th><th>Microbatch</th><th>Source/window s</th><th>Batch wall s</th><th>Aggregate processed-audio RTF</th><th>Batch wall / new stream interval</th></tr>']
    for row in report['timings']:
        duration=row['source_seconds'];interval=1.6 if duration==3.2 else duration
        parts.append('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in [row['method'],row['n'],row.get('microbatch','—'),duration,
            f'{row["seconds"]:.3f}',f'{row["aggregate_rtf"]:.3f}',f'{row["seconds"]/interval:.3f}'])+'</tr>')
    parts.append('</table></div><p>Two independent copies with two CUDA streams were benchmarked on the same card. Their small gain must be weighed against doubled model memory and more difficult scheduling; model replication is not enabled in serving merely because a test finished faster.</p>')
    if bf16:
        parts.append('<h2>Fast serving option: BF16 flow only</h2><p>The optional VC stage now defaults to BF16 only in the flow estimator, with an explicit FP32 comparison selector in the UI. Tokenizer, conditioning encoder and vocoder remain FP32; the original FP32 estimator is retained intact for switching. No reduction in the original ten flow steps. The small quality-proxy check below is promising, not universal preservation proof.</p><div class="scroll"><table><tr><th>N</th><th>Source/window s</th><th>Batch wall s</th><th>Processed-audio aggregate RTF</th><th>Wall / new chunk interval</th></tr>')
        for row in bf16['rows']:
            duration=row['source_seconds'];interval=1.6 if duration==3.2 else duration
            parts.append('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in [row['n'],duration,f'{row["seconds"]:.3f}',f'{row["aggregate_rtf"]:.3f}',f'{row["seconds"]/interval:.3f}'])+'</tr>')
        parts.append('</table></div><p>At eight streams with 1.6-second new chunks plus 1.6-second old context, VC alone takes 0.943 s per round (round RTF 0.590). Sixteen streams take 1.886 s (round RTF 1.179), so they do not keep up at that interval. Neither number establishes full TTS/search/scoring/network real-time performance.</p>')
    if hardware:
        parts.append('<h2>Isolated hardware experiments — not production defaults</h2><pre>'+esc(json.dumps(hardware,indent=2))+'</pre><p>TF32 and compilation can change numerical output. A smaller wall time alone does not prove preserved voice identity, clarity or emotion. Compilation can introduce substantial cold latency for new shapes.</p>')
    if quality:
        parts.append('<h2>Eight short-clip quality-proxy checks</h2><pre>'+esc(json.dumps(quality,indent=2))+'</pre><p>ECAPA and Ears Medium learned proxies; 1.6-second clips, one conversion target. These are numerical-optimization safeguards, not human MOS or an emotion-realization study. No meaningful full-transcript WER is claimed for these truncated benchmark inputs.</p>')
    parts.append('<h2>Listen to numerical variants</h2>')
    for source in sorted(root.glob('hardware-*-n8-0.wav'))+sorted(root.glob('parity-n8-0-*.wav')):
        mp3=source.with_suffix('.mp3')
        subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(source),'-ac','1','-c:a','libmp3lame','-b:a','96k',str(mp3)],check=True)
        encoded=base64.b64encode(mp3.read_bytes()).decode('ascii')
        parts.append('<p>'+esc(source.stem)+'</p><audio controls preload="none" src="data:audio/mpeg;base64,'+encoded+'"></audio>')
    live_path=settings.HERE/'test-results/catalogue-vc-live.json'
    if live_path.exists():
        live=json.loads(live_path.read_text())
        parts.append('<h2>Real HTTPS integration checks</h2><p>Actual TTS → optional VC → post-conversion alignment/rewards → mono 96 kbit/s MP3. Individual request times include cold work and must not be compared as warmed VC benchmarks. The first new TTS shape took about 39 seconds despite startup warm-up; the first VC request also included model loading. No numerical or request failures occurred, but some non-VC takes omitted the second sentence (WER 0.50): technical success is not universal speech reliability.</p><div class="scroll"><table><tr><th>Case</th><th>Chunks before result</th><th>Total s</th><th>First chunk s</th><th>VC s</th><th>Largest GPU microbatch</th><th>Final WER</th></tr>')
        for item in live:
            result=item['result'];vc=result['voice_conversion'];chunks=item['chunks'];winner=result['candidates'][result['winner']]
            values=[item['case'],len(chunks),f'{result["total_ms"]/1000:.3f}',f'{chunks[0]["seconds"]:.3f}' if chunks else '—',
                f'{vc.get("conversion_seconds",0):.3f}',vc.get('max_microbatch_size','—'),winner.get('wer','—')]
            parts.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in values)+'</tr>')
        parts.append('</table></div>')
        for label in ('vc-single-live','vc-single-fp32','vc-winner4','vc-beam4'):
            source=settings.HERE/'test-results'/f'catalogue-{label}-take.mp3'
            if source.is_file():
                encoded=base64.b64encode(source.read_bytes()).decode('ascii')
                parts.append('<p>'+esc(label)+'</p><audio controls preload="none" src="data:audio/mpeg;base64,'+encoded+'"></audio>')
    parts.append('<h2>Reproduce</h2><pre>python vc_performance_study.py\npython vc_hardware_study.py\npython vc_quality_check.py\npython vc_performance_report.py\npython catalogue_vc_live_test.py</pre>')
    parts.append('<details><summary>Raw timings / seeds and parity caveats</summary><pre>'+esc(json.dumps(report,indent=2))+'</pre></details><p>Batch-vs-serial waveform output is not bit-exact. Per-candidate generators prevent arbitrary batch RNG coupling, but floating-point kernels and token quantization can still differ.</p>')
    parts.append('<p><a href="https://github.com/LAION-AI/chatterbox-voice-conversion">LAION Chatterbox VC implementation</a> · <a href="results.json">Raw benchmark JSON</a> · <a href="/">Playground</a> · <a href="/catalogue-steering-study/">Speaker-vector study</a></p></html>')
    temp=root/'index.build.html';temp.write_text('\n'.join(parts));temp.replace(root/'index.html')
    print('Report:',root/'index.html')


if __name__=='__main__':main()
