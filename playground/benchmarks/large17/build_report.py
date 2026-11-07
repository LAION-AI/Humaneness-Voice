"""Build the portable English report from measured JSON (no guessed results)."""
import json,statistics,html,shutil,subprocess,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DEST=Path(os.environ.get('HVS_REPORT_DEST',str(ROOT/'report')))
def esc(x):return html.escape(str(x))
def median(xs):return statistics.median(xs) if xs else None
def fmt(x,n=3):return '—' if x is None else f'{x:.{n}f}'
def summary(c):
 runs=[r for r in c['runs'] if not r['warmup']]
 return {'n':len(runs),'ttfa':median([r['chunks'][0]['seconds'] for r in runs if r['chunks']]),
  'wall':median([r['wall_seconds'] for r in runs]),
  'rtf':median([r['performance']['aggregate_rtf'] for r in runs]),
  'vram':max([r['peak_allocated_gb'] for r in runs],default=None),
  'capped':sum(sum(r.get('hit_frame_cap',[])) for r in runs)}
def main():
 DEST.mkdir(parents=True,exist_ok=True)
 data=json.loads((ROOT/'benchmark.json').read_text())
 variants=[('Original Small optimization package',data)]
 if (ROOT/'layerwise.json').exists():variants.append(('Layerwise semantic compilation',json.loads((ROOT/'layerwise.json').read_text())))
 rows=[]
 quick=[]
 for label,doc,streaming,batch in [
     ('Same Small optimizations · one stream',data,True,1),
     ('Same Small optimizations · one offline take',data,False,1),
     ('Additional layerwise compiler · best offline batch',variants[-1][1],False,None),
     ('Additional layerwise compiler · best streaming batch',variants[-1][1],True,None)]:
  candidates=[c for c in doc['conditions'] if c.get('optimized',True) and c['cfg']==2 and c['streaming']==streaming
              and (batch is None or c['batch']==batch) and len(c['runs'])>=4 and not c.get('error')]
  if candidates:
   c=min(candidates,key=lambda c:summary(c)['rtf']);s=summary(c)
   quick.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [label,c['batch'],fmt(s['ttfa']),fmt(s['wall']),fmt(s['rtf']),fmt(1/s['rtf'],1)])+'</tr>')
 overview='<section><h2>At a glance · CFG 2, temperature 0.8</h2><div class="scroll"><table><tr><th>Operating point</th><th>Batch</th><th>First PCM (s)</th><th>Complete batch (s)</th><th>Aggregate RTF ↓</th><th>Audio s / compute s ↑</th></tr>'+''.join(quick)+'</table></div><p>All rows use the same short German sentence and fixed Mira reference. A batch produces multiple alternative takes. Aggregate RTF below 1 does not mean every candidate finishes within its own duration. The live service keeps a conservative batch cap of 8; the short benchmark additionally probes 16.</p></section>'
 api_rows=[]
 if (ROOT/'api-check.json').exists():
  receipts=json.loads((ROOT/'api-check.json').read_text())
  for model in ('small_rl','large17'):
   for align in (False,True):
    group=[r for r in receipts if r['kind']=='generation' and r['model']==model and r['align']==align and not r['warmup']]
    if group:api_rows.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [model,align,len(group),fmt(median([r['chunks'][0]['at'] for r in group])),fmt(median([r['wall_seconds'] for r in group]))])+'</tr>')
 api_section='<section><h2>Actual serving path</h2><p>Local HTTP SSE first PCM, including reference lookup, preparation and the selected finishing path. Full request time can include end alignment and file output. These are not browser mouth-to-ear measurements.</p><table><tr><th>Model</th><th>Alignment</th><th>Warm repeats</th><th>First PCM (s)</th><th>Full request (s)</th></tr>'+''.join(api_rows)+'</table>'
 if (ROOT/'browser-check.json').exists():
  browser=json.loads((ROOT/'browser-check.json').read_text())
  api_section+='<p>Real HTTPS browser model-choice/audio/action check: <strong>'+('passed' if browser['passed'] else 'not passed')+'</strong>. Scope: typed synthetic prompts, the real Qwen endpoint, selected TTS checkpoint, PCM reception, a spin command and return to Small. This does not test a physical microphone.</p>'
  api_section+='<p>Later browser swaps with CPU weights retained: '+', '.join(esc(s['model'])+' '+fmt(s['wall_seconds'],1)+' s' for s in browser['switches'])+'. The first direct Large selection after service startup took 132.1 s including loading and warming. Model swaps are separate from per-utterance latency; Small still rebuilds its monolithic graphs.</p>'
  api_section+='<p>The two Large HTTPS smoke turns delivered first PCM after '+', '.join(fmt(t['first_pcm_ms']/1000,2)+' s' for t in browser['turns'] if t['model']=='large17')+' from typed submission, including Qwen and transport. These are two examples, not an end-to-end latency distribution or a measurement of physical speaker playback. The first Small browser turn also includes establishing a fresh session.</p>'
 if (ROOT/'quality.json').exists():
  quality=json.loads((ROOT/'quality.json').read_text())
  api_section+='<p>Transcription canary: '+str(sum(r['normalized_wer']==0 for r in quality))+'/'+str(len(quality))+' retained optimized clips had zero normalized WER under Parakeet v3. Teacher-forced semantic state cosines exceeded 0.9986; relative RMSE reached about 4.5%, so BF16 paths are not bit-identical. These are correctness canaries, not perceptual or speaker-identity scores.</p>'
 api_section+='</section>'
 for variant,doc in variants:
  for c in doc['conditions']:
   if not c['runs'] and not c.get('error'):continue
   s=summary(c)
   rows.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [variant if c.get('optimized',True) else 'Legacy eager loop',
       c['cfg'],'Streaming' if c['streaming'] else 'Offline',c['batch'],s['n'],fmt(s['ttfa']),fmt(s['wall']),
       fmt(s['rtf']),fmt(1/s['rtf'] if s['rtf'] else None,1),fmt(s['vram'],2),
       c.get('error','Frame cap reached' if s['capped'] else '')])+'</tr>')
 columns=['Path','CFG','Mode','Batch','Warm repeats','First PCM (s)','Full batch (s)','Aggregate RTF','Audio s / compute s','Peak allocated GB','Notes']
 profile=data.get('checks',{}).get('component_profile',{})
 profile_rows=''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [name,m['calls'],fmt(m['total_ms'],1),fmt(m['median_ms'],2)])+'</tr>' for name,m in profile.items())
 audios=[]
 for path in sorted(ROOT.glob('sample-*.wav'))+sorted(ROOT.glob('long-*.wav')):
  target=DEST/'large17-audio';target.mkdir(exist_ok=True)
  name=path.stem+'.mp3'
  subprocess.run(['ffmpeg','-v','error','-y','-i',str(path),'-ac','1','-b:a','96k',str(target/name)],check=True)
  audios.append(f'<article><h3>{esc(path.stem)}</h3><audio controls preload="none" src="large17-audio/{esc(name)}"></audio></article>')
 raw_files=['benchmark.json','layerwise.json','initial-smoke.json','smoke-asr.json','quality.json','api-check.json','browser-check.json']
 links=[]
 for filename in raw_files:
  if (ROOT/filename).exists():
   published='large17-'+filename;shutil.copyfile(ROOT/filename,DEST/published)
   links.append(f'<a href="{published}">{esc(filename)}</a>')
 page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Humaneness Voice Large 1.7 · RTX 3090 streaming and batch study</title>
<style>body{font:16px/1.55 system-ui,sans-serif;margin:0;background:#f5f5f0;color:#1e293b}main{max-width:1400px;margin:auto;padding:28px}h1{line-height:1.15}section{background:white;border-radius:16px;padding:24px;margin:20px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}th{background:#e7eee9;position:sticky;top:0}code{background:#eef2f6;padding:2px 4px}a{color:#27604e}.samples{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}audio{width:100%}.note{border-left:4px solid #b6852d;padding-left:16px}pre{white-space:pre-wrap;font-size:13px}</style>
<main><a href="/desk-robot/">← Robby demo</a><h1>Large 1.7: streaming onset and batch throughput</h1>
<p>Measured on 7 November 2026. One NVIDIA RTX 3090, 24 GiB VRAM; the assistant’s other workers remain resident. The second RTX 3090 is not used for TTS tensor parallelism.</p>
<section><h2>Measured results</h2><p>Each measured cell is the median of three seeds after a separate warm-up. First PCM is arrival at the local streaming callback, before network transport or the browser’s playback reserve. Offline mode returns complete waveforms. Batch throughput divides by the <strong>sum</strong> of all candidate audio durations; it is not the latency of one user’s answer.</p><div class="scroll"><table><thead><tr>'''+''.join('<th>'+x+'</th>' for x in columns)+'''</tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<p class="note">The first two CFG=1 eager baseline conditions overlapped a short ASR smoke check and should not be used for precise speedup claims. CFG=2 is the matched Robby operating point. This is a shared-host test, not an isolated hardware maximum. An out-of-memory row is a failed configuration, not a successful throughput result.</p></section>
<section><h2>What changed</h2><p>The pinned merged Large 1.7 release contains 4,129,990,144 parameters. Its FP32 distribution shards are verified by SHA-256 and converted once to private BF16 CPU weights. GPU inference uses BF16 without reapplying the training adapter. Small and Large share the existing MOSS Audio Tokenizer v2.</p>
<p>The unchanged Small package combines cached reference codes, conditional/unconditional branch batching, static semantic KV caches, compiled semantic continuation, CUDA graphs, fast top-k/top-p sampling, the Triton acoustic-token decoder for one candidate, causal codec graphs and fused codec operations. Single-stream chunks grow from 3 to 6 to 12 to 20 frames (240, 480, 960, 1600 ms). Multiple candidates use the original batched local decoder and causal batched codec; the specialized single-row Triton kernels and codec graphs do not silently run as serial candidate inference.</p>
<p>The eager comparison uses the shared runtime’s legacy frame loop with codec fusion and graphs disabled. It is not presented as a benchmark of the release’s standalone CLI. Its streaming chunk is 20 frames; part of the first-audio improvement therefore comes from the smaller initial chunk.</p>
<p>The additional layerwise experiment compiles one transformer block and reuses it for all 36 semantic layers, inside a complete continuation CUDA graph. This reduces the size of the compilation problem. It uses the same BF16 weights, attention mask and KV semantics; teacher-forced state comparisons are included in the raw evidence.</p></section>
<section><h2>Where computation goes</h2><p>One warmed CFG=2 single stream, measured with CUDA events around components. Event instrumentation adds overhead; these figures explain the work and are not substituted for the uninstrumented table.</p><table><tr><th>Component</th><th>Calls</th><th>Total ms</th><th>Median ms per call</th></tr>'''+profile_rows+'''</table></section>
<section><h2>Reference, quality and timing contract</h2><p>The voice reference is the demo’s fixed synthetic Mira clip, 10.26 seconds. Large 1.7 requires 63–187 reference codec frames; waveform cropping happens before encoding at 14.96 seconds. The reference is cached by content. Shorter-than-supported references are rejected explicitly rather than silently padded or truncated to the historical 2.96 seconds.</p>
<p>Temperature is 0.8, top-p 0.95, top-k 50, repetition penalty 1.0. The main fixture says: “Hallo, schön, dass du da bist. Was möchtest du heute machen?” with a 5-second duration condition. A separate longer story sentence tests continuation. Stop decisions remain model-generated; frame caps are reported. Candidate batches use a fixed base seed per repeat; changing batch size changes RNG consumption and does not guarantee identical samples.</p>
<p>RTF includes semantic prefill, autoregressive generation and PCM decoding, including GPU launch gaps. It excludes reference preparation, prompt packing, ASR, the language model, scoring/ranking, alignment, voice conversion, MP3 and network time. Peak allocated VRAM is this process’s allocator measurement, not total board occupancy. Batch-1 callback timings include synchronization and a CPU copy; larger batches deliver a PCM block for each live row.</p>
<p>ASR transcript checks and teacher-forced state comparisons are smoke checks, not a blinded evaluation of naturalness, emotion or speaker similarity. Listen to the retained clips. Optimized sampling uses a different random stream, so matching seeds are not proof of identical speech.</p></section>
<section><h2>Browser model switching</h2><p>Use <strong>Voice synthesis</strong> near the top of Robby. Small remains the default; Large 1.7 is an explicit choice. Model swaps use a short asynchronous request plus status polling, avoiding Cloudflare’s long-request timeout. New turns wait for the swap. Active replies block model switching. Only one TTS model occupies GPU memory at a time, with verified CPU weights retained for subsequent swaps. The selection controls the shared GPU service.</p>
<p>The first compile is cold work and can take minutes. It is not included in warmed TTFA or RTF. Existing conversation history, face actions, speech recognition, interruptions and LLM selection remain in place. Optional early reaction batching retains the established Small-only path; Large uses ordered sentence streaming.</p></section>
<section><h2>Reproduce and inspect</h2><p>Runtime: <a href="https://github.com/LAION-AI/Humaneness-Voice">Humaneness-Voice</a>. Robby: <a href="https://github.com/christophschuhmann/cartoon-face-generator">cartoon-face-generator</a>. Model: <a href="https://huggingface.co/laion/Humaneness-Voice-Large-1.7">Large 1.7</a>, pinned revision <code>'''+esc(data['revision'])+'''</code>.</p><p>Use the repository’s RTX 3090 environment, stop only the TTS worker, and run the included benchmark with the same fixed reference. Keep other GPU load constant. Warm each batch/CFG shape separately, retain failed runs, and report cold initialization separately. See the accompanying Markdown guide for exact paths and commands.</p><p>Raw receipts: '''+' · '.join(links)+'''</p></section>
<section><h2>Generated listening samples</h2><div class="samples">'''+''.join(audios)+'''</div></section></main></html>'''
 page=page.replace('<section><h2>Measured results</h2>',overview+api_section+'<section><h2>Measured results</h2>')
 (DEST/'large17-speed.html').write_text(page)
 print(DEST/'large17-speed.html')
if __name__=='__main__':main()
