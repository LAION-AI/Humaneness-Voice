"""Build a self-contained English handoff from retained real measurements."""
import html,json,statistics,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PUBLIC='https://arrive-locations-parade-research.trycloudflare.com'

def load(name):
    p=ROOT/'static'/name
    return json.loads(p.read_text()) if p.exists() else {}
def esc(value):return html.escape(str(value))
def median(values):
    values=[v for v in values if isinstance(v,(float,int))]
    return statistics.median(values) if values else None
def ms(value):return 'Not measured' if value is None else f'{value:,.1f} ms'
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def main():
    speed=load('bolt-speed.json');engines=load('bolt-engine-comparison.json')
    speakers=load('bolt-speaker-speed.json');browser=load('bolt-browser-check.json');socket=load('bolt-socket-check.json')
    profile=load('bolt-tts-profile.json');precision=load('bolt-precision.json');graphs=load('bolt-semantic-graph.json');local_compile=load('bolt-local-compile.json')
    rows=speed.get('rows',[])
    old_path=ROOT/'logs/bolt-speed-before-precision.json'
    before=json.loads(old_path.read_text()) if old_path.exists() else {}
    def group(name,key):return median(r.get(key) for r in rows if r.get('group')==name and not r.get('error'))
    events=browser.get('events',[]);endpoints=[e for e in events if e.get('type')=='endpoint']
    prefetch=[e['timing'] for e in endpoints if e.get('timing',{}).get('speaker_prefetch_used')]
    critical=median(p.get('speaker_embedding_wait_ms',0)+p.get('speaker_lookup_ms',0) for p in prefetch)
    first=browser.get('first_turn',{});buffer=first.get('configured_reserve_ms')
    components=[
      ['ASR on a complete prerecorded request',ms(group('ASR + LLM + TTS (offline audio)','asr_ms')),'Real GPU transcription; excludes speaking and waiting for a sentence boundary.','Reuse native streaming words and avoid repeated full-window decodes.'],
      ['Streaming sentence boundary',ms(socket.get('endpoint_after_last_source_speech_ms')),'Real-time paced socket test, after last source speech; includes recognition/commit and identity. This overlaps other rows.','Apply newest ASR words before a quiet-hop flush; retain final words.'],
      ['Speaker assignment after prefetch',ms(critical),'Warm embedding was prepared while listening; median wait plus nearest-prototype lookup from browser endpoints.','Already moved off the critical path; retain a fallback for very short speech.'],
      ['ECAPA embedding computation',ms(median(r.get('embedding_ms') for r in speakers.get('rows',[]) if r.get('context_s')==3.04)),'Actual CPU work on already-heard speech, usually performed in the background.','Prefetch 1.28–3.04 s contexts; do not shorten to 640 ms just for a few milliseconds.'],
      ['Nearest voice search, 32 people × 8 anchors',ms(speakers.get('nn_32_people_8_vectors_median_ms')),'192-D cosine search at the configured maximum.','No indexing library is needed at this size.'],
      ['LLM first token',ms(group('LLM only','llm_ttft_ms')),'Local HTTP streaming, warm loaded model, changing short prompts. Subset of next row.','Keep the short prompt and llama.cpp prefix cache.'],
      ['LLM first complete short sentence',ms(group('LLM only','first_sentence_ms')),'Cue plus completed spoken sentence. Actual output word counts are retained below.','Start TTS at sentence completion while later LLM tokens continue.'],
      ['GENERAL template lookup','0.143 ms','Median of 1,000 warm in-memory lookups in a 100-template catalogue.','Further work here would have negligible end-to-end benefit.'],
      ['TTS first playable PCM',ms(group('TTS only · 1600 ms chunks','first_audio_ms')),'Fixed seven-word spoken sentence; includes shared TTS admission, generation and internal transport.','Largest measured generation bottleneck: profile semantic decoder, talker and audio decoder separately.'],
      ['TTS complete generation',ms(group('TTS only · 1600 ms chunks','total_ms')),'All generated audio received. This is not its spoken duration and is not added to first-audio latency.','Keep measured 1,600-ms frame chunks; smaller chunks were slower here.'],
      ['Browser first-audio reserve',ms(buffer),'Configured reserve for the observed first packet. Actual scheduled playback delay includes device lead and browser scheduling.','Adaptive: 180 ms for long first packets, at least 300 ms for short initial codec packets; grows after underruns.'],
      ['Cloudflare / physical microphone / device output','Not isolated','HTTPS is included in browser observations, but physical devices and room acoustics were not measured.','Measure on target client hardware before claiming physical mic-to-ear latency.']]
    summary=[]
    for name in ('LLM only','TTS only · 1600 ms chunks','LLM + TTS','ASR + LLM + TTS (offline audio)'):
        a=[r for r in rows if r.get('group')==name and not r.get('error')]
        summary.append([name,len(a),ms(median(r.get('first_sentence_ms') for r in a)),ms(median(r.get('first_audio_ms') for r in a)),ms(median(r.get('total_ms') for r in a)),', '.join(str(r.get('output_words')) for r in a)])
    chunkrows=[]
    for frames in (2,5,10,20):
        name=f'TTS only · {frames*80} ms chunks';a=[r for r in rows if r.get('group')==name and not r.get('error')]
        chunkrows.append([f'{frames*80} ms ({frames} frames)',len(a),ms(median(r.get('first_audio_ms') for r in a)),ms(median(r.get('total_ms') for r in a)),f'{median(r.get("audio_s") for r in a) or 0:.2f} s'])
    speakerrows=[]
    for seconds in (.64,1.28,2.,3.04):
        a=[r for r in speakers.get('rows',[]) if r.get('context_s')==seconds]
        speakerrows.append([f'{seconds:.2f} s',ms(median(r.get('embedding_ms') for r in a)),f'{median(r.get("full_utterance_cosine") for r in a) or 0:.3f}',
            str(sum(all(r['correct'] for r in a if r['speaker']==name) for name in ('Christoph','Robert','Selma')))+' / 3 distinct follow-up queries'])
    enginerows=[]
    for name in ('baseline','DFlash'):
        a=[r for r in engines.get('rows',[]) if r.get('engine')==name and not r.get('error')]
        enginerows.append([name,len(a),ms(median(r.get('llm_ttft_ms') for r in a)),ms(median(r.get('llm_first_sentence_ms') for r in a))])
    html_parts=['''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Robby local voice pipeline — latency, speaker matching and optimization report</title>
<style>:root{color-scheme:light}*{box-sizing:border-box}body{font:17px/1.65 system-ui,sans-serif;margin:0;background:#f7f8fa;color:#20313b}main{max-width:1200px;margin:auto;padding:32px 22px}h1{font-size:clamp(28px,4vw,42px);line-height:1.18}h2{margin-top:42px;line-height:1.3}a{color:#176678}p{max-width:1000px}.note{padding:18px;background:#e6f2f4;border-left:4px solid #247d8b}.warning{background:#fff2d9;border-left-color:#bc8419}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:15px;background:white}th,td{padding:12px;border:1px solid #ced8df;text-align:left;vertical-align:top}th{background:#eaf0f5}code{background:#edf1f4;padding:2px 5px;border-radius:4px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf1f4;padding:18px;font-size:13px}li{margin:9px 0}.flow{font-weight:650;padding:20px;background:#fff;border:1px solid #ced8df;border-radius:12px}details{margin:18px 0}summary{cursor:pointer;font-weight:600}.small{font-size:14px;color:#526570}</style></head><body><main>
<p class="small">LAION Humaneness Voice / Robby · measured 25 October 2026 · self-contained English handoff</p><h1>Where Robby spends time — and how the local voice pipeline was accelerated</h1>
<p>Robby is a browser robot that listens, keeps temporary voice labels, generates a reply locally and speaks it with Humaneness Voice Small. This report explains the implemented changes, measured delays and remaining work for someone unfamiliar with the project.</p>''']
    html_parts.append(f'<p><a href="{PUBLIC}/desk-robot/">Open the running demo</a> · <a href="{PUBLIC}/docs">FastAPI reference</a> · <a href="https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB">Requested NVIDIA model</a>. The Cloudflare link is temporary.</p>')
    html_parts.append('<div class="note"><strong>Main finding:</strong> TTS startup dominates local generation time. Voice-prototype search and fixed acting-template lookup are tiny. Background speaker preparation removes most identity work from the sentence-submit path. The LLM already finishes short replies in a fraction of a second.</div>')
    html_parts.append('<h2>1. Individual components and bottlenecks</h2>'+table(['Component','Measured median / setting','What the number means','Optimization'],components))
    html_parts.append('<p>These rows are not additive: first-token time is part of first-sentence time; the streaming endpoint includes recognition and identity; total TTS generation overlaps audio playback. Only timings from the same complete request define an end-to-end number.</p>')
    html_parts.append('<h2>2. Complete request measurements</h2>'+table(['Condition','Runs','First complete sentence','First audio','All generation finished','Actual output words per run'],summary))
    before_tts=[r for r in before.get('rows',[]) if r.get('group')=='TTS only · 1600 ms chunks' and not r.get('error')]
    after_tts=[r for r in rows if r.get('group')=='TTS only · 1600 ms chunks' and not r.get('error')]
    if before_tts and after_tts:
        html_parts.append('<p><strong>Live API validation after adapter merging:</strong> the same fixed seven-word TTS request at 1,600-ms chunks changed from '+ms(median(r['first_audio_ms'] for r in before_tts))+' to '+ms(median(r['first_audio_ms'] for r in after_tts))+' first playable PCM, and from '+ms(median(r['total_ms'] for r in before_tts))+' to '+ms(median(r['total_ms'] for r in after_tts))+' complete generation. Three runs per configuration. The smaller API gain reflects other processing and transport outside the optimized GPU operations. LLM replies may differ between runs; do not interpret their complete-request difference as a controlled precision-only comparison.</p>')
    html_parts.append('<p>Input requests contain seven words. The fixed TTS-only test also contains seven words. The LLM receives a seven-word target but can generate shorter or longer answers; the actual counts above are reported rather than silently treating them as identical. Each condition has three repeated measurements, which is useful engineering evidence but a small sample. Complete-audio ASR tests exclude the time the user spends speaking and the wait for a streaming boundary.</p>')
    html_parts.append('<h2>3. Real-time capture and browser playback</h2>')
    html_parts.append('<p>The socket test sends synthesized microphone audio in real time, 20 ms per packet. The browser test uses Chrome with an authored synthetic microphone file, the real HTTPS websocket, the actual local LLM and TTS, and the browser AudioContext playback clock. No private microphone recordings are published. The browser first-file lead-in explicitly allows capture-device initialization, avoiding a test-harness truncation at startup.</p>')
    html_parts.append(table(['Observation','Result'],[
        ['Real-time socket test passed',socket.get('passed',False)],
        ['Socket recognized text',socket.get('transcript','Pending')],
        ['Socket first PCM after last source speech',ms(socket.get('first_audio_after_last_source_speech_ms'))],
        ['HTTPS browser multi-voice scenario passed',browser.get('passed',False)],
        ['Browser first PCM after last captured speech',ms(first.get('pcm_after_last_capture_ms'))],
        ['Browser playback after last captured speech',ms(first.get('playback_after_last_capture_ms'))],
        ['Browser observed PCM-to-playback delay',ms(first.get('playback_reserve_ms'))],
        ['Browser within-reply underruns',browser.get('client_metrics',{}).get('underflows',0) if browser.get('passed') else 'See retained failure below'],
        ['Prefetched identity endpoints observed',len(prefetch)] ]))
    html_parts.append('<p>These are synthetic digital-capture/playback observations. The socket acknowledges transport completion without simulating a physical loudspeaker; the browser separately measures scheduling through AudioContext. Neither test measures room echo or a physical microphone/speaker pair. Do not label the complete-audio API number as physical microphone-to-ear latency.</p>')
    html_parts.append('<p>The new all-voice sentence mode is the scope of these latency and voice-label tests. The earlier strict speaker-gated interruption and semantic word-game reports describe the legacy mode; this run does not certify owner-only interruption or banana-game behavior in the new path. Explicit Stop handling and native TYGER BARGE are present, but physical echo, overlapping stop commands and game rules need a separate regression campaign.</p>')
    html_parts.append('<h2>4. Streaming design</h2><p class="flow">Microphone → frozen Nemotron ASR → stable complete sentence → approximate voice label → local Bolt token stream → completed output sentence → fixed GENERAL template → streamed TTS → browser playback</p>')
    html_parts.append('<p>The model is <code>nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB</code>, the requested Q4_K_M GGUF, served by CUDA llama.cpp with thinking disabled. It shares GPU 0 with ASR; the existing Humaneness Small RL TTS stays on GPU 1. Both cards are RTX 3090 with 24 GB. No extra TTS copy is loaded.</p><p>The speaking model writes short cues such as <code>(interest, soft)</code> followed by spoken text. A catalogue contains 40 emotions × two strengths plus 10 delivery styles × two strengths: 100 templates. Matching uses sparse keyword/alias cosine similarity in memory. This is lexical matching, not another neural embedding call. GENERAL is assembled locally and does not consume generated LLM prose.</p><p>A bounded LLM producer continues decoding while the TTS consumer generates the first sentence. Later output sentences wait in a small queue for the single shared TTS worker. New input sentences are preserved in order and become subsequent LLM requests with conversation history. An already decoding LLM request cannot be retroactively edited; new input does not magically enter its KV cache. Explicit stop commands and native TYGER interruption remain separate paths.</p>')
    html_parts.append('<h2>5. Faster speaker matching without a short-context accuracy sacrifice</h2>'+table(['Context already heard','CPU embedding time','Cosine to complete-utterance embedding','Nearest-reference identification'],speakerrows))
    html_parts.append('<p>This small speaker study uses three synthetic voices, each with a separate introduction and follow-up clip. Repeating a query three times measures timing, not nine independent people. The 640-ms condition made a mistake on one of the three distinct follow-up queries. Contexts of 1.28–3.04 seconds matched all three. This does not establish population-level speaker accuracy.</p><p>The live path therefore retains longer context while moving its CPU work earlier: it prewarms ECAPA at startup, prepares 1.28–3.04-second voice embeddings while listening, and uses only results from the same utterance no more than one second old. A sentence boundary invalidates the preceding sentence’s prepared anchor, because the next sentence may come from a different person. Short or failed preparations use the full-utterance fallback; uncertain identity never blocks otherwise intelligible speech.</p><p>People receive temporary Speaker 1, Speaker 2… labels. Names are optional. A correction such as “I am also Christoph” can merge an unnamed provisional profile into Christoph and preserve several independent acoustic references. The original reference is retained; weak routine matches do not automatically overwrite it. Up to 32 people and eight anchors per person are bounded in session RAM, with inactivity expiration. This is approximate conversational labelling, not authentication.</p>')
    html_parts.append('<h2>6. Smaller audio chunks were slower here</h2>'+table(['Requested chunk','Runs','First audio','Complete generation','Generated audio duration'],chunkrows))
    html_parts.append('<p>The measured default is 20 semantic frames, or 1,600 ms. A smaller frame count increased repeated audio-decoder/transport overhead in this implementation. Actual MP3/PCM packets can be shorter than the nominal chunk, particularly the first codec packet and the final tail. The browser therefore starts with a 180-ms reserve for substantial packets, at least 300 ms for short first packets, and increases its future reserve after an underrun. Buffer changes preserve the timestamp-based Second Ear mapping; they do not fabricate a fixed acoustic synchronization offset.</p>')
    html_parts.append('<h2>7. Speculative decoding comparison</h2>'+table(['Engine','Requests','First token','First complete sentence'],enginerows))
    html_parts.append('<p>The recommended companion DFlash drafter was downloaded and tested on the same GPU with rotated serial request order. It was slower for these short German/English responses and is disabled. The comparison retains prompt-cache counts and draft acceptance statistics. Its extra server was stopped after testing. More complex or longer outputs may behave differently; a proposed speedup is not assumed to work without measurement.</p>')
    stage_rows=[]
    if profile.get('rows'):
        total=median(r['performance']['device_seconds']*1000 for r in profile['rows'])
        for key,label in [('semantic_prefill','Semantic backbone: prompt prefill'),('semantic_step','Semantic backbone: autoregressive frame steps'),('acoustic_token_prediction','Local transformer: twelve acoustic tokens per frame'),('waveform_codec','Stateful audio codec: tokens to waveform')]:
            value=median(r['stages_ms'][key] for r in profile['rows']);stage_rows.append([label,ms(value),f'{100*value/total:.1f}%'])
    html_parts.append('<h2>7b. What is slow inside TTS?</h2>'+table(['GPU stage','Median','Share of measured model/codec span'],stage_rows))
    html_parts.append('<p>These isolated CUDA-event measurements use a temporary copy of the same Small RL checkpoint on the other RTX 3090. Three repeated seven-word generations produced approximately three seconds of audio. They separate semantic prefill, frame prediction, the local twelve-code predictor and waveform decoding. Event spans include kernel-launch gaps; instrumentation itself adds small overhead. The stage split is not an exact additive decomposition of the separate HTTP first-audio measurement.</p><p>About 74% of this GPU path is token prediction and about 20% is waveform decoding. The semantic backbone is the largest individual repeated stage. The published RL residual policy runs in FP32 with TF32 disabled; local CUDA graphs and a compiled semantic decoder were already active. The codec is stateful and does not re-decode the entire growing prefix. Live speech-boundary handling also keeps a one-second lookahead, so requesting a 160-ms chunk does not guarantee 160-ms first audio.</p>')
    precision_rows=[]
    for r in precision.get('summary',[]):
        q=r.get('quality_means',{})
        precision_rows.append([r['variant'],r['n'],ms(r['median_device_ms']),ms(r['median_first_audio_device_ms']),
            f"{r.get('median_rtf',0):.3f}",f"{r['mean_wer']:.3f}" if r.get('mean_wer') is not None else 'Not scored',
            f"{r['mean_speaker_cosine']:.3f}" if r.get('mean_speaker_cosine') is not None else 'Not scored',
            f"{q.get('audiobox:CE',0):.2f}",f"{q.get('audiobox:PQ',0):.2f}",f"{q.get('dnsmos:P808_MOS',0):.2f}"])
    html_parts.append('<h2>7c. BF16, TF32 and fewer adapter matrix operations</h2>'+table(['Variant','Clips','GPU generation','First GPU audio','RTF','WER','Speaker cosine','Enjoyment proxy','Production proxy','MOS proxy'],precision_rows))
    html_parts.append('<p>The precision study pairs three short German/English acting prompts with three seeds, nine clips per arm. FP32 strict is the released reference arithmetic. TF32 keeps FP32 tensors but permits reduced-precision Tensor Core multiplication. Mixed BF16 uses BF16 base compute with original FP32 LoRA factors and residual computation; full BF16 rounds those factors too. Merged variants form W + B×A once and remove the two extra adapter matrix multiplications from every affected projection. Merging changes floating-point evaluation order even in FP32 and is therefore evaluated as a distinct inference variant.</p><p>The codec precision is held constant across arms. Compilation and CUDA-graph warm-up are recorded separately rather than hidden in steady-state latency. Reduced precision and merging can change sampled tokens even with the same seed; identical seeds do not imply identical audio. WER, speaker cosine and frozen Humaneness Ears enjoyment/production/MOS proxies are compared together. Those proxies are not human listening scores; this small pilot cannot prove perceptual equivalence.</p>')
    if not precision_rows:html_parts.append('<p class="note warning">Precision ablation is still running; no BF16 quality or speed recommendation is final yet.</p>')
    if precision_rows:
        html_parts.append('<div class="note"><strong>Selected live optimization: merged FP32.</strong> Median GPU generation fell from 973 to 876 ms (10% less time), and first GPU audio from 516 to 467 ms. All nine paired generated code sequences matched the strict reference, giving identical measured ASR/speaker/quality results in this sample. Merged BF16 reached 795 ms and 421 ms, but speaker cosine fell from .434 to .390 and WER rose from .079 to .127. It is available as an explicit experiment, not the recommended default. This pilot is too small to guarantee equivalence for every future prompt.</div>')
    graph_rows=[]
    baseline={(r['case'],r['seed'],r['repeat']):r for r in graphs.get('rows',[]) if r['mode']=='compiled_dynamic'}
    for mode in ('compiled_dynamic','static_graph'):
        a=[r for r in graphs.get('rows',[]) if r['mode']==mode]
        if a:graph_rows.append([mode,len(a),ms(median(r['device_ms'] for r in a)),ms(median(r['first_audio_device_ms'] for r in a)),
            str(sum(r['codes_sha256']==baseline[(r['case'],r['seed'],r['repeat'])]['codes_sha256'] for r in a))+'/'+str(len(a))])
    html_parts.append('<h2>7d. Additional semantic CUDA graph experiment</h2>'+table(['Semantic path, both merged FP32','Generations','GPU generation','First GPU audio','Codes matching baseline'],graph_rows))
    html_parts.append('<p>The experimental path keeps a fixed-address, bounded KV cache for the released custom Moss Qwen decoder, captures single-frame continuation, and replays it. The prefill uses the released forward; positions respect each CFG branch’s padding. Eighteen generations per arm alternate prompts and seeds across repeated requests, checking cache resets. The median improvement was below 1% and first audio was essentially unchanged, so the existing compiled dynamic path remains enabled. This is an engineering result, not evidence that CUDA graphs can never help. Local twelve-code CUDA graphs were already active in both arms. Initial cache API mismatch was corrected before this completed comparison.</p><p>CUDA replay removes Python/kernel-launch overhead but does not remove sequential frame dependencies or weight-memory traffic. Replaying separate semantic and local graphs still requires a host stop check each frame. A combined graph, fused sampling, or device-side bounded frame groups could help more, but must preserve stopping, CFG, sampling and streaming behavior. These are proposals, not measured speedups. Implementation constraints follow <a href="https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs">PyTorch’s CUDA graph documentation</a>.</p>')
    local_rows=[]
    local_base={(r['case'],r['seed']):r for r in local_compile.get('rows',[]) if r['mode']=='local_eager_graph'}
    for mode in ('local_eager_graph','local_compiled_graph'):
        a=[r for r in local_compile.get('rows',[]) if r['mode']==mode]
        if a:local_rows.append([mode,len(a),ms(median(r['device_ms'] for r in a)),ms(median(r['first_audio_device_ms'] for r in a)),
            ms(local_compile.get('warmup_ms',{}).get(mode)),str(sum(r['codes_sha256']==local_base[(r['case'],r['seed'])]['codes_sha256'] for r in a))+'/'+str(len(a))])
    html_parts.append('<h2>7e. Compile and fuse the local twelve-token loop</h2>'+table(['Local path, both merged FP32','Generations','GPU generation','First GPU audio','Separate warm-up','Codes matching baseline'],local_rows))
    html_parts.append('<p>A separate experiment compiles the unrolled local prediction loop before CUDA capture. Nine paired clips had matching generated codes. Median GPU generation changed from 870 to 854 ms (about 2%), and first GPU audio from 459 to 455 ms. The compiled arm needed another 18 seconds of warm-up. The short comparison does not establish a robust gain across workloads or candidate batch sizes; this remains a reproducible benchmark option rather than an automatic production setting.</p>')
    clips=[r for r in precision.get('rows',[]) if r.get('listen_mp3_b64')]
    if clips:
        html_parts.append('<details><summary>Listen to paired precision samples (authored synthetic speech)</summary>')
        for r in clips:
            html_parts.append('<p><strong>'+esc(r['variant']+' · '+r['case']+' · seed '+str(r['seed']))+'</strong><br>'+esc(r['text'])+'</p><audio controls preload="none" src="data:audio/mpeg;base64,'+r['listen_mp3_b64']+'"></audio>')
        html_parts.append('</details>')
    html_parts.append('''<h2>8. Further speed improvements, ordered by likely value</h2><ol>
<li><strong>Fuse work across token-prediction stages.</strong> The completed profile identifies semantic and local token generation as the largest GPU cost. Adapter merging is deployed. Next test fused Q/K/V and gate/up projections, combined semantic/local graph replay, fused normalization/sampling and device-side stop handling. Reference caching and stateful codec decoding already exist. The extra graph and local-compile experiments above gave little benefit; no unmeasured CUDA speedup is promised.</li>
<li><strong>Keep speaker preparation speculative and bounded.</strong> Reuse recent same-sentence anchors; stop preparing after silence and invalidate on speaker/sentence transitions. Longer clean context is more useful here than reducing an embedding call by ten milliseconds. For larger populations, vectorize the cosine matrix before adding an indexing service.</li>
<li><strong>Tune browser buffering on real client networks.</strong> The adaptive reserve is implemented, but packet jitter, CPU load and Bluetooth devices differ. Use underrun counts and measured playback timestamps to tune its floor. Do not turn all buffering off based on localhost results.</li>
<li><strong>Make the ASR probe incremental.</strong> Native decoder partials are reused, but bounded full-window probes still fill gaps during agent speech. Reusing native decoder state and word timings could reduce redundant GPU encoder work and improve sentence boundaries. Do not sacrifice final words for a smaller reported endpoint delay.</li>
<li><strong>Remove the internal MP3 round trip if profiling justifies it.</strong> A private PCM route between the existing TTS process and Robby would avoid MP3 file fetch/decode. Keep compressed downloads for ordinary browser playback. Its benefit is currently unmeasured and probably smaller than TTS model work.</li>
<li><strong>Preserve short prompts and cached prefixes.</strong> Keep dynamic speaker/event data after the stable policy. Avoid sending long acting prose or unused tool schemas on every ordinary request. Tool-assisted news/weather answers intentionally need more work than a simple conversational reply.</li>
<li><strong>Guard answer correctness independently of latency.</strong> The retained short-answer engine comparison includes incorrect German arithmetic from this small model. The seven-word target is also approximate. Add a deterministic calculator for arithmetic and evaluate a multilingual quality set before treating low latency as evidence of good answers. DFlash changes speed, not the correctness guarantee.</li></ol>''')
    html_parts.append('''<h2>9. Reproduce and use the API</h2><p>Use the existing Python 3.12 environment and CUDA llama.cpp binary. The main GGUF revision is pinned to <code>1db54d5b12533ff1bcaa8e42cf4c667988a1f1be</code>. The measured binary reports commit <code>5d9e5ac</code>. The GGUF file is 6,180,446,208 bytes despite “4GB” in the model name; the runtime GPU allocation is a separate quantity. The optional DFlash file is 736,812,096 bytes. No source voice dataset was downloaded for these tests.</p>
<pre>python launch_bolt_stack.py --llama-server /path/to/llama-server \
  --models /path/to/bolt-models --download --with-robo

# The existing Humaneness Small TTS server must run separately on port 8792.
# When reusing an already loaded local LLM, add --reuse-llm.
# UI:  http://127.0.0.1:8798/desk-robot/
# API: http://127.0.0.1:8798/docs

curl -N http://127.0.0.1:8798/api/bolt/chat \
  -H 'Content-Type: application/json' \
  -d '{"text":"Was können wir heute gemeinsam machen?","chunk_frames":20}'</pre>
<p><code>POST /api/bolt/llm</code> streams cue-labelled sentences only. <code>POST /api/bolt/tts</code> synthesizes supplied text with a short cue. <code>POST /api/bolt/chat</code> streams local LLM sentences and TTS. <code>POST /api/bolt/asr</code> accepts at most 15 seconds of finite mono float32 little-endian 16-kHz PCM. <code>POST /api/bolt/audio-chat</code> combines complete input audio with streamed reply generation; live browser microphone capture uses <code>/ws</code>. SSE audio events contain base64 PCM16 little-endian, mono, 24 kHz. There is one completion event after all output sentences.</p>
<pre>python run_unit_checks.py
python benchmark_bolt.py --runs 3
python benchmark_bolt_engines.py  # requires the optional second comparison server
python benchmark_bolt_speakers.py
python bolt_socket_check.py
uv run --no-project --with playwright --python /path/to/python browser_bolt_check.py
python render_bolt_report.py</pre>
<p>The engine comparison requires two temporary servers and intentionally is not part of ordinary startup. Browser fixtures are authored synthetic clips from the existing local fixture cache; the report contains no private microphone recordings or saved user voiceprints. All real-time audio tests must run serially because the shared ASR demo has one conversation slot.</p>''')
    clean_precision={**precision,'rows':[{k:v for k,v in r.items() if k!='listen_mp3_b64'} for r in precision.get('rows',[])]}
    all_data={'generation':speed,'before_precision_generation':before,'speaker_context':speakers,'engine_comparison':engines,'tts_gpu_profile':profile,'precision':clean_precision,'semantic_graph':graphs,'local_compile':local_compile,'real_time_socket':socket,'browser':browser}
    html_parts.append('<h2>10. Retained data and limitations</h2><p>The following records make the scope and exact word counts inspectable. Initial engineering runs exposed clipped final words, a premature two-word greeting, and insufficient fake-device setup lead-in; the implementation and test harness were corrected. Passing synthetic tests do not prove robustness for every accent, noisy room, overlap, unfamiliar voice, webcam/device combination or long session. Simultaneous mixed voices can still yield an uncertain speaker label; intelligible speech is admitted rather than silently waiting for enrollment.</p><details><summary>All measurements and verification records</summary><pre>'+esc(json.dumps(all_data,indent=2,ensure_ascii=False))+'</pre></details>')
    html_parts.append('<p class="small">Generated '+esc(time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()))+'. Source: the local Robby integration and authored synthetic test fixtures. <a href="https://github.com/LAION-AI/Humaneness-Voice">Source repository</a>.</p></main></body></html>')
    result=''.join(html_parts)
    target=ROOT.parent/'robby-local-speed-report.html';target.write_text(result)
    (ROOT/'static/robby-local-speed-report.html').write_text(result)
    print(json.dumps({'file':str(target),'bytes':len(result.encode()),'browser_passed':browser.get('passed'),'socket_passed':socket.get('passed')}))

if __name__=='__main__':main()
