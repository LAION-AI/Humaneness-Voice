"""English engineering receipt; synthetic short speech is not an ASR benchmark."""
import html,json,statistics,time
from pathlib import Path
from asr_catalog import MODELS
ROOT=Path(__file__).resolve().parent
def esc(x):return html.escape(str(x))
def read(p):return json.loads(p.read_text()) if p.exists() else {}

def main():
    checked=read(ROOT/'static/asr-model-check.json');browser=read(ROOT/'static/asr-model-browser.json')
    controls=read(ROOT/'static/asr-controls-browser.json')
    p=ROOT/'logs/asr-options-unit.log';unit=json.loads(p.read_text().splitlines()[-1]) if p.exists() else {}
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Robby — selectable German ASR models</title>
<style>body{font:17px/1.6 system-ui;max-width:1120px;margin:30px auto;padding:20px;background:#f6f8fa;color:#20333a}h1,h2{line-height:1.25}table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #bfcbd0;text-align:left;padding:10px}.scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eaf0f4;padding:18px;font-size:14px}a{color:#166a78}.note{padding:18px;background:#e3f0ed}</style>
<p><a href="https://arrive-locations-parade-research.trycloudflare.com/desk-robot/">Open Robby</a> · <a href="https://github.com/LAION-AI/Humaneness-Voice/tree/main/robby">Source and setup</a></p>
<h1>Five recognizers, German replies, shared turn-taking</h1>
<p class="note">The Speech recognition menu above the robot selects the final recognizer. German is the default reply language; an explicit request for English overrides it. The original ASR remains the default recognizer so users can make their own comparisons. Selection is remembered in this browser.</p>
<h2>Measured short-clip checks</h2><p>One RTX 3090, three authored synthetic German clips (3.39–4.51 s), twice each. These figures are engineering smoke tests, not a representative German ASR evaluation or a recommendation for every microphone. WER compares intended synthesis text with ASR output and can include TTS pronunciation errors. Timing excludes speech collection, the configured pause, LLM, TTS and browser playback.</p>
<div class="scroll"><table><tr><th>Recognizer</th><th>Checks</th><th>Median ASR HTTP time</th><th>Mean toy WER</th><th>One measured model switch</th><th>HTTPS microphone → spoken reply</th></tr>'''
    for key,spec in MODELS.items():
        rows=[r for r in checked.get('rows',[]) if r['model']==key]
        switch=next((r for r in checked.get('switches',[]) if r['model']==key),{})
        b=next((r for r in browser.get('rows',[]) if r['model']==key),{})
        values=[spec['label'],str(len(rows))+' passed' if rows else 'Pending',
            f"{statistics.median(r['asr_ms'] for r in rows):.1f} ms" if rows else 'Pending',
            f"{statistics.mean(r['wer'] for r in rows)*100:.1f}%" if rows else 'Pending',
            str(switch.get('switch_ms','Pending'))+' ms',str(b.get('passed','Pending'))]
        page+='<tr>'+''.join('<td>'+esc(v)+'</td>' for v in values)+'</tr>'
    page+='''</table></div><p>Model switching includes eviction of the previous extra recognizer and host-to-device loading. These are single switch observations; CUDA/allocator warm-up, weight size and the previously selected model affect them. Resident transcription calls do not reload the weights each time. Original Nemotron uses the existing optimized CUDA-graph decoder; the German telephony model is a separately loaded FP32 recognizer.</p>
<h2>Pipeline and buffering</h2><pre>microphone ─┬─ original Nemotron → live preview / TYGER / speaker features / Stop
            └─ small CPU Silero VAD → speech pause → selected final recognizer
                                                   ↓
                                       corrected final text + voice label
                                                   ↓
                                       local LLM → sentence-streamed TTS</pre>
<p>The previous Nemotron encoder and TYGER GRU stay resident and unchanged. Their features also power existing speaker matching. The final recognizer is interchangeable: the selected model's transcript, not the preview, is submitted to the LLM. Pauses still use the existing neural CPU VAD slider (400 ms default, 200–2,000 ms). Energy-only background noise does not reset this clock; actual background speech can.</p>
<p>Whisper is used as a buffered recognizer after speech pauses. Nemotron continues live recognition and interruptions alongside it. The same pause-final adapter is used for the other alternatives; this integration does not claim native incremental Moonshine/Parakeet decoding. A final alternative pass avoids an unnecessary second native-ASR pass. If speech resumes or the model changes during finalization, stale results are discarded and retained input is checked again.</p>
<p>Continuous input is held only in bounded session RAM (up to 1,000 existing 320-ms records, about 20 MB of float32 audio). Alternative-model calls use at most 28 seconds, with a short overlap and conservative duplicate-word removal for unusually long uninterrupted input. Typical sentence-sized input is one call. Audio is never saved by the live adapter. Long continuous monologues beyond that RAM window retain the earlier implementation's bounded-input limitation.</p>
<h2>CPU RAM and GPU offloading</h2><p>The optional loopback worker on port 8801 preloads all four additional models into CPU RAM, roughly 6.6 GiB of weight payload plus runtime overhead. Exactly one extra recognizer is active on GPU; switching evicts it to CPU before moving the new model to GPU. CTranslate2 uses unload_model(to_cpu=True) and load_model(keep_cache=True), retaining its CPU cache. PyTorch models use CPU/GPU transfers. The original Nemotron/TYGER, local LLM and shared TTS remain available.</p>
<p>Cold startup downloads only pinned inference files (about 6.7 GiB total); subsequent swaps use in-memory models. Progress/readiness is visible in the menu's details panel. Loading runs on worker threads in a separate process. Failure keeps the previous selection, and original Nemotron remains a recovery option even when the optional worker is offline. A selection change invalidates an in-flight stale result without deleting the pending utterance.</p>
<h2>Models and exact source revisions</h2><table><tr><th>Model</th><th>Revision / source</th></tr>'''
    for key,spec in MODELS.items():
        page+='<tr><td><a href="https://huggingface.co/'+esc(spec['repo'])+'">'+esc(spec['label'])+'</a></td><td>'+esc(spec['revision'])+'</td></tr>'
    page+='''</table><p>Original Nemotron is the frozen copy in the existing pinned TYGER bundle, not a fresh download of an evolving main branch. The telephony artifact is converted strictly in memory using the released NeMo-to-Transformers key layout; all weights, every tokenizer ID and the German language prompt are checked. Nonpersistent positional buffers are initialized on CPU. Moonshine retains the required attention mask and a duration-based output cap. Whisper uses German recognition with no custom name/keyword prompt.</p><p>An initial decoder smoke test caught three integration errors before release: a base Moonshine class without a generation head, uninitialized nonpersistent meta buffers in the converted model, and a too-small RNN-T output budget that cut final words. The worker now uses the conditional Moonshine class, CPU-initialized buffers, the pinned generation configuration, and bounded budgets that include acoustic blank/time-advance steps. Measurements below use the corrected budget. The telephony model's published improvements concern telephone-style audio; three clean synthetic clips do not determine its usefulness in that domain.</p>
<h2>Actual browser and language checks</h2>'''
    page+='<p>Unit checks: '+esc(unit.get('tests','Pending'))+'; passed='+esc(unit.get('passed'))+'. HTTPS CPU Chrome test: passed='+esc(browser.get('passed'))+'. Each model is selected through the actual visible dropdown; a synthetic microphone utterance enters the real ASR/identity/LLM/TTS pipeline. The receipt verifies the final recognizer ID, one submitted turn, actual PCM, saved selection and no browser/service errors.</p>'
    page+='<p>Additional real mobile-browser controls: passed='+esc(controls.get('passed'))+'. The selector fits a 390-pixel viewport, the choice survives an actual reload, and repeated Whisper → Parakeet → original switches complete.</p>'
    for row in checked.get('languages',[]):page+='<details><summary>'+esc(row['input'])+' → '+esc(row['expected'])+'</summary><pre>'+esc(row['reply'])+'</pre></details>'
    page+='''<h2>Reproduction</h2><pre># Existing Robby Python 3.12 environment and pinned TYGER assets are required.
python prepare_asr_models.py
pip install -r requirements-asr.txt
python asr_model_worker.py  # loopback 8801; CPU preload runs in the background
# Existing server.py + bolt_api.py + LLM + TTS must also be running.
python run_unit_checks.py
python validate_asr_models.py
uv run --no-project --with playwright --python /path/to/venv/bin/python python browser_asr_models_check.py</pre>
<p>The live installation keeps the four additional dependencies in an isolated ROBO_ASR_DEPS directory; the shared TTS environment was not upgraded. ROBO_ASR_CACHE selects the model cache drive and ROBO_ASR_DEVICE selects the optional worker GPU. The launcher also accepts --with-asr-models. API clients can POST mono 16-kHz float32 PCM to /api/bolt/asr?model=whisper_turbo (or another catalogue ID); /api/asr/models reports cache readiness. Private development ASR endpoints remain blocked by the public facade.</p>
<h2>Limits and provenance</h2><p>Physical-room microphone quality, dialects, overlap and natural conversations still need user comparison. This check uses synthetic speech and a CPU Chromium fake microphone, not private recordings. It verifies integration and timing, not human ASR accuracy. Native preview quality and rapid Stop recognition still depend on the original Nemotron/TYGER path. ASR/voice labels are approximate, and the new recognizers do not authenticate a person.</p>
<p>Primary implementation references: <a href="https://huggingface.co/moonshine-ai/moonshine-streaming-small-de">Moonshine German usage and mask requirements</a>; <a href="https://huggingface.co/BotgenossenGmbH/nemotron-3.5-asr-de-telephony">German telephony adaptation</a>; <a href="https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2">Whisper CT2 weights</a>; <a href="https://opennmt.net/CTranslate2/python/ctranslate2.models.Whisper.html">CTranslate2 CPU cache/offload API</a>; <a href="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3">Parakeet v3</a>. Their source licenses remain separate; no model weights or private biometric data are bundled in the source release.</p>
<details><summary>Machine-readable test receipts</summary><pre>'''
    page+=esc(json.dumps({'recognition':checked,'https_browser':browser,'controls_browser':controls,'unit':unit},ensure_ascii=False,indent=2))+'</pre></details>'
    page+='<p>Generated '+esc(time.strftime('%Y-%m-%d %H:%M UTC',time.gmtime()))+'.</p></html>'
    (ROOT/'static/asr-model-update.html').write_text(page);(ROOT.parent/'robby-german-asr-report.html').write_text(page)
    print(json.dumps({'asr_passed':checked.get('passed'),'browser_passed':browser.get('passed'),'unit_passed':unit.get('passed')}))

if __name__=='__main__':main()
