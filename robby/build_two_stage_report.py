"""Build the public English report from authored, retained experiment receipts."""
import html,json,re,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent;STATIC=ROOT/'static'
def read(name):return json.loads((STATIC/name).read_text())
def esc(value):return html.escape(str(value))
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def main():
    previous=STATIC/'two-stage-report.html'
    # Exported JSON omits duplicate MP3 data. Preserve its already reviewed
    # embedded clips when rebuilding the public report from compact receipts.
    previous_audio=dict(re.findall(r'<article data-case="([^"]+)">.*?src="data:audio/mpeg;base64,([A-Za-z0-9+/=]+)"',previous.read_text() if previous.exists() else '',re.S))
    browser=read('two-stage-browser.json');memory=read('two-stage-memory-reactions-final.json');mem_browser=read('two-stage-memory-browser.json');cancel=read('two-stage-cancel.json')
    warm=browser['turns'][:2];onsets=[r['scheduled_playback_ms'] for r in warm]
    overview=[
        ('First reaction uses memory','20 authored local probes, plus actual HTTPS recall with zero history budget','Known Bob/Jazz/Alice facts are used; unidentified visitors remain uncertain.'),
        ('Warm two-stage browser onset',f'{min(onsets):.0f}–{max(onsets):.0f} ms in {len(onsets)} typed turns','Scheduled playback, not a microphone-to-ear measurement.'),
        ('Browser underruns',str(sum(r['new_underflows'] for r in warm))+' in the two warm staged turns','The first post-restart browser trial took 2.47 s including service admission; no universal 900-ms guarantee.'),
        ('TTS default','0.8 temperature; CFG 2.0; two active streams; one weight instance','Three streams hurt foreground continuity; two were selected.'),
        ('TTS batching pilot','2.46 s interleaved versus 2.04 s batched warm (~17% less wall time)','Truncated deterministic 18-frame control, not a quality or scale benchmark.'),
        ('Stop during new stages',f'{len(cancel["trials"])} browser control-path checks passed','Worker idle after '+', '.join(str(t['worker_idle_ms'])+' ms' for t in cancel['trials'])+'; no stale playback.'),
        ('Regression checks','501 Robby unit checks + 3 CPU KV/admission checks passed','Browser: staged speech, poem, disabled option, persistence, mobile controls, memory reconnect/reset.'),
    ]
    parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Robby: memory-informed reactions and continuous speech batching</title>
<style>body{font:17px/1.55 system-ui,sans-serif;background:#f5f7fb;color:#172437;margin:0}main{max-width:1120px;margin:auto;padding:36px 22px}h1{line-height:1.15}h2{margin-top:2em}table{border-collapse:collapse;background:white;width:100%;font-size:15px}td,th{padding:12px;border-bottom:1px solid #dce3ed;text-align:left;vertical-align:top}th{background:#dce8fa}.scroll{overflow:auto}article,.flow{background:white;border-radius:16px;padding:22px;margin:20px 0}audio{width:100%}code{background:#e6ecf5;padding:2px 5px}a{color:#135ac1}.muted{color:#53657a}.flow{font-size:21px;font-weight:650}.note{border-left:5px solid #e2ad4a;padding-left:18px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><main>
<h1>Memory-informed first words, then an informed transition</h1><p>Robby engineering study · 27 October 2026 · two shared RTX 3090 GPUs · authored synthetic dialogue only.</p>
<p><a href="/desk-robot/">Live demo (on the demo host)</a> · <a href="https://github.com/LAION-AI/Humaneness-Voice/blob/main/robby/TWO_STAGE_SPEECH.md">Architecture, source and reproduction</a></p>''',table(['Finding','Measured result / decision','Interpretation'],overview),
    '''<h2>What the user hears</h2><div class="flow">Memory-aware local reaction → Qwen-informed transition → full Qwen answer</div>
<p>The first local call sees the current request, recent conversation, approximate speaker identity and compact excerpts of the already retrieved wiki entries. It does not see Qwen's answer yet. It can say a tiny grounded answer such as “Deine Freundin heißt Alice” instead of a thinking/waiting announcement. With no matching evidence it must not invent recognition. Current-session self-introductions work even with persistent memory off.</p>
<p>The second local call sees the actual first one or two Qwen sentences, plus the reaction already chosen. It writes a short contextual transition. For jokes only the setup is supplied, so the transition does not receive the punchline. Exact-output tasks and poems bypass both introductions. Invalid, repeated or inappropriate prefaces are skipped rather than replaced by canned filler.</p>
<h2>Why parallel text generation alone did not remove the audio gap</h2>
<p>Previously the single shared TTS server finished the first utterance before admitting another. Having Qwen's text ready did not make its audio ready. The new internal WebSocket admits independent utterances to one CUDA owner and one set of weights. Global semantic and local acoustic-token decoding share real tensor batches; each utterance retains separate KV and codec state. Codec decoding itself restores one stream's state at a time. Main-answer audio is prefetched immediately, while the second local model call runs.</p>
<p>Two active streams are selected. Three can improve aggregate work while slowing the stream the listener currently needs. That produced early underruns in the retained three-stream runs. The foreground emits a 160-ms first audio block, then 400-ms blocks; the browser starts with a 220-ms reserve. Main utterances retain final forced alignment and fades. Short prefaces use 10-ms fades without alignment holdback. All conversational trials below use TTS temperature 0.8; the isolated deterministic cache test used zero.</p>
<h2>Actual HTTPS browser timing</h2><p>Times start from typed submission. Playback is the browser's scheduled AudioContext start, not a measured physical sound event. ASR is not part of this test.</p>''']
    rows=[]
    for t in browser['turns']:
        rows.append((t['input'],t['enabled'],f"{t['first_pcm_ms']:.0f} ms",f"{t['scheduled_playback_ms']:.0f} ms",t['new_underflows'],f"{t['max_gap_ms']:.1f} ms"))
    parts.append(table(['Request','Prefaces enabled','First received PCM','Scheduled playback','Underruns','Largest scheduling gap'],rows))
    parts.append('<p class="note">Startup is different: the retained first post-restart trial began scheduled playback after 2,473 ms. Its TTS first-PCM service time was about 189 ms; admission/readiness waiting accounted for most of the extra onset. The two subsequent warm staged trials above do not establish a tail-latency guarantee. Network jitter, concurrent users, model switches and compilation can add latency.</p>')
    parts.append('<h2>Server experiments and retained variants</h2>')
    rows=[]
    for name,label in [('two-stage-audio-v1.json','Initial 3-stream run'),('two-stage-audio-warm.json','Warm 3-stream'),('two-stage-audio-pair.json','2-stream selection'),('two-stage-audio-final.json','2-stream + revised link')]:
        for r in read(name)['rows']:
            d=next((e['value'] for e in r['events'] if e['event']=='done'),{})
            rows.append((label,r['case'],r['first_pcm_ms'],d.get('answer_audio_ready_ms'),d.get('answer_first_audio_ms'),len(r['virtual_gaps']),round(max([g['gap_ms'] for g in r['virtual_gaps']] or [0]),1)))
    parts.append(table(['Variant','Case','First PCM ms','Main PCM ready ms','Main PCM delivered ms','Virtual gaps','Largest gap ms'],rows))
    parts.append('<p>These earlier server runs used a virtual 140-ms playback reserve; the selected browser uses 220 ms. They are stochastic, not equal-text/equal-length paired quality trials. The main PCM-ready and delivered columns distinguish useful prefetch from the intentionally later playback order. Earlier three-stream runs were retained even when a later case could not start because the shared service was occupied.</p>')
    parts.append('<h2>Memory grounding: all final local outputs</h2><p>Two calls per authored case. These examples use fictional test facts, never the visitor’s real wiki. Earlier failed variants are linked below: they were too cautious about known speakers, mishandled English or missed session-only evidence.</p>')
    parts.append(table(['Case','Trial','Local first reaction','Generation ms'],[(r['case'],r['repeat']+1,r['result'].get('text') or '[skipped]',r['result']['ms']) for r in memory['rows']]))
    reaction=mem_browser['turns'][-1]['reactions'][0]
    parts.append('<article><h3>Actual memory browser check</h3><p>Saved authored Bob/Alice facts, reconnected with a <strong>zero recent-history budget</strong>, then asked: '+esc(mem_browser['turns'][-1]['user'])+'</p><p>First spoken reaction: <strong>'+esc(reaction['sentence_text'])+'</strong>. It received '+str(reaction['timing']['quick_reply_memory_entries'])+' retrieved entries. Opt-in, disable and reset passed; the isolated test workspace was reset afterward.</p></article>')
    parts.append('<h2>Listening examples</h2><p>Mono 96-kbps MP3 embedded below. Audio is concatenated in spoken order: these files demonstrate wording and voice, not network pauses. Receipts retain the chunk timing for those pauses. Short-preface prosody may still differ from the following sentence.</p>')
    for r in read('two-stage-audio-final.json')['rows']:
        clip=r.get('listen_mp3_b64') or previous_audio.get(r['case'])
        parts.append('<article data-case="'+esc(r['case'])+'"><h3>'+esc(r['input'])+'</h3>')
        parts.append('<audio controls preload="none" src="data:audio/mpeg;base64,'+clip+'"></audio>' if clip else '<p>Rerun benchmark_two_stage.py to generate this listening clip.</p>')
        for e in r['events']:
            if e['event']=='sentence':parts.append('<p><strong>'+esc(e['value']['sentence_role'])+':</strong> '+esc(e['value']['sentence_text'])+'</p>')
        if r['case']=='tired':parts.append('<p class="note">Retained failure of substantive answer quality: Qwen announced ideas but did not actually provide them in this trial. Batching does not solve main-model completeness or factual reliability.</p>')
        parts.append('</article>')
    parts.append('''<h2>Reproduction, evidence and limits</h2><p>Run the scripts from the published <code>robby/</code> folder using its pinned stack. The internal TTS transport is loopback-only, shares the existing engine admission lock and does not replace the normal Playground API. No private recordings, credentials or real user memories are published.</p>
<pre>python run_unit_checks.py
python benchmark_live_batch.py
python benchmark_two_stage.py --output two-stage-audio-new.json
python benchmark_reaction_memory.py --output two-stage-memory-new.json
python browser_two_stage_check.py --url https://YOUR-DEMO
python browser_two_stage_memory.py --url https://YOUR-DEMO
python browser_two_stage_cancel.py  # edit URL for another deployment</pre>
<p class="note">One repeated memory browser run failed before the memory assertion because the existing exact-word TTS path rejected a SCRIPT/transcript mismatch. Its original model wording was not captured, so the cause is not established. The harness now retains all events and surfaces memory errors promptly. A fresh run passed; this does not prove that formatting failure is fixed.</p><p>First reaction text remains generated at temperature 0.35; informed-link text at 0.25. TTS is 0.8. One local model is reused, not a second local LLM download. The truncated temperature-zero batching pilot had sample-identical PCM across its tested schedules; that is not a proof of stochastic or perceptual equivalence at 0.8. Some links remain generic or grammatically thin; no human MOS/listening panel was run. Speaker matches and retrieved facts remain fallible. These typed/browser tests do not newly certify acoustic Stop, noisy-room ASR, speaker recognition or universal sub-900-ms latency.</p><ul>''')
    for p in sorted(STATIC.glob('two-stage-*.json')):parts.append('<li><a href="'+p.name+'">'+p.name+'</a></li>')
    parts.append('</ul><p><a href="https://github.com/LAION-AI/Humaneness-Voice/blob/main/robby/TWO_STAGE_SPEECH.md">Full implementation notes and privacy boundaries</a></p></main></html>')
    (STATIC/'two-stage-report.html').write_text('\n'.join(parts))
    print('Wrote',STATIC/'two-stage-report.html')
if __name__=='__main__':main()
