"""Publish only explicitly enumerated synthetic receipts, never private sessions."""
import datetime,html,json,statistics
from pathlib import Path
from final_verify import runtime_signature

ROOT=Path(__file__).resolve().parent
FILES=['director-speed-api','director-speed-mercury-tuned','director-speed-gemma4-e4b',
 'director-speed-gemma4-12b','director-speed-gemini','director-speed-integrated','director-speed-routed',
 'audio-decoder-speed','pipeline-speed-before','pipeline-speed-after','pipeline-speed-routed',
 'speed-baseline-prompt','knowledge-director-live-before','knowledge-director-live',
 'open-speakers-browser-before-greeting','open-speakers-browser','knowledge-browser',
 'speaker-variant-live-before-new-guest-fix','speaker-variant-live-before-claim-prefix-fix','speaker-variant-live','unit-checks',
 'browser-lifecycle-checks','reliability-ui-checks']

def read(name):
    p=ROOT/'logs'/(name+'.json');return json.loads(p.read_text()) if p.exists() else {}
def e(value):return html.escape(str(value))
def q(values,p):
    v=sorted(values);pos=(len(v)-1)*p;i=int(pos)
    return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(pos-i)
def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+e(v)+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+e(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'

def main():
    folder=ROOT/'static/speed-study-data';folder.mkdir(exist_ok=True)
    for name in FILES:
        p=ROOT/'logs'/(name+'.json')
        if p.exists():(folder/p.name).write_bytes(p.read_bytes())
    out='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Robby: speed, public tools and session speakers</title>
    <style>body{font:16px/1.55 system-ui;max-width:1100px;margin:30px auto;padding:0 20px;color:#20313a;background:#f8faf9}h1,h2{line-height:1.2}table{border-collapse:collapse;background:white;width:100%}th,td{padding:10px;border:1px solid #dce3e4;text-align:left}th{background:#e6efeb}.scroll{overflow:auto}a{color:#155d88}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{background:white;padding:12px;margin:10px 0}.note{padding:14px;background:#fff3cf;border-radius:10px}code{overflow-wrap:anywhere}</style>
    <p><a href="/desk-robot/">← Open Robby</a> · <a href="/static/worklog.html">Task log</a> · <a href="/downloads/robby-source.zip">Source code</a></p><h1>Robby: speed, public tools and flexible speaker memory</h1>'''
    out+='<p>Built '+e(datetime.datetime.now(datetime.timezone.utc).isoformat())+'. Public synthetic tests; no private microphone recordings.</p>'
    out+='<h2>Director comparison: speed is not enough</h2>'
    rows=[];failures=[]
    configs=[('director-speed-api','original live request'),('director-speed-mercury-tuned','instant + strict schema'),
        ('director-speed-gemma4-e4b','local QAT Q4'),('director-speed-gemma4-12b','local QAT Q4'),('director-speed-gemini','original live request')]
    for filename,label in configs:
        data=read(filename)
        for model in data.get('models',[]):
            attempts=[r for r in data['rows'] if r['model']==model];times=[r['total_ms']/1000 for r in attempts]
            rows.append([model,label,len(times),sum(r['passed'] for r in attempts),f'{statistics.median(times):.3f}',f'{q(times,.95):.3f}'])
            failures.extend((model,r) for r in attempts if not r['passed'])
    out+=table(['Model','Configuration','Attempts','All checks pass','Median seconds','p95 seconds'],rows)
    out+='<p class="note"><strong>Decision: keep Luna as the production default.</strong> DeepSeek passed the same checks but was slower here. Mercury, Gemini and the tested local Gemmas were often faster, but failed important instruction/tool checks. A claim that an interruption rule was installed is not equivalent to executing its real tool. These results describe our gateway, prompt and parser—not a claim that a vendor model is fundamentally incapable of tool use.</p>'
    out+='<p>Fourteen explicit tasks cover German/English, exact words, arithmetic, two-sentence stories, speaker labels, context, honest tool limitations and installing/cancelling real interruption rules. Three repetitions per API model and Gemma 12B; two for E4B. Every failure counts, including invalid JSON. Luna/Mercury/DeepSeek were serially interleaved with rotated order. Gemini, local models and the Mercury adaptation were later sessions, not simultaneous matched-load experiments. Timing includes HTTP, full director output, validated tool round-trip and bounded existing provider retries. It is not token throughput, TTFT, ASR latency or time to audible speech. This small explicit-check set is not a broad human quality evaluation.</p>'
    out+='<p>API models used the already configured Hyprlab gateway; direct native vendor endpoints were not tested. Mercury additionally used its documented instant effort, a strict three-field JSON schema and a 2,048-token output budget. Baseline used the existing none effort/900-token JSON-object request. No real session rules were changed by the synthetic model tests.</p>'
    out+='<h2>Local Gemma feasibility</h2><p>Gemma <strong>4</strong> E4B and 12B instruction models, official QAT Q4_0 GGUFs, text only. Both loaded and generated on this RTX 3090 server, using GPU 0 only, one at a time. Runtime: llama.cpp 5d9e5ac; 8,192 context, one slot, 99 GPU layers requested, 8 threads, batch/ubatch 256, flash attention on, reasoning off. No multimodal projector. These shared-machine tests do not establish performance under simultaneous heavy ASR load. Temporary LLM servers are stopped; neither replaces the shared TTS model.</p>'
    out+='<p><a href="https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/tree/4b4a2c1d584be7264f87aac328a1bc739ce81b6c">E4B revision 4b4a2c1…</a>, file gemma-4-E4B_q4_0-it.gguf (5,154,941,280 bytes). <a href="https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf/tree/29d097773436b69ff9feafd636ab4cf873786537">12B revision 29d0977…</a>, file gemma-4-12b-it-qat-q4_0.gguf (6,975,879,296 bytes). Weights remain outside the source export; no whole-repository download.</p>'
    out+='<h2>Pipeline optimizations and measured limits</h2>'
    audio=read('audio-decoder-speed')
    out+=table(['Chunk seconds','Old subprocess median ms','New worker median ms','Maximum PCM difference','Padding samples removed'],[
        [r['duration_s'],round(statistics.median(r['ffmpeg_ms']),3),round(statistics.median(r['soundfile_worker_ms']),3),f"{r['max_pcm_error']:.2g}",r['trailing_padding_removed_samples']] for r in audio.get('rows',[])])
    out+='<p>Thirty repetitions per duration. Existing SoundFile/libsndfile decodes mono 48-kHz transport off the event loop. Other rates/layouts or unsupported files retain ffmpeg fallback; cancellation is tested. Audio agrees with seekable ffmpeg within 5e-7. The old pipe decoder retained extra encoder padding. This optimization changes neither model weights nor TTS sampling. HTTP connections now survive 120-second conversational pauses. Irrelevant tool schemas are omitted while current/recent lookup context keeps follow-up tools available.</p>'
    timings=[]
    for name,label in [('pipeline-speed-before','Original'),('pipeline-speed-after','Decoder + full tool schemas'),('pipeline-speed-routed','Decoder + routed tool schemas')]:
        d=read(name);rs=[r for r in d.get('rows',[]) if r.get('chunks') and r.get('reply_at')]
        if rs:timings.append([label,len(rs),round(statistics.median(r['timing']['director_ms'] for r in rs),1),round(statistics.median(r['chunks'][0]['at']-r['reply_at'] for r in rs),1),round(statistics.median(r['chunks'][0]['at']-r['start'] for r in rs),1),sum(c['gap_ms']>20 for r in rs for c in r['chunks'])])
    out+=table(['HTTPS typed turn','N','Director median ms','Reply text → first PCM ms','Submit → first PCM ms','Scheduling gaps >20 ms'],timings)
    out+='<p class="note">The decoder win does not prove a faster whole conversation: API latency varies, and adding tools can offset local savings. Only three repeated typed turns per condition are available. The browser adds a real 450-ms initial reserve plus device lead before audible playback; this is not included in first-PCM arrival. Normal chunks remain 1.6 seconds, with a roughly 0.6-second initial codec block. Fake/headless device scheduling is not a physical listening test. Known-speaker endpointing still targets one second of that speaker’s silence; first-time enrollment has a longer evidence/grace period. ASR/identity latency is additional to this typed-turn table.</p>'
    routed=read('director-speed-routed')
    if routed.get('rows'):out+='<p>Current routed Luna regression: '+str(sum(r['passed'] for r in routed['rows']))+'/'+str(len(routed['rows']))+' checks passed. This is a separate current-integration run, not substituted into the baseline comparison.</p>'
    out+='<h2>Wake-free speakers and explicit voice variants</h2><p>The new checkbox defaults ON. Known-reference recovery runs first; sufficiently long, coherent, dominant new foreground speech can join as a temporary Guest without “Hey Robbie”. Robby answers and asks a name once. “My name is Alice” names a guest. “It’s me, Bob” / “Ich bin’s doch, Bob”, in a clear longer sentence, explicitly attaches an unnamed guest’s new fast/long/ECAPA anchor to one existing unambiguous name. Up to four separate variants preserve the original anchor. Two named people are not automatically merged. The UI displays variant counts.</p><p>These are user-asserted conversational labels, <strong>not authentication</strong>; another person can impersonate a spoken claim. Open enrollment can also admit dominant TV/background speech. Switch OFF to restore wake-required new enrollment. Overlap, final word-origin, ECAPA and playback-echo checks remain. All new anchors are session RAM only, expiring/clearing with the person; no new biometric persistence. Face VAD/Face ID remain OFF by default, with camera gaze independent.</p>'
    out+='<h2>Public information tools</h2><p>News: official RSS from Tagesschau, BBC, Deutsche Welle, Deutschlandfunk and The Guardian, fetched concurrently. Topic matching covers recent feed entries, not the complete web. Weather: Open-Meteo geocoding plus current/three-day forecasts, explicit location, dates, units and WMO descriptions. A guessed location is blocked before HTTP. Wikipedia: German/English MediaWiki search and short extracts. Actual links appear under “Sources consulted”; this does not guarantee every generated claim is grounded.</p><p>Bounded allowlisted HTTPS GETs, checked redirects, 1.5-MB responses, 6-second per-request/10-second per-tool budgets, strict arguments and one bounded tool round. Public-result cache: 128 entries, three minutes, or ten for weather. No voiceprints/audio/history is sent to sources. Model summaries still need fact checking for consequential decisions. Open-Meteo’s free hosted API is non-commercial; publisher feed terms and Wikipedia attribution/CC BY-SA apply.</p>'
    checks=[]
    for name,label in [('unit-checks','Offline contracts'),('open-speakers-browser','HTTPS Chrome mic: wake-free ON / strict OFF'),('knowledge-director-live','Live LLM + public lookups + missing-location/guest controls'),('knowledge-browser','HTTPS sources + TTS + persisted checkbox'),('speaker-variant-live','Real ASR/ECAPA: explicit asserted variant, repeated claim clip, original-voice follow-up')]:
        d=read(name);current=not d.get('runtime_source_sha256') or d.get('runtime_source_sha256')==runtime_signature()
        checks.append([label,d.get('passed','Not run'),current,d.get('tests',len(d.get('cases',d.get('rows',d.get('turns',[])))))])
    out+=table(['Integration test','Passed','Matches current source if signed','Tests / cases'],checks)
    lifecycle=read('browser-lifecycle-checks');ui=read('reliability-ui-checks')
    out+='<p>Additional regressions: '+e(lifecycle.get('passed','not run'))+' microphone/playback lifecycle checks passed; camera/follow controls, stable face tracking, reference images, independent voice/face status, privacy opt-out and persisted settings: '+e(ui.get('passed','not run'))+'. These use fake browser devices and deterministic landmarks, not a physical camera.</p>'
    out+='<p>Retained development failures: first Wikipedia request lacked an acceptable identifying user agent; first guest responses omitted the name question; a first weather test guessed Berlin despite no user location. The latter two now have explicit code guards. First variant replay was attempted while ASR was starting (HTTP 503); the test now waits for readiness. A later replay exposed an old known-person veto against a second qualified new guest. Another lost the explicit claim prefix at a diarization-channel boundary. Prefix recovery now requires independent transcription and agreement between two non-overlapping acoustic windows, and never bypasses overlap masking. The variant test intentionally asserts one name from two different synthetic voices, then repeats the same claim clip and tests the original voice with different words. It tests session association and anchor retention, not held-out accuracy. Real-room accuracy and language/emotion generalization of the new variants are not established by synthetic tests.</p>'
    out+='<h2>Inspect failed director attempts</h2><details><summary>All retained failures (not hidden retries)</summary>'
    for model,row in failures:out+='<p><strong>'+e(model+' / '+row['case'])+'</strong>: '+e(row.get('error') or row.get('result',{}).get('text'))+'</p>'
    out+='</details><h2>Reproducibility receipts</h2><ul>'
    for name in FILES:
        if (folder/(name+'.json')).exists():out+='<li><a href="/static/speed-study-data/'+name+'.json">'+e(name)+'</a></li>'
    out+='</ul><p>Source: speed_study/benchmark.py, knowledge_tools.py, open_speakers.py, speaker_live/variant_check.py and browser checks in the downloadable source bundle. Historical prompt/schema snapshot is included above; new harness runs use the current integrated prompt. Optional operator-only ROBO_DIRECTOR_MODEL / ROBO_DIRECTOR_BASE settings do not alter TTS or the separate turn-taking decision model. A different remote endpoint requires its own environment API key.</p>'
    out+='<h2>Primary documentation</h2><ul>'
    for title,url in [('Luna','https://developers.openai.com/api/docs/models/gpt-5.6-luna'),('Mercury effort controls','https://docs.inceptionlabs.ai/capabilities/reasoning-efforts'),('DeepSeek updates','https://api-docs.deepseek.com/updates/'),('Gemini 3.5 Flash-Lite','https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite'),('Gemma','https://ai.google.dev/gemma/docs/core'),('Tagesschau feeds','https://www.tagesschau.de/infoservices/rssfeeds'),('BBC feeds','https://support.bbc.co.uk/platform/feeds/NewsFeeds.htm'),('Open-Meteo terms','https://open-meteo.com/en/pricing'),('MediaWiki API','https://www.mediawiki.org/wiki/API:Search')]:out+='<li><a href="'+url+'">'+title+'</a></li>'
    out+='</ul></html>'
    (ROOT/'static/speed-and-tools.html').write_text(out)
    print('Built static/speed-and-tools.html')

if __name__=='__main__':main()
