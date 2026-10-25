"""English before/after report; only authored test conversations are embedded."""
import html,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text()) if path.exists() else {}
def esc(x):return html.escape(str(x))
def main():
    checked=read(ROOT/'static/long-replies-check.json');browser=read(ROOT/'static/long-replies-browser.json')
    before=read(ROOT/'logs/long-replies-before.json')
    unit=read_unit=ROOT/'logs/long-replies-unit.log'
    unit=json.loads(read_unit.read_text().splitlines()[-1]) if read_unit.exists() else {}
    rows=''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [r['case'],r.get('passed'),len(r.get('parts',[])),
        r.get('metrics',{}).get('completion_repairs',0),r.get('metrics',{}).get('response_complete'),r.get('error','')])+'</tr>' for r in checked.get('rows',[]))
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Robby — complete answers and playful conversations</title>
<style>body{font:17px/1.65 system-ui;max-width:1120px;margin:30px auto;padding:20px;background:#f6f8fa;color:#20333a}h1,h2{line-height:1.25}table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #bfcbd0;text-align:left;padding:10px}.scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eaf0f4;padding:18px;font-size:14px}a{color:#166a78}.note{padding:18px;background:#e3f0ed}</style>
<p><a href="https://arrive-locations-parade-research.trycloudflare.com/desk-robot/">Open Robby</a> · <a href="https://github.com/LAION-AI/Humaneness-Voice/tree/main/robby">Source and setup</a></p>
<h1>Complete poems, longer answers and a warmer robot</h1>
<p class="note">Robby now completes requested creative work and detailed answers instead of reducing them to a short conversational acknowledgment. Its personality is warm, playful and emotionally expressive in character. Sentence-level TTS streaming is retained; the first line can play while the rest is generated.</p>
<h2>What caused the problem?</h2><ol>
<li>The released GGUF template appends “Respond as briefly as possible” when conversational_mode is true. It was enabled in addition to our own short-response guidance.</li>
<li>The ordinary director allowed only 320 output tokens. The tool fallback had a separate 120-word validation ceiling.</li>
<li>The stream parser expected full-stop sentence boundaries and could not correctly handle verse lines without punctuation or preserve their display formatting.</li>
<li>Generic negation such as “Keine Einleitung” could incorrectly route a poem into the interruption-rule tool path. That path also collapsed line breaks.</li></ol>
<p>Production timing logs and runtime code were inspected. They do not store the user's full private transcript, so this report does not pretend to reproduce the exact last human conversation. The examples below are newly authored regression prompts.</p>
<h2>Implemented changes</h2><p>The local prompt prioritizes completing the actual task, respects requested structure and supports playful simulated emotions without repetitive disclaimers. conversational_mode is disabled while thinking stays disabled. Ordinary output has a 2,048-token ceiling; deliberately tiny benchmarks/avatar reactions remain bounded separately. Recent history is kept in order with a context budget.</p>
<p>Poetry uses complete verse lines; prose uses sentences or bounded phrases. Acting cues stay out of spoken text. Newlines and stanza breaks remain in the UI and history. The reply language is retained across individual lines. For counted poems/stories, a structure guard can request just the missing continuation at most twice; it does not replace earlier speech with a canned answer. A pilot that stopped at six of eight lines motivated this guard. The UI marks an actual response limit if completion still fails.</p>
<h2>Real local model checks</h2><div class="scroll"><table><tr><th>Case</th><th>Passed</th><th>Streamed parts</th><th>Continuation repairs</th><th>Complete</th><th>Error</th></tr>'''+rows+'''</table></div>'''
    page+='<p>HTTPS CPU Chrome test passed: <strong>'+esc(browser.get('passed','Pending'))+'</strong>. It requests eight poem lines, waits for actual playback completion, checks every displayed/spoken part, then requests a four-line continuation using the same conversation. Audio chunks received: '+esc(browser.get('audio_chunks','Pending'))+'. Unit checks: '+esc(unit.get('tests','Pending'))+'; passed: '+esc(unit.get('passed','Pending'))+'.</p>'
    page+='<h2>Authored before/after examples</h2>'
    if isinstance(before,list):
        for row in before:
            text=' '.join(e.get('text','') for e in row.get('events',[]) if e.get('manual_prompt'))
            page+='<details><summary>Before: '+esc(row['input'])+'</summary><pre>'+esc(text)+'</pre></details>'
    for row in checked.get('rows',[]):
        page+='<details><summary>After: '+esc(row['input'])+'</summary><pre>'+esc(row.get('text',''))+'</pre></details>'
    page+='''<h2>Scope and limits</h2><p>These are real local-model and browser/TTS checks with synthetic authored requests, not a human evaluation of literary quality. Structural completion does not guarantee perfect rhyme, grammar, factual accuracy or instruction following. The small local model can still produce awkward wording. Responses remain bounded by available context/output tokens; the guard is bounded, not an infinite retry loop. Real-room microphone behavior is covered separately by the speech-pause report; this browser check uses typed requests and actual audio playback.</p>
<p>The source contains the exact local prompt, compact tool recipe, counted-task planner, stream parser and validation scripts. No private microphone recordings, voiceprints or human conversation transcripts are included.</p><details><summary>Raw test receipts</summary><pre>'''
    page+=esc(json.dumps({'llm':checked,'browser':browser,'unit':unit,'authored_before':before},indent=2,ensure_ascii=False))+'</pre></details>'
    page+='<p>Generated '+esc(time.strftime('%Y-%m-%d %H:%M UTC',time.gmtime()))+'.</p></html>'
    (ROOT/'static/long-replies-update.html').write_text(page);(ROOT.parent/'robby-long-replies-report.html').write_text(page)
    print(json.dumps({'llm_passed':checked.get('passed'),'browser_passed':browser.get('passed'),'unit_passed':unit.get('passed')}))
if __name__=='__main__':main()
