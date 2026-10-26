"""Build the public English quality report from complete synthetic receipts."""
import collections,html,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
S=ROOT/'static'
esc=lambda x:html.escape(str(x))
def read(name):return json.loads((S/name).read_text())
def stats(rows):
    times=[r['reaction']['ms'] for r in rows if r['eligible']]
    played=[r['reaction']['text'] for r in rows if r['reaction'].get('text')]
    return {'attempts':len(times),'ready':len(played),'distinct':len(set(played)),
            'median':round(statistics.median(times),1),'p95':round(sorted(times)[int(.95*len(times))],1)}
def main():
    base=read('quick-quality-baseline.json');final=read('quick-quality-final.json');judged=read('quick-quality-final-ratings.json')
    ratings={r['case']:r for r in judged['rows']};wins=collections.Counter();scores=collections.defaultdict(list)
    for r in judged['rows']:
        if 'rating' not in r:continue
        z=r['rating'];wins[r['order'].get(z['winner'],'tie')]+=1
        for label,version in r['order'].items():
            for metric in ['relevance','naturalness','tone','engagement']:scores[version,metric].append(z[label][metric])
    text=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robby: Are the quick reactions helpful and entertaining?</title>
<style>body{overflow-wrap:anywhere;max-width:1100px;margin:2em auto;padding:0 18px;font:17px/1.55 system-ui;color:#20323b;background:#fafafa}h1{line-height:1.15}table{border-collapse:collapse;width:100%;margin:1em 0}th,td{text-align:left;padding:10px;border-bottom:1px solid #ccd7df;vertical-align:top}details{padding:12px;background:white;border:1px solid #ddd;margin:12px 0}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere}audio{max-width:100%}.scroll{overflow:auto}.note{padding:16px;background:#e8f2f5;border-radius:10px}a{color:#006d85}</style>
<h1>Robby: better short reactions, with measured limitations</h1>
<p>The existing local Nemotron Bolt E4B writes up to six words while Qwen produces the full answer. This study tests the words, their context and the handoff. It is not a human listening panel. All inputs are authored synthetic conversations.</p>
<div class="note"><b>Decision:</b> retain low temperature (0.35), use small tone-specific prompt examples, preserve joke follow-ups, suppress repeated openings and unsupported action promises, and keep the complete main answer. Playful replies receive a playful TTS cue; serious replies use soft delivery. Examples guide the model; no phrase is selected directly for playback, and there is no extra online judge or retry.</div>
<h2>Results at a glance</h2><div class="scroll"><table><tr><th>Measure</th><th>Before</th><th>Selected version</th></tr>''']
    a,b=stats(base['rows']),stats(final['rows'])
    for key,label in [('attempts','Actual local-model calls'),('ready','Valid reactions available (before the main-answer race)'),('distinct','Distinct valid text strings'),('median','Median local generation, ms'),('p95','P95 local generation, ms')]:
        text.append(f'<tr><td>{label}</td><td>{a[key]}</td><td>{b[key]}</td></tr>')
    for m in ['relevance','naturalness','tone','engagement']:
        text.append(f'<tr><td>Model-rated {m} (0–4, advisory)</td><td>{statistics.mean(scores["baseline",m]):.2f}</td><td>{statistics.mean(scores["selected",m]):.2f}</td></tr>')
    text.append(f'''</table></div><p>Blinded, randomized A/B judgments: {wins['selected']} prefer the selected version, {wins['baseline']} prefer the baseline, {wins['tie']} ties; {len(judged['rows'])-sum(wins.values())} malformed judge responses are retained and excluded. The judge was Qwen 3.8 27B. These are model preferences, not human accuracy percentages or statistically established improvements.</p>
<p><b>The trade-off is explicit:</b> engagement and relevance improved on the model rating; naturalness decreased slightly. Some phrases such as “Das lohnt einen Blick” remain generic. Several joke openings are charming, but not every attempt is funny. The evaluator sometimes prefers an action promise that our policy disallows; ratings do not override the runtime checks.</p>
<h2>Method and changes</h2><p>Forty fixed situations, with three seeds per eligible situation (110 baseline rows; 106 selected rows, including explicit skips). Twenty-eight were development situations; twelve were reserved until selecting a candidate. The first held-out pass exposed an English-language parsing bug. That bug and a compound-joke follow-up bug were repaired; the final rerun is therefore a regression evaluation, not untouched held-out evidence. Eight development prompt variants including baseline were explored; all receipts remain below.</p>
<ul><li>Increasing temperature or asking for more imaginative wording often produced unnatural word combinations or questions. Those variants were rejected.</li><li>Short tone hints cover play, jokes, distress, support, kindness, achievements, corrections, casual chat and curiosity. They are keyword heuristics, not psychological diagnoses.</li><li>Recent assistant openings are supplied as examples to avoid. Exact repeated openings are suppressed without a second model call.</li><li>Urgent help, simple arithmetic, exact-output requests, poems and Stop skip the optional preface. Invalid output never blocks the full answer.</li><li>“Katzenwitz”, “Pinguinwitz” and “Und noch einen” preserve the joke task. Explicit “Please speak English” works for both models.</li><li>Duplicate short condolences at the handoff are removed, including “Oh, das tut mir so, so sehr leid.” Advice or substantive sentences are preserved.</li></ul>
<h2>Every final text case</h2><p>Each list contains all three samples, or the deliberate no-call skip. Repeated strings here are independent trials; the dialogue probe below tests repetition across actual turns.</p>''')
    for name in dict.fromkeys(r['name'] for r in final['rows']):
        rows=[r for r in final['rows'] if r['name']==name];c=rows[0];before=[r for r in base['rows'] if r['name']==name]
        text.append(f'<details><summary>{esc(name)} · {esc(c["text"])}</summary><p>{esc(c["expect"])}</p>')
        for label,rr in [('Before',before),('After',rows)]:
            text.append(f'<b>{label}</b><ul>'+''.join('<li>'+esc(r['reaction'].get('text') or 'No preface — '+r['reaction']['status'])+'</li>' for r in rr)+'</ul>')
        z=ratings.get(name,{})
        text.append('<p>Model reviewer: '+esc(z.get('rating',{}).get('reason',z.get('error','unavailable')))+'</p></details>')
    audio=read('quick-quality-audio.json')
    text.append('<h2>Real Qwen and TTS handoffs: listening examples</h2><p>Four full live API/TTS trials. The main answer won the race in three; the bereavement bridge reached first PCM at about 861 ms. Other first PCM times were 1.39–1.53 s without a bridge. Timing begins at text submission and excludes ASR, endpointing, browser scheduling and physical speaker latency. This is not a renewed universal sub-900-ms claim.</p><p>The bereavement clip revealed a repeated condolence from Qwen. The narrow deduplication repair was made afterwards and covered by regression tests; this original clip is intentionally retained. Full-answer joke quality and novelty also remain imperfect and are separate from bridge quality.</p>')
    for r in audio['rows']:
        final_text=next((e.get('text','') for e in reversed(r['events']) if e.get('sentence_role')),'')
        text.append(f'<details><summary>{esc(r["case"])} · first PCM {r["first_pcm_ms"]:.0f} ms</summary><p>{esc(r["input"])}</p><audio controls preload="none" src="data:audio/mpeg;base64,{r["listen_mp3_b64"]}"></audio><p>{esc(final_text)}</p></details>')
    dialog=read('quick-quality-dialogue.json')
    text.append('<h2>Six-turn actual-model dialogue</h2><p>All preceding generated answers are carried forward. No selected bridge exactly repeated a recent opening. However, Qwen repeated a whole earlier joke and produced weak punchlines: this test does not certify the main model’s humor or novelty.</p>')
    for r in dialog['rows']:
        text.append(f'<details><summary>{esc(r["input"])}</summary><p>Selected reaction: {esc(r["played_reaction"] or "none")}</p><p>{esc(" ".join(r["main_answer"]))}</p></details>')
    if (S/'quick-quality-browser.json').exists():
        browser=read('quick-quality-browser.json')
        text.append('<h2>Final HTTPS browser regression</h2><p>Status: '+esc(browser.get('passed'))+'. Real Qwen/local-model/TTS, full answers, a four-line poem, disabling/persistence, mobile layout and manual cancellation. Typed synthetic turns; no physical microphone or acoustic-interruption claim. See the full browser receipt.</p>')
    if (S/'quick-quality-unit.json').exists():
        unit=read('quick-quality-unit.json')
        text.append('<p>Offline checks: '+esc(unit.get('tests'))+' tests, '+esc(unit.get('failures'))+' failures, '+esc(unit.get('errors'))+' errors.</p>')
    text.append('<h2>Reproduce and inspect</h2><p>Use the pinned local model and existing Qwen credential from the repository setup. There is no new model or dependency. Only synthetic fixtures should be used in public receipts.</p><pre>python benchmark_quick_quality.py --label new-run --split all\npython judge_quick_quality.py --after static/quick-quality-new-run.json --output static/new-ratings.json\npython benchmark_quick_dialogue.py\npython benchmark_quick_quality_audio.py\npython run_unit_checks.py\npython browser_quick_reaction_check.py --url https://YOUR-DEMO</pre><p>The benchmarks refuse to overwrite existing runs. Rename output receipts before an intentional rerun. Baseline prompt/sampling is archived in experiments/quick-reaction-v1.json; rejected implementations are in experiments/quick-reaction-candidate*.py. The baseline code is also available in Git commit ae1377918544f2b4f3c9719350580a6a9deb6fe8. Model/API outputs remain stochastic.</p><ul>')
    for path in sorted(S.glob('quick-quality-*.json')):
        text.append(f'<li><a href="{esc(path.name)}">{esc(path.name)}</a></li>')
    text.append('</ul><p><a href="/desk-robot/">Live Robby</a> · <a href="quick-reaction-report.html">Earlier latency study</a></p></html>')
    (S/'quick-quality-report.html').write_text('\n'.join(text))
if __name__=='__main__':main()
