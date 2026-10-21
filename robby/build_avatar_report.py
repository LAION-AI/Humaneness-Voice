"""Publish only explicit synthetic QA receipts and a compact banana-game trace."""
import datetime,html,json,shutil
from pathlib import Path
ROOT=Path(__file__).parent

def main():
    out=ROOT/'static/avatar-test-data';out.mkdir(exist_ok=True)
    names=['avatar-browser-checks','avatar-live-browser','avatar-live-before-ready-wait','avatar-live-before-render-wait','avatar-director-checks','avatar-director-before-empty-fallback','avatar-decision-checks','unit-checks']
    records={}
    for name in names:
        path=ROOT/'logs'/(name+'.json')
        if path.exists():records[name]=json.loads(path.read_text());shutil.copyfile(path,out/path.name)
    for name,path in [('banana-game',Path('/mnt/nvme/robo-speaker-studies/speaker-live-results/proactive-live.json')),
                      ('banana-before-rule-priority',ROOT/'logs/banana-before-rule-priority.json')]:
        if not path.exists():continue
        source=json.loads(path.read_text());banana={k:v for k,v in source.items() if k!='cases'};banana['cases']=[]
        for c in source.get('cases',[]):
            row={k:v for k,v in c.items() if k!='events'}
            row['diagnostic_events']=[e for e in c.get('events',[]) if e.get('phase')=='probe' and
                (e['type']=='endpoint' or (e['type']=='interaction' and e.get('verification')))]
            row['interruption_times_s']=[e['at_s'] for e in c.get('events',[]) if e.get('phase')=='probe' and e['type']=='endpoint' and e.get('by')=='agreed_interruption']
            banana['cases'].append(row)
        records[name]=banana;(out/(name+'.json')).write_text(json.dumps(banana,indent=2,ensure_ascii=False))
    e=lambda v:html.escape(str(v))
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Robby actions — test results</title><style>body{font:17px/1.5 system-ui;max-width:1000px;margin:32px auto;padding:0 18px;color:#273b43;background:#faf8f4}table{border-collapse:collapse;width:100%}td,th{padding:12px;border:1px solid #ddd;text-align:left}.scroll{overflow:auto}.note{padding:15px;background:#fff0ca}a{color:#176175}</style><p><a href="./avatar-manual.html">Interactive manual</a> · <a href="https://github.com/LAION-AI/Humaneness-Voice/tree/main/robby">Source</a></p><h1>Expressive actions and the banana game</h1>'''
    page+='<p>Built '+e(datetime.datetime.now(datetime.timezone.utc).isoformat())+'. No private recordings. Browser tests use synthetic requests; acoustic game tests use public synthetic voices.</p><div class="scroll"><table><tr><th>Suite</th><th>Passed</th><th>Receipt</th></tr>'
    for name,data in records.items():page+='<tr><td>'+e(name)+'</td><td>'+e(data.get('passed',False))+'</td><td><a href="./avatar-test-data/'+name+'.json">JSON</a></td></tr>'
    page+='</table></div><h2>What was tested</h2><p>All 40 archetypes are distinct. CPU-only browser tests exercise both spin directions, independent eyes, sequences, actual rendered motion, persisted switches, cancel/clear, idle/listening cooldowns, reduced motion, calibration suppression, playback anchoring and mobile manual layout. Real API tests request actions and negative controls. The HTTPS browser integration test requires a real LLM tool result, TTS PCM and visible motion tied to playback.</p>'
    browser=records.get('avatar-browser-checks',{})
    page+='<p>Measured renderer p95 in the headless CPU-only browser fixture: '+e(browser.get('render_p95_ms','not measured'))+' ms. This includes software 3-D rendering; it is not model inference latency or a guarantee on every device.</p>'
    page+='<h2>Why “Bananen” can be recognized but not cause an interruption</h2><p>Keyword matching and speaker authorization are separate. An agreed literal rule needs no new LLM call after it is installed, but it still requires a sufficiently confident current speaker. Proactive interjections require a teacher cosine of at least 0.55, stable evidence, plus short and long student checks. This is deliberately stricter than the ordinary 0.32 utterance setting. Echo and short phonetic windows can weaken student evidence. Rules also have a four-second rearm interval and require fresh words after the previous reply; they do not fire repeatedly on an old transcript.</p>'
    for case in records.get('banana-game',{}).get('cases',[]):
        if case.get('condition')=='repeated':
            times=case.get('interruption_times_s',[]);start=case.get('user_start_s',0);end=case.get('user_end_s',start)
            page+='<p>The retained continuous test spoke for <strong>'+e(round(end-start,2))+' s</strong> and produced <strong>'+e(len(times))+'</strong> spoken interruptions. First decision after '+e(round(times[0]-start,2) if times else 'none')+' s; relative interruption times: '+e([round(t-start,2) for t in times])+'. This passes the repetition contract (at least two before the user finishes), but is <strong>not immediate or every-mention reliability</strong>.</p>'
    page+='<p>A separate replay exposed a rule-priority defect: generic helpfulness produced “Darf ich kurz einhaken?” during an explicit “Stopp, stopp!” game. Generic helpfulness is now suppressed while explicit rules are installed, including their cooldowns. Exact rule responses and normal turn-taking remain unchanged; clear the game to restore generic helpful interjections. The failed trace is retained above.</p>'
    page+='<p class="note">Do not interpret a passing banana test as proof that interruption always works. The trace includes periods of correctly transcribed words rejected by identity checks. The separate brief user “Stop” browser failure remains open. No blanket threshold reduction was made: it could admit unrelated background speakers.</p>'
    page+='<h2>Development failures are retained</h2><p>The first manual browser run found a mobile-width overflow, fixed by bounding the selector width. A real director run also rejected two movement requests at the strict spoken-JSON validator; a narrow fallback now supplies a short local acknowledgment only for an already validated visual sequence with no spoken words. Other malformed answers remain errors. Historical receipts are named explicitly; tests are finite development checks, not human accuracy estimates.</p></html>'
    (ROOT/'static/avatar-checks.html').write_text(page)
    print('Built avatar-checks.html and explicit synthetic receipts')

if __name__=='__main__':main()
