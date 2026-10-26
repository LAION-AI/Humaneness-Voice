"""Build a standalone English report from actual archived stress receipts."""
import argparse, html, json
from pathlib import Path

def esc(x):return html.escape(str(x))

def summary(record):
    return {k:record.get(k) for k in ('run_id','started_utc','finished_utc','configuration','summary','passed','all_cases_passed','source_unchanged','runtime_unchanged','models_unchanged')}

def build(records, output, audit=None, tools=None):
    current=records[-1]
    body=[]
    for r in records:
        s=r.get('summary',{})
        body.append(f'<tr><td>{esc(r["run_id"])}</td><td>{esc(s.get("passed_steps",0))} / {esc(s.get("executed_steps",0))}</td><td>{esc(s.get("blocked_steps",0))}</td><td>{"PASS" if r["passed"] else "FAIL"}</td><td>{"PASS" if r.get("all_cases_passed") else "FAIL"}</td></tr>')
    table=''.join(body)
    controls=[]
    if audit:
        rows=[]
        for row in audit.get('cases',[]):
            rows.append('<tr><td>'+esc(row.get('id',row.get('name','')))+ '</td><td><pre>'+esc(json.dumps(row,ensure_ascii=False,indent=2))+'</pre></td></tr>')
        controls.append('<section><h2>Clean-input audit</h2><p>All 26 fixed takes were retained. An audit pass means processing completed, not that every input is valid or every voice is consistent. Stop versus Stopp is an orthographic word-error penalty, not an unrecognized stop intent. Conversely, the negated-stop take produced an unrelated clean ASR transcript, so its lack of cancellation cannot establish correct understanding of negation.</p><details><summary>Per-input clean transcription and voice-consistency evidence</summary><table>'+''.join(rows)+'</table></details><p><a href="adversarial-input-audit.json">Complete clean-input audit</a></p></section>')
    if tools:
        rows=[]
        for row in tools.get('cases',[]):
            rows.append('<tr><td>'+esc(row['id'])+'</td><td>'+('PASS' if row.get('passed') else 'FAIL')+'</td><td>'+esc(', '.join(k for k,v in row.get('checks',{}).items() if not v) or row.get('error') or 'All automated checks passed')+'</td><td>'+esc(row.get('text',''))+'</td></tr>')
        controls.append('<section><h2>Isolated real-tool controls</h2><p>Typed browser requests isolate public-tool and speech-output behavior from microphone recognition. These results are a separate, later attempt, not replacements for failed noisy-audio cases. Actual lookup receipts and result hashes are required. The Wikipedia check additionally requires a relevant article title. Weather numbers are compared against returned tomorrow min/max values, allowing 0.6 °C rounding. Human source review remains necessary.</p><div class="scroll"><table><tr><th>Tool</th><th>Result</th><th>Failed checks</th><th>Actual reply</th></tr>'+''.join(rows)+'</table></div><p><a href="adversarial-tool-control.json">Latest tool receipt</a> · <a href="adversarial-tool-before-fix.json">Original failing attempt</a></p><p>The original Wikipedia domain-only check passed despite returning unrelated articles: Professor Moriarty, the Dreyfus affair and Apsley Cherry-Garrard. Source-title inspection by the coding agent rejected that answer. The original weather reply split decimals into separate integers, and the original news reply had no real lookup. These failures motivated stronger checks; a source URL alone is insufficient.</p></section>')
    groups=[]
    for scenario in current['scenarios']:
        rows=[];details=[]
        for step in ([scenario['setup']] if scenario.get('setup') else [])+scenario['steps']:
            failed=[k for k,v in step.get('checks',{}).items() if not v]
            if step.get('error'):failed.append(step['error'])
            metric=[]
            if 'wer' in step:metric.append(f'WER {step["wer"]:.3f}')
            for name in ('energy_end_to_submit_ms','energy_end_to_first_pcm_ms','energy_end_to_cancel_ms'):
                if name in step:metric.append(name.replace('_',' ')+f': {step[name]:.0f}')
            passed=step.get('passed',False)
            rows.append(f'<tr><td><a href="#{esc(scenario["id"]+"-"+step["id"])}">{esc(step["id"])}</a></td><td class="{"pass" if passed else "fail"}">{"PASS" if passed else "FAIL"}</td><td>{"Required" if step.get("required",True) else "Diagnostic"}</td><td>{esc("; ".join(failed) or "All declared checks passed")}</td><td>{esc("; ".join(metric))}</td></tr>')
            events=[e for e in step.get('events',[]) if e.get('type') in ('asr_sentence','endpoint','speaker_focus','ignored','error','reply_cancelled','speaker_permission')]
            details.append(f'<details id="{esc(scenario["id"]+"-"+step["id"])}"><summary>{esc(step["id"])} — {"pass" if passed else "fail"}</summary><p><b>Authored input:</b> {esc(step.get("input_script",""))}</p><p><b>Accepted transcript:</b> {esc(step.get("transcript","—"))}</p><p><b>Reply:</b></p><pre>{esc(step.get("reply","—"))}</pre><p><b>Source links:</b> {esc(json.dumps(step.get("sources",[]),ensure_ascii=False))}</p><details><summary>Checks, soundscape and relevant events</summary><pre>{esc(json.dumps({"checks":step.get("checks"),"mixture":step.get("mixture"),"events":events},indent=2,ensure_ascii=False))}</pre></details></details>')
        groups.append(f'<section><h2>{esc(scenario["id"])}</h2><p>{esc(scenario["description"])}</p><p>{esc(scenario.get("error",""))}</p><div class="scroll"><table><thead><tr><th>Case</th><th>Result</th><th>Scope</th><th>Failed checks</th><th>Measured values (ms)</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>{"".join(details)}</section>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Robby — adversarial conversation tests</title>
<style>body{font:16px/1.6 system-ui,sans-serif;background:#f5f7fc;color:#18233d;max-width:1200px;margin:auto;padding:28px}h1,h2{line-height:1.2}section,header{background:white;border:1px solid #d8e0ef;border-radius:16px;padding:24px;margin:20px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #d8e0ef;text-align:left;vertical-align:top}th{background:#eef2f9}.pass{color:#08603b;font-weight:bold}.fail{color:#a12525;font-weight:bold}a{color:#154ca4}details{border-top:1px solid #d8e0ef;padding:12px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;word-break:break-word;font-size:13px;max-height:650px;overflow:auto}.notice{border-left:4px solid #d08a21;padding:12px;background:#fff8df}code{font-size:14px}</style>
<style>body{overflow-wrap:anywhere}section,header{min-width:0}@media(max-width:600px){body{padding:12px}section,header{padding:16px}th,td{padding:6px;overflow-wrap:anywhere}table{font-size:12px}}</style>
<header><h1>Robby: adversarial conversation tests</h1><p>Reproducible, stateful evaluation of conversation quality under competing speakers, television-style speech, instrumental music, colored noise and transient sounds. Real browser, ASR, speaker recognition, language model, public lookup tools and streamed speech synthesis.</p>
<p><a href="VOICE_ASSISTANT_TEST_PROTOCOL.md">Full test protocol and exact commands</a> · <a href="adversarial-check.json">Current complete machine-readable receipt</a> · <a href="adversarial-fixtures.json">Pinned synthetic inputs</a> · <a href="https://github.com/LAION-AI/Humaneness-Voice/tree/main/robby">Source code</a></p>
<div class="notice"><b>This is a development stress test, not a reliability certificate.</b> Failures remain visible. Digital microphone injection bypasses real microphone hardware and the acoustic loudspeaker/echo path. Source “TV” is authored synthetic speech, not an actual television recording. Word-error rate uses authored TTS scripts pending independent listening review. Human humor/coherence and source-fact review remain separate.</div></header>
<section><h2>Findings and interpretation</h2><p>Two audio-context defects were corrected: speaker recognition previously sampled trailing background speech instead of the foreground utterance; the quiet-speaker veto could dilute a short genuine request with a long television pre-roll. Short literal Stop cues now use a bounded shorter voice context. Existing competing-speaker and echo checks remain active.</p><p>The strict eight-phase Stop/Wait test still exposes misses. Overlapping voices can contaminate the transcript, prevent submission or trigger an unintended topic/tool. Successful permission revocation after a failed invitation does not prove a successful guest conversation: inspect the entire scenario. The totals below count declared assertions, not independent demonstrations of field reliability.</p><p>Creative-task repair requests at most one extra continuation for an unfinished ordinary joke/story, preserving the actual preceding text. Explicit one-line requests are exempt. A sentence count cannot judge humor or coherence. Some fixture voices themselves show low clean similarity, requiring independent listening before attributing every rejection to the application.</p></section>
<section><h2>Run ledger</h2><table><thead><tr><th>Attempt</th><th>Passed / executed steps</th><th>Blocked</th><th>Required suite</th><th>Including diagnostic cases</th></tr></thead><tbody>'''+table+'''</tbody></table>
<p>The initial harness attempt incorrectly required a two-sentence introduction to be submitted exactly once; its blocked steps are not model failures. The corrected procedure permits two distinct turns for explicitly annotated multi-sentence inputs, while still rejecting duplicate submissions and additional background turns. Every attempt is retained.</p></section>
<section><h2>What is measured</h2><p>Seven scenarios declare 34 steps: multi-turn jokes and topic recall; a longer poem; real Wikipedia, weather and news tools; explicit guest invitation/revocation; Stop/Wait at four 320-ms-hop phases; negative interruption controls; and two-speaker acoustic sweeps at 20, 12, 6 and 0 dB measured target/interferer SNR. The 6/0-dB conditions are diagnostic boundaries, not omitted failures.</p><p>Speaker admission, transcript completeness, meaningful task content, source metadata, completed TTS and interruption behavior are scored separately. A recognized person with corrupted words cannot pass. Automated structure/keyword checks do not prove a joke is funny or a tool answer is factually correct.</p>
<p>Timing uses one browser clock. Energy-end → submission/PCM/cancel includes ASR/identity/queue costs. “Energy end” is the isolated source's 20-ms RMS support, <b>not an independently annotated last word</b>. File-onset phase differs from first-phoneme phase if a clip has leading silence. Background false-event exposure and finite-sample uncertainty are recorded rather than presenting a few successful trials as 100% reliable.</p></section>
<section><h2>Separate post-fix validation</h2><p>The frozen targeted run <code>20261026T144116Z-50ba538a</code> passed 453 offline checks, three typed real-tool controls and all eleven combined-browser cases, with no planning error. The earlier full run's separate 18-turn continuity check also passed. These successes do not replace the failed 34-step noisy/phase sweep. The corrected Wikipedia reply still added an unrequested penguin joke; automated checks do not certify all factual prose or conversational taste.</p><p><a href="adversarial-targeted-run.json">Targeted run manifest</a> · <a href="adversarial-full-run.json">Complete earlier run manifest</a> · <a href="https://github.com/LAION-AI/Humaneness-Voice/releases/tag/robby-test-fixtures-v2">Pinned synthetic audio pack</a></p></section>
<section><h2>Reproduce and extend</h2><pre>python -m unittest test_adversarial_protocol
python run_voice_assistant_checks.py --suites unit,adversarial --seeds 42 --keep-going
python run_voice_assistant_checks.py --suites adversarial \\
  --adversarial-scenarios interruptions,background_only --seeds 43</pre><p>Restore the pinned fixture pack, run on the configured host, and serialize all live audio tests. Use a new run ID for each attempt. The code records source/model/fixture hashes and does not restart occupied production services. Full instructions, evaluation limits and physical-room/soak extensions are in the protocol.</p></section>
'''+''.join(controls)+''.join(groups)+'''<section><h2>Configuration and evidence</h2><pre>'''+esc(json.dumps([summary(r) for r in records],indent=2,ensure_ascii=False))+'''</pre><p>Tables are generated from saved test receipts; interpretation is explicitly separated from automated assertions. No API keys, private microphone recordings or enrolled human voiceprints are included.</p></section></html>'''
    output.write_text(page)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipts',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--audit',type=Path);p.add_argument('--tools',type=Path);a=p.parse_args()
    build([json.loads(x.read_text()) for x in a.receipts],a.output,
          json.loads(a.audit.read_text()) if a.audit else None,json.loads(a.tools.read_text()) if a.tools else None)
