"""Public report from authored synthetic trials; no microphone recordings."""
import html,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
S=ROOT/'static'
def load(name):return json.loads((S/name).read_text()) if (S/name).exists() else {}
def esc(x):return html.escape(str(x))
def main():
 a=load('quick-reaction-audio.json');m=load('quick-reaction-models.json');b=load('quick-reaction-browser.json');onset=load('quick-reaction-onset.json');chunking=load('quick-reaction-chunking.json')
 parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Robby: contextual quick reactions</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:auto;padding:24px;background:#f7f5ef;color:#24322e}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:9px;border-bottom:1px solid #c9cec8}article{border:1px solid #c9cec8;padding:16px;border-radius:12px;margin:16px 0}audio{width:min(100%,500px)}pre{white-space:pre-wrap;overflow-wrap:anywhere}.scroll{overflow:auto}a{color:#15635a}</style><h1>Contextual quick reactions, not a phrase bank</h1><p>The existing local Nemotron Bolt E4B races the main Qwen/Cerebras answer to produce at most six empathetic words. The full answer continues independently. The main answer wins ties; invalid or late reactions are skipped. Same reference voice, direct handoff.</p><h2>Measured audio latency</h2><p>Three paired real API/TTS trials on a shared dual-RTX-3090 host. PCM arrival at the HTTP client, excluding physical playback. Different stochastic answers: these are small observational measurements, not a controlled quality study.</p><div class="scroll"><table><tr><th>Input</th><th>Quick reaction</th><th>First audio (ms)</th><th>First main-answer audio (ms)</th></tr>']
 for r in a.get('rows',[]):parts.append(f'<tr><td>{esc(r["input"])}</td><td>{r["quick_replies"]}</td><td>{r["first_pcm_ms"]}</td><td>{r["first_answer_pcm_ms"]}</td></tr>')
 parts.append('</table></div><p><strong>A faster first sound delays the substantive answer by the time needed for the reaction.</strong> This is optional. There is no claim of perfectly seamless voice continuity.</p>')
 rows=m.get('rows',[])
 if rows:
  vals=[r['reaction']['ms'] for r in rows];wins=sum(r['local_ready_first'] for r in rows)
  parts.append(f'<h2>Text-generation experiment</h2><p>{len(rows)} actual local/API races; local generation {min(vals):.0f}–{max(vals):.0f} ms, median {statistics.median(vals):.0f} ms. Local was ready first in {wins}/{len(rows)} cases. Outputs remain imperfect: factual questions can receive generic acknowledgments. Three longer transition probes added a separate call and redundant speech; direct handoff was selected.</p>')
  for r in rows:
   if 'longer_transition'in r:parts.append('<p>'+esc(r['case'])+': '+esc(r['longer_transition'])+'</p>')
 onset_rows=onset.get('rows',[]);onset_ms=[r['first_pcm_ms'] for r in onset_rows if r.get('role')=='reaction']
 if onset_ms:
  parts.append(f'<h2>Can the reaction start below 900 ms?</h2><p>{len(onset_ms)} measured reaction onsets: {min(onset_ms):.0f}–{max(onset_ms):.0f} ms, median {statistics.median(onset_ms):.0f} ms; {sum(v<900 for v in onset_ms)}/{len(onset_ms)} below 900 ms. These HTTP diagnostics deliberately cancel after the first PCM block and do not evaluate the handoff. The full browser trials below include playback scheduling. An earlier post-restart trial took 1311 ms: this is not an unconditional latency guarantee.</p>')
 parts.append('<h2>What changed after testing</h2><p>Initial alignment held back short reactions. Removing that lookahead improved first sound, but a mistakenly long duration condition stretched four words to nearly seven seconds. The final reaction uses a 2–2.4 s duration condition, a 160 ms initial block, then 400 ms blocks and 10 ms fades; main answers retain 1600 ms chunks and existing alignment. A malformed optional acting cue also surfaced; it now falls back to a neutral cue without discarding the answer. Internal server-to-server PCM eliminates MP3 encoding and download round trips. Single live replies skip ranking and final-file export. Same-seed 2.48-second speech took 2862 ms with uniform 160-ms blocks versus 1975 ms with a 160-ms first / 400-ms following schedule. Codec waveforms have block-size-dependent numerical differences; perceptual equivalence is not proven. Earlier measurements and failures are retained.</p><h2>Browser verification</h2><pre>'+esc(json.dumps({k:b.get(k) for k in ('passed','error','scope','layout','mobile_controls_accessible','off_setting_persists','source_unchanged')},indent=2))+'</pre>')
 for r in b.get('turns',[]):
  parts.append('<p>'+esc(r['input'])+' — '+esc({k:r.get(k) for k in ('passed','first_pcm_ms','scheduled_playback_ms','new_underflows','max_gap_ms')})+'</p>')
 parts.append('<p>Playback scheduling gaps still occurred in one of the latest two reaction turns. The earlier start does not establish a gap-free transition on this shared host. Device acoustics were not measured.</p>')
 parts.append('<h2>Listen to authored test outputs</h2><p>Only synthetic prompts and generated speech are included. No private microphone audio.</p>')
 for r in a.get('rows',[]):
  parts.append('<article><b>'+esc(r['input'])+'</b><p>Quick reaction: '+str(r['quick_replies'])+'</p><audio controls preload="none" src="data:audio/mpeg;base64,'+r['listen_mp3_b64']+'"></audio></article>')
 parts.append('<h2>Reproduce and inspect</h2><p><a href="https://github.com/LAION-AI/Humaneness-Voice/blob/main/robby/QUICK_REACTIONS.md">Setup, design and test commands</a> · <a href="quick-reaction-models.json">Text trial receipt</a> · <a href="quick-reaction-audio.json">Audio timing receipt</a> · <a href="quick-reaction-browser.json">HTTPS browser receipt</a>.</p><p>Manual cancellation is tested separately from spoken interruption. Previously documented noisy acoustic Stop failures are not certified fixed by this change.</p></html>')
 (S/'quick-reaction-report.html').write_text('\n'.join(parts))
if __name__=='__main__':main()
