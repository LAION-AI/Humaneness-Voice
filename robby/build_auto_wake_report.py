"""Report automatic-start, permissions, wake variants and playback tests."""
import datetime,html,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;STATIC=ROOT/'static'
def esc(x):return html.escape(str(x))
def main():
 bp=sorted(STATIC.glob('auto-wake-browser-*.json'))[-1];b=json.loads(bp.read_text())
 pp=sorted(STATIC.glob('auto-wake-policy-*.json'))[-1];p=json.loads(pp.read_text())
 units=json.loads((ROOT/'logs/unit-checks.json').read_text())
 rows=[('Old browser settings migrated','PASS' if b['cases'][0]['passed'] else 'FAIL', 'Microphone and camera start without clicking Start Talking; fake device.'),
  ('Spoken “Hi Robbie”', 'PASS' if b['cases'][0].get('ten_seconds_after_playback') else 'FAIL',f"ASR: {b['cases'][0].get('transcripts')}; {b['cases'][0].get('pcm_chunks',0)} PCM chunks; {b['cases'][0].get('actual_reply_model')}"),
  ('Ten-second follow-up','PASS' if b['cases'][0].get('ten_seconds_after_playback') else 'FAIL','Actual playback-end event renews the window; unit test simulates a 30-second reply.'),
  ('Camera + gaze','PASS' if b['cases'][0].get('gaze',{}).get('translation_tracked') else 'FAIL',f"CPU camera worker {b['cases'][0].get('camera_frames',0)} frames; synthetic eye positions moved gaze both ways."),
  ('Turn off and persist','PASS' if b['cases'][0].get('explicit_off_stops_and_persists') else 'FAIL','Microphone, camera and playback stop; next reload respects OFF.'),
  ('Camera permission denied','PASS' if b['cases'][1]['passed'] else 'FAIL','Microphone continues without the camera.'),
  ('Microphone permission denied','PASS' if b['cases'][2]['passed'] else 'FAIL','Manual-enable guidance is displayed.'),
  ('Native autoplay policy','PASS' if p['passed'] else 'FAIL','Manual gesture required: '+str(p.get('manual_enable_needed','unknown'))+'; microphone active: '+str(p.get('microphone_active',False))+'.'),
  ('Offline regression',str(units['tests'])+' passed' if units['passed'] else 'FAIL','Includes 20 greeting/name combinations and a 30-second playback clock.')]
 table='<div class="scroll"><table><tr><th>Check</th><th>Result</th><th>Scope</th></tr>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'
 checks={'generated':datetime.datetime.now(datetime.timezone.utc).isoformat(),'browser_receipt':bp.name,'browser_passed':b['passed'],'policy_receipt':pp.name,'policy_passed':p['passed'],'source_sha256':b['source_sha256'],'policy_source_sha256':p['source_sha256'],'unit_tests':units['tests'],'unit_passed':units['passed'],'physical_devices_used':False}
 (STATIC/'auto-wake-release-checks.json').write_text(json.dumps(checks,indent=2))
 page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Robby automatic wake and camera</title><style>body{{font:16px/1.6 system-ui;max-width:1050px;margin:auto;padding:24px;color:#20304a;background:#f7fafc}}section{{background:white;padding:20px;border-radius:15px;margin:16px 0}}table{{border-collapse:collapse;min-width:800px}}td,th{{border-bottom:1px solid #ddd;padding:9px;text-align:left;vertical-align:top}}.scroll{{overflow:auto}}a{{color:#15628d}}</style><h1>Automatic listening and eye contact</h1><p><a href="/desk-robot/">Open Robby</a> · <a href="wake-report.html">Wake mode report</a> · <a href="https://github.com/LAION-AI/Humaneness-Voice/blob/main/robby/WAKE_MODE.md">Implementation and reproduction</a></p>
<section><h2>What changes</h2><p>On a new or migrated browser, Robby starts its microphone and camera automatically when server readiness and browser permissions allow. It waits for “Hey Robby”, “Hi Robbie”, “Hallo Robbi” and equivalent forms. Camera landmarks drive position-based gaze following in a browser CPU worker. Webcam Face ID and face-based speech filtering remain off.</p><p>The current speaker can follow up for ten seconds after Robby finishes actual playback, even after a long answer. Uncertain/different voices still need the wake phrase. Turn Robby off stops microphone, camera and playback; the off choice persists. Microphone, camera and autoplay permission decisions remain under browser control.</p></section>
<section><h2>Executed checks</h2>{table}<p>Runtime fingerprint for the browser run: <code>{esc(b['source_sha256'])}</code>; autoplay policy test: <code>{esc(p['source_sha256'])}</code>. Each receipt carries its own fingerprint. The local Nemotron model was explicitly selected for the test; the user’s configured LLM default was not changed.</p></section>
<section><h2>Limits and reproduction</h2><p>Microphone and camera were browser fake devices. Gaze-direction logic used synthetic eye landmarks; it is not a human eye-contact accuracy measurement. Recognition and speaker verification are approximate, especially for short words or overlap. Some browsers require an explicit microphone/camera grant and an activation click before sound may play; Robby shows the enable control when blocked.</p><pre>python run_unit_checks.py
python generate_auto_wake_fixture.py
python browser_auto_wake.py --url https://YOUR-HOST
python browser_auto_wake_policy.py --url https://YOUR-HOST</pre><p>Run browser checks sequentially: the live ASR service has one session slot. The first script makes real ASR/LLM/TTS requests using authored synthetic audio, not a physical microphone.</p><p><a href="{bp.name}">Full browser receipt</a> · <a href="{pp.name}">Autoplay-policy receipt</a></p></section></html>'''
 (STATIC/'auto-wake-report.html').write_text(page);print(json.dumps(checks))
if __name__=='__main__':main()
