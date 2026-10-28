"""HTTPS synthetic-audio wake, owner, timeout, UI and optional sound check."""
import argparse,asyncio,datetime,json
from pathlib import Path
from types import SimpleNamespace
from playwright.async_api import async_playwright
from browser_adversarial_check import BrowserSession
from generate_adversarial_fixtures import decode,fixture_root
from environment_fixtures import music_fixture
from run_voice_assistant_checks import deployed_source_hash
ROOT=Path(__file__).resolve().parent
async def main(url,local):
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 report={'scope':__doc__,'source_sha256':deployed_source_hash(),'cases':[],'injections':[],'passed':False,'llm':'local Nemotron Bolt E4B' if local else 'configured default'}
 meta={r['name']:r for r in json.loads((fixture_root()/'manifest.json').read_text())['cases']}
 clips={n:decode(fixture_root()/meta[n]['file']) for n in ('background','selma-followup-clear','christoph-intro','selma-intro')}
 clips['who-calm']=decode(ROOT/'logs/social-fixtures/who-calm.mp3')
 clips['wake-short']=decode(ROOT/'logs/wake-short.mp3')
 clips['sound-question']=decode(ROOT/'logs/environment-question.mp3')
 args=SimpleNamespace(url=url,threshold=.28,pause_ms=300,asr_model='parakeet_v3',seed=42,focus=False,wake=True)
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required']);page=await browser.new_page();s=BrowserSession(page,clips,meta,report,args)
  if local:await page.add_init_script('localStorage.setItem("robby.llmSelection",JSON.stringify({model:"local",provider:"local"}))')
  async def turn(label,clip,accept,expected=None):
   mark=await s.mark();await s.feed(clips[clip],record=True);await s.idle_audio(2.)
   await s.wait(f'stressEvents.slice({mark["event"]}).some(e=>e.type==="ignored"||e.type==="turn_done"||e.type==="error")',60)
   if accept:await s.settle(90)
   events=await s.events(mark);ends=[e for e in events if e['type']=='endpoint'];replies=[e for e in events if e['type']=='reply'];ignored=[e for e in events if e['type']=='ignored']
   row={'case':label,'endpoints':[{'text':e.get('transcript'),'speaker':e.get('speaker_name')} for e in ends],
    'replies':[e.get('text') for e in replies[-1:]],'ignored':[{'source':e.get('source'),'reason':e.get('reason')} for e in ignored],
    'pcm_chunks':(await page.evaluate('stressPCM.length'))-mark['pcm']}
   row['passed']=bool(ends and replies and row['pcm_chunks'] and (not expected or all(e['speaker']==expected for e in row['endpoints']))) if accept else bool(not ends and not replies and any(e['source']=='wake_mode' for e in row['ignored']))
   report['cases'].append(row);print(json.dumps(row,ensure_ascii=False),flush=True);assert row['passed'],label
  try:
   await s.open();assert await page.locator('#wake-mode').is_checked();assert not await page.locator('#environment-enabled').is_checked()
   assert await page.locator('#speech-pause').input_value()=='300';report['defaults']={'wake':True,'automatic_sounds':False,'speech_pause_ms':300}
   await turn('before-wake','who-calm',False)
   await turn('christoph-wake','wake-short',True,'Christoph')
   await turn('owner-followup','who-calm',True,'Christoph')
   await turn('foreign-no-wake','selma-followup-clear',False)
   await s.idle_audio(11.)
   await turn('expired-owner','who-calm',False)
   await turn('second-speaker-wake','selma-intro',True,'Selma')
   await turn('previous-owner-without-wake','who-calm',False)
   await page.locator('#wake-mode').uncheck();await turn('open-mode','who-calm',True)
   report['cases'][-1]['expected_speaker']='Christoph'
   report['cases'][-1]['speaker_label_correct']=all(e['speaker']=='Christoph' for e in report['cases'][-1]['endpoints'])
   # Explicit sound question still works while automatic sound reactions are off.
   mark=await s.mark();await s.feed(clips['sound-question'],record=True);await s.idle_audio(.35);await s.feed(music_fixture(4.),record=True)
   await s.wait(f'stressEvents.slice({mark["event"]}).some(e=>e.type==="turn_done")',60);await s.settle(90)
   events=await s.events(mark);captions=[e for e in events if e['type']=='sound_caption' and e.get('requested')];replies=[e for e in events if e['type']=='reply']
   row={'case':'asked-sound-automatic-off','captions':[{'kind':e.get('kind'),'caption':e.get('caption'),'requested':e.get('requested')} for e in captions],
    'reply':replies[-1]['text'] if replies else '', 'pcm_chunks':(await page.evaluate('stressPCM.length'))-mark['pcm']}
   row['passed']=bool(captions and replies and row['pcm_chunks']);report['cases'].append(row);print(json.dumps(row,ensure_ascii=False),flush=True);assert row['passed']
   report['passed']=not s.errors;report['page_errors']=s.errors
  except Exception as exc:
   report['error']=type(exc).__name__+': '+str(exc)[:500]
   report['recent_events']=await page.evaluate('stressEvents.slice(-30).filter(e=>["ignored","endpoint","reply","asr_settings","error"].includes(e.type))')
  finally:
   await page.evaluate('robo.stop()');await browser.close();report['source_unchanged']=report['source_sha256']==deployed_source_hash();report['passed'] &= report['source_unchanged']
 path=ROOT/'static'/('wake-browser-'+stamp+'.json');path.write_text(json.dumps(report,ensure_ascii=False,indent=2));print({'passed':report['passed'],'receipt':path.name,'error':report.get('error')},flush=True)
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com');parser.add_argument('--local',action='store_true');a=parser.parse_args();asyncio.run(main(a.url,a.local))
