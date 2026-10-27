"""Real HTTPS role/voice-profile replay with authored synthetic voices only.

Includes actual TTS and browser profile persistence. No voiceprint arrays,
private microphone recordings or session credentials are written to the receipt.
"""
import argparse,asyncio,datetime,json,re
from pathlib import Path
from types import SimpleNamespace
from playwright.async_api import async_playwright
from browser_adversarial_check import BrowserSession
from generate_adversarial_fixtures import decode,fixture_root
from run_voice_assistant_checks import deployed_source_hash
ROOT=Path(__file__).resolve().parent
async def main(url):
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');report={'scope':__doc__,'passed':False,'source_sha256':deployed_source_hash(),'injections':[],'turns':[]}
 meta={r['name']:r for r in json.loads((fixture_root()/'manifest.json').read_text())['cases']}
 clips={name:decode(fixture_root()/meta[name]['file']) for name in ('background','selma-followup-clear','christoph-intro','selma-intro')}
 for name in ('who-calm','who-playful'):clips[name]=decode(ROOT/'logs/social-fixtures'/f'{name}.mp3')
 args=SimpleNamespace(url=url,threshold=.28,pause_ms=700,asr_model='parakeet_v3',seed=42,focus=False)
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required']);context=await browser.new_context();page=await context.new_page()
  page.on('dialog',lambda d:asyncio.create_task(d.accept()))
  s=BrowserSession(page,clips,meta,report,args)
  async def voice(name,expected):
   mark=await s.mark();await s.feed(clips[name],record=True);await s.idle_audio(2.)
   await s.wait(f'stressEvents.slice({mark["event"]}).some(e=>e.type==="turn_done")',60);await s.settle(90)
   events=await s.events(mark);ends=[e for e in events if e['type']=='endpoint'];replies=[e for e in events if e['type']=='reply']
   row={'clip':name,'expected':expected,'transcripts':[e['transcript'] for e in ends],'names':[e.get('speaker_name') for e in ends],
    'reply':replies[-1]['text'] if replies else '', 'voice_captions':[e for e in events if e['type']=='voice_caption'],
    'identity':[e for e in events if e['type']=='identity'],'model_label':await page.locator('#recent-reply-model').inner_text(),
    'pcm_chunks':(await page.evaluate('stressPCM.length'))-mark['pcm'],'errors':[e for e in events if e['type']=='error']}
   row['passed']=bool(ends and replies and row['pcm_chunks']) and all(n==expected for n in row['names']) and not row['errors']
   if name.startswith('who-'):row['passed'] &= expected.lower() in row['reply'].lower() and not re.search(r'\bich (?:bin|heiße|heisse)(?: also)? '+expected,row['reply'],re.I)
   report['turns'].append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('identity','voice_captions')},ensure_ascii=False),flush=True)
   assert row['passed'],name
  try:
   await s.open();assert await page.locator('#environment-impressions').is_checked();assert not await page.locator('#camera-auto').is_checked();assert not await page.locator('#faceIdEnabled').is_checked();assert not await page.locator('#remember-voice-profiles').is_checked()
   await voice('christoph-intro','Christoph')
   await voice('who-calm','Christoph');await voice('who-playful','Christoph')
   assert any(e.get('vocal_traits') for row in report['turns'] for e in row['voice_captions'])
   await page.locator('#social-profiles').evaluate('e=>e.open=true');await page.locator('#remember-voice-profiles').check()
   await page.wait_for_function('JSON.parse(localStorage.getItem("robby.consentedVoiceProfiles.v1")||"null")?.profiles.length>=1',timeout=10000)
   report['saved_count']=await page.evaluate('JSON.parse(localStorage.getItem("robby.consentedVoiceProfiles.v1")).profiles.length')
   report['profile_summary']=await page.locator('#social-profile-list').inner_text()
   await page.evaluate('robo.stop()');await page.wait_for_timeout(700)
   s=BrowserSession(page,clips,meta,report,args);await s.open()
   await s.wait('stressEvents.some(e=>e.type==="identity"&&e.speakers?.some(p=>p.name==="Christoph"))',10)
   await voice('who-calm','Christoph');report['restored_voice_without_new_introduction']=True
   await voice('selma-intro','Selma');await voice('who-calm','Christoph')
   await page.locator('#social-profiles').evaluate('e=>e.open=true');await page.locator('#remember-voice-profiles').uncheck()
   assert await page.evaluate('localStorage.getItem("robby.consentedVoiceProfiles.v1")') is None
   await page.locator('#forget-voice-profiles').click();await s.wait('stressEvents.filter(e=>e.type==="identity").at(-1)?.speakers.length===0',10);report['disable_and_forget']=True
   report['passed']=not s.errors;report['page_errors']=s.errors
  except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)[:1500]
  finally:
   try:await page.evaluate('localStorage.removeItem("robby.consentedVoiceProfiles.v1");robo.stop()')
   except Exception:pass
   await browser.close();report['source_unchanged']=report['source_sha256']==deployed_source_hash();report['passed'] &= report['source_unchanged']
 path=ROOT/'static'/('social-browser-'+stamp+'.json');path.write_text(json.dumps(report,ensure_ascii=False,indent=2));print({'passed':report['passed'],'report':path.name,'error':report.get('error')},flush=True)
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com');asyncio.run(main(parser.parse_args().url))
