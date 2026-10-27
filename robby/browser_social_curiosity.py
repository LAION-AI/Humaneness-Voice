"""Real foreground introduction followed by repeated quiet synthetic background speech."""
import argparse,asyncio,json,datetime,re
from pathlib import Path
from types import SimpleNamespace
from playwright.async_api import async_playwright
from browser_adversarial_check import BrowserSession
from generate_adversarial_fixtures import fixture_root,decode
from run_voice_assistant_checks import deployed_source_hash
ROOT=Path(__file__).resolve().parent
async def main(url):
 report={'scope':__doc__,'passed':False,'source_sha256':deployed_source_hash(),'injections':[]}
 meta={r['name']:r for r in json.loads((fixture_root()/'manifest.json').read_text())['cases']};clips={n:decode(fixture_root()/meta[n]['file']) for n in ('background','selma-followup-clear','christoph-intro')}
 args=SimpleNamespace(url=url,threshold=.28,pause_ms=700,asr_model='parakeet_v3',seed=42,focus=False)
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required']);page=await browser.new_page();s=BrowserSession(page,clips,meta,report,args)
  try:
   await s.open();await s.feed(clips['christoph-intro']);await s.wait('stressEvents.some(e=>e.type==="turn_done")',40);await s.settle()
   mark=await s.mark()
   for _ in range(2):await s.feed(clips['background']*.1,record=True);await s.idle_audio(2.)
   await s.wait('stressEvents.slice('+str(mark['event'])+').some(e=>e.type==="environment_observation"&&e.kind==="unfamiliar_voice"&&e.status==="responding")',30)
   await s.settle();events=await s.events(mark)
   replies=[e for e in events if e['type']=='reply'];report['events']=events;report['reply']=replies[-1]['text'] if replies else ''
   report['pcm_chunks']=await page.evaluate('stressPCM.length')-mark['pcm']
   report['background_not_enrolled']=not any(e['type']=='endpoint' for e in events)
   report['routing_passed']=bool(replies and report['pcm_chunks'] and report['background_not_enrolled']) and not s.errors
   report['semantic_question_check']=bool('?' in report['reply'] and re.search(r'stimme|fernseh|jemand|\bwer\b|sprech|\btv\b|voice|someone|\bwho\b|television',report['reply'],re.I))
   report['passed']=report['routing_passed'] and report['semantic_question_check']
  except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)[:500];report['events']=await page.evaluate('window.stressEvents||[]')
  finally:
   await page.evaluate('window.robo?.stop()');await browser.close()
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');path=ROOT/'static'/('social-curiosity-'+stamp+'.json');path.write_text(json.dumps(report,ensure_ascii=False,indent=2));print({k:v for k,v in report.items() if k not in ('events','injections')})
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com');asyncio.run(main(parser.parse_args().url))
