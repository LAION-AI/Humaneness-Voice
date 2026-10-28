"""Explicit sound questions, automation off, authored speech and music via HTTPS."""
import argparse,asyncio,datetime,json,re
from pathlib import Path
from types import SimpleNamespace
from playwright.async_api import async_playwright
from browser_adversarial_check import BrowserSession
from generate_adversarial_fixtures import decode,fixture_root
from environment_fixtures import music_fixture
from run_voice_assistant_checks import deployed_source_hash
ROOT=Path(__file__).resolve().parent
async def main(url,local):
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');report={'scope':__doc__,'source_sha256':deployed_source_hash(),'passed':False,'cases':[],'injections':[],'llm':'local Nemotron Bolt E4B' if local else 'configured default'}
 meta={r['name']:r for r in json.loads((fixture_root()/'manifest.json').read_text())['cases']}
 clips={n:decode(fixture_root()/meta[n]['file']) for n in ('background','selma-followup-clear','christoph-intro')}
 clips['wake']=decode(ROOT/'logs/wake-short.mp3');clips['question']=decode(ROOT/'logs/environment-question.mp3')
 args=SimpleNamespace(url=url,threshold=.28,pause_ms=300,asr_model='parakeet_v3',seed=42,focus=False,wake=True)
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required']);page=await browser.new_page();s=BrowserSession(page,clips,meta,report,args)
  if local:await page.add_init_script('localStorage.setItem("robby.llmSelection",JSON.stringify({model:"local",provider:"local"}))')
  try:
   await s.open();assert not await page.locator('#environment-enabled').is_checked()
   mark=await s.mark();await s.feed(clips['wake']);await s.wait('stressEvents.slice('+str(mark['event'])+').some(e=>e.type==="turn_done")',60);await s.settle()
   for mode in ('after','before'):
    # The same wake speaker follows after Robby's reply; the playback end
    # renews the ten-second window even when the answer is lengthy.
    assert await page.locator('#wake-mode').is_checked()
    mark=await s.mark()
    if mode=='before':await s.feed(music_fixture(2.5),record=True);await s.idle_audio(.15)
    await s.feed(clips['question'],record=True)
    if mode=='after':await s.idle_audio(.35);await s.feed(music_fixture(4.),record=True)
    await s.idle_audio(3.)
    await s.wait('stressEvents.slice('+str(mark['event'])+').some(e=>e.type==="turn_done"||e.type==="error")',60);await s.settle(90)
    events=await s.events(mark);caps=[e for e in events if e['type']=='sound_caption' and e.get('requested')];ends=[e for e in events if e['type']=='endpoint'];replies=[e for e in events if e['type']=='reply']
    row={'case':mode,'captions':[{'kind':x.get('kind'),'caption':x.get('caption')} for x in caps],
     'transcripts':[x.get('transcript') for x in ends], 'reply':replies[-1].get('text') if replies else '', 'pcm_chunks':(await page.evaluate('stressPCM.length'))-mark['pcm']}
    row['passed']=bool(caps and replies and row['pcm_chunks'] and any(x.get('kind')=='music' for x in caps)
      and re.search(r'musik|melodi|lied|klavier|synth|keyboard|musikstück|music|melody|instrument',row['reply'],re.I))
    report['cases'].append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    if not row['passed']:raise AssertionError(mode)
   report['passed']=not s.errors;report['page_errors']=s.errors
  except Exception as exc:
   report['error']=type(exc).__name__+': '+str(exc)[:500]
   report['events']=await page.evaluate('stressEvents.slice(-80).filter(e=>["endpoint","reply","ignored","error","sound_caption","environment_observation"].includes(e.type))')
  finally:
   await page.evaluate('robo.stop()');await browser.close();report['source_unchanged']=report['source_sha256']==deployed_source_hash();report['passed'] &= report['source_unchanged']
 path=ROOT/'static'/('wake-sound-'+stamp+'.json');path.write_text(json.dumps(report,ensure_ascii=False,indent=2));print({'passed':report['passed'],'receipt':path.name,'error':report.get('error')},flush=True)
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com');parser.add_argument('--local',action='store_true');a=parser.parse_args();asyncio.run(main(a.url,a.local))
