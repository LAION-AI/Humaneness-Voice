"""HTTPS memory-informed FIRST reaction after reconnect with zero recent history."""
import argparse,asyncio,datetime,fcntl,json
from pathlib import Path
from playwright.async_api import async_playwright
from run_voice_assistant_checks import deployed_source_hash,public_state
ROOT=Path(__file__).resolve().parent
async def main(args):
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 report={'passed':False,'scope':__doc__,'source_sha256':deployed_source_hash(),'turns':[]};errors=[]
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
  page=await browser.new_page(viewport={'width':1366,'height':950});page.on('pageerror',lambda e:errors.append(str(e)))
  page.on('dialog',lambda d:asyncio.create_task(d.accept()))
  await page.add_init_script("localStorage.setItem('robby.autoStart','false');localStorage.setItem('robby.environment','false');")
  await page.add_init_script("localStorage.setItem('robby.quickReplies','false');localStorage.setItem('robby.avatar-idle-enabled','false');localStorage.setItem('robby.avatar-pet-invites-enabled','false');")
  async def open_page():
   await page.goto(args.url+'/desk-robot/');await page.wait_for_function('window.robotFace?.memoryControls&&!document.querySelector("#call").disabled',timeout=60000)
   await page.locator('#memory-settings').evaluate('e=>e.open=true')
   await page.evaluate('''()=>{window.memEvents=[];const old=robo.on.message;robo.on.message=m=>{memEvents.push(m);old(m)};}''')
  async def turn(text):
   mark=await page.evaluate('memEvents.length');await page.fill('#message',text);await page.locator('#form').evaluate('f=>f.requestSubmit()')
   await page.wait_for_function('(n)=>memEvents.slice(n).some(e=>e.type==="turn_done"||e.type==="error")',arg=mark,timeout=90000)
   await page.wait_for_function('(n)=>memEvents.slice(n).some(e=>e.type==="memory_status"&&(e.state==="error"||(e.state==="ready"&&e.updated!==undefined)))',arg=mark,timeout=30000)
   await page.wait_for_function('!robo.sources.length',timeout=90000)
   ev=await page.evaluate('(n)=>memEvents.slice(n)',mark);replies=[e for e in ev if e['type']=='reply']
   row={'user':text,'reply':replies[-1]['text'] if replies else '', 'memory_events':[e for e in ev if e['type'].startswith('memory_')],
        'reactions':[e for e in ev if e['type']=='reply' and e.get('sentence_role')=='reaction'],'events':ev,
        'errors':[e for e in ev if e['type']=='error'],'pcm_chunks':await page.evaluate('robo.metrics.chunks')}
   report['turns'].append(row);assert replies and not row['errors'],row
   assert row['memory_events'][-1]['state']=='ready',row
   return row
  try:
   await open_page();assert not await page.locator('#memory-enabled').is_checked();report['default_off']=True
   assert await page.locator('#memory-knowledge-budget').input_value()=='10000'
   assert await page.locator('#memory-history-budget').input_value()=='5000'
   await page.locator('#memory-history-budget').evaluate('e=>{e.value=0;e.dispatchEvent(new Event("input"));e.dispatchEvent(new Event("change"));}')
   await page.locator('#memory-enabled').check()
   await page.wait_for_function('memEvents.some(e=>e.type==="memory_status"&&e.state==="ready"&&e.settings.enabled)',timeout=20000)
   await page.evaluate('robo.ctx.resume()');assert await page.evaluate('robo.ctx.state')=='running'
   await turn('Hallo, ich bin Bob. Meine Freundin heißt Alice und ich wohne in Bremen. Alice mag Jazz. Bitte antworte nur: Danke.')
   await page.click('#memory-browse');await page.wait_for_function('memEvents.some(e=>e.type==="memory_entries"&&e.total>=2)',timeout=10000)
   titles=await page.locator('#memory-entries').inner_text();assert 'Bob' in titles and 'Alice' in titles;report['explorer']=titles
   await page.locator('#memory-entries button').first.click();await page.wait_for_function('!document.querySelector("#memory-entry").hidden')
   report['evidence_visible']=await page.locator('#memory-entry').inner_text();assert 'evidence' in report['evidence_visible']
   await page.screenshot(path=str(ROOT/'logs/two-stage-memory-explorer.png'),full_page=True)
   await page.evaluate('robo.stop()');await page.wait_for_timeout(500);await open_page()
   assert await page.locator('#memory-enabled').is_checked() and await page.locator('#memory-history-budget').input_value()=='0'
   await page.click('#message');await page.evaluate('robo.start(false)');await page.evaluate('robo.ctx.resume()')
   await page.wait_for_function('memEvents.some(e=>e.type==="memory_status"&&e.state==="ready"&&e.entries>=2)',timeout=15000)
   await page.locator('#model-settings').evaluate('e=>e.open=true')
   await page.locator('#quick-replies').check()
   row=await turn('Ich bin Bob. Weißt du noch, wie meine Freundin heißt?')
   assert row['reactions'],row['reply']
   assert 'alice' in row['reactions'][0]['sentence_text'].casefold(),row['reactions'][0]['sentence_text']
   assert row['reactions'][0]['timing']['quick_reply_memory_entries']>=1
   report['recall_without_recent_history']=True
   await page.locator('#memory-enabled').uncheck();await page.wait_for_function('memEvents.at(-1)?.type==="memory_status"&&memEvents.at(-1).state==="off"',timeout=10000)
   report['disable_verified']=True
   await page.click('#memory-reset');await page.wait_for_function('memEvents.some(e=>e.type==="memory_status"&&e.reset&&e.entries===0)',timeout=10000)
   await page.click('#memory-browse');await page.wait_for_function('memEvents.filter(e=>e.type==="memory_entries").at(-1)?.total===0',timeout=10000)
   report['reset_verified']=True
   await page.evaluate('robo.stop()');await page.wait_for_timeout(500);await open_page()
   assert not await page.locator('#memory-enabled').is_checked();report['off_persists']=True
   await page.set_viewport_size({'width':390,'height':844});assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth+1');report['mobile_fits']=True
   report['passed']=not errors
  except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)[:1500]
  finally:
   try:report["all_events"]=await page.evaluate("memEvents")
   except Exception:pass
   try:
    # Never retain the authored test wiki on the public demo after this check.
    await page.evaluate('window.robotFace?.memoryControls?.command("reset")');await page.wait_for_timeout(500);await page.evaluate('window.robo?.stop()')
   except Exception:pass
   await browser.close();report['errors']=errors;report['source_unchanged']=report['source_sha256']==deployed_source_hash();report['passed'] &= report['source_unchanged']
   raw=json.dumps(report,indent=2,ensure_ascii=False);(ROOT/'logs'/('two-stage-memory-browser-'+stamp+'.json')).write_text(raw);(ROOT/'static/two-stage-memory-browser.json').write_text(raw)
 print(json.dumps({'passed':report['passed'],'error':report.get('error'),'turns':len(report['turns'])}),flush=True)
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com');a=parser.parse_args()
 lock=(ROOT/'logs/voice-assistant-runs/.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);asyncio.run(main(a))
