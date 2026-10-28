"""Test native Chromium autoplay policy and manual recovery with fake devices."""
import argparse,asyncio,datetime,json
from pathlib import Path
from playwright.async_api import async_playwright
from run_voice_assistant_checks import deployed_source_hash
ROOT=Path(__file__).resolve().parent
async def main(url):
 report={'scope':__doc__,'source_sha256':deployed_source_hash(),'passed':False,'physical_devices_used':False}
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream','--autoplay-policy=document-user-activation-required'])
  page=await browser.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  try:
   await page.add_init_script("localStorage.setItem('robby.avatar-idle-enabled','false');localStorage.setItem('robby.avatar-pet-invites-enabled','false')")
   r=await page.goto(url.rstrip('/')+'/desk-robot/');assert r.status==200
   # Allow the browser's own autoplay policy to run before a test gesture.
   await asyncio.sleep(4)
   await page.wait_for_function('window.robotFace?.environmentControls&&window.robo',timeout=60000)
   visible=await page.locator('#enable-robby').is_visible();report['manual_enable_needed']=visible
   report['initial_notice']=await page.locator('#auto-start-status').inner_text()
   if visible:await page.locator('#enable-robby').click()
   await page.wait_for_function('robo.stream&&robo.ws?.readyState===1&&robo.metrics.mic_blocks>10',timeout=60000)
   report['microphone_active']=True;report['page_errors']=errors;report['passed']=not errors
  except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)[:500]
  finally:
   try:await page.evaluate('window.robo?.stop();window.robotFace?.cameraControls?.stop()')
   except Exception:pass
   await browser.close()
 report['source_unchanged']=report['source_sha256']==deployed_source_hash();report['passed'] &= report['source_unchanged']
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');path=ROOT/'static'/('auto-wake-policy-'+stamp+'.json');path.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(report,flush=True)
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com');a=parser.parse_args();asyncio.run(main(a.url))
