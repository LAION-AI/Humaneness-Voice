"""End-to-end HTTPS lab test with Chromium fake camera and synthetic microphone."""
import argparse,asyncio,json,datetime
from pathlib import Path
from playwright.async_api import async_playwright
from run_voice_assistant_checks import deployed_source_hash
ROOT=Path(__file__).resolve().parent
async def main(url):
 errors=[];report={'passed':False,'scope':__doc__,'source_sha256':deployed_source_hash()}
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream','--use-file-for-fake-audio-capture='+str(ROOT/'.runtime/lfm3-tests/synthetic.wav'),'--autoplay-policy=no-user-gesture-required'])
  page=await browser.new_page(viewport={'width':1200,'height':900});page.on('pageerror',lambda e:errors.append(str(e)))
  await page.add_init_script("localStorage.setItem('robby.autoStartDefaultVersion','1');localStorage.setItem('robby.autoStart','false');localStorage.setItem('robby.cameraAutoVersion','3');localStorage.setItem('robby.cameraAuto','false');localStorage.setItem('robby.memoryDefaultVersion','1');localStorage.setItem('robby.memorySettings',JSON.stringify({enabled:false}));localStorage.setItem('robby.avatar-idle-enabled','false');")
  try:
   response=await page.goto(url.rstrip('/')+'/desk-robot/');assert response.status==200
   await page.wait_for_selector('#lfm-open',state='attached',timeout=45000);await page.locator('#lfm-open').evaluate('e=>e.closest("details").open=true');await page.click('#lfm-open')
   assert await page.locator('#lfm-win').input_value()=='3';assert await page.locator('#lfm-stride').input_value()=='1'
   await page.locator('#lfm-win').evaluate("e=>{e.value='2.5';e.dispatchEvent(new Event('input'))}")
   await page.locator('#lfm-stride').evaluate("e=>{e.value='0.5';e.dispatchEvent(new Event('input'))}")
   await page.click('#lfm-record');await page.wait_for_function('document.querySelector("#lfm-status").textContent.includes("Recording locally")',timeout=20000)
   await page.wait_for_timeout(5400);await page.click('#lfm-stop');await page.wait_for_function('!document.querySelector("#lfm-analyze").disabled',timeout=20000)
   await page.click('#lfm-analyze');await page.wait_for_function('document.querySelector("#lfm-status").textContent.includes("windows analyzed")||document.querySelector("#lfm-status").textContent.includes("Analysis failed")',timeout=180000)
   status=await page.locator('#lfm-status').inner_text();report['status']=status;assert 'windows analyzed' in status,status
   report['windows']=await page.locator('.lfm-card').count();assert report['windows']>=6
   report['ui_fields']=await page.locator('.lfm-card').first.inner_text();assert 'Audio' in report['ui_fields'] and 'Video' in report['ui_fields'] and 'tokens/s' in report['ui_fields']
   await page.locator('.lfm-card video').first.evaluate('(v)=>v.play()');await page.wait_for_function('document.querySelectorAll(".lfm-card video")[0].currentTime>0',timeout=10000);report['playback_time']=await page.locator('.lfm-card video').first.evaluate('v=>v.currentTime')
   await page.locator('.lfm-card video').nth(1).evaluate('(v)=>v.play()');await page.wait_for_function('document.querySelectorAll(".lfm-card video")[1].currentTime>=0.5',timeout=10000);report['second_window_playback_time']=await page.locator('.lfm-card video').nth(1).evaluate('v=>v.currentTime');assert report['second_window_playback_time']<3
   await page.locator('#lfm-close').click();assert not await page.locator('.lfm-dialog').evaluate('d=>d.open');report['passed']=not errors
  except Exception as e:
   report['error']=type(e).__name__+': '+str(e)[:700]
   try:report['failure_status']=await page.locator('#lfm-status').inner_text(timeout=1000)
   except Exception:pass
  finally:await browser.close()
 report['page_errors']=errors;report['source_unchanged']=report['source_sha256']==deployed_source_hash();report['passed'] &= report['source_unchanged']
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');(ROOT/'static'/('lfm3-browser-'+stamp+'.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps(report,ensure_ascii=False),flush=True)
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com');args=parser.parse_args();asyncio.run(main(args.url))
