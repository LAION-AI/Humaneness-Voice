"""Additional real HTTPS checks: persisted choice, model changes, mobile layout."""
import asyncio,json,time
from pathlib import Path
from playwright.async_api import async_playwright

ROOT=Path(__file__).resolve().parent
URL='https://arrive-locations-parade-research.trycloudflare.com'
async def main():
    report={'passed':False,'started':time.time(),'physical_microphone':False};errors=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page(viewport={'width':390,'height':844});page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script("localStorage.setItem('robby.llmSelection',JSON.stringify({model:'local',provider:'local'}));localStorage.setItem('robby.cameraAuto','false');localStorage.setItem('robby.avatar-idle-enabled','false');localStorage.setItem('robby.avatar-pet-invites-enabled','false')")
        try:
            await page.goto(URL+'/desk-robot/')
            await page.wait_for_function('window.robotFace?.asrControls && !document.querySelector("#call").disabled',timeout=60000)
            await page.evaluate("document.querySelector('#model-settings')?.setAttribute('open','')")
            box=await page.locator('#asr-model').bounding_box()
            assert box and box['x']>=0 and box['x']+box['width']<=391,box
            report['mobile_select_fits']=True
            await page.evaluate("document.querySelector('#model-settings')?.setAttribute('open','')")
            await page.select_option('#asr-model','whisper_turbo')
            await page.wait_for_function('robotFace.asrControls.active==="whisper_turbo"&&!document.querySelector("#asr-model-status").textContent.startsWith("Loading")',timeout=20000)
            assert await page.evaluate("localStorage.getItem('robby.asrModel')")=='whisper_turbo'
            await page.evaluate('robo.stop()');await page.reload()
            await page.wait_for_function('window.robotFace?.asrControls && !document.querySelector("#call").disabled',timeout=60000)
            assert await page.locator('#asr-model').input_value()=='whisper_turbo'
            report['reload_persists']=True
            await page.evaluate("document.querySelector('#model-settings')?.setAttribute('open','')")
            await page.select_option('#asr-model','parakeet_v3')
            await page.wait_for_function('robotFace.asrControls.active==="parakeet_v3"&&!document.querySelector("#asr-model-status").textContent.startsWith("Loading")',timeout=20000)
            await page.evaluate("document.querySelector('#model-settings')?.setAttribute('open','')")
            await page.select_option('#asr-model','nemotron')
            await page.wait_for_function('robotFace.asrControls.active==="nemotron"&&!document.querySelector("#asr-model-status").textContent.startsWith("Loading")',timeout=20000)
            report['repeat_switches']=True
            assert not errors,errors;report['passed']=True
        except Exception as exc:report['error']=str(exc)
        finally:
            report['errors']=errors
            try:await page.evaluate('window.robo?.stop()')
            except Exception:pass
            await browser.close()
    report['finished']=time.time();(ROOT/'static/asr-controls-browser.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':asyncio.run(main())
