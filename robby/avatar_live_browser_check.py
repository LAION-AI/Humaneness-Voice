"""Real UI -> configured LLM tool -> TTS PCM -> playback-triggered motion."""
import argparse,asyncio,json,time
from pathlib import Path
from playwright.async_api import async_playwright
ROOT=Path(__file__).resolve().parent

async def main(url):
    report={'passed':False,'url':url,'typed_synthetic_request':True,'real_llm_and_tts':True}
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page(viewport={'width':1280,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script("localStorage.setItem('robby.cameraAuto','false');localStorage.setItem('robby.avatar-actions-enabled','true')")
        try:
            await page.goto(url+'/desk-robot/');await page.wait_for_function('window.robo && window.robotFace?.actions')
            await page.wait_for_function('!document.getElementById("call").disabled',timeout=45000)
            await page.evaluate('''()=>{window.avatarEvents=[];window.avatarObserved=[];const old=robo.on.message;robo.on.message=m=>{avatarEvents.push(m);old(m)};
              window.avatarTimer=setInterval(()=>{const a=robotFace.actions.active,s=robotFace.rig.snapshot();if(a)avatarObserved.push({kind:a.kind,source:a.source,pan:s.visualPan,replyChunks:robo.metrics.reply_chunks})},50)}''')
            await page.fill('#message','Drehe deinen Kopf bitte einmal um 360 Grad gegen den Uhrzeigersinn. Sage dazu nur: Schau mal!')
            await page.locator('#form').evaluate('(e)=>e.requestSubmit()')
            await page.wait_for_function("avatarEvents.some(m=>m.type==='reply'&&m.avatar_actions?.some(s=>s.kind==='spin'))",timeout=65000)
            await page.wait_for_function("avatarObserved.some(a=>a.source==='reply'&&a.kind==='spin'&&Math.abs(a.pan)>90&&a.replyChunks>0)",timeout=35000)
            await page.wait_for_function("avatarObserved.length>10&&!robotFace.actions.active&&Math.abs(robotFace.rig.snapshot().visualPan-robotFace.rig.snapshot().pan)<.01",timeout=15000)
            report.update(await page.evaluate('''()=>({replies:avatarEvents.filter(m=>m.type==='reply').map(m=>({text:m.text,actions:m.avatar_actions})),observed_frames:avatarObserved.length,max_visual_pan:Math.max(...avatarObserved.map(a=>Math.abs(a.pan))),actual_pcm_chunks:robo.metrics.reply_chunks,restored:Math.abs(robotFace.rig.snapshot().visualPan-robotFace.rig.snapshot().pan)<.01})'''))
            assert report['restored'];assert not errors,errors;report.update(passed=True,errors=errors,finished=time.time())
        except Exception as exc:
            report.update(error=str(exc),errors=errors)
            try:report['events']=await page.evaluate('window.avatarEvents||[]')
            except Exception:pass
        finally:
            try:await page.evaluate('clearInterval(window.avatarTimer);window.robo?.stop()')
            except Exception:pass
            await browser.close();(ROOT/'logs/avatar-live-browser.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com');asyncio.run(main(p.parse_args().url.rstrip('/')))
