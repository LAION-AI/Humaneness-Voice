"""One actual HTTPS typed Luna/TTS turn; measures PCM packet duration and
browser scheduling underflows, not subjective sound quality or human speech."""
import asyncio,json,time
from pathlib import Path
from playwright.async_api import async_playwright
from final_verify import runtime_signature

async def main():
    report={'passed':False,'runtime_source_sha256':runtime_signature(),'started':time.time(),'physical_audio_device':False}
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,
            args=['--no-sandbox','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script("localStorage.setItem('robby.cameraAuto','false')")
        try:
            await page.goto('https://diane-asylum-understood-invoice.trycloudflare.com/desk-robot/')
            await page.wait_for_function('window.robo && !document.querySelector("#call").disabled',timeout=90000)
            await page.evaluate('''()=>{
              window.chunkProbe=[];window.chunkTurnDone=false;
              const play=robo.play.bind(robo),message=robo.on.message;
              robo.play=raw=>{const id=new DataView(raw).getInt32(0,true);if(id)chunkProbe.push({
                seconds:(raw.byteLength-4)/2/24000,arrival_ms:performance.now(),
                gap_ms:chunkProbe.length?Math.max(0,robo.ctx.currentTime-robo.next)*1000:0});play(raw);};
              robo.on.message=m=>{if(m.type==='turn_done')chunkTurnDone=true;message(m);};
            }''')
            await page.locator('#message').fill('Bitte lies genau diese zwei Sätze vor: Heute machen wir einen kleinen Spaziergang durch den Park. Danach setzen wir uns auf eine Bank und genießen ganz in Ruhe die warme Nachmittagssonne.')
            await page.locator('#form').evaluate('form=>form.requestSubmit()')
            await page.wait_for_function('window.chunkTurnDone && chunkProbe.length>=2',timeout=90000)
            chunks=await page.evaluate('chunkProbe')
            assert any(1.5<c['seconds']<1.7 for c in chunks),chunks
            assert not errors,errors
            report.update(chunks=chunks,scheduled_gap_count=sum(c['gap_ms']>20 for c in chunks),
                max_scheduled_gap_ms=max(c['gap_ms'] for c in chunks),page_errors=errors,
                passed=report['runtime_source_sha256']==runtime_signature())
        finally:
            await page.evaluate('window.robo?.stop()');await browser.close()
            report['finished']=time.time();Path('logs/tts-chunk-checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))

if __name__=='__main__':asyncio.run(main())
