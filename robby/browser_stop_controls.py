"""HTTPS control persistence and real streamed-TTS cancellation, fake device.

Speech-triggered cancellation is covered separately by speaker_live.stop_test.
This checks that the browser actually releases queued audio and refuses late
packets after a long answer is cancelled. No private recordings are used.
"""
import asyncio,json,struct,time
from pathlib import Path
from playwright.async_api import async_playwright
from final_verify import runtime_signature

async def main():
    report={'passed':False,'started':time.time(),'runtime_source_sha256':runtime_signature(),'synthetic_only':True}
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page();errors=[];events=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        def frame(raw):
            if isinstance(raw,str):
                item=json.loads(raw)
                if item.get('type') in ('reply','reply_cancelled','barge','turn_done','stop_feedback_setting','error'):events.append(item)
        page.on('websocket',lambda ws:ws.on('framereceived',frame))
        await page.add_init_script("localStorage.setItem('robby.cameraAuto','false')")
        try:
            await page.goto('https://diane-asylum-understood-invoice.trycloudflare.com/desk-robot/')
            await page.wait_for_function('window.robo && !document.querySelector("#call").disabled',timeout=90000)
            assert not await page.locator('#stop-feedback').is_checked()
            assert await page.locator('#stop-feedback').is_visible()
            await page.locator('#stop-feedback').check();await page.reload()
            await page.wait_for_function('window.robo && !document.querySelector("#call").disabled',timeout=90000)
            assert await page.locator('#stop-feedback').is_checked()
            await page.locator('#message').fill('Erkläre in ungefähr hundert Wörtern, warum ein Regenbogen entsteht. Lies ruhig und sanft vor.')
            await page.locator('#form').evaluate('f=>f.requestSubmit()')
            await page.wait_for_function('robo.metrics.reply_chunks>=3 && robo.sources.length>0',timeout=90000)
            assert any(e.get('type')=='stop_feedback_setting' and e['enabled'] for e in events)
            assert len(next(e['text'] for e in events if e['type']=='reply').split())>=70
            await page.locator('#stop-feedback').uncheck();await page.wait_for_timeout(200)
            assert any(e.get('type')=='stop_feedback_setting' and not e['enabled'] for e in events)
            start=time.perf_counter();await page.locator('#interrupt').click()
            await page.wait_for_function('robo.sources.length===0',timeout=3000)
            report['button_to_empty_queue_ms']=(time.perf_counter()-start)*1000
            for _ in range(80):
                if any(e.get('type')=='reply_cancelled' for e in events):break
                await page.wait_for_timeout(50)
            cancelled=next(e['reply_id'] for e in events if e['type']=='reply_cancelled')
            await page.evaluate('''id=>{const b=new ArrayBuffer(484);new DataView(b).setInt32(0,id,true);robo.play(b)}''',cancelled)
            await page.wait_for_timeout(1500)
            assert await page.evaluate('robo.sources.length===0')
            assert not errors,errors
            report.update(passed=runtime_signature()==report['runtime_source_sha256'],events=events,page_errors=errors,
                late_cancelled_packet_ignored=True,checkbox_persisted=True)
            await page.screenshot(path='logs/stop-controls.png',full_page=True)
        finally:
            await page.evaluate('window.robo?.stop()');await browser.close()
            report['finished']=time.time();Path('logs/stop-controls-browser.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'passed':report['passed'],'button_to_empty_queue_ms':report.get('button_to_empty_queue_ms')}))

if __name__=='__main__':asyncio.run(main())
