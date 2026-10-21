"""Actual HTTPS text -> public lookup -> sourced UI -> shared TTS test."""
import asyncio,json,time
from pathlib import Path
from playwright.async_api import async_playwright
from final_verify import runtime_signature

async def main():
    result=dict(started=time.time(),passed=False,runtime_source_sha256=runtime_signature(),physical_microphone_used=False)
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page();errors=[];events=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        def frame(raw):
            if isinstance(raw,str):
                m=json.loads(raw)
                if m.get('type') in ('reply','error','turn_done','identity'):events.append(m)
        page.on('websocket',lambda ws:ws.on('framereceived',frame))
        await page.add_init_script("localStorage.setItem('robby.cameraAuto','false')")
        try:
            await page.goto('https://diane-asylum-understood-invoice.trycloudflare.com/desk-robot/')
            await page.wait_for_function('window.robo && !document.querySelector("#call").disabled',timeout=90000)
            assert await page.locator('#open-speakers').is_checked()
            await page.locator('#open-speakers').evaluate('e=>e.click()')
            assert await page.evaluate("localStorage.getItem('robby.openSpeakers')==='false'")
            await page.reload();await page.wait_for_function('window.robo && !document.querySelector("#call").disabled',timeout=90000)
            assert not await page.locator('#open-speakers').is_checked()
            await page.locator('#message').fill('Schau bitte bei Wikipedia nach Bananen. Erzähl mir einen kurzen Fakt dazu.')
            await page.locator('#form').evaluate('f=>f.requestSubmit()')
            await page.wait_for_function('document.querySelector(".reply-sources a") && robo.metrics.reply_chunks>0',timeout=90000)
            links=await page.locator('.reply-sources a').evaluate_all('a=>a.map(x=>({text:x.textContent,url:x.href,rel:x.rel}))')
            assert any('wikipedia.org/wiki/' in a['url'] for a in links),links
            assert all('noopener' in a['rel'] and a['url'].startswith('https://') for a in links)
            assert not errors,errors
            await page.locator('#open-speakers').evaluate('e=>e.click()')
            await page.wait_for_timeout(500)
            assert any(e.get('type')=='identity' and e.get('open_enrollment') is True for e in events)
            result.update(links=links,events=events,page_errors=errors,metrics=await page.evaluate('robo.metrics'),
                passed=runtime_signature()==result['runtime_source_sha256'])
            await page.screenshot(path='logs/knowledge-ui.png',full_page=True)
        finally:
            await page.evaluate('window.robo?.stop()');await browser.close()
            result['finished']=time.time();Path('logs/knowledge-browser.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({'passed':result['passed'],'links':len(result.get('links',[]))}))

if __name__=='__main__':asyncio.run(main())
