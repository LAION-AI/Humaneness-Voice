"""Real HTTPS UI status regression; injected synthetic messages, no ASR/LLM call."""
import argparse,asyncio,json,time
from pathlib import Path
from playwright.async_api import async_playwright

async def main(url):
    report={'started':time.time(),'synthetic_messages_only':True,'physical_microphone_used':False,'passed':False,'cases':[]}
    target=Path(__file__).parent/'logs/browser-ui-status.json'
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
        page=await browser.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script("localStorage.setItem('robby.cameraAuto','false')")
        async def send(value):await page.evaluate('(value)=>window.robo.on.message(value)',value)
        try:
            await page.goto(url);await page.wait_for_function('window.robo&&!document.querySelector("#call").disabled',timeout=90000)
            await page.locator('#recognition-settings > summary').click()
            await page.locator('.interaction-controls details > summary').click()
            assert await page.locator('#allow-proactive').is_checked()
            assert await page.locator('#allow-proactive').is_visible()
            assert await page.locator('#allow-proactive').evaluate('el=>el.getBoundingClientRect().top')<500
            await page.evaluate("window.testInteractionCommands=[];window.robo.send=s=>window.testInteractionCommands.push(s)")
            await page.locator('#allow-proactive').uncheck()
            assert await page.evaluate("window.testInteractionCommands.includes('interaction:off')")
            assert 'still interrupt' in await page.locator('#interaction-status').text_content()
            await page.locator('#allow-proactive').check()
            assert await page.evaluate("window.testInteractionCommands.includes('interaction:on')")
            await send({'type':'interaction','enabled':True,'rules':[{'condition':'Talks about bananas','reply':'Stopp, stopp!'}],
                        'decision_interval_ms':320,'decision_model':'Synthetic test controller'})
            assert 'bananas' in await page.locator('#interaction-rules').text_content()
            assert '320 ms' in await page.locator('#interaction-status').text_content()
            await page.locator('#clear-interaction-rules').click()
            assert await page.evaluate("window.testInteractionCommands.includes('interaction:clear')")
            report['cases'].append('Prominent default-on proactive checkbox, safe rule display, on/off and clear commands; user stops remain available')
            assert await page.locator('#identity-model').input_value()=='v4-long3'
            assert await page.locator('#utterance-strictness').input_value()=='0.32'
            assert await page.locator('#utterance-strictness').is_visible()
            report['cases'].append('Defaults: 3.04-s context and visible 0.32 full-turn cosine')
            await page.locator('#utterance-strictness').evaluate("el=>{el.value='.40';el.dispatchEvent(new Event('input'));el.dispatchEvent(new Event('change'));}")
            assert await page.locator('#utterance-threshold-value').text_content()=='0.40'
            await page.reload();await page.wait_for_function('window.robo&&!document.querySelector("#call").disabled')
            await page.locator('#recognition-settings > summary').click()
            assert await page.locator('#utterance-strictness').input_value()=='0.4'
            await send({'type':'identity','verification':{'reference_scores':[{'name':'Synthetic user','cosine':.46}], 'threshold':.40,'view_cosine':.91}})
            assert 'voice match 0.46 / required 0.40' in await page.locator('#speaker-match-feedback').text_content()
            await page.get_by_role('button',name='Reset voice matching',exact=True).click()
            assert await page.locator('#utterance-strictness').input_value()=='0.32'
            report['cases'].append('Slider persists across reload, score is visible, reset restores 0.32')
            await send({'type':'endpoint','transcript':'Synthetic foreground request.','speaker_name':'Test user'})
            await send({'type':'ignored','reason':'Synthetic background rejection.'})
            await send({'type':'asr_preview','text':'Unrelated background words.','source':'native_asr','status':'not_submitted'})
            assert await page.locator('#phase-text').text_content()=='Thinking of a reply'
            assert await page.locator('#asr-preview').text_content()=='Synthetic foreground request.'
            report['cases'].append('Background rejection/preview cannot overwrite an accepted pending turn')
            await send({'type':'barge'});await send({'type':'ignored','reason':'New uncertain foreground.'})
            assert await page.locator('#phase-text').text_content()=='Waiting for an enrolled speaker'
            report['cases'].append('Interruption restores normal uncertainty feedback')
            await send({'type':'endpoint','transcript':'A synthetic stop test.'})
            await send({'type':'stop_command','speaker_name':'Test user'})
            assert await page.locator('#phase-text').text_content()=='Paused · I’m listening'
            assert 'No new spoken reply.' in await page.locator('#asr-status').text_content()
            report['cases'].append('Verified stop clears pending UI state without a new spoken reply')
            await send({'type':'endpoint','transcript':'Another synthetic request.'})
            await send({'type':'error','error':'Synthetic test error'})
            await send({'type':'asr_preview','text':'New recognized speech.','source':'native_asr','status':'not_submitted'})
            assert await page.locator('#asr-preview').text_content()=='New recognized speech.'
            assert not errors,errors
            report['cases'].append('Error clears pending state; later recognition remains visible')
            report.update(passed=True,finished=time.time(),page_errors=errors)
        finally:
            target.write_text(json.dumps(report,indent=2));await browser.close()
    print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com/desk-robot/')
    asyncio.run(main(p.parse_args().url))
