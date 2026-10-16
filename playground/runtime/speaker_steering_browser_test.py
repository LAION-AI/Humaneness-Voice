"""Read-only HTTPS UI audit: capture requests without generating GPU audio.

Run with the Playwright-enabled Python after deployment. A fresh isolated
browser session is used; /api/turn is intercepted and returns a diagnostic
error, never reaching the TTS server. No upload or existing user data changed.
"""
import json
import sys

from playwright.sync_api import sync_playwright


def main(url):
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,
            args=['--no-sandbox'])
        context=browser.new_context(viewport={'width':1440,'height':1000})
        context.add_init_script('localStorage.clear();')
        page=context.new_page();errors=[];requests=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        def intercept(route):
            requests.append(route.request.post_data_json)
            route.fulfill(status=200,content_type='text/event-stream',body=
                'data: '+json.dumps({'event':'error','message':'Read-only browser test: no generation submitted.'})+'\n\n')
        page.route('**/api/turn',intercept)
        page.goto(url,wait_until='domcontentloaded',timeout=60000)
        page.wait_for_function('ready && !running',timeout=240000)
        page.wait_for_function('globalThis.smallVoiceCatalog && globalThis.smallSteeringCatalog',timeout=30000)
        page.locator('#speaker-steering').wait_for(state='visible')
        assert page.locator('#vc-precision').input_value()=='bf16'
        assert page.evaluate('buildTurnRequest("Hello").vc_precision')=='bf16'
        assert page.locator('#speaker-steering').is_checked()
        catalog=page.evaluate('globalThis.smallSteeringCatalog')
        voices=page.evaluate('Object.keys(globalThis.smallVoiceCatalog)')
        assert len(voices)==10
        assert set(voices)<=set(catalog),'Missing catalog steering status after deployment'
        for voice in voices:
            page.select_option('#voice-choice',voice)
            payload=page.evaluate('buildTurnRequest("Please say hello.")')
            assert payload['speaker_steering'] is True and payload['voice_profile']==voice
            assert 'Layer 16' in page.locator('#speaker-steering-status').inner_text()
        page.locator('#speaker-steering').uncheck()
        payload=page.evaluate('buildTurnRequest("Please say hello.")')
        assert payload['speaker_steering'] is False
        assert page.locator('#speaker-steering-status').inner_text().startswith('Aus:')
        page.locator('#recommended').click()
        assert page.locator('#speaker-steering').is_checked()
        page.select_option('#voice-choice','none')
        assert 'kein Sprecher-Vektor' in page.locator('#speaker-steering-status').inner_text()
        payload=page.evaluate('buildTurnRequest("Please say hello.")')
        assert payload['speaker_steering'] and payload['voice_profile'] is None
        page.select_option('#voice-choice','custom')
        assert 'kein Katalog-Vektor' in page.locator('#speaker-steering-status').inner_text()
        page.select_option('#voice-choice',voices[0])
        page.locator('#vc').check()
        page.select_option('#vc-precision','fp32')
        payload=page.evaluate('buildTurnRequest("Hello")')
        assert payload['voice_conversion'] and payload['vc_precision']=='fp32'
        page.select_option('#vc-precision','bf16')
        page.locator('#vc').uncheck()
        page.locator('[data-lora-preset="dpo"]').click()
        assert 'Mit LoRA nicht aktiv' in page.locator('#speaker-steering-status').inner_text()
        page.locator('[data-lora-preset="base"]').click()
        modes=['single','best','all','search_winner','search_beam']
        for mode in modes:
            page.select_option('#generation',mode)
            for streaming in ['offline','streaming']:
                if not mode.startswith('search_'):
                    page.select_option('#streaming',streaming)
                payload=page.evaluate('buildTurnRequest("Please say hello.")')
                assert payload['generation']==mode and payload['speaker_steering'] is True
        page.select_option('#generation','single')
        page.select_option('#inputmode','manual')
        page.locator('#q').fill('CAPTION: Calm conversation.\nTRANSCRIPT: "Hello."')
        page.wait_for_function('!document.getElementById("send").disabled',timeout=30000)
        page.locator('#send').click()
        page.wait_for_function('document.querySelector(".error")?.textContent.includes("Read-only browser test")',timeout=10000)
        assert len(requests)==1 and requests[0]['speaker_steering'] is True
        mobile=[]
        for width in (390,320):
            page.set_viewport_size({'width':width,'height':900})
            page.locator('#speaker-steering').scroll_into_view_if_needed()
            assert page.locator('#speaker-steering').is_visible()
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            mobile.append(width)
        assert not errors,errors
        print(json.dumps({'url':url,'voices':len(voices),'default_on':True,
            'explicit_opt_out':True,'recommended_restores_on':True,
            'no_ref_custom_lora_statuses':True,'generation_modes':modes,
            'streaming_and_offline_payloads':True,'turns_intercepted_no_gpu':len(requests),
            'mobile_widths':mobile,'page_errors':errors},indent=2))
        browser.close()


if __name__=='__main__':
    main(sys.argv[1] if len(sys.argv)>1 else 'https://switch-surprised-clock-sparc.trycloudflare.com/')
