"""Real HTTPS UI + six fixed-reference GPU smokes; not a quality experiment."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
URL = 'https://switch-surprised-clock-sparc.trycloudflare.com'
IDENTITY = "Use the same speaker as the reference audio. Preserve that speaker's voice identity."


def main():
    report = {'time': datetime.now(timezone.utc).isoformat(), 'url': URL,
        'scope': 'Real UI/GPU integration, not a perceptual quality or speaker study', 'cases': []}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True,
            args=['--no-sandbox','--disable-dev-shm-usage'])
        context = browser.new_context(viewport={'width':1440,'height':1050})
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        assert page.goto(URL).status == 200
        page.wait_for_function('ready && !running && !!smallVoiceCatalog && !!smallLoras', timeout=300000)
        state = context.request.get(URL+'/api/state').json()
        assert state['guidance_default'] == 2 and state['reference_policy_default'] == 'fixed'
        assert state['checkpoint']['step'] == 26774 and state['ears_loaded'] == ['base','medium']
        defaults = page.evaluate('buildTurnRequest("defaults")')
        assert defaults['guidance'] == 2 and defaults['temperature'] == 1 and defaults['seconds'] is None
        assert defaults['format'] == 'caption' and defaults['generation'] == 'best' and defaults['n'] == 3
        assert defaults['voice_profile'] == 'emolia_c0542' and defaults['adapters']['id'] is None
        assert not defaults['adapters']['ref15'] and defaults['chunk_frames'] == 20
        report['defaults'] = defaults
        catalog = context.request.get(URL+'/api/voices/catalog').json()
        promoted = [x for x in catalog['voices'] if x['recommended']]
        assert len(promoted) == 6 and not any(x['dynamic'] for x in catalog['voices'])
        assert page.locator('#voice-choice optgroup').first.locator('option').count() == 6
        page.evaluate('document.querySelectorAll(".settings-card").forEach(x=>x.open=true)')
        page.locator('#autoplay').uncheck()
        for voice in promoted:
            page.locator('#voice-choice').select_option(voice['id'])
            preview = page.locator('#catalog-preview')
            assert preview.is_visible() and preview.get_attribute('src').endswith(voice['preview_url'])
            response = context.request.get(URL+voice['preview_url'])
            assert response.ok and response.headers['content-type'].startswith('audio/mpeg')
            file = ROOT/'test-results'/f"fixed-preview-{voice['id']}.mp3"
            file.write_bytes(response.body())
            stream = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(file)]))['streams'][0]
            assert stream['channels'] == 1 and int(stream['bit_rate']) == 96000
        assert context.request.get(URL+'/api/voices/reference/unknown').status == 404
        assert context.request.get(URL+'/fixed-references/selection.json').status == 404
        # Recommendation resets duration but preserves explicit voice selection.
        page.locator('#seconds').fill('9')
        page.locator('#recommended').click()
        assert page.locator('#seconds').input_value() == ''
        assert page.locator('#voice-choice').input_value() == promoted[-1]['id']
        page.locator('#voice-choice').select_option('emolia_c0542')
        page.locator('#inputmode').select_option('manual')
        page.locator('#template').click()
        assert IDENTITY in page.locator('#q').input_value()
        page.locator('#voice-choice').select_option('none')
        page.locator('#template').click()
        assert IDENTITY not in page.locator('#q').input_value()
        page.locator('#voice-choice').select_option('emolia_c0542')
        page.locator('#inputmode').select_option('chat')
        page.evaluate('''() => {const original=takes;takes=function(box,result){
            window.actualResult=result;return original(box,result);};}''')

        def turn(label, prompt):
            page.wait_for_function('ready && !running', timeout=300000)
            page.evaluate('window.actualResult=null')
            page.locator('#q').fill(prompt)
            page.locator('#send').click()
            page.wait_for_function('actualResult!==null || !!document.querySelector(".error")', timeout=300000)
            assert not page.locator('.error').count(), page.locator('.error').all_text_contents()
            page.wait_for_function('!running', timeout=10000)
            result = page.evaluate('actualResult')
            ref = result['reference']
            expected = next(x for x in catalog['voices'] if x['id'] == ref['voice'])
            assert ref['policy'] == 'fixed' and not ref['dynamic']
            assert ref['reference_sha256'] == expected['reference_sha256']
            assert abs(ref['tts_context']['used_seconds'] - expected['reference_seconds']) < .02
            assert not result['adapters']['active'] and result['checkpoint']['step'] == 26774
            assert result['guidance'] == 2
            assert all(x['ears']['available'] for x in result['candidates'])
            candidate = next(x for x in result['candidates'] if x['index'] == result['winner'])
            audio = context.request.get(URL+candidate['url'])
            assert audio.ok and audio.headers['content-type'].startswith('audio/mpeg')
            other = browser.new_context()
            assert other.request.get(URL+candidate['url']).status == 404
            other.close()
            output = ROOT/'test-results'/f'fixed-reference-{label}.mp3'
            output.write_bytes(audio.body())
            report['cases'].append({'name':label,'result':result,
                'output_sha256':hashlib.sha256(audio.body()).hexdigest()})
            (ROOT/'test-results/fixed-references-live.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
            print(json.dumps({'case':label,'voice':ref['name'],'reference_seconds':ref['tts_context']['used_seconds'],
                'takes':len(result['candidates']),'generation_ms':result['generation_ms']}), flush=True)
            return result

        first = turn('default-selma-best3', 'Reply in English in one short sentence. Comfort a friend warmly after a difficult day.')
        assert len(first['candidates']) == 3 and first['generation'] == 'best'
        page.locator('#inputmode').select_option('manual')
        page.locator('#generation').select_option('single')
        page.locator('#language').select_option('en')
        angry = turn('selma-anger-single', 'CAPTION: '+IDENTITY+' Intense anger, firm emphasis, then a softer ending.\nTRANSCRIPT: "I trusted you. Tell me what really happened."')
        assert first['reference']['reference_sha256'] == angry['reference']['reference_sha256']
        assert angry['live_stream']['enabled']
        for voice in promoted[1:]:
            page.locator('#voice-choice').select_option(voice['id'])
            turn(voice['id']+'-single', 'CAPTION: '+IDENTITY+' Calm natural conversation, a brief hesitation, then warm relief.\nTRANSCRIPT: "You came back. I am glad to see you."')
        for width in [390,320]:
            page.set_viewport_size({'width':width,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        assert not errors, errors
        report['passed'] = True
        report['javascript_errors'] = errors
        (ROOT/'test-results/fixed-references-live.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
        browser.close()


if __name__ == '__main__':
    main()
