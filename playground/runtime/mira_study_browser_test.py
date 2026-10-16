"""Read-only HTTPS report/UI verification; never requests a generation."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='https://switch-surprised-clock-sparc.trycloudflare.com')
    parser.add_argument('--takes', type=int, required=True)
    args = parser.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/google-chrome',
            headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width':1440, 'height':1050})
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        response = page.goto(args.url.rstrip('/') + '/mira-activation-study/',
            wait_until='domcontentloaded', timeout=180000)
        assert response.status == 200
        page.wait_for_function('document.querySelector("#count")?.textContent.includes("takes shown")', timeout=60000)
        assert page.locator('article').count() == args.takes
        assert page.locator('article audio').count() == args.takes
        assert page.locator('audio').count() == args.takes + 2
        assert page.locator('select[data-filter="block"]').count() == 1
        page.locator('select[data-filter="phase"]').select_option('validation')
        page.locator('select[data-filter="block"]').select_option('0')
        assert page.locator('article:visible').count() == 5
        audio = page.locator('article:visible audio').first
        assert audio.get_attribute('src').startswith('data:audio/mpeg;base64,')
        duration = audio.evaluate('a => new Promise((resolve,reject)=>{a.onloadedmetadata=()=>resolve(a.duration);a.onerror=()=>reject(Error("MP3 decode"));a.load();})')
        assert duration > 0
        audio.evaluate('a=>a.play()')
        page.wait_for_timeout(300)
        assert audio.evaluate('a=>a.currentTime') > 0
        audio.evaluate('a=>a.pause()')
        for width in (1440, 390, 320):
            page.set_viewport_size({'width':width, 'height':900})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), width
        state = page.request.get(args.url.rstrip('/') + '/api/state').json()
        assert state['ready'] and state['checkpoint']['step'] == 26774
        assert state['adapters']['default'] is None
        assert state['guidance_default'] == 2.0 and state['temperature_default'] == 1.0
        assert state['reference_policy_default'] == 'fixed'
        assert state['voice_default'] == 'emolia_c0542'
        assert state['n_default'] == 3
        assert page.request.get(args.url.rstrip('/') + '/mira-activation-study/audio/private.wav').status == 404
        assert not errors, errors
        result = {'https':args.url, 'takes':args.takes, 'embedded_audio':args.takes,
            'matched_validation_block_visible':5, 'decoded_mp3_seconds':duration,
            'actual_playback':True, 'viewport_widths':[1440,390,320],
            'javascript_errors':errors, 'live_defaults_unchanged':True, 'private_paths_blocked':True}
        (Path(__file__).parent / 'mira-activation-study/browser-check.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        browser.close()


if __name__ == '__main__':
    main()
