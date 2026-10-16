"""Read-only HTTPS checks of the public static study Space."""
import json
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
BASE = "https://laion-humaneness-voice-small-studies.static.hf.space"


def main():
    with urlopen(BASE + "/manifest.json", timeout=30) as response:
        manifest = json.load(response)
    assert sum(x['path'].startswith('study/audio/') for x in manifest) == 1320
    assert sum(x['path'].startswith('search/audio/') for x in manifest) == 129
    assert not any(x['path'].endswith('.wav') or 'runtime/' in x['path'] for x in manifest)
    for file in ['performance/sweep.json', 'study/summary.json', 'study/results.json']:
        with urlopen(BASE + '/' + file, timeout=30) as response:
            value = json.load(response)
        if file.endswith('results.json'):
            assert len(value) == 660
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/snap/bin/chromium', headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        assert page.goto(BASE + '/index.html').status == 200
        page.get_by_role('link', name='Read throughput study →').click()
        assert page.url == BASE + '/performance/index.html'
        assert page.locator('h1').is_visible()
        page.get_by_role('link', name='Emotion / CFG study', exact=True).click()
        assert page.url == BASE + '/study/index.html'
        assert page.evaluate('getComputedStyle(document.body).color') == 'rgb(236, 236, 241)'
        page.locator('#filter').fill('Fear caption g1.5 t1-s777 main')
        page.wait_for_function('document.querySelectorAll("#takes details").length === 1')
        assert page.locator('#takes details').count() == 1
        page.locator('#takes summary').click()
        for index in [0, 1]:
            audio = page.locator('#takes audio').nth(index)
            audio.evaluate('e => e.load()')
            page.wait_for_function('(i) => document.querySelectorAll("#takes audio")[i].readyState >= 1', arg=index)
            assert audio.evaluate('e => e.duration') > 0
            audio.evaluate('e => e.play()')
            page.wait_for_function('(i) => document.querySelectorAll("#takes audio")[i].currentTime > 0', arg=index)
            audio.evaluate('e => e.pause()')
        page.evaluate('document.activeElement.blur(); window.scrollTo(0,0)')
        page.screenshot(path=str(ROOT/'test-results'/'hf-study-desktop.png'))
        for report in ['index.html', 'performance/index.html', 'study/index.html']:
            assert page.goto(BASE + '/' + report).status == 200
            page.set_viewport_size({'width':390, 'height':844})
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        assert page.goto(BASE+'/search/index.html').status==200
        assert page.locator('h1').is_visible()
        page.locator('#filter').fill('quality affection_de b2 c20 s2027')
        page.wait_for_function('document.querySelectorAll("#takes details").length===1')
        page.locator('#takes summary').click()
        audio=page.locator('#takes audio').first
        audio.evaluate('e=>e.load()')
        page.wait_for_function('document.querySelector("#takes audio").readyState>=1')
        assert audio.evaluate('e=>e.duration')>0
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not errors, errors
        browser.close()
    print('PASS: public HF HTTPS landing, three reports, cross-links, 660 CFG takes, 129 search outputs, 1449 MP3 manifest entries, raw/processed audio playback, mobile layout')


if __name__ == '__main__':
    main()
