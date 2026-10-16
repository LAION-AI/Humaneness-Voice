"""Real HTTPS report/filter/audio/privacy verification; no listening verdict."""
import base64
import json
from pathlib import Path
import re
import urllib.error
import urllib.request

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent/'full-ft-speaker-study'
URL='https://switch-surprised-clock-sparc.trycloudflare.com/full-ft-speaker-study/'


def main():
    with urllib.request.urlopen(URL,timeout=60) as response:
        assert response.headers['content-type'].startswith('text/html');markup=response.read().decode()
    rows=sum((json.loads((ROOT/f'results-{phase}.json').read_text()) for phase in ('pilot','validation','diagnostic','followup')),[])
    good=[r for r in rows if not r['failed']]
    audio=re.findall(r'src="data:audio/mpeg;base64,([A-Za-z0-9+/=]+)"',markup)
    assert len(audio)==len(good)
    for encoded in audio:assert len(base64.b64decode(encoded,validate=True))>1000
    assert 'no DPO' in markup and 'Fixed-reference cosine' in markup
    assert not re.search(r'\bsrc="https?://',markup)
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,
            args=['--no-sandbox','--disable-dev-shm-usage','--autoplay-policy=no-user-gesture-required'])
        page=browser.new_page(viewport={'width':1440,'height':1050});errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(URL,timeout=60000).status==200
        assert page.locator('article').count()==len(rows) and page.locator('audio').count()==len(good)
        page.locator('[data-filter="voice"]').select_option('selma')
        page.locator('[data-filter="phase"]').select_option('pilot')
        page.locator('[data-filter="condition"]').select_option('baseline')
        assert page.locator('#count').inner_text()==f'10 of {len(rows)} takes shown'
        page.locator('[data-filter="emotion"]').select_option('joy')
        page.locator('[data-filter="language"]').select_option('en')
        clip=page.locator('article:not([hidden]) audio');assert clip.count()==1
        clip.evaluate('(el)=>el.play()')
        page.wait_for_function('''() => {let a=document.querySelector('article:not([hidden]) audio');
            return a.currentTime>.15 && !a.error && Number.isFinite(a.duration);}''',timeout=30000)
        clip.evaluate('(el)=>el.pause()')
        for select in page.locator('[data-filter]').all():select.select_option('')
        for width in (1440,390,320):
            page.set_viewport_size({'width':width,'height':900})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        assert not errors,errors
        browser.close()
    for name in ('summary-pilot.json','summary-validation.json','summary-followup.json','followup-design.json',
                 'selection.json','protocol-amendment.json','measurement-sanity.json','backend-check.json'):
        with urllib.request.urlopen(URL+name,timeout=60) as response:json.loads(response.read())
    for private in ('private-provenance.json','references/idris-15.wav','selma-15.wav','scores.json'):
        try:urllib.request.urlopen(URL+private,timeout=20)
        except urllib.error.HTTPError as error:assert error.code==404
        else:raise AssertionError('Reference route exposed: '+private)
    result={'url':URL,'passed':True,'all_base64_valid':True,'embedded_audio_count':len(good),
        'filters_and_actual_audio_playback':True,'private_routes_blocked':True,'javascript_errors':errors}
    (ROOT/'browser-check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
