"""Real HTTPS/self-contained HTML/audio/filter/mobile report verification."""
import argparse
import base64
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess
import urllib.request

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent/'speaker-study'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='https://switch-surprised-clock-sparc.trycloudflare.com/speaker-study/')
    args=parser.parse_args()
    with urllib.request.urlopen(args.url,timeout=60) as response:
        assert response.headers['Content-Type'].startswith('text/html');markup=response.read().decode()
    embedded=re.findall(r'src="data:audio/mpeg;base64,([A-Za-z0-9+/=]+)"',markup)
    assert len(embedded)==240
    plan=json.loads((ROOT/'design.json').read_text());assert plan['total_trials']==240
    for sample in embedded:assert len(base64.b64decode(sample,validate=True))>1000
    assert not re.search(r'\bsrc="https?://',markup),'Report audio must be self-contained'
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',
        str(next((ROOT/'audio').glob('*.mp3')))]))['streams'][0]
    assert probe['channels']==1 and int(probe['bit_rate'])==96000
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,
            args=['--no-sandbox','--disable-dev-shm-usage','--autoplay-policy=no-user-gesture-required'])
        page=browser.new_page(viewport={'width':1440,'height':1050});errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        assert page.goto(args.url,timeout=60000).status==200
        assert page.locator('audio').count()==240 and page.locator('tbody').first.locator('tr').count()==12
        assert page.locator('#count').inner_text()=='240 of 240 takes shown'
        page.locator('[data-filter="voice"]').select_option('selma')
        assert page.locator('#count').inner_text()=='120 of 240 takes shown'
        page.locator('[data-filter="adapter"]').select_option('dpo')
        page.locator('[data-filter="condition"]').select_option('caption_affect')
        assert page.locator('#count').inner_text()=='10 of 240 takes shown'
        page.locator('[data-filter="emotion"]').select_option('anger')
        page.locator('[data-filter="language"]').select_option('de')
        assert page.locator('#count').inner_text()=='1 of 240 takes shown'
        audio=page.locator('article:not([hidden]) audio')
        assert audio.count()==1
        audio.evaluate('(el)=>el.play()')
        page.wait_for_function('''() => {const a=document.querySelector('article:not([hidden]) audio');
            return a.currentTime>.15 && !a.error && Number.isFinite(a.duration) && a.duration>1;}''',timeout=30000)
        duration=audio.evaluate('(el)=>el.duration');audio.evaluate('(el)=>el.pause()')
        for select in page.locator('[data-filter]').all():select.select_option('')
        for width,height in [(1440,1050),(390,844),(320,740)]:
            page.set_viewport_size({'width':width,'height':height})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),f'Mobile overflow at {width}'
            page.screenshot(path=str(ROOT/f'report-{width}.png'))
        assert not errors,errors
        browser.close()
    for name in ('design.json','summary.json','results.json','reference-controls.json'):
        with urllib.request.urlopen(args.url+name,timeout=60) as response:json.loads(response.read())
    for private in ('idris.wav','selma.wav','references.json','scores.json'):
        try:urllib.request.urlopen(args.url+private,timeout=10)
        except urllib.error.HTTPError as error:assert error.code==404
        else:raise AssertionError('Private reference path exposed: '+private)
    report={'timestamp':datetime.now(timezone.utc).isoformat(),'url':args.url,'embedded_audio_count':240,
        'all_base64_valid':True,'https_status':200,'filters_verified':True,'actual_embedded_audio_playback':True,
        'tested_audio_duration':duration,'mp3_mono_96kbps':True,'viewport_widths':[1440,390,320],
        'javascript_errors':errors,'source_reference_routes_rejected':True}
    (ROOT/'browser-check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
