"""Read-only HTTPS report rendering and embedded MP3 decoding tests."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extended',action='store_true')
    args=parser.parse_args();anchor='strict-tails' if args.extended else 'seed-scaling'
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,
                                 args=['--no-sandbox','--autoplay-policy=no-user-gesture-required'])
        page=browser.new_page(viewport={'width':1440,'height':1000});errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        response=page.goto('https://switch-surprised-clock-sparc.trycloudflare.com/catalogue-steering-study/#'+anchor,
                           wait_until='domcontentloaded',timeout=180000)
        assert response.status==200
        scope=page.locator('#'+anchor);assert scope.is_visible()
        assert scope.locator('table').count()>=4
        assert scope.locator('audio').count()>=82
        case=scope.locator('details').first;case.locator('summary').click()
        audio=case.locator('audio').first
        seconds=audio.evaluate('a=>new Promise((ok,bad)=>{a.onloadedmetadata=()=>ok(a.duration);a.onerror=()=>bad(Error("MP3 decode failed"));a.load()})')
        assert seconds>0
        audio.evaluate('a=>a.play()');page.wait_for_timeout(300)
        assert audio.evaluate('a=>a.currentTime')>0
        audio.evaluate('a=>a.pause()')
        for width in (1440,390,320):
            page.set_viewport_size({'width':width,'height':900})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),width
        state=page.request.get('https://switch-surprised-clock-sparc.trycloudflare.com/api/state').json()
        assert state['ready'] and state['checkpoint']['step']==26774
        assert state['voice_default']=='emolia_c0542' and state['n_default']==3
        assert not errors,errors
        output={'https':True,'anchor':anchor,'audio_elements':scope.locator('audio').count(),
                'decoded_seconds':seconds,'actual_playback':True,'widths':[1440,390,320],
                'javascript_errors':errors,'live_defaults_unchanged':True}
        folder='strict-tail-study' if args.extended else 'seed-scaling-study'
        (Path(__file__).parent/folder/'browser-check.json').write_text(json.dumps(output,indent=2))
        print(json.dumps(output,indent=2));browser.close()


if __name__=='__main__':main()
