"""Real HTTPS/API checks for 24k defaults and post-conversion streaming."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import httpx

ROOT=Path(__file__).resolve().parent
URL='https://switch-surprised-clock-sparc.trycloudflare.com'


def browser_check():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1000});errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(URL,wait_until='domcontentloaded',timeout=60000)
        page.wait_for_function('globalThis.smallVcAvailable && !document.getElementById("vc").disabled',timeout=60000)
        assert page.locator('#vc').is_checked()
        assert page.evaluate('buildTurnRequest("default test").voice_conversion')
        assert '24k' in page.locator('#vc-help').inner_text()
        page.locator('#vc').uncheck();assert not page.evaluate('buildTurnRequest("disabled test").voice_conversion')
        page.reload(wait_until='domcontentloaded');page.wait_for_function('globalThis.smallVcAvailable',timeout=60000)
        assert not page.locator('#vc').is_checked()  # Explicit choice persists.
        page.locator('#vc').check()
        page.locator('#voice-choice').select_option('none')
        assert page.locator('#vc').is_disabled() and not page.locator('#vc').is_checked()
        assert not page.evaluate('buildTurnRequest("no reference").voice_conversion')
        page.locator('#voice-choice').select_option('emolia_c0542')
        assert page.locator('#vc').is_checked() and page.locator('#vc').is_enabled()
        for width in (1440,390,320):
            page.set_viewport_size({'width':width,'height':900})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not errors,errors
        browser.close()
        return {'https':True,'default_on':True,'opt_out_persists':True,'none_reference_disables':True,
            'restoring_profile_restores_preference':True,'widths':[1440,390,320],'js_errors':errors}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--browser',action='store_true');args=parser.parse_args()
    records=[]
    with httpx.Client(base_url=URL,timeout=240) as client:
        deadline=time.monotonic()+420
        while True:
            try:
                state=client.get('/api/state').json()
                if state.get('error'):raise RuntimeError(state['error'])
                if state['ready'] and not state['busy']:break
            except (httpx.HTTPError,ValueError):pass
            if time.monotonic()>deadline:raise RuntimeError('Live startup timeout')
            time.sleep(1)
        assert state['voice_conversion']['default'] and state['voice_conversion']['step']==24000
        assert state['voice_conversion']['arm']=='speaker_mix'
        for label,extra in [
            ('default-stream',{'generation':'single','streaming':True}),
            ('best2-disabled',{'generation':'best','n':2,'streaming':False,'voice_conversion':False}),
            ('winner4-default',{'generation':'search_winner','search_n':4,'search_seconds':4.8,'search_chunk_frames':20}),
            ('fp32-default',{'generation':'single','streaming':False,'vc_precision':'fp32'})]:
            # VC flag intentionally omitted in on-cases to verify API default.
            body={'message':'CAPTION: Warm, natural conversation with clear phrasing.\nTRANSCRIPT: "I am glad you came back. We can talk now."',
                'input_mode':'manual','language':'en','format':'caption','seed':24017,**extra}
            tick=time.monotonic();events=[];chunks=[]
            with client.stream('POST','/api/turn',json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith('data: '):continue
                    event=json.loads(line[6:]);events.append(event)
                    if event['event']=='error':raise RuntimeError((label,event))
                    if event['event'] in ('audio_chunk','search_chunk'):
                        chunks.append({'time_seconds':time.monotonic()-tick,'event':event})
            result=next(e for e in events if e['event']=='result')
            on=extra.get('voice_conversion',True)
            assert result['voice_conversion']['enabled']==on
            if on:
                assert result['voice_conversion']['step']==24000 and result['voice_conversion']['arm']=='speaker_mix'
                assert result['voice_conversion']['scoring_audio']=='converted output, never raw TTS'
            if label in ('default-stream','winner4-default'):
                assert chunks and events.index(chunks[0]['event'])<events.index(result)
            if label=='winner4-default':assert result['voice_conversion']['max_microbatch_size']>1
            url=result['candidates'][result['winner']]['url']
            audio=client.get(url);audio.raise_for_status()
            path=ROOT/'test-results'/('grow24k-live-'+label+'.mp3');path.write_bytes(audio.content)
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)]))['streams'][0]
            assert info['codec_name']=='mp3' and info['channels']==1 and int(info['bit_rate'])==96000
            records.append({'case':label,'request':body,'chunks':chunks,'result':result})
            (ROOT/'test-results/grow24k-live.json').write_text(json.dumps(records,indent=2))
            print(label,'chunks',len(chunks),'wall',round(time.monotonic()-tick,2),flush=True)
    if args.browser:
        proof=browser_check();(ROOT/'test-results/grow24k-browser.json').write_text(json.dumps(proof,indent=2))
        print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
