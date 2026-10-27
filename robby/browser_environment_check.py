"""Real HTTPS browser + GPU captioner + local/API LLM + TTS; authored fake mic.

Three independent fresh browsers cover permitted autoplay, explicit browser
activation, and microphone denial. No physical-room accuracy is claimed.
"""
import argparse,asyncio,datetime,json
from pathlib import Path
from playwright.async_api import async_playwright
from environment_fixtures import microphone_file
from run_voice_assistant_checks import deployed_source_hash,public_state

ROOT=Path(__file__).resolve().parent
INIT='''window.envEvents=[];window.envPCM=[];window.cameraRequests=0;
const W=window.WebSocket;window.WebSocket=class extends W{constructor(...args){super(...args);
this.addEventListener('message',e=>{if(typeof e.data==='string')envEvents.push({...JSON.parse(e.data),at:performance.now()});
else if(e.data instanceof ArrayBuffer)envPCM.push({at:performance.now(),id:new DataView(e.data).getInt32(0,true),samples:(e.data.byteLength-4)/2});});}};
const gum=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);navigator.mediaDevices.getUserMedia=x=>{if(x.video)cameraRequests++;return gum(x);};
localStorage.setItem('robby.avatar-idle-enabled','false');localStorage.setItem('robby.avatar-pet-invites-enabled','false');'''

async def main(args):
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    fixture=microphone_file(ROOT/'logs/environment-microphone.wav')
    report={'passed':False,'started':stamp,'scope':__doc__,'source_sha256':deployed_source_hash(),
        'server':public_state(args.url),'cases':[]}
    async with async_playwright() as pw:
        async def browser_case(mode):
            flags=['--no-sandbox','--disable-gpu','--use-fake-device-for-media-stream',
                '--use-file-for-fake-audio-capture='+str(fixture)]
            if mode!='denied':flags+=['--use-fake-ui-for-media-stream']
            flags+=['--autoplay-policy=document-user-activation-required' if mode=='activation' else '--autoplay-policy=no-user-gesture-required']
            browser=await pw.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=flags)
            page=await browser.new_page(viewport={'width':1366,'height':950});errors=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            await page.add_init_script(INIT)
            if mode=='denied':
                # Browser permission denial is deterministic; typed TTS still uses
                # a real AudioContext and server connection, not mock synthesis.
                await page.add_init_script("navigator.mediaDevices.getUserMedia=()=>Promise.reject(new DOMException('Permission denied by user','NotAllowedError'))")
            row={'mode':mode,'passed':False}
            try:
                response=await page.goto(args.url+'/desk-robot/');assert response.status==200,response.status
                # Playwright evaluate/wait_for_function can mark a synthetic
                # user activation. Let native autoplay policy run FIRST.
                if mode=='activation':await asyncio.sleep(2.)
                await page.wait_for_function('window.robotFace?.environmentControls&&window.robo',timeout=45000)
                assert await page.locator('#robby-auto-start').is_checked()
                assert await page.locator('#environment-enabled').is_checked()
                assert not await page.locator('#camera-auto').is_checked()
                if mode in ('activation','denied'):
                    await page.locator('#enable-robby').wait_for(state='visible',timeout=20000)
                    row['permission_message']=await page.locator('#auto-start-status').inner_text()
                    assert not await page.evaluate('robo.stream')
                    assert not await page.evaluate('envPCM.length')
                    if mode=='activation':await page.click('#enable-robby')
                    else:
                        await page.fill('#message','Bitte sage nur: Hallo.');await page.locator('#form').evaluate('f=>f.requestSubmit()')
                        await page.wait_for_function('envEvents.some(e=>e.type==="turn_done")',timeout=60000)
                        assert await page.evaluate('envPCM.length')>0
                        row['typed_speech_works_after_denial']=True
                if mode!='denied':
                    await page.wait_for_function('envEvents.some(e=>e.type==="environment_observation"&&e.kind==="welcome")',timeout=20000)
                    await page.wait_for_function('envEvents.some(e=>e.type==="turn_done")',timeout=60000)
                    assert await page.evaluate('robo.ctx.state')=='running'
                    assert await page.evaluate('envPCM.length')>0
                    row['automatic_welcome']=True
                if mode=='permitted':
                    await page.wait_for_function('envEvents.some(e=>e.type==="sound_caption"&&e.status==="available")',timeout=45000)
                    await page.wait_for_function('envEvents.some(e=>e.type==="environment_observation"&&e.kind==="sound"&&e.status==="responding")',timeout=12000)
                    await page.wait_for_function('envEvents.filter(e=>e.type==="turn_done").length>=2',timeout=60000)
                    row['caption_text']=await page.locator('#environment-caption').text_content();assert row['caption_text']
                    row['latency_card']=await page.locator('#latency-sound').inner_text();assert 'ms' in row['latency_card']
                    events=await page.evaluate('envEvents');pcm=await page.evaluate('envPCM')
                    obs=next(e for e in events if e['type']=='environment_observation' and e['kind']=='sound' and e['status']=='responding')
                    replies=[e for e in events if e['type']=='reply' and e['at']>=obs['at']]
                    assert replies,replies
                    row['sound_reply']=replies[-1]['text'];rid=replies[0]['reply_id']
                    first=next(x for x in pcm if x['id']==rid)
                    caption=next(e for e in events if e['type']=='sound_caption')
                    row['caption_to_first_pcm_ms']=round(first['at']-caption['at'],2)
                    # Approximate onset on the browser receive clock; capture
                    # timestamps and model request wall time are explicitly shown.
                    row['onset_to_first_pcm_estimate_ms']=round(row['caption_to_first_pcm_ms']+caption['event_age_ms'],2)
                    assert len([e for e in events if e['type']=='sound_caption'])==1
                    assert not [e for e in events if e['type']=='endpoint'], 'Authored impacts must not create a user utterance'
                    await page.locator('#environment-settings').evaluate('e=>e.open=true')
                    await page.locator('#environment-enabled').uncheck()
                    await page.wait_for_function('envEvents.filter(e=>e.type==="environment_settings").at(-1)?.enabled===false')
                    await page.locator('#sound-prominence').fill('18');await page.locator('#sound-prominence').dispatch_event('change')
                    await page.locator('#environment-impressions').check()
                    row['settings_verified']=True
                    await page.screenshot(path=str(ROOT/'logs/environment-desktop.png'),full_page=True)
                    await page.set_viewport_size({'width':390,'height':844})
                    assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
                    await page.screenshot(path=str(ROOT/'logs/environment-mobile.png'),full_page=True)
                assert await page.evaluate('cameraRequests')==0
                row['events']=await page.evaluate('envEvents');row['pcm']=await page.evaluate('envPCM')
                await page.click('#disable-robby');await page.wait_for_timeout(500)
                assert await page.evaluate('!robo.stream&&!robo.sources.length&&!robo.ws')
                await page.reload();await page.wait_for_function('window.robotFace?.environmentControls')
                await page.wait_for_timeout(1800)
                assert not await page.locator('#robby-auto-start').is_checked()
                assert await page.evaluate('envEvents.length')==0
                if mode=='permitted':
                    assert not await page.locator('#environment-enabled').is_checked()
                    assert await page.locator('#sound-prominence').input_value()=='18'
                row['off_persists_and_stops_mic_playback']=True;row['passed']=not errors
            except Exception as exc:
                row['error']=type(exc).__name__+': '+str(exc)[:1500]
                row['events']=await page.evaluate('window.envEvents||[]');row['pcm']=await page.evaluate('window.envPCM||[]')
                await page.screenshot(path=str(ROOT/'logs'/f'environment-failure-{mode}.png'),full_page=True)
            finally:
                row['page_errors']=errors
                try:await page.evaluate('window.robo?.stop()')
                except Exception:pass
                await browser.close();await asyncio.sleep(.5)
            report['cases'].append(row)
        for mode in ('permitted','activation','denied'):await browser_case(mode)
    report['passed']=all(c['passed'] for c in report['cases'])
    out=ROOT/'static'/f'environment-browser-{stamp}.json';out.write_text(json.dumps(report,ensure_ascii=False,separators=(',',':')))
    print(json.dumps({'report':str(out),'passed':report['passed'],'cases':[{k:v for k,v in c.items() if k not in ('events','pcm')} for c in report['cases']]},indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com')
    asyncio.run(main(p.parse_args()))
