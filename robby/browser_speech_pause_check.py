"""Real HTTPS/CPU Chrome test of the pause slider and noisy multi-sentence mic."""
import asyncio,json,tempfile,time
from pathlib import Path
import numpy as np,soundfile as sf
from scipy.signal import resample_poly
from playwright.async_api import async_playwright

ROOT=Path(__file__).resolve().parent
URL='https://arrive-locations-parade-research.trycloudflare.com'

async def main():
    report={'passed':False,'physical_microphone':False,'url':URL,'started':time.time()}
    wave=np.r_[np.zeros(5*16000),np.load(ROOT/'logs/speech-pause/longer_setting_holds.npy'),np.zeros(10*16000)].astype(np.float32)
    with tempfile.TemporaryDirectory(prefix='robby-pause-browser-') as temp:
        wav=Path(temp)/'authored-mic.wav';sf.write(wav,resample_poly(wave,3,1),48000,subtype='PCM_16')
        async with async_playwright() as p:
            browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=[
                '--no-sandbox','--disable-gpu','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream',
                '--use-file-for-fake-audio-capture='+str(wav),'--autoplay-policy=no-user-gesture-required'])
            page=await browser.new_page(viewport={'width':1280,'height':1000});errors=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            await page.add_init_script("localStorage.setItem('robby.llmSelection',JSON.stringify({model:'local',provider:'local'}));localStorage.setItem('robby.cameraAuto','false');localStorage.setItem('robby.avatar-idle-enabled','false');localStorage.setItem('robby.avatar-pet-invites-enabled','false')")
            try:
                await page.goto(URL+'/desk-robot/')
                await page.wait_for_function('window.robo && !document.querySelector("#call").disabled',timeout=60000)
                await page.locator('#conversation-options').evaluate('(e)=>e.open=true')
                assert await page.locator('#speech-pause').input_value()=='400'
                assert await page.locator('#speech-pause-controls').is_visible()
                await page.locator('#speech-pause').fill('950');await page.locator('#speech-pause').dispatch_event('change')
                assert await page.locator('#speech-pause-value').text_content()=='950 ms'
                await page.reload();await page.wait_for_function('window.robo && !document.querySelector("#call").disabled',timeout=60000)
                await page.locator('#conversation-options').evaluate('(e)=>e.open=true')
                assert await page.locator('#speech-pause').input_value()=='950'
                report['default_and_persistence']=True
                await page.evaluate('''()=>{window.pauseEvents=[];const old=robo.on.message;robo.on.message=m=>{if(['asr_sentence','asr_settings','endpoint','error'].includes(m.type))pauseEvents.push({...m,at:performance.now()});old(m)}}''')
                await page.click('#call')
                await page.wait_for_function('pauseEvents.some(m=>m.type==="asr_settings"&&m.speech_pause_ms===950)',timeout=10000)
                await page.locator('#speech-pause').fill('1000');await page.locator('#speech-pause').dispatch_event('change')
                await page.wait_for_function('pauseEvents.some(m=>m.type==="asr_settings"&&m.speech_pause_ms===1000)',timeout=5000)
                report['live_settings_ack']=True
                await page.wait_for_function('pauseEvents.some(m=>m.type==="asr_sentence")',timeout=25000)
                await page.wait_for_timeout(1000)
                events=await page.evaluate('pauseEvents');sent=[m for m in events if m['type']=='asr_sentence']
                assert len(sent)==1,sent
                assert 'gemeinsam machen' in sent[0]['text'].casefold() and 'spiel vor' in sent[0]['text'].casefold(),sent
                assert sent[0]['observed_quiet_ms']>=1000,sent
                await page.wait_for_function('pauseEvents.some(m=>m.type==="endpoint")',timeout=10000)
                await page.wait_for_function('robo.metrics.chunks>0',timeout=20000)
                report['events']=events;report['errors']=errors;report['audio_chunks']=await page.evaluate('robo.metrics.chunks')
                assert not errors and not [m for m in events if m['type']=='error']
                await page.locator('#speech-pause').fill('300');await page.locator('#speech-pause').dispatch_event('change')
                await page.wait_for_function('pauseEvents.some(m=>m.type==="asr_settings"&&m.speech_pause_ms===300)',timeout=5000)
                report['live_change_to_300']=True;report['passed']=True
            except Exception as exc:
                report['error']=str(exc);report['errors']=errors
                try:report['events']=await page.evaluate('window.pauseEvents||[]')
                except Exception:pass
            finally:
                await page.evaluate('window.robo?.stop()');await browser.close()
    report['finished']=time.time();(ROOT/'static/speech-pause-browser.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':asyncio.run(main())
