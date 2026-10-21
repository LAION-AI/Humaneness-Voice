"""Fake device/synthetic-landmark behavior + real CPU-model smoke test on public
NASA/skimage imagery. Not a physical conversation/recognition-accuracy study."""
import argparse,asyncio,json,time
from pathlib import Path
from playwright.async_api import async_playwright

async def main(url):
    report={'started':time.time(),'passed':False,'physical_devices_used':False,'cases':[]}
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream'])
        page=await browser.new_page(viewport={'width':1440,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script("localStorage.setItem('robby.cameraAuto','false');localStorage.setItem('robby.multimodal.settings.v1',JSON.stringify({ignoreOffscreenSpeakers:true,lookAtActiveSpeaker:true,greetUnknownSpeakers:true}))")
        try:
            await page.goto(url);await page.wait_for_function('window.robo&&window.robotFace.multimodal',timeout=90000)
            await page.locator('#recognition-settings > summary').click()
            for key in ['lookAtActiveSpeaker','greetUnknownSpeakers']:assert await page.locator('#'+key).is_checked()
            for key in ['ignoreOffscreenSpeakers','faceIdEnabled']:assert not await page.locator('#'+key).is_checked()
            assert await page.locator('#quick-controls #ignoreOffscreenSpeakers').is_visible()
            assert await page.locator('#quick-controls #faceIdEnabled').is_visible()
            await page.locator('#ignoreOffscreenSpeakers').check()
            assert not await page.locator('#remember-biometric-profiles').is_checked()
            await page.locator('#ignoreOffscreenSpeakers').uncheck();await page.reload();await page.wait_for_function('window.robotFace?.multimodal')
            await page.locator('#recognition-settings > summary').click()
            assert not await page.locator('#ignoreOffscreenSpeakers').is_checked();assert await page.locator('#lookAtActiveSpeaker').is_checked();await page.locator('#ignoreOffscreenSpeakers').check()
            await page.locator('#faceIdEnabled').check()
            report['cases'].append('Old v1 camera gate migrated OFF; face ID OFF; gaze ON; independent v2 opt-in persists')
            report['geometry']=await page.evaluate("async()=> (await import('/static/multimodal-test-cases.js')).behaviorChecks()")
            report['disabled_activity']=await page.evaluate('''async()=>{
              const {VisualSpeakers}=await import('/static/visual-speakers.js'),{testFace}=await import('/static/multimodal-test-cases.js');
              const v=new VisualSpeakers();for(let i=0;i<20;i++)v.update([testFace(.3+i*.002,i%2?.3:.01)],i*100,false);
              const t=[...v.tracks.values()][0];if(t.speaking||t.samples.length||!v.gazeFace(1900))throw Error('Disabled VAD blocked position tracking or still analyzed lips');return true;
            }''')
            report['storage_denied']=await page.evaluate('''async()=>{
              const {People}=await import('/static/visual-speakers.js');
              const denied=()=>{throw Error('Synthetic storage denied');},p=new People({getItem:denied,setItem:denied,removeItem:denied}),d=Array(128).fill(0);d[0]=1;
              p.bind('person-test','Synthetic person',d);if(p.profiles.size!==1)throw Error('Session memory failed with denied storage');
              p.setConsent(true);if(p.consent||!p.storageError)throw Error('Failed storage was presented as persisted');p.clear();if(p.profiles.size)throw Error('Session clear failed');return true;
            }''')
            report['active_gaze']=await page.evaluate('''async()=>{
              const {testFace}=await import('/static/multimodal-test-cases.js'),m=robotFace.multimodal;
              const result={faceLandmarks:[testFace(.3),testFace(.7)]},at=performance.now(),video={videoWidth:0,videoHeight:0};
              m.frame(result,at,video,0);const target=m.tracks.find(t=>t.box.x>.5);
              m.audioCandidate={id:target.id,since:at-800};m.lastLookSwitch=at-800;
              m.message({type:'visual_active',track_id:target.id});
              const selected=m.frame(result,at+1,video,0);if(selected?.faceLandmarks[0]!==result.faceLandmarks[1])throw Error('Gaze did not select the audio-correlated second face');
              m.settings.lookAtActiveSpeaker=false;const fallback=m.frame(result,at+2,video,0);if(fallback?.faceLandmarks[0]!==result.faceLandmarks[0])throw Error('Disabled active-speaker gaze did not retain primary face');
              m.settings.lookAtActiveSpeaker=true;m.visual.clear();m.tracks=[];return {second_face_selected:true,disabled_option_respected:true};
            }''')
            await page.evaluate("()=>{const d=Array(128).fill(0);d[0]=1;robotFace.multimodal.people.bind('person-ui','Synthetic test person',d);}")
            async def decline(dialog):await dialog.dismiss()
            page.on('dialog',decline);await page.locator('#remember-biometric-profiles').click()
            assert not await page.locator('#remember-biometric-profiles').is_checked();page.remove_listener('dialog',decline)
            async def accept(dialog):await dialog.accept()
            page.on('dialog',accept);await page.locator('#remember-biometric-profiles').check()
            assert await page.evaluate("!!localStorage.getItem('robby.consentedPeople.v1')");page.remove_listener('dialog',accept)
            await page.reload();await page.wait_for_function('window.robotFace?.multimodal')
            await page.locator('#recognition-settings > summary').click()
            assert await page.locator('#remember-biometric-profiles').is_checked();assert 'Synthetic test person' in await page.locator('#visual-person-list').text_content()
            await page.locator('#remember-biometric-profiles').uncheck();assert await page.evaluate("localStorage.getItem('robby.consentedPeople.v1')===null")
            await page.locator('#forget-multimodal-people').click();assert await page.evaluate('robotFace.multimodal.people.profiles.size===0')
            report['cases'].append('Real UI consent rejection/acceptance, persisted reload, withdrawal and Forget all')
            await page.locator('#camera').click()
            await page.wait_for_function("robotFace.cameraStats.inferences>3||document.querySelector('#camera-status').textContent.includes('unavailable')",timeout=60000)
            status=await page.locator('#camera-status').text_content();assert 'unavailable' not in status,status
            report['camera_worker']=await page.evaluate('({stats:robotFace.cameraStats,ms:robotFace.multimodal.metrics.landmarkMs,capture:document.querySelector("#video").srcObject.getVideoTracks()[0].getSettings()})')
            await page.locator('#camera').click()
            report['real_models']=await page.evaluate("async()=> (await import('/static/multimodal-test-cases.js')).realModels()")
            report['cases'].append('Real MediaPipe two-face detection + repeatable face-api 128-D descriptors, CPU workers')
            assert not errors,errors
            report.update(passed=True,finished=time.time(),page_errors=errors)
        finally:
            (Path(__file__).parent/'logs/multimodal-browser-checks.json').write_text(json.dumps(report,indent=2));await browser.close()
    print(json.dumps(report))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8793/desk-robot/');asyncio.run(main(parser.parse_args().url))
