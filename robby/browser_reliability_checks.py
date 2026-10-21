"""Current UI defaults, photos/privacy, following and navigation. Fake browser
camera + deterministic face landmarks; real camera worker separately smoke-tested.
"""
import asyncio,json,time
from pathlib import Path
from playwright.async_api import async_playwright

ROOT=Path(__file__).parent
async def main():
    report={'started':time.time(),'passed':False,'physical_camera_used':False}
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream'])
        page=await browser.new_page(viewport={'width':1440,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script("const BaseWorker=Worker;window.__cameraFrameTimes=[];window.Worker=class extends BaseWorker{postMessage(d,...args){if(d.type==='frame')__cameraFrameTimes.push({at:d.at,now:performance.now()});return super.postMessage(d,...args)}}")
        try:
            await page.goto('http://127.0.0.1:8793/desk-robot/');await page.wait_for_function('window.robotFace?.cameraControls.active',timeout=60000)
            assert await page.locator('#mirror').is_checked();assert await page.locator('#camera-auto').is_checked()
            assert await page.locator('#mirror').is_visible();assert await page.locator('#camera').bounding_box()
            assert not await page.locator('#recognition-settings').evaluate('e=>e.open')
            assert await page.locator('#allow-helpful').is_checked()
            assert await page.locator('#allow-helpful').is_visible()
            await page.locator('#allow-helpful').uncheck()
            assert await page.evaluate("localStorage.getItem('robby.helpfulInterruptions')==='false'")
            report['defaults']={'camera_automatically_active':True,'follow_checked':True,'primary_controls_visible':True}
            report['camera_timestamps']=[]
            for expanded in (False,True,False):
                await page.locator('aside > details.settings:not(#recognition-settings)').evaluate('(e,value)=>e.open=value',expanded)
                start=await page.evaluate('__cameraFrameTimes.length')
                await page.wait_for_function('(start)=>__cameraFrameTimes.length>=start+8',arg=start,timeout=30000)
                frames=await page.evaluate('(start)=>__cameraFrameTimes.slice(start)',start)
                assert all(b['at']>a['at'] for a,b in zip(frames,frames[1:])),frames
                assert frames[-1]['at']-frames[0]['at']>400,frames
                report['camera_timestamps'].append({'advanced_expanded':expanded,'strictly_increasing':True,'capture_span_ms':round(frames[-1]['at']-frames[0]['at'],1)})
            await page.locator('#mirror').uncheck();await page.locator('#follow-center').click();assert await page.locator('#mirror').is_checked()
            report['people']=await page.evaluate('''async()=>{
              const {testFace}=await import('/static/multimodal-test-cases.js'),m=robotFace.multimodal,video=document.querySelector('#video');
              const faceOption=document.getElementById('faceIdEnabled');faceOption.checked=true;faceOption.onchange();
              const faces={faceLandmarks:[testFace(.3),testFace(.7)]};let at=performance.now();m.peopleView.lastImages=-Infinity;m.frame(faces,at,video,0);const first=m.followedTrack;
              const again=m.frame({faceLandmarks:[faces.faceLandmarks[1],faces.faceLandmarks[0]]},at+100,video,0);
              if(m.followedTrack!==first)throw Error('Face following changed after landmark order changed');
              const t=m.tracks.find(t=>t.id===first),d=Array(128).fill(0);d[0]=1;
              t.embedding=d;t.referenceImage=t.liveImage;m.bind({track_id:t.id,person_id:'person-ui-test',name:'Synthetic test person',speaker_id:'voice-ui'});
              m.message({type:'visual_profiles',profiles:[{person_id:'person-ui-test',speaker_id:'voice-ui',name:'Synthetic test person'}]});
              m.message({type:'identity',speakers:[{id:'voice-ui',name:'Synthetic test person'}],verification:{reference_scores:[{name:'Synthetic test person',cosine:.67}]}});
              m.peopleView.render(true);const card=[...document.querySelectorAll('.person-card')].find(e=>e.textContent.includes('Synthetic test person'));
              if(!card||card.querySelectorAll('img').length!==2)throw Error('Missing reference/current picture');
              if(!card.textContent.includes('Voice:')||!card.textContent.includes('Face:'))throw Error('Missing independent voice/face status');
              if(localStorage.getItem('robby.consentedPeople.v1'))throw Error('Photos saved without consent');
              faceOption.checked=false;faceOption.onchange();m.peopleView.render(true);
              if(m.worker||m.pending.size||m.tracks.some(t=>t.personId||t.embedding))throw Error('Face ID continued after opt-out');
              if(!m.peopleView.voices.some(v=>v.id==='voice-ui')||!m.people.profiles.has('person-ui-test'))throw Error('Face opt-out deleted voice/profile memory');
              if(!document.querySelector('#people-overview').textContent.includes('recognized by voice'))throw Error('Voice-only gallery missing');
              return {stable_primary_face:true,reference_and_current_images:true,separate_voice_face_status:true,no_unconsented_storage:true,face_opt_out_preserves_voice_memory:true};
            }''')
            await page.screenshot(path=str(ROOT/'logs/reliability-ui-desktop.png'))
            await page.set_viewport_size({'width':390,'height':844});await page.screenshot(path=str(ROOT/'logs/reliability-ui-mobile.png'))
            assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            await page.locator('#camera').click();assert not await page.evaluate('robotFace.cameraControls.active')
            await page.reload();await page.wait_for_function('window.robotFace?.cameraControls');assert not await page.locator('#camera-auto').is_checked()
            assert not await page.locator('#allow-helpful').is_checked()
            report['helpful_checkbox_independent_and_persisted']=True
            assert not await page.evaluate('robotFace.cameraControls.active')
            report['camera_opt_out_persists']=True;assert not errors,errors
            report.update(passed=True,finished=time.time(),page_errors=errors)
        finally:(ROOT/'logs/reliability-ui-checks.json').write_text(json.dumps(report,indent=2));await browser.close()
    print(json.dumps(report))
if __name__=='__main__':asyncio.run(main())
