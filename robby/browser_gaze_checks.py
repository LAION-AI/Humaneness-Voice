"""Actual Chromium UI with synthetic landmarks/fake video, not gaze accuracy."""
import argparse,asyncio,json,time
from pathlib import Path
from playwright.async_api import async_playwright

ROOT=Path(__file__).parent
MOCK='''export const FilesetResolver={forVisionTasks:async()=>({})};export const FaceLandmarker={createFromOptions:async()=>({detectForVideo:()=>window.gazeTestResult||{faceLandmarks:[]}})};'''

async def main(url):
    report={'started':time.time(),'passed':False,'physical_camera_used':False,
        'landmark_model_mocked':True,'scope':'Real UI/keyboard/renderer and fake browser camera; synthetic landmarks, not human perceptual eye-contact accuracy.'}
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=[
            '--no-sandbox','--disable-gpu','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream'])
        page=await browser.new_page(viewport={'width':1440,'height':1100});errors=[]
        await page.add_init_script("localStorage.setItem('robby.cameraAuto','false')")
        # UI/calibration remains explicitly synthetic; real CPU workers have
        # their own separate smoke/performance test.
        await page.add_init_script('''{
          const NativeWorker=window.Worker;
          window.Worker=class extends NativeWorker {
            constructor(url,opts){
              if(String(url).includes('face-camera-worker.js'))return {
                onmessage:null,onerror:null,terminate(){},
                postMessage(d){d.frame?.close();queueMicrotask(()=>this.onmessage?.({data:d.type==='load'?{type:'ready'}:{type:'result',result:window.gazeTestResult||{faceLandmarks:[]},at:d.at,ms:0,generation:d.generation}}));}
              };super(url,opts);
            }
          };
        }''')
        page.on('pageerror',lambda e:errors.append(str(e)))
        await page.route('**/vision_bundle.mjs',lambda r:r.fulfill(status=200,content_type='text/javascript',body=MOCK))
        try:
            await page.goto(url);await page.wait_for_function('window.robotFace?.eyeContact&&robotFace.rig.snapshot().renderCount>3')
            await page.locator('#recognition-settings > summary').click()
            await page.locator('details.settings:not(.recognition-settings)>summary').click()
            assert await page.locator('#utterance-strictness').input_value()=='0.32'
            await page.locator('#utterance-strictness').fill('0.4')
            await page.locator('#utterance-strictness').dispatch_event('change')
            assert await page.evaluate('localStorage.getItem("robby.voiceThreshold")')=='0.4'
            await page.get_by_role('button',name='Reset voice matching',exact=True).click()
            assert await page.locator('#utterance-strictness').input_value()=='0.32'
            report['speaker_default']={'cosine':.32,'custom_setting_persists':True,'reset_restores_default':True}
            report['math']=await page.evaluate('''async()=>{
             const {eyeObservation,predictGaze,fitGaze,GazeTracker,PRIOR}=await import('/static/gaze.js');
             function face(u=.5,v=.5,span=.08,roll=0,nose=.5){
              const points=Array.from({length:478},()=>({x:u,y:v,z:0}));
              for(const [a,b,sign] of [[33,133,-1],[362,263,1]])for(const index of [a,b])points[index]={x:u+sign*span/2*Math.cos(roll),y:v+sign*span/2*Math.sin(roll)*640/480,z:0};
              points[1]={x:nose,y:.4,z:-.2};return {faceLandmarks:[points],faceBlendshapes:[]};
             }
             window.makeGazeFace=face;window.gazeTestResult=face();
             const observe=r=>eyeObservation(r,640,480),o=observe(face()),turned=observe(face(.5,.5,.08,.7,.7));
             const a=predictGaze(o),b=predictGaze(turned);
             if(Math.abs(a.pan-b.pan)>.001||Math.abs(a.tilt-b.tilt)>.001||Math.abs(o.depth-turned.depth)>.001)throw Error('Rotation/nose incorrectly changes gaze');
             const left=predictGaze(observe(face(.3))),right=predictGaze(observe(face(.7))),up=predictGaze(observe(face(.5,.3))),down=predictGaze(observe(face(.5,.7)));
             if(!(left.pan>0&&right.pan<0&&up.tilt>0&&down.tilt<0))throw Error('Wrong position-follow axes');
             const near=observe(face(.5,.5,.16)),far=observe(face(.5,.5,.06));if(!(near.depth<far.depth))throw Error('Distance cue reversed');
             const target=o=>({pan:Math.atan(.08+o.rayX*.85+.035*(1/o.depth-1/.65))*180/Math.PI,tilt:Math.atan(-.03+o.rayY*.75-.015*(1/o.depth-1/.65))*180/Math.PI});
             const samples=[];for(const u of [.25,.5,.75])for(const v of [.3,.5,.7])for(const span of [.06,.10]){const observation=observe(face(u,v,span));samples.push({observation,...target(observation)});}
             const coefficients=fitGaze(samples);let old=0,improved=0;
             for(const u of [.35,.6])for(const v of [.4,.6])for(const span of [.07,.12]){const o=observe(face(u,v,span)),t=target(o),a=predictGaze(o),b=predictGaze(o,coefficients);old+=Math.hypot(a.pan-t.pan,a.tilt-t.tilt);improved+=Math.hypot(b.pan-t.pan,b.tilt-t.tilt);}
             if(improved>old*.35)throw Error('Calibration did not improve held-out synthetic formula');
             const tr=new GazeTracker(null);tr.observe(face(),640,480,100);if(!tr.read(100)||tr.read(1600)!==null)throw Error('Lost-face timeout failed');
             const {buildDeskGeometry}=await import('/static/desk-robot3d.js'),meshes=buildDeskGeometry();
             const black=meshes.find(m=>m.part==='bottom plate'),white=meshes.find(m=>m.part==='base shell');
             const ring=m=>m.points.filter(p=>Math.abs(p[1]-12)<1e-7&&Math.abs(Math.hypot(p[0],p[2])-109.5)<1e-5).map(p=>p.map(v=>Math.round(v*1e6)).join('/')).sort();
             if(ring(black).length!==96||JSON.stringify(ring(black))!==JSON.stringify(ring(white)))throw Error('Base material seam rings differ');
             return {rotation_changes_gaze:false,translation_axes_correct:true,near_depth_m:near.depth,far_depth_m:far.depth,
              synthetic_heldout_mean_error_before:old/8,synthetic_heldout_mean_error_after:improved/8,shared_base_ring_vertices:96,lost_face_returns_neutral:true};
            }''')
            factory=await page.evaluate('makeGazeFace.toString()')
            await page.add_init_script('window.makeGazeFace='+factory+';window.gazeTestResult=makeGazeFace();')
            await page.locator('.gaze-controls summary').click();await page.locator('#gaze-start').click()
            await page.wait_for_function('robotFace.eyeContact.state.active&&robotFace.gaze.latest')
            rate_before=await page.evaluate('({count:robotFace.cameraStats.inferences,at:performance.now()})')
            await page.wait_for_timeout(2200)
            rate_after=await page.evaluate('({count:robotFace.cameraStats.inferences,at:performance.now(),settings:document.querySelector("#video").srcObject.getVideoTracks()[0].getSettings()})')
            observed_fps=(rate_after['count']-rate_before['count'])*1000/(rate_after['at']-rate_before['at'])
            assert 0<observed_fps<=10.5,observed_fps
            assert rate_after['settings']['frameRate']<=10.1,rate_after
            report['camera_rate']={'inference_fps':observed_fps,'capture_fps':rate_after['settings']['frameRate'],'maximum_requested_fps':10}
            assert await page.locator('#gaze-tour').is_visible()
            assert '1 / 8' in await page.locator('.gaze-tour-step').text_content()
            await page.wait_for_timeout(900)
            await page.keyboard.press('x');assert await page.evaluate('!!robotFace.eyeContact.state.frozen')
            before=await page.evaluate('robotFace.eyeContact.state.frozen.observation')
            await page.evaluate('window.gazeTestResult=makeGazeFace(.25,.35,.16)');await page.wait_for_timeout(400)
            assert before==await page.evaluate('robotFace.eyeContact.state.frozen.observation')
            await page.keyboard.press('Shift+ArrowRight');await page.keyboard.press('ArrowUp');await page.wait_for_timeout(900)
            corrected=await page.evaluate('({pan:robotFace.rig.snapshot().pan,tilt:robotFace.rig.snapshot().tilt})')
            assert corrected['pan']>4.5 and corrected['tilt']>.8,corrected
            await page.keyboard.press('y');assert not await page.evaluate('!!robotFace.eyeContact.state.frozen')
            assert await page.evaluate('robotFace.gaze.samples.length')==1
            await page.fill('#message','');await page.locator('#message').press('x');await page.locator('#message').press('y')
            assert not await page.evaluate('!!robotFace.eyeContact.state.frozen')
            assert await page.locator('#message').input_value()=='xy'
            await page.locator('.desk-robot-canvas').focus();await page.keyboard.press('Escape')
            assert not await page.evaluate('robotFace.eyeContact.state.active')
            stored=await page.evaluate('localStorage.getItem("robby.eyeContact.v1")')
            assert len(stored)<1500 and 'image' not in stored.lower()
            await page.reload();await page.wait_for_function('window.robotFace?.eyeContact&&robotFace.rig.snapshot().renderCount>3')
            await page.locator('details.settings:not(.recognition-settings)>summary').click()
            assert await page.evaluate('robotFace.gaze.samples.length')==1
            await page.locator('.gaze-controls summary').click();await page.locator('#gaze-reset').click()
            assert await page.evaluate('robotFace.gaze.samples.length')==0
            await page.locator('#gaze-start').click();await page.wait_for_function('robotFace.eyeContact.state.active&&robotFace.gaze.latest')
            for u,v,span in [(.5,.5,.08),(.3,.5,.08),(.7,.5,.08),(.5,.35,.08),(.5,.65,.08),(.5,.5,.14),(.5,.5,.055),(.35,.4,.1)]:
                await page.evaluate('([u,v,s])=>window.gazeTestResult=makeGazeFace(u,v,s)',[u,v,span])
                await page.wait_for_timeout(250);await page.keyboard.press('y')
            assert not await page.evaluate('robotFace.eyeContact.state.active')
            assert await page.evaluate('robotFace.eyeContact.state.done')
            await page.keyboard.press('Enter');assert not await page.locator('#gaze-tour').is_visible()
            report['ui']={'freeze_keeps_original_observation':True,'arrow_correction':corrected,'y_saves_and_resumes':True,
                'typing_xy_does_not_calibrate':True,'numeric_local_persistence':True,'reset':True,'eight_step_keyboard_tour':True}
            await page.set_viewport_size({'width':390,'height':844})
            await page.locator('#gaze-start').click();await page.wait_for_function('robotFace.eyeContact.state.active')
            await page.wait_for_timeout(700)
            mobile=await page.evaluate('''()=>{
              const robot=document.querySelector('.desk-robot-canvas').getBoundingClientRect(),card=document.querySelector('#gaze-tour').getBoundingClientRect();
              return {no_overflow:document.documentElement.scrollWidth<=innerWidth,
                card_inside_view:card.left>=0&&card.right<=innerWidth&&card.top>=0&&card.bottom<=innerHeight,
                face_not_covered:robot.top+robot.height*.32<card.top};
            }''')
            assert all(mobile.values()),mobile
            report['ui']['mobile_tutorial']=mobile
            await page.keyboard.press('Escape');await page.set_viewport_size({'width':1440,'height':1100})
            report['renderer']=await page.evaluate('''()=>{
             const {rig,pose}=robotFace;rig.setManual(0,0);const input={pose:pose('neutral',1,0)};
             for(let i=0;i<100;i++)rig.update(input,100,1000);
             const canvas=document.querySelector('.desk-robot-canvas'),images=[];
             for(let i=0;i<5;i++){rig.update(input,100,1000);images.push(canvas.toDataURL());}
             const stable=new Set(images).size===1;if(!stable)throw Error('Static frame flicker');
             const timings=[];for(let i=0;i<50;i++){rig.setManual(-35+i*1.4,Math.sin(i/8)*10);rig.update(input,100,1000);timings.push(rig.snapshot().renderMs);}
             timings.sort((a,b)=>a-b);const s=rig.snapshot();rig.center();
             return {static_identical_frames:stable,motion_render_p95_ms:timings[Math.floor(timings.length*.95)],
                 motion_render_p50_ms:timings[Math.floor(timings.length*.5)],smooth_shading:s.smoothShading,depth_buffer:s.depthBuffer,mesh_faces:s.meshFaces};
            }''')
            assert report['renderer']['motion_render_p95_ms']<70,report['renderer']
            await page.screenshot(path=str(ROOT/'logs/gaze-rendering-desktop.png'),full_page=True)
            assert not errors,errors
            report.update(passed=True,finished=time.time(),page_errors=errors)
        finally:
            (ROOT/'logs/gaze-rendering-checks.json').write_text(json.dumps(report,indent=2));await browser.close()
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8793/desk-robot/');asyncio.run(main(p.parse_args().url))
