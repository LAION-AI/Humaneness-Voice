"""Actual camera callback -> CPU landmarks -> face selection -> 3D head.

Public NASA image on a moving canvas camera, not physical gaze accuracy.
"""
import asyncio,json,time
from pathlib import Path
from playwright.async_api import async_playwright

async def main():
    report={'passed':False,'physical_camera':False,'real_landmark_model':True,'started':time.time()}
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu'])
        page=await browser.new_page(viewport={'width':1280,'height':1000});errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script('''
          localStorage.setItem('robby.cameraAuto','true');localStorage.setItem('robby.followPosition','true');
          const original=HTMLVideoElement.prototype.requestVideoFrameCallback;
          if(original)HTMLVideoElement.prototype.requestVideoFrameCallback=function(cb){return original.call(this,(now,m)=>cb(now,window.testBrokenClock?{...m,captureTime:1}:m));};
          navigator.mediaDevices.getUserMedia=async()=>{
            const image=new Image();image.crossOrigin='anonymous';image.src='https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/astronaut.png';await image.decode();
            const canvas=document.createElement('canvas');canvas.width=640;canvas.height=480;
            const ctx=canvas.getContext('2d');window.cameraPosition={x:320,y:240};
            const paint=()=>{ctx.fillStyle='#bbb';ctx.fillRect(0,0,640,480);const {x,y}=cameraPosition;ctx.drawImage(image,100,0,300,300,x-150,y-150,300,300);};
            paint();setInterval(paint,100);return canvas.captureStream(10);
          };
        ''')
        try:
            await page.goto('http://127.0.0.1:8793/desk-robot/');await page.wait_for_function('window.robotFace?.cameraStats.inferences>5',timeout=60000)
            samples=[]
            for x,y in [(320,240),(190,240),(450,240),(320,160),(320,320)]:
                await page.evaluate('([x,y])=>Object.assign(cameraPosition,{x,y})',[x,y])
                await page.wait_for_timeout(2200)
                samples.append(await page.evaluate('''()=>({position:{...cameraPosition},gaze:robotFace.gaze.read(),latest:robotFace.gaze.latest,age:performance.now()-robotFace.gaze.at,
                  pan:robotFace.rig.state.pan,tilt:robotFace.rig.state.tilt,active:robotFace.rig.snapshot().gazeActive,faces:robotFace.multimodal.tracks.length,
                  status:document.querySelector('#follow-status').textContent,manual:robotFace.rig.state.manual})'''))
            report['samples']=samples
            assert all(s['active'] and s['faces'] for s in samples),samples
            assert samples[1]['pan']>samples[0]['pan']+8 and samples[2]['pan']<samples[0]['pan']-8,samples
            assert samples[3]['tilt']>samples[4]['tilt']+8,samples
            await page.evaluate('window.testBrokenClock=true;cameraPosition={x:190,y:240}')
            await page.wait_for_timeout(2200)
            report['stalled_capture_clock']=await page.evaluate('({pan:robotFace.rig.state.pan,active:robotFace.rig.snapshot().gazeActive,fallback:robotFace.cameraStats.timestampFallback})')
            assert report['stalled_capture_clock']['active'] and report['stalled_capture_clock']['pan']>10 and report['stalled_capture_clock']['fallback']
            await page.evaluate('robotFace.eyeContact.start()');await page.wait_for_timeout(400)
            await page.evaluate('robotFace.eyeContact.freeze()');assert await page.evaluate('!!robotFace.eyeContact.state.frozen')
            await page.locator('#follow-center').click();await page.wait_for_timeout(500)
            assert not await page.evaluate('robotFace.eyeContact.state.frozen||robotFace.eyeContact.state.active||robotFace.rig.state.manual')
            assert await page.evaluate('robotFace.rig.snapshot().gazeActive')
            report['resume_exits_frozen_calibration']=True
            report['audio_first_defaults']=await page.evaluate('''()=>({face_vad:robotFace.multimodal.settings.ignoreOffscreenSpeakers,
              face_id:robotFace.multimodal.settings.faceIdEnabled,face_id_worker:!!robotFace.multimodal.worker,
              embedding_calls:robotFace.multimodal.metrics.embeddingCalls})''')
            assert report['audio_first_defaults']=={'face_vad':False,'face_id':False,'face_id_worker':False,'embedding_calls':0}
            assert not errors,errors
            report['passed']=True
        finally:
            report.update(page_errors=errors,finished=time.time());Path('logs/follow-pipeline-checks.json').write_text(json.dumps(report,indent=2));await browser.close()
    print(json.dumps(report,indent=2))
if __name__=='__main__':asyncio.run(main())
