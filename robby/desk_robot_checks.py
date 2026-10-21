"""Real Chromium software-only 3D, kinematics, expressions and responsive QA.

No microphone, GPU model, camera or hosted API is used by this test. --disable-gpu
proves that the model does not require hardware 3D acceleration. Timings are
observations of this host, not promises for every phone or Raspberry Pi.
"""
import argparse,asyncio,json,time
from pathlib import Path
from playwright.async_api import async_playwright

ROOT=Path(__file__).resolve().parent

async def run(args):
    (ROOT/'logs/desk-robot-checks.json').write_text(json.dumps({'passed':False,'running':True,'url':args.url}))
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,
            args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
        try:
            page=await browser.new_page(viewport={'width':1440,'height':1100});errors=[];requests=[]
            page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:requests.append(r.url))
            await page.goto(args.url);await page.wait_for_function('window.robotFace?.rig.snapshot().renderCount>3',timeout=30000)
            await page.wait_for_timeout(700)
            await page.screenshot(path=str(ROOT/'logs/desk-robot-desktop.png'))
            result=await page.evaluate('''async()=>{
              const {rig,pose,profiles}=window.robotFace;
              const {DIMENSIONS,buildDeskGeometry}=await import('/static/desk-robot3d.js');
              const meshes=buildDeskGeometry(),baseline=rig.snapshot();
              if(Math.abs(baseline.viewDegrees)>.001||Math.abs(baseline.elevationDegrees)>.001||baseline.eyeContactErrorDegrees>.15)throw Error('Initial gaze is not frontal and eye-level: '+JSON.stringify(baseline));
              rig.setManual(130,30);for(let i=0;i<100;i++)rig.update({pose:pose()},100,0);
              const limit=rig.snapshot();if(limit.pan>90||limit.tilt>15||limit.roll!==0)throw Error('Mechanical limits violated');
              rig.setManual(-130,-30);for(let i=0;i<100;i++)rig.update({pose:pose()},100,0);
              const negative=rig.snapshot();if(negative.pan < -90||negative.tilt < -15)throw Error('Negative limits violated');
              rig.setManual(0,0,false);for(let i=0;i<100;i++)rig.update({pose:pose(),x:1,y:1},100,0);
              const gaze=rig.snapshot();if(gaze.pan<40||gaze.tilt> -10||gaze.roll!==0)throw Error('Face-follow control did not steer pan/tilt');
              rig.center();rig.setManual(0,0);for(let i=0;i<70;i++)rig.update({pose:pose()},100,0);
              const geometries=[],pixels=[],captures=[];
              for(const profile of profiles){const value=pose(profile.slug,.95,0,42);for(let i=0;i<25;i++)rig.update({pose:value},100,0);
                const s=rig.snapshot();geometries.push(JSON.stringify(s.geometry.points.slice(12,89).map(p=>p.map(x=>Math.round(x*100)))));
                const canvas=document.querySelector('.desk-robot-canvas');const image=canvas.toDataURL();pixels.push(image);captures.push({name:profile.label,image});
              }
              const uniqueGeometry=new Set(geometries).size,uniqueRenders=new Set(pixels).size;
              if(uniqueGeometry!==40||uniqueRenders!==40)throw Error('Expressions collapsed: '+uniqueGeometry+'/'+uniqueRenders);
              rig.center();for(let i=0;i<50;i++)rig.update({pose:pose()},100,0);
              window.deskCaptures=captures;
              return {renderer:baseline.renderer,dimensions:DIMENSIONS,mesh_faces:meshes.reduce((n,m)=>n+m.polys.length,0),
                positive_limits:{pan:limit.pan,tilt:limit.tilt,roll:limit.roll},negative_limits:{pan:negative.pan,tilt:negative.tilt,roll:negative.roll},
                simulated_gaze_follow:{pan:gaze.pan,tilt:gaze.tilt,roll:gaze.roll,physical_camera_accuracy_measured:false},
                unique_expression_geometry:uniqueGeometry,unique_expression_renders:uniqueRenders,
                initial_eye_contact_error_degrees:baseline.eyeContactErrorDegrees,
                initial_view_degrees:baseline.viewDegrees,initial_elevation_degrees:baseline.elevationDegrees,
                snapshot:rig.snapshot(),no_svg_arms:!document.querySelector('#arm-left,#arm-right'),webgl_used:false};
            }''')
            # Actual animation cadence (not just a requested 30-fps setting).
            before=await page.evaluate('window.robotFace.rig.snapshot().renderCount');t0=time.monotonic()
            await page.wait_for_timeout(2500)
            after=await page.evaluate('window.robotFace.rig.snapshot()')
            result['observed_render_fps']=(after['renderCount']-before)/(time.monotonic()-t0)
            result['render_p95_ms']=after['renderP95Ms'];result.pop('snapshot',None)
            await page.click('.desk-controls:not(.gaze-controls) summary');await page.click('#desk-threequarter')
            await page.fill('#desk-explode','0.65');await page.dispatch_event('#desk-explode','input');await page.wait_for_timeout(200)
            result['exploded_control']=await page.evaluate('window.robotFace.rig.snapshot().exploded')
            await page.screenshot(path=str(ROOT/'logs/desk-robot-exploded.png'))
            await page.click('#desk-center');await page.fill('#desk-explode','0');await page.dispatch_event('#desk-explode','input')
            await page.wait_for_timeout(600)
            assert await page.evaluate('Math.abs(robotFace.rig.state.view)<1e-6&&Math.abs(robotFace.rig.state.elevation)<1e-6')
            await page.click('#desk-threequarter');await page.locator('.desk-robot-canvas').press('Home')
            assert await page.evaluate('Math.abs(robotFace.rig.state.view)<1e-6&&Math.abs(robotFace.rig.state.elevation)<1e-6')
            await page.click('#desk-threequarter');await page.click('#desk-front')
            assert await page.evaluate('Math.abs(robotFace.rig.state.view)<1e-6&&Math.abs(robotFace.rig.state.elevation)<1e-6')
            result['reset_home_and_front_are_frontal']=True
            await page.set_viewport_size({'width':390,'height':844});await page.wait_for_timeout(300)
            result['mobile_no_overflow']=await page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            await page.screenshot(path=str(ROOT/'logs/desk-robot-mobile.png'))
            assert result['mobile_no_overflow'],result
            captures=await page.evaluate('window.deskCaptures')
            gallery=await browser.new_page(viewport={'width':1300,'height':950})
            await gallery.set_content('<body style="margin:0;background:#f7f5ef"></body>')
            await gallery.evaluate('''captures=>{const sheet=document.createElement('section');sheet.id='desk-contact-sheet';sheet.style.cssText='display:grid;grid-template-columns:repeat(8,150px);gap:6px;background:#f7f5ef;padding:15px;width:1260px';for(const item of captures){const cell=document.createElement('div');const image=document.createElement('img');image.src=item.image;image.style.cssText='width:150px;height:150px;object-fit:contain';const caption=document.createElement('p');caption.textContent=item.name;caption.style.cssText='font:10px system-ui;text-align:center;margin:0';cell.append(image,caption);sheet.append(cell);}document.body.append(sheet);}''',captures)
            await gallery.wait_for_timeout(200)
            await gallery.locator('#desk-contact-sheet').screenshot(path=str(ROOT/'logs/desk-robot-40-expressions.png'))
            await gallery.close()
            result['page_errors']=errors;result['hardware_gpu_disabled']=True
            result['remote_mesh_downloads']=[u for u in requests if any(ext in u.lower() for ext in ('.glb','.fbx','.3mf','.obj'))]
            assert not errors and not result['remote_mesh_downloads'],result
            assert result['render_p95_ms']<50,'Software renderer exceeded 20-fps frame budget on this host'
            result['passed']=True
            (ROOT/'logs/desk-robot-checks.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(result,indent=2))
        finally:await browser.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8793/desk-robot/');asyncio.run(run(p.parse_args()))
