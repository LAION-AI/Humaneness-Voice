"""CPU-only Playwright animation, UI, comfort and manual regression checks."""
import argparse,asyncio,json,time
from pathlib import Path
from playwright.async_api import async_playwright
ROOT=Path(__file__).resolve().parent

async def main(url):
    report={'passed':False,'url':url,'camera_or_mic_used':False,'gpu_disabled':True}
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu'])
        page=await browser.new_page(viewport={'width':1280,'height':1000});errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script("localStorage.setItem('robby.cameraAuto','false')")
        try:
            await page.goto(url+'/desk-robot/');await page.wait_for_function('window.robotFace?.actions && window.robotFace.rig.snapshot().renderCount>2')
            report['contracts']=await page.evaluate('''async()=>{
             const {AvatarActions,sampleAction,validateSteps}=await import('/static/avatar-actions.js');
             const {pose,profiles}=await import('/static/emotions.js');const {geometry}=await import('/static/rig.js');
             const assert=(v,m)=>{if(!v)throw Error(m);},a=new AvatarActions(),base=pose();
             assert(validateSteps([{kind:'shell',duration_s:2}]).length===0,'unknown action');
             assert(validateSteps([{kind:'spin',direction:'up',duration_s:2}]).length===0,'direction');
             assert(validateSteps(Array(6).fill({kind:'nod',duration_s:6})).length===0,'budget');
             const cw=sampleAction({kind:'spin',direction:'clockwise'},.5,base),ccw=sampleAction({kind:'spin',direction:'counterclockwise'},.5,base);
             assert(cw.motion.spin===-180&&ccw.motion.spin===180,'direction signs');
             assert(sampleAction({kind:'spin',direction:'clockwise'},1,base).motion.spin===-360,'full turn');
             assert(sampleAction({kind:'spin',direction:'clockwise'},.5,base,true).motion.spin===0,'reduced motion');
             const g=sampleAction({kind:'grimace',variant:'crossed'},.5,base),points=geometry(g.pose,g.controls).pupils;
             assert(points[0][0][0]>270&&points[1][0][0]<530,'independent crossed eyes');
             for(const kind of ['nod','shake','eye_roll','grimace'])for(const t of [0,.01,.25,.5,.75,.99,1]){
              const v=sampleAction({kind,variant:'opposite'},t,base);assert(Math.abs(v.motion.pan)<=17&&Math.abs(v.motion.tilt)<=9,'motion bounds');
              assert(geometry(v.pose,v.controls).points.flat().every(Number.isFinite),'finite geometry');
             }
             const archetypes=profiles.map(p=>{const x=sampleAction({kind:'emotion',emotion:p.slug},.5,base);assert(JSON.stringify(x.pose.actions)===JSON.stringify(pose(p.slug,1,0).actions),'archetype '+p.slug);return JSON.stringify(geometry(x.pose).points);});
             assert(new Set(archetypes).size===40,'40 distinct expressions');
             a.play([{kind:'nod',duration_s:1},{kind:'shake',duration_s:1}],'preview',100);
             assert(a.frame(600,base).kind==='nod'&&a.frame(1600,base).kind==='shake','sequence order');
             a.clear();assert(!a.queue.length&&!a.plans.size,'clear');
             a.set(12,[{kind:'nod',duration_s:2,sentence:1}],{sentences:[{start_s:0},{start_s:4}]});
             a.frame(100,base,{position:{id:12,sample:24000}});assert(!a.queue.length,'not network arrival');
             a.frame(200,base,{position:{id:12,sample:96000}});assert(a.queue.length===1,'playback sentence trigger');
             a.frame(300,base,{position:{id:12,sample:100000}});assert(a.queue.length===1,'no duplicates');
             a.clear();a.nextIdle=0;a.frame(50000,base,{busy:true});assert(!a.queue.length,'no idle while busy');
             a.nextIdle=0;a.frame(50001,base);assert(a.active.source==='idle','idle available');a.touch(50002);assert(!a.queue.length,'speech cancels idle');
             a.nextIdle=0;a.frame(50003,base,{reduced:true});assert(!a.queue.length,'no reduced-motion idle');
             a.clear();a.enabled=false;assert(!a.play([{kind:'nod',duration_s:2}]),'master disable');a.enabled=true;
             assert(a.listen('nod',100000)&&!a.listen('shake',100001),'listening cooldown');a.clear();a.listeningEnabled=false;assert(!a.listen('nod',120000),'listening disabled');
             a.set(2,[{kind:'nod',duration_s:2}],{});a.frame(1,base,{calibrating:true});assert(!a.plans.size,'calibration clears');
             a.play([{kind:'nod',duration_s:2}]);a.frame(2,base,{hidden:true});assert(!a.queue.length,'hidden clears');
             return {unique_archetypes:40,sequence_order:true,independent_eyes:true,playback_anchored:true,reduced_motion:true,disable_and_clear:true,idle_and_listening_cooldowns:true};
            }''')
            await page.locator('#avatar-action-controls').evaluate('(e)=>e.open=true')
            await page.select_option('#avatar-trick','2');await page.click('#avatar-perform');await page.wait_for_timeout(1300)
            report['actual_spin_visual_pan']=await page.evaluate('robotFace.rig.snapshot().visualPan')
            assert abs(report['actual_spin_visual_pan'])>10,'Visible spin did not move'
            await page.click('#avatar-stop');await page.wait_for_timeout(250)
            assert await page.evaluate('Math.abs(robotFace.rig.snapshot().visualPan-robotFace.rig.snapshot().pan)<.01')
            await page.uncheck('#avatar-actions-enabled');await page.reload();await page.wait_for_function('window.robotFace?.actions')
            assert await page.evaluate('robotFace.actions.enabled===false'),'Setting not persisted'
            await page.locator('#avatar-action-controls').evaluate('(e)=>e.open=true');await page.check('#avatar-actions-enabled')
            await page.select_option('#avatar-trick','7');await page.click('#avatar-perform');await page.wait_for_timeout(650)
            await page.screenshot(path=str(ROOT/'logs/avatar-grimace.png'))
            report['render_p95_ms']=await page.evaluate('robotFace.rig.snapshot().renderP95Ms')
            await page.goto(url+'/static/avatar-manual.html');await page.wait_for_function('window.manualPreview?.profiles.length===40')
            assert await page.locator('#expression-cards .card').count()==40
            await page.select_option('#manual-action','11');await page.click('#manual-play');await page.wait_for_timeout(800)
            await page.screenshot(path=str(ROOT/'logs/avatar-manual.png'))
            await page.set_viewport_size({'width':390,'height':844});await page.wait_for_timeout(300)
            assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'Mobile overflow'
            assert not errors,errors;report.update(passed=True,errors=errors,finished=time.time())
        except Exception as exc:report.update(error=str(exc),errors=errors)
        finally:
            await browser.close();(ROOT/'logs/avatar-browser-checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8793');asyncio.run(main(p.parse_args().url.rstrip('/')))
