"""CPU-only browser checks. Clock fast-forward is explicit; no real minute wait."""
import asyncio,json,time
from pathlib import Path
from playwright.async_api import async_playwright
ROOT=Path(__file__).resolve().parent

async def main():
    report={'passed':False,'synthetic_pointer':True,'idle_clock_fast_forwarded':True,'real_microphone':False}
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu'])
        page=await browser.new_page(viewport={'width':1280,'height':1000});errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script("localStorage.setItem('robby.cameraAuto','false')")
        try:
            await page.goto('https://diane-asylum-understood-invoice.trycloudflare.com/desk-robot/')
            await page.wait_for_function('window.robotFace?.petting && robotFace.rig.snapshot().renderCount>2')
            # Do not occupy the live audio session during visual-only tests.
            await page.evaluate('robo.start=async()=>{}')
            report['contracts']=await page.evaluate('''async()=>{
             const {AvatarActions,sampleAction,GRIMACES,carouselProgress}=await import('/static/avatar-actions.js');
             const {StrokeTracker}=await import('/static/avatar-petting.js');
             const {pose}=await import('/static/emotions.js');const {geometry}=await import('/static/rig.js');
             const ok=(v,m)=>{if(!v)throw Error(m)},base=pose(),signatures=[];
             for(const variant of GRIMACES){const s={kind:'grimace',variant};
              const q=sampleAction(s,.375,base);signatures.push(JSON.stringify(geometry(q.pose,q.controls).points));
              for(let i=0;i<=100;i++){const v=sampleAction(s,i/100,base);ok(geometry(v.pose,v.controls).points.flat().every(Number.isFinite),'finite');}
             }ok(new Set(signatures).size===8,'eight distinct faces');
             const a=new AvatarActions();ok(a.idleDelayMs===20000,'20-second default');a.lastActivity=0;a.setIdleDelay(20);
             a.frame(19999,base);ok(!a.active,'not before 20 seconds');a.frame(20000,base);ok(a.active?.source==='idle','idle play at 20 seconds');
             ok(a.queue.some(q=>q.step.kind==='carousel'),'idle carousel');ok(a.nextIdle===a.queue.at(-1).end+20000,'pause after sequence');
             a.touch(20001);ok(!a.queue.length,'user cancels idle');
             a.setIdleDelay(60);ok(a.nextIdle===80001,'longer setting applies');a.setIdleDelay(5);ok(a.nextIdle===25001,'shorter setting applies');
             ok(!a.setIdleDelay(NaN)&&a.idleDelayMs===5000,'invalid setting rejected');
             a.nextIdle=0;a.frame(20002,base,{reduced:true});ok(!a.queue.length,'no reduced-motion idle');
             ok(Math.abs(sampleAction({kind:'carousel',direction:'clockwise'},1,base).motion.spin+1080)<1e-6,'three turns');
             ok(sampleAction({kind:'carousel',direction:'clockwise'},.5,base,true).motion.spin===0,'reduced carousel');
             const speed=p=>(carouselProgress(p+.001)-carouselProgress(p))/.001;
             ok(speed(.1)<speed(.3)&&speed(.3)<speed(.6)&&speed(.6)<speed(.78),'accelerates through ride');
             ok(speed(.99)<speed(.8),'gentle final brake');
             const early=sampleAction({kind:'carousel'},.25,base),late=sampleAction({kind:'carousel'},.8,base);
             ok(late.pose.actions.smile>early.pose.actions.smile&&late.pose.actions.mouth_open>early.pose.actions.mouth_open,'growing elation');
             ok(sampleAction({kind:'afterglow'},.3,base).pose.actions.smile>.8,'happy afterglow');
             ok(sampleAction({kind:'puppy'},.5,base).pose.actions.gaze_y<0,'puppy looks upward');
             ok(!a.setIntensity('playIntensity',NaN),'reject invalid intensity');
             const low={playIntensity:.25,cuddleIntensity:.25},high={playIntensity:1,cuddleIntensity:1};
             for(const kind of ['grimace','carousel','afterglow','pet','puppy']){
              const s={kind,variant:'goofy',direction:'clockwise'};
              const l=sampleAction(s,.5,base,false,low),h=sampleAction(s,.5,base,false,high);
              ok(l.weight<h.weight,'intensity changes '+kind);
             }
             ok(Math.abs(sampleAction({kind:'carousel',direction:'clockwise'},1,base,false,low).motion.spin+360)<1e-6,'low intensity one complete turn');
             a.setIntensity('playIntensity',0);a.nextIdle=0;a.frame(99999,base);ok(!a.queue.length,'zero disables idle play');
             a.setIntensity('cuddleIntensity',0);ok(sampleAction({kind:'pet'},.5,base,false,a).weight===0,'zero cuddle expression');
             a.setIntensity('cuddleIntensity',.5);ok(a.petCooldownMs===24000&&a.inviteStartMs===390000,'scaled cuddling cooldowns');
             a.setIntensity('playIntensity',1);a.setIntensity('cuddleIntensity',1);
             a.play([{kind:'carousel',direction:'clockwise',duration_s:6}],'optional',100000);a.frame(101000,base,{busy:true});ok(a.queue.length,'own audio keeps animation');a.touch(101001);ok(!a.queue.length,'user activity cancels optional animation');
             const s=new StrokeTracker();s.begin(0,0,0);ok(s.finish(1000)===null,'stationary hold');
             s.begin(0,0,0);for(let i=1;i<=8;i++)s.move(i*10,0,i*100,false);ok(s.finish(900)===null,'off-head ignored');
             s.begin(0,0,0);for(let i=1;i<=8;i++)s.move(i*10,0,i*100,true);
             ok(s.sample(850)?.moves===8&&s.active,'reaction possible while mouse held');ok(s.finish(900)?.moves===8,'real stroke');
             return {grimaces:8,idle_default_seconds:20,adjustable_delay:true,carousel:true,reduced_motion:true,stroke_validation:true};
            }''')
            canvas=page.locator('#avatar canvas');await canvas.scroll_into_view_if_needed()
            h=await page.evaluate('''()=>{const r=document.querySelector('#avatar canvas').getBoundingClientRect(),h=robotFace.rig.headRegion();return {x:r.left+h.x,y:r.top+h.y,rx:h.rx,ry:h.ry,view:robotFace.rig.state.view}}''')
            await page.mouse.move(h['x'],h['y']-h['ry']*.45)
            assert 'data:image/svg+xml' in await canvas.evaluate('(e)=>e.style.cursor')
            await page.mouse.down()
            for i in range(12):
                await page.mouse.move(h['x']+(-1 if i%2 else 1)*h['rx']*.35,h['y']-h['ry']*.45,steps=3)
                await page.wait_for_timeout(90)
            state=await page.evaluate('robotFace.rig.snapshot()')
            report['petted_eye_open']=state['pose']['actions']['eye_open']
            assert report['petted_eye_open']<.6,'No soft petting expression'
            assert abs(state['viewDegrees']-h['view']*180/3.14159265)<.01,'Petting accidentally orbited view'
            await page.screenshot(path=str(ROOT/'logs/personality-petting.png'))
            await page.mouse.up();await page.wait_for_timeout(3000)
            assert not await page.evaluate('robotFace.petting.stroke.active')
            await page.locator('#avatar-action-controls').evaluate('(e)=>e.open=true')
            assert await page.locator('#avatar-idle-delay').input_value()=='20'
            await page.locator('#avatar-idle-delay').evaluate("e=>{e.value='75';e.dispatchEvent(new Event('input',{bubbles:true}))}")
            assert await page.evaluate('robotFace.actions.idleDelayMs===75000')
            for id,value in [('avatar-play-intensity','25'),('avatar-cuddle-intensity','60')]:
                await page.locator('#'+id).evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}))}',value)
            assert await page.evaluate('robotFace.actions.playIntensity===.25&&robotFace.actions.cuddleIntensity===.6')
            await page.uncheck('#avatar-petting-enabled');await page.reload();await page.wait_for_function('window.robotFace?.petting')
            assert await page.evaluate('!robotFace.actions.pettingEnabled'),'Persistence'
            assert await page.evaluate('robotFace.actions.idleDelayMs===75000'),'Idle delay persistence'
            assert await page.evaluate('robotFace.actions.playIntensity===.25&&robotFace.actions.cuddleIntensity===.6'),'Intensity persistence'
            assert await page.locator('#avatar-cuddle-intensity-value').text_content()=='60%'
            assert await page.locator('#avatar-idle-delay-value').text_content()=='75 seconds'
            assert not errors,errors
            report.update(passed=True,errors=errors,finished=time.time())
        except Exception as e:report.update(error=str(e),errors=errors)
        finally:
            await page.evaluate('window.robo?.stop()');await browser.close()
            (ROOT/'logs/personality-browser-checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':asyncio.run(main())
