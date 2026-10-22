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
             const {AvatarActions,sampleAction,GRIMACES}=await import('/static/avatar-actions.js');
             const {StrokeTracker}=await import('/static/avatar-petting.js');
             const {pose}=await import('/static/emotions.js');const {geometry}=await import('/static/rig.js');
             const ok=(v,m)=>{if(!v)throw Error(m)},base=pose(),signatures=[];
             for(const variant of GRIMACES){const s={kind:'grimace',variant};
              const q=sampleAction(s,.375,base);signatures.push(JSON.stringify(geometry(q.pose,q.controls).points));
              for(let i=0;i<=100;i++){const v=sampleAction(s,i/100,base);ok(geometry(v.pose,v.controls).points.flat().every(Number.isFinite),'finite');}
             }ok(new Set(signatures).size===8,'eight distinct faces');
             const a=new AvatarActions();a.lastActivity=0;a.nextIdle=60000;
             a.frame(59999,base);ok(!a.active,'not before one minute');a.frame(60000,base);ok(a.active?.source==='idle','idle play at minute');
             ok(a.queue.some(q=>q.step.kind==='carousel'),'idle carousel');a.touch(60001);ok(!a.queue.length,'user cancels idle');
             a.nextIdle=0;a.frame(60002,base,{reduced:true});ok(!a.queue.length,'no reduced-motion idle');
             ok(sampleAction({kind:'carousel',direction:'clockwise'},1,base).motion.spin===-720,'two turns');
             ok(sampleAction({kind:'carousel',direction:'clockwise'},.5,base,true).motion.spin===0,'reduced carousel');
             const s=new StrokeTracker();s.begin(0,0,0);ok(s.finish(1000)===null,'stationary hold');
             s.begin(0,0,0);for(let i=1;i<=8;i++)s.move(i*10,0,i*100,false);ok(s.finish(900)===null,'off-head ignored');
             s.begin(0,0,0);for(let i=1;i<=8;i++)s.move(i*10,0,i*100,true);ok(s.finish(900)?.moves===8,'real stroke');
             return {grimaces:8,minute_boundary:true,carousel:true,reduced_motion:true,stroke_validation:true};
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
            await page.uncheck('#avatar-petting-enabled');await page.reload();await page.wait_for_function('window.robotFace?.petting')
            assert await page.evaluate('!robotFace.actions.pettingEnabled'),'Persistence'
            assert not errors,errors
            report.update(passed=True,errors=errors,finished=time.time())
        except Exception as e:report.update(error=str(e),errors=errors)
        finally:
            await page.evaluate('window.robo?.stop()');await browser.close()
            (ROOT/'logs/personality-browser-checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':asyncio.run(main())
