"""Real pointer -> event -> configured LLM -> streamed TTS, plus optional cancellation.

Audio activity for cancellation is injected at the UI callback, not a physical
microphone or an ASR-accuracy test. Uses the single live conversation slot.
"""
import asyncio,json,time
from pathlib import Path
from playwright.async_api import async_playwright
ROOT=Path(__file__).resolve().parent

async def main():
    report={'passed':False,'real_llm_tts':True,'synthetic_pointer':True,'cancel_activity_injected':True,'physical_microphone':False}
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page(viewport={'width':1280,'height':1000});errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script("localStorage.setItem('robby.cameraAuto','false')")
        try:
            await page.goto('https://diane-asylum-understood-invoice.trycloudflare.com/desk-robot/')
            await page.wait_for_function('window.robotFace?.petting && !document.getElementById("call").disabled',timeout=60000)
            await page.evaluate('''()=>{window.petEvents=[];const old=robo.on.message;robo.on.message=m=>{petEvents.push({...m,at:performance.now()});old(m);}}''')
            await page.locator('#avatar canvas').scroll_into_view_if_needed()
            h=await page.evaluate('''()=>{const r=document.querySelector('#avatar canvas').getBoundingClientRect(),h=robotFace.rig.headRegion();return {x:r.left+h.x,y:r.top+h.y,rx:h.rx,ry:h.ry}}''')
            await page.mouse.move(h['x'],h['y']-h['ry']*.4);await page.mouse.down()
            await page.wait_for_function('petEvents.some(m=>m.type==="ready")',timeout=45000)
            for i in range(12):
                await page.mouse.move(h['x']+(-1 if i%2 else 1)*h['rx']*.35,h['y']-h['ry']*.4,steps=3)
                await page.wait_for_timeout(80)
            await page.wait_for_function('petEvents.some(m=>m.type==="avatar_event"&&m.event==="petted"&&m.accepted)',timeout=10000)
            report['reaction_started_while_held']=await page.evaluate('robotFace.petting.stroke.active')
            assert report['reaction_started_while_held']
            await page.mouse.up()
            await page.wait_for_function('petEvents.some(m=>m.type==="reply")&&robo.metrics.reply_chunks>0',timeout=45000)
            report['reply']=await page.evaluate('petEvents.find(m=>m.type==="reply").text')
            report['audio_chunks']=await page.evaluate('robo.metrics.reply_chunks')
            await page.evaluate('robo.on.input({rms:.02,sent:0,state:"running"})')
            await page.wait_for_function('petEvents.some(m=>m.type==="barge")',timeout=10000)
            await page.wait_for_function('petEvents.some(m=>m.type==="turn_done")',timeout=15000)
            report['cancelled_on_activity']=True
            await page.evaluate('robo.send("avatar:"+JSON.stringify({event:"petted",duration_ms:1500}))')
            await page.wait_for_function('petEvents.filter(m=>m.type==="avatar_event").length>=2')
            report['repeat_rejected']=await page.evaluate('petEvents.filter(m=>m.type==="avatar_event").at(-1).accepted===false')
            assert report['repeat_rejected'],'Missing server cooldown'
            await page.wait_for_timeout(12500)
            await page.locator('#avatar canvas').scroll_into_view_if_needed()
            await page.mouse.move(h['x'],h['y']-h['ry']*.4);await page.mouse.down()
            for i in range(12):
                await page.mouse.move(h['x']+(-1 if i%2 else 1)*h['rx']*.35,h['y']-h['ry']*.4,steps=3)
                await page.wait_for_timeout(80)
            await page.mouse.up()
            await page.wait_for_function('petEvents.filter(m=>m.type==="reply").length>=2&&robo.metrics.reply_chunks>0',timeout=45000)
            report['second_stroke_reply']=await page.evaluate('petEvents.filter(m=>m.type==="reply").at(-1).text')
            await page.evaluate('robo.on.input({rms:.02,sent:0,state:"running"})')
            await page.wait_for_function('petEvents.filter(m=>m.type==="turn_done").length>=2',timeout=15000)
            # Fast-forward only the browser idle timer. LLM, server cooldown,
            # TTS, PCM playback and animation timing below remain real.
            await page.evaluate('robotFace.actions.idleIndex=0;robotFace.actions.nextIdle=0')
            await page.wait_for_function('petEvents.some(m=>m.type==="reply"&&m.avatar_event==="play_vocal")',timeout=45000)
            report['carousel_reply']=await page.evaluate('petEvents.find(m=>m.type==="reply"&&m.avatar_event==="play_vocal").text')
            await page.wait_for_function('robotFace.actions.active?.kind==="carousel"&&Math.abs(robotFace.rig.snapshot().visualPan)>500',timeout=15000)
            report['carousel_playback_pose']=True
            await page.wait_for_function('robotFace.actions.active?.kind==="afterglow"&&robotFace.rig.snapshot().pose.actions.smile>.7',timeout=10000)
            report['afterglow_visible']=True
            await page.evaluate('robo.send("avatar:"+JSON.stringify({event:"idle_play",variant:"fish",carousel:true}))')
            await page.wait_for_timeout(1200)
            await page.fill('#message','Was machst du mit deinem kleinen Karussell und dem Fischgesicht? Erklär es kurz.')
            await page.locator('#form').evaluate('(e)=>e.requestSubmit()')
            await page.wait_for_function('petEvents.filter(m=>m.type==="reply").length>=4',timeout=45000)
            report['explanation']=await page.evaluate('petEvents.filter(m=>m.type==="reply").at(-1).text')
            await page.wait_for_function('robo.sources.length===0&&petEvents.filter(m=>m.type==="turn_done").length>=4',timeout=45000)
            await page.fill('#message','Dreh dich bitte um 360 Grad gegen den Uhrzeigersinn und rede dabei.')
            await page.locator('#form').evaluate('(e)=>e.requestSubmit()')
            await page.wait_for_function('petEvents.filter(m=>m.type==="reply").length>=5',timeout=45000)
            report['requested_spin_reply']=await page.evaluate('petEvents.filter(m=>m.type==="reply").at(-1).text')
            await page.wait_for_function('robotFace.actions.active?.kind==="spin"&&robo.sources.length>0&&robotFace.rig.snapshot().visualPan>40',timeout=20000)
            report['requested_spin_while_real_tts_plays']=True
            await page.wait_for_function('robotFace.rig.snapshot().visualPan>330',timeout=10000)
            report['requested_counterclockwise_full_turn']=True
            assert not errors,errors
            report.update(passed=True,errors=errors,finished=time.time())
        except Exception as exc:
            report.update(error=str(exc),errors=errors)
            try:report['events']=await page.evaluate('window.petEvents||[]')
            except Exception:pass
        finally:
            await page.evaluate('window.robo?.stop()');await browser.close()
            (ROOT/'logs/personality-live-browser.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps(report,indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':asyncio.run(main())
