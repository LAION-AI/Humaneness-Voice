"""Real HTTPS/ASR/TTS focus regression in CPU Chrome; no microphone recordings."""
import asyncio,base64,json,time
from pathlib import Path
import numpy as np
from playwright.async_api import async_playwright
from speaker_live.live_test import audio
ROOT=Path(__file__).resolve().parent
URL='https://arrive-locations-parade-research.trycloudflare.com'

async def main():
    report={'passed':False,'physical_microphone_used':False,'started':time.time(),'cases':[]};errors=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page(viewport={'width':1440,'height':1100});page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script("localStorage.setItem('robby.llmSelection',JSON.stringify({model:'local',provider:'local'}));localStorage.setItem('robby.cameraAuto','false');localStorage.setItem('robby.avatar-idle-enabled','false');localStorage.setItem('robby.avatar-pet-invites-enabled','false');localStorage.setItem('robby.speechPauseMs','600')")
        async def events():return await page.evaluate('focusEvents')
        async def feed(x=None,seconds=0):
            wave=np.asarray(x if x is not None else np.zeros(round(seconds*16000)),dtype='<f4')
            data=base64.b64encode(wave.tobytes()).decode()
            await page.evaluate('''async data=>{
             const raw=Uint8Array.from(atob(data),c=>c.charCodeAt(0)),x=new Float32Array(raw.buffer),start=performance.now();
             for(let at=0;at<x.length;at+=320){robo.mic(robo.ctx.currentTime,x.subarray(at,at+320));await new Promise(r=>setTimeout(r,Math.max(0,start+(at+320)/16-performance.now())))}
            }''',data)
        async def settle(limit=45):
            for _ in range(limit):
                await feed(seconds=1.)
                if await page.evaluate('focusEvents.filter(e=>e.type==="endpoint").length<=focusEvents.filter(e=>e.type==="turn_done").length&&!robo.sources.length'):return
            raise TimeoutError('Reply did not settle')
        try:
            await page.goto(URL+'/desk-robot/');await page.wait_for_function('window.robo&&!document.querySelector("#call").disabled',timeout=60000)
            await page.evaluate('''()=>{window.focusEvents=[];const old=robo.on.message;robo.on.message=m=>{if(!['state','input_status','asr_preview','speech_activity'].includes(m.type))focusEvents.push({...m,at:performance.now()});old(m)}}''')
            assert await page.is_checked('#conversation-focus')
            await page.evaluate('robo.start(false)');await page.wait_for_function('focusEvents.some(e=>e.type==="ready")')
            await feed(seconds=1.);await feed(audio('christoph-intro'));await feed(seconds=1.8)
            await page.wait_for_function('focusEvents.some(e=>e.type==="endpoint"&&e.speaker_name==="Christoph")',timeout=15000)
            await settle();first=next(e for e in await events() if e['type']=='endpoint')
            report['cases'].append({'case':'wake-free-compatible introduction','speaker_id':first['speaker_id'],'passed':True})
            count=sum(e['type']=='endpoint' for e in await events())
            await feed(audio('selma-followup'));await feed(seconds=2.5)
            assert sum(e['type']=='endpoint' for e in await events())==count,'Background took the turn'
            assert any(e['type']=='speaker_focus' and not e.get('accepted') for e in await events())
            report['cases'].append({'case':'different voice while listening does not take over','passed':True})
            await feed(audio('christoph-followup'));await feed(seconds=2.)
            await page.wait_for_function('(n)=>focusEvents.filter(e=>e.type==="endpoint").length>n',arg=count,timeout=15000)
            endpoints=[e for e in await events() if e['type']=='endpoint'];assert endpoints[-1]['speaker_id']==first['speaker_id']
            await page.wait_for_function('robo.sources.length>0',timeout=30000)
            report['cases'].append({'case':'original speaker returns with a different sentence','passed':True})
            await page.click('#interrupt');await feed(seconds=.7)
            await page.fill('#message','Erzähle eine freundliche Geschichte mit zwölf vollständigen Sätzen über einen Roboter im Wald.');await page.locator('#form').evaluate('f=>f.requestSubmit()')
            await page.wait_for_function('robo.sources.length>0',timeout=30000)
            cancelled=sum(e['type']=='reply_cancelled' for e in await events())
            await feed(audio('selma-followup'));await feed(seconds=1.)
            assert sum(e['type']=='reply_cancelled' for e in await events())==cancelled,'Background speech cancelled the story'
            report['cases'].append({'case':'background voice during a real long TTS reply','passed':True})
            await feed(audio('christoph-stop'));await feed(seconds=1.2)
            await page.wait_for_function('(n)=>focusEvents.filter(e=>e.type==="reply_cancelled").length>n',arg=cancelled,timeout=10000)
            report['cases'].append({'case':'current speaker spoken Stop cancels real streamed TTS','passed':True})
            await page.fill('#message','Erzähle eine Geschichte mit zehn Sätzen über eine Katze.');await page.locator('#form').evaluate('f=>f.requestSubmit()')
            await page.wait_for_function('robo.sources.length>0',timeout=30000)
            await page.click('#interrupt');await page.wait_for_function('robo.sources.length===0',timeout=3000)
            report['cases'].append({'case':'manual Stop still cancels immediately','passed':True})
            await page.locator('#conversation-focus-controls details').evaluate('e=>e.open=true')
            await page.locator('#conversation-focus-threshold').fill('0.35');await page.locator('#conversation-focus-threshold').dispatch_event('change')
            await page.wait_for_function('focusEvents.some(e=>e.type==="identity"&&e.conversation_focus?.threshold===.35)',timeout=10000)
            await page.click('#release-conversation-focus')
            await page.wait_for_function('focusEvents.at(-1)?.type==="identity"&&!focusEvents.at(-1).conversation_focus.speaker_id',timeout=10000)
            await page.uncheck('#conversation-focus');await page.wait_for_timeout(200)
            assert await page.evaluate('localStorage.getItem("robby.conversationFocus")==="false"')
            await page.check('#conversation-focus');await page.wait_for_timeout(200)
            report['controls_work']=True
            await page.locator('#avatar').scroll_into_view_if_needed()
            avatar=await page.locator('#avatar').bounding_box();recent=await page.locator('#live-conversation').bounding_box();latency=await page.locator('#latency-dashboard').bounding_box()
            assert recent['x']>avatar['x']+avatar['width']*.7 and min(avatar['y']+avatar['height'],recent['y']+recent['height'])>max(avatar['y'],recent['y'])
            assert latency['y']>=avatar['y']+avatar['height']-20
            report['desktop_layout_preserved']=True
            await page.locator('#chat-history').evaluate('e=>e.open=true');await page.click('#clear');assert await page.locator('#recent-asr').inner_text()=='Waiting for speech…'
            assert await page.locator('#latency-llm').inner_text()=='—';report['clear_resets_context_and_metrics']=True
            await page.reload();await page.wait_for_function('window.robo&&!document.querySelector("#call").disabled',timeout=60000)
            assert await page.is_checked('#conversation-focus');assert await page.locator('#conversation-focus-threshold').input_value()=='0.35'
            report['focus_preferences_persist']=True
            await page.set_viewport_size({'width':390,'height':844});assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            report['mobile_no_overflow']=True;assert not errors,errors;report['passed']=True
        except Exception as exc:
            report['error']=str(exc)
            try:report['events']=await events()
            except Exception:pass
            await page.screenshot(path=str(ROOT/'logs/conversation-focus-failure.png'))
        finally:
            await page.evaluate('window.robo?.stop()');await browser.close()
    report.update(errors=errors,finished=time.time());(ROOT/'static/conversation-focus-browser.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in report.items() if k!='events'},indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':asyncio.run(main())
