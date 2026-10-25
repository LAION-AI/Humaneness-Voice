"""HTTPS CPU-browser evidence of real requested animation and spoken replies."""
import asyncio,json,time
from pathlib import Path
from playwright.async_api import async_playwright
ROOT=Path(__file__).resolve().parent
URL='https://arrive-locations-parade-research.trycloudflare.com'

async def main():
    report={'passed':False,'physical_microphone':False,'url':URL,'turns':[],'started':time.time()}
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page(viewport={'width':1280,'height':1000});errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script("localStorage.setItem('robby.cameraAuto','false');localStorage.setItem('robby.avatar-idle-enabled','false');localStorage.setItem('robby.avatar-pet-invites-enabled','false');localStorage.setItem('robby.avatar-actions-enabled','true')")
        try:
            await page.goto(URL+'/desk-robot/')
            await page.wait_for_function('window.robotFace&&window.robo&&!document.querySelector("#call").disabled',timeout=60000)
            await page.evaluate('''()=>{
              window.actionEvents=[];window.actionAudio=[];window.actionFrames=[];
              const old=robo.on.message,play=robo.play.bind(robo),frame=robotFace.actions.frame.bind(robotFace.actions);
              robo.on.message=m=>{if(['avatar_action','reply','turn_done','error','barge'].includes(m.type))actionEvents.push({type:m.type,at:performance.now(),text:m.text,steps:m.steps,action_id:m.action_id,avatar_action_id:m.avatar_action_id});old(m)};
              robo.play=raw=>{actionAudio.push(performance.now());play(raw)};
              robotFace.actions.frame=(...args)=>{const result=frame(...args),active=robotFace.actions.active;if(active)actionFrames.push({at:performance.now(),kind:active.kind,source:active.source,cartoon:result.controls.cartoon||0,eyeDifference:Math.abs((result.controls.eyeLeftX||0)-(result.controls.eyeRightX||0))});return result};
            }''')
            cases=[('Mach bitte eine lustige Grimasse und sag etwas Lustiges dazu.',['grimace']),('Mach noch eine Grimasse!',['grimace']),
                ('Schiele bitte und rolle dann mit den Augen.',['grimace','eye_roll']),
                ('Guck bitte wütend und mach dann eine lustige Grimasse.',['emotion','grimace']),
                ('Please make a goofy face.',['grimace'])]
            variants=[]
            for index,(text,kinds) in enumerate(cases):
                await page.evaluate('actionEvents=[];actionAudio=[];actionFrames=[]')
                await page.locator('#message').fill(text);await page.locator('#form button').click()
                await page.wait_for_function('actionEvents.some(e=>e.type==="turn_done")',timeout=60000)
                await page.wait_for_function('robotFace.actions.queue.length===0',timeout=15000)
                data=await page.evaluate('({events:actionEvents,audio:actionAudio,frames:actionFrames})')
                action=next(e for e in data['events'] if e['type']=='avatar_action')
                assert [s['kind'] for s in action['steps']]==kinds,action
                variants.append(action['steps'][0].get('variant'))
                assert data['audio'] and action['at']<data['audio'][0],data['events']
                assert all(any(f['kind']==kind and f['source']=='requested' for f in data['frames']) for kind in kinds),data['frames']
                assert not [e for e in data['events'] if e['type'] in ('error','barge')],data['events']
                assert any(e['type']=='reply' and e.get('text') for e in data['events'])
                if 'grimace' in kinds:assert max(f['cartoon'] for f in data['frames'])>.4
                report['turns'].append({'request':text,'steps':action['steps'],'reply':next(e['text'] for e in reversed(data['events']) if e['type']=='reply'),
                    'action_before_audio_ms':round(data['audio'][0]-action['at'],1),'pcm_chunks':len(data['audio']),
                    'rendered_kinds':list(dict.fromkeys(f['kind'] for f in data['frames']))})
                assert await page.evaluate('robotFace.actions.request('+json.dumps(action['action_id'])+','+json.dumps(action['steps'])+')') is False
            assert variants[0]!=variants[1],variants
            # Disabled actions must explain themselves while speech still works.
            await page.evaluate('document.querySelector("#avatar-action-controls").open=true;actionEvents=[];actionAudio=[];actionFrames=[]')
            await page.locator('#avatar-actions-enabled').uncheck()
            await page.locator('#message').fill('Mach eine Grimasse.');await page.locator('#form button').click()
            await page.wait_for_function('actionEvents.some(e=>e.type==="turn_done")',timeout=60000)
            assert await page.locator('#action-notice').is_visible()
            assert 'disabled' in (await page.locator('#action-notice').text_content()).lower()
            assert await page.evaluate('actionAudio.length>0 && !actionFrames.some(f=>f.source==="requested")')
            report['disabled_control_explained']=True
            await page.locator('#avatar-actions-enabled').check()
            # Fault injection is explicitly labelled; the preceding turns used real TTS.
            fault=await page.evaluate('''()=>{robo.on.message({type:'avatar_action',action_id:'synthetic-failure',steps:[{kind:'grimace',variant:'crossed',duration_s:3}]});robo.on.message({type:'error',error:'Synthetic TTS failure'});const kept=robotFace.actions.queue[0]?.source==='requested';robo.on.message({type:'barge'});return {speech_error_keeps_action:kept,interrupt_clears:robotFace.actions.queue.length===0}}''')
            assert all(fault.values()),fault;report['synthetic_fault_injection']=fault
            report['errors']=errors;assert not errors,errors
            report['passed']=True
        except Exception as exc:
            report['error']=str(exc);report['errors']=errors
            try:report['last_events']=await page.evaluate('window.actionEvents||[]')
            except Exception:pass
        finally:
            await page.evaluate('window.robo?.stop()');await browser.close()
    report['finished']=time.time();(ROOT/'static/avatar-command-browser.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':asyncio.run(main())
