"""HTTPS regression: quiet TV, voice changes, real handover, tricks and decisions.

All microphone samples are public synthetic fixtures. Real ASR, selected
Qwen/Cerebras, TTS and CPU browser rendering are used without canned replies.
"""
import asyncio,base64,json,time,os
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from playwright.async_api import async_playwright
from speaker_live.live_test import audio
from conversation_stress import music,foley,repeat
ROOT=Path(__file__).resolve().parent
URL=os.environ.get('ROBBY_TEST_URL','http://127.0.0.1:8793').rstrip('/')
class SuiteComplete(Exception):pass

async def main():
    report={'passed':False,'physical_microphone':False,'started':time.time(),'cases':[]};errors=[]
    quick=os.environ.get('ROBBY_TEST_QUICK_STOP')=='1'
    permissions_mode=os.environ.get('ROBBY_TEST_PERMISSIONS')=='1'
    threshold=float(os.environ.get('ROBBY_TEST_THRESHOLD','.28'));pause_ms=int(os.environ.get('ROBBY_TEST_PAUSE_MS','700'))
    explicit_settings='ROBBY_TEST_THRESHOLD' in os.environ or 'ROBBY_TEST_PAUSE_MS' in os.environ
    report['configuration']={'focus_threshold':threshold,'speech_pause_ms':pause_ms,'url':URL}
    background=audio('background');offset=0
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=os.environ.get('ROBBY_TEST_CHROME','/usr/bin/google-chrome'),headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page(viewport={'width':1440,'height':1080});page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script("localStorage.setItem('robby.cameraAuto','false');localStorage.setItem('robby.avatar-idle-enabled','false');localStorage.setItem('robby.avatar-pet-invites-enabled','false');localStorage.setItem('robby.avatar-actions-enabled','true');localStorage.setItem('robby.focusThreshold','0.32')")
        await page.add_init_script('localStorage.setItem("robby.asrDefaultVersion","2");localStorage.setItem("robby.asrModel",'+json.dumps(os.environ.get('ROBBY_TEST_ASR_MODEL','nemotron'))+');')
        if explicit_settings:
            await page.add_init_script('localStorage.setItem("robby.focusBalanceVersion","2");localStorage.setItem("robby.pauseBalanceVersion","2");localStorage.setItem("robby.focusThreshold",'+json.dumps(str(threshold))+');localStorage.setItem("robby.speechPauseMs",'+json.dumps(str(pause_ms))+');')
        async def ev():return await page.evaluate('events')
        async def count(kind):return sum(e['type']==kind for e in await ev())
        async def feed(x=None,seconds=0,tv=.0):
            nonlocal offset
            wave=np.asarray(x if x is not None else np.zeros(round(seconds*16000)),dtype=np.float32).copy()
            if tv:
                wave+=background[(np.arange(len(wave))+offset)%len(background)]*tv
            offset+=len(wave);data=base64.b64encode(wave.astype('<f4').tobytes()).decode()
            await page.evaluate('''async data=>{
                const x=new Float32Array(Uint8Array.from(atob(data),c=>c.charCodeAt(0)).buffer),start=performance.now();
                for(let at=0;at<x.length;at+=320){robo.mic(robo.ctx.currentTime,x.subarray(at,at+320));await new Promise(r=>setTimeout(r,Math.max(0,start+(at+320)/16-performance.now())))}
            }''',data)
        async def until(predicate,seconds=40,tv=0.):
            for _ in range(round(seconds/.4)):
                if await page.evaluate(predicate):return
                await feed(seconds=.4,tv=tv)
            raise TimeoutError(predicate)
        async def settle(tv=0.):
            stable=0.
            for _ in range(175):
                await feed(seconds=.4,tv=tv)
                idle=await page.evaluate('!window.turnBusy&&!robo.sources.length')
                stable=stable+.4 if idle else 0.
                if stable>=1.2:return
            raise TimeoutError('The current and queued replies did not settle')
        async def typed(text):
            await settle()
            await page.fill('#message',text);await page.locator('#form').evaluate('f=>f.requestSubmit()')
        async def permission_cases():
            if await page.evaluate('robo.sources.length>0'):await page.click('#interrupt')
            await feed(seconds=.7)
            await feed(audio('focus-invite-alice'),tv=.08);await feed(seconds=1.4,tv=.08)
            await until('events.some(e=>e.type==="speaker_permission"&&e.action==="invite"&&e.name==="Alice"&&e.applied)',25,.08)
            await settle(.08)
            n=await count('endpoint');await feed(audio('selma-followup-clear'),tv=.08);await feed(seconds=1.4,tv=.08)
            await until(f'events.filter(e=>e.type==="endpoint").length>{n}&&events.filter(e=>e.type==="endpoint").at(-1).speaker_name==="Alice"',25,.08)
            await settle(.08)
            members=await page.evaluate('events.filter(e=>e.type==="identity"&&e.conversation_focus).at(-1).conversation_focus.participants')
            assert {'Alice','Christoph'}<={m['name'] for m in members},members
            report['cases'].append({'case':'spoken invitation enrolls Alice while retaining Christoph','participants':members,'passed':True})
            n=await count('endpoint');await feed(audio('christoph-followup'));await feed(seconds=1.2)
            await until(f'events.filter(e=>e.type==="endpoint").length>{n}&&events.filter(e=>e.type==="endpoint").at(-1).speaker_name==="Christoph"',25)
            report['cases'].append({'case':'original participant can speak again after the guest','passed':True})
            await until('robo.sources.length>0',25);await page.click('#interrupt');await feed(seconds=.7)
            permissions=await count('speaker_permission');n=await count('endpoint')
            await feed(audio('focus-background-invite')*.08);await feed(seconds=2.)
            assert await count('speaker_permission')==permissions and await count('endpoint')==n,'Background voice granted access'
            report['cases'].append({'case':'unadmitted background voice cannot invite another person','passed':True})
            await feed(audio('focus-ignore-onlyme'));await feed(seconds=1.4)
            await until('events.filter(e=>e.type==="identity"&&e.conversation_focus).at(-1)?.conversation_focus.participants?.length===1',25)
            await settle();n=await count('endpoint')
            await feed(audio('selma-followup-clear'));await feed(seconds=2.)
            assert await count('endpoint')==n,'Revoked guest was still answered'
            report['cases'].append({'case':'spoken only-me request revokes the guest and suppresses later replies','passed':True})
            assert not errors,errors
        try:
            await page.goto(URL+'/desk-robot/');await page.wait_for_function('window.robo&&robotFace.llmControls?.catalog&&!document.querySelector("#call").disabled',timeout=60000)
            await page.evaluate('''()=>{
                window.events=[];window.motionFrames=[];window.audioTimes=[];window.turnBusy=false;
                const old=robo.on.message,update=robotFace.rig.update.bind(robotFace.rig),play=robo.play.bind(robo);
                robo.on.message=m=>{if(['endpoint','latency_start'].includes(m.type))turnBusy=true;if(['turn_done','error','reply_cancelled','barge'].includes(m.type))turnBusy=false;if(!['state','speech_activity','input_status'].includes(m.type))events.push({...m,at:performance.now()});old(m)};
                robotFace.rig.update=(input,...rest)=>{if(Math.abs(input.actionMotion?.spin||0)>1||robotFace.actions.active)motionFrames.push({at:performance.now(),spin:input.actionMotion?.spin||0,kind:robotFace.actions.active?.kind,eye:input.controls?.cartoon});return update(input,...rest)};
                robo.play=raw=>{audioTimes.push(performance.now());play(raw)};
            }''')
            assert abs(float(await page.input_value('#conversation-focus-threshold'))-threshold)<.001
            assert int(await page.input_value('#speech-pause'))==pause_ms
            report['old_default_migrated']=not explicit_settings
            await page.evaluate('robo.start(false)');await until('events.some(e=>e.type==="ready")',10)
            await page.evaluate('robo.send("interaction:helpful:off")')
            await feed(seconds=.64);await feed(audio('christoph-intro'),tv=.12);await feed(seconds=2.,tv=.12)
            await until('events.some(e=>e.type==="endpoint"&&e.speaker_name==="Christoph")',20,.12)
            endpoints=[e for e in await ev() if e['type']=='endpoint'];first=endpoints[-1]
            assert len(first['transcript'].split())>=5 and 'Christoph' in first['transcript'],first
            await settle(.12)
            report['cases'].append({'case':'whole introduction with quiet competing speech','transcript':first['transcript'],'passed':True})
            if permissions_mode:
                await permission_cases();raise SuiteComplete()
            if quick:
                for command in ('christoph-stop','christoph-wait','christoph-halt','christoph-pause-clear'):
                    await typed('Erzähle eine freundliche Geschichte mit zwölf Sätzen über ein Kätzchen.')
                    await until('robo.sources.length>0',25)
                    cancellations=await count('reply_cancelled');at=await page.evaluate('performance.now()')
                    await feed(audio(command));await feed(seconds=.8)
                    await until(f'events.filter(e=>e.type==="reply_cancelled").length>{cancellations}',8)
                    stopped=[e for e in await ev() if e['type']=='reply_cancelled'][-1]
                    report['cases'].append({'case':command+' stops real long TTS','capture_to_cancel_ms':round(stopped['at']-at,1),'passed':True})
                await feed(seconds=.8);n=await count('endpoint');await feed(audio('christoph-followup'));await feed(seconds=1.2)
                await until(f'events.filter(e=>e.type==="endpoint").length>{n}',20)
                await until('robo.sources.length>0',25)
                report['cases'].append({'case':'voice conversation resumes after four consecutive Stops','passed':True})
                raise SuiteComplete()
            n=await count('endpoint')
            # 4% speed/pitch shift is still the same synthetic identity.
            changed=resample_poly(audio('christoph-followup'),25,26)
            changed=changed+repeat(audio('selma-followup-clear'),len(changed),8192)*.07+music(len(changed))*.04+foley(len(changed))*.025
            await feed(changed,tv=.12);await feed(seconds=1.2,tv=.12)
            await until(f'events.filter(e=>e.type==="endpoint").length>{n}',20,.12)
            returning=[e for e in await ev() if e['type']=='endpoint'][-1]
            assert returning['speaker_id']==first['speaker_id'] and len(returning['transcript'].split())>=5 and 'Geschichte' in returning['transcript'],returning
            report['cases'].append({'case':'same voice with changed delivery, two overlapping background voices, procedural music and foley','transcript':returning['transcript'],'passed':True})
            # Handover WHILE a story is playing, then a new voice without a wake phrase.
            await until('robo.sources.length>0',20,.12)
            await page.locator('#conversation-focus-controls details').evaluate('e=>e.open=true')
            await page.click('#release-conversation-focus');await feed(seconds=.5)
            n=await count('endpoint');await feed(audio('selma-followup-clear'),tv=.10);await feed(seconds=1.6,tv=.10)
            await until(f'events.filter(e=>e.type==="endpoint").length>{n}',20,.10)
            joined=[e for e in await ev() if e['type']=='endpoint'][-1]
            assert joined['speaker_id']!=first['speaker_id'] and len(joined['transcript'].split())>=4,joined
            report['cases'].append({'case':'handover accepts next different voice without introduction','transcript':joined['transcript'],'speaker_id':joined['speaker_id'],'passed':True})
            await settle(.1)
            await page.evaluate('motionFrames=[];audioTimes=[]')
            done=await count('turn_done');await typed('Dreh mal im Kreis und sag dabei etwas Lustiges!')
            await until(f'events.filter(e=>e.type==="turn_done").length>{done}',35)
            await page.wait_for_function('motionFrames.some(f=>Math.abs(f.spin)>280)',timeout=10000)
            frames=await page.evaluate('motionFrames');times=await page.evaluate('audioTimes')
            assert times and any(f['at']<=times[0]<=f['at']+6000 for f in frames)
            report['cases'].append({'case':'real 3D full rotation with concurrent speech','max_yaw_degrees':max(abs(f['spin']) for f in frames),'pcm_chunks':len(times),'passed':True})
            await page.evaluate('motionFrames=[]');done=await count('turn_done')
            await typed('Guck bitte wütend, mach dann eine lustige Grimasse und nicke.')
            await until(f'events.filter(e=>e.type==="turn_done").length>{done}',35)
            await page.wait_for_function('["emotion","grimace","nod"].every(k=>motionFrames.some(f=>f.kind===k))',timeout=12000)
            report['cases'].append({'case':'emotion, grimace and nod actually rendered','passed':True})
            # Return floor to the original public voice before the interruption game.
            await page.click('#release-conversation-focus');await feed(seconds=.4);await feed(audio('christoph-intro'));await feed(seconds=1.2)
            await until('events.filter(e=>e.type==="endpoint").at(-1)?.speaker_name==="Christoph"',20)
            await settle()
            done=await count('turn_done')
            await typed('Wenn ich über Bananen rede, unterbrich mich sofort und sage Stopp, stopp!')
            await until('events.some(e=>e.type==="interaction"&&e.rules?.length)',20)
            await until(f'events.filter(e=>e.type==="turn_done").length>{done}',30)
            start=await page.evaluate('performance.now()');banana=audio('christoph-bananas')
            await feed(banana);await feed(seconds=.5)
            interjections=[e for e in await ev() if e['type']=='endpoint' and e.get('by')=='agreed_interruption' and e['at']>=start]
            assert interjections,'No live banana interruption while the user continued'
            first_live=interjections[0]
            assert first_live['at']-start<len(banana)/16,first_live
            pcm_times=await page.evaluate('audioTimes');first_pcm=next((at-start for at in pcm_times if at>=first_live['at']),None)
            report['cases'].append({'case':'banana rule interrupts an ongoing monologue before its end','decision_from_input_start_ms':round(first_live['at']-start,1),'first_interjection_pcm_ms':round(first_pcm,1) if first_pcm is not None else None,'input_duration_ms':round(len(banana)/16,1),'interjections':len(interjections),'passed':True})
            await settle();done=await count('turn_done');await typed('Beende bitte das Bananen-Spiel und lösche die Unterbrechungsregel.')
            await until('events.filter(e=>e.type==="interaction").at(-1)?.rules?.length===0',20)
            await until(f'events.filter(e=>e.type==="turn_done").length>{done}',30)
            await typed('Erzähle eine freundliche Geschichte mit zwölf Sätzen über ein Kätzchen.')
            await until('robo.sources.length>0',25)
            cancellations=await count('reply_cancelled');stop_at=await page.evaluate('performance.now()')
            await feed(audio('christoph-stop'));await feed(seconds=1.)
            await until(f'events.filter(e=>e.type==="reply_cancelled").length>{cancellations}',12)
            stopped=[e for e in await ev() if e['type']=='reply_cancelled'][-1]
            await feed(seconds=.8);n=await count('endpoint');await feed(audio('christoph-followup'));await feed(seconds=1.2)
            await until(f'events.filter(e=>e.type==="endpoint").length>{n}',20)
            await until('robo.sources.length>0',25)
            report['cases'].append({'case':'spoken Stop cancels and the next spoken request still gets audio','cancel_from_capture_start_ms':round(stopped['at']-stop_at,1),'passed':True})
            await permission_cases()
            assert not errors,errors
            assert not [e for e in await ev() if e['type']=='error'],[e for e in await ev() if e['type']=='error']
            report['passed']=True
        except SuiteComplete:
            report['passed']=True
        except Exception as exc:
            report['error']=str(exc);report['events']=await ev()
            await page.screenshot(path=str(ROOT/'logs/rebalance-browser-failure.png'))
        finally:
            try:await page.evaluate('window.robo?.stop()')
            except Exception:pass
            await browser.close()
    report.update(errors=errors,finished=time.time());(ROOT/'static'/('rebalance-permissions-browser.json' if permissions_mode else 'rebalance-stop-browser.json' if quick else 'rebalance-browser.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in report.items() if k!='events'},indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':asyncio.run(main())
