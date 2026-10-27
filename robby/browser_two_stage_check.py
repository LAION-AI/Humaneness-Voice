"""Actual HTTPS UI, two-stage shared TTS, continuity and cancellation.

Typed authored turns only; this does not certify acoustic interruption. Every
attempt keeps its own receipt. The validated reaction must precede the main answer and be present in its context.
"""
import argparse,asyncio,datetime,fcntl,json,re,time
from pathlib import Path
from playwright.async_api import async_playwright
from run_voice_assistant_checks import deployed_source_hash,public_state

ROOT=Path(__file__).resolve().parent
async def main(args):
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report={'passed':False,'started':stamp,'scope':'Actual HTTPS browser, typed synthetic turns, real local/API/TTS. No physical microphone or acoustic-stop assertion.',
        'source_sha256':deployed_source_hash(),'server':public_state(args.url),'turns':[]};errors=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page(viewport={'width':1366,'height':950});page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script("localStorage.setItem('robby.autoStart','false');localStorage.setItem('robby.environment','false');")
        await page.add_init_script('''localStorage.setItem('robby.avatar-idle-enabled','false');localStorage.setItem('robby.avatar-pet-invites-enabled','false');
            window.cameraRequests=0;const gum=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);navigator.mediaDevices.getUserMedia=x=>{if(x.video)cameraRequests++;return gum(x);};''')
        try:
            await page.goto(args.url+'/desk-robot/');await page.wait_for_function('window.robo&&document.querySelector("#model-settings")&&!document.querySelector("#call").disabled',timeout=60000)
            ids=re.findall(r'id="([^"]+)"',(ROOT/'static/index.html').read_text())
            missing=await page.evaluate('(ids)=>ids.filter(id=>document.querySelectorAll(`[id="${id}"]`).length!==1)',ids)
            assert not missing,missing
            boxes={k:await page.locator(v).bounding_box() for k,v in {'avatar':'#avatar','conversation':'.conversation-context','latency':'#latency-dashboard','models':'#model-settings'}.items()}
            assert boxes['avatar']['y']<170 and boxes['conversation']['x']>boxes['avatar']['x']+boxes['avatar']['width']
            assert boxes['latency']['y']>boxes['avatar']['y']+boxes['avatar']['height']
            assert boxes['models']['y']>boxes['latency']['y']+boxes['latency']['height']
            assert await page.evaluate('cameraRequests')==0
            assert await page.locator('#asr-model').input_value()=='parakeet_v3'
            assert not await page.locator('#camera-auto').is_checked()
            assert not await page.locator('#quick-replies').is_checked()
            report['layout']={'all_static_controls_retained':True,'boxes':boxes,'camera_requests':0}
            await page.screenshot(path=str(ROOT/'logs/two-stage-desktop.png'),full_page=True)
            await page.evaluate('''()=>{
                window.qEvents=[];window.qPCM=[];const old=robo.on.message,play=robo.play.bind(robo);
                robo.on.message=m=>{qEvents.push({...m,at:performance.now()});old(m);};
                robo.play=raw=>{const at=performance.now(),before=robo.next,now=robo.ctx?.currentTime||0;play(raw);qPCM.push({at,reply_id:new DataView(raw).getInt32(0,true),samples:(raw.byteLength-4)/2,previous_end:before,now,next:robo.next,underflows:robo.metrics.underflows||0});};
            }''')
            await page.evaluate('robo.start(false)')
            await page.click('#message')
            await page.evaluate('robo.ctx.resume()')
            assert await page.evaluate('robo.ctx.state')=='running'
            async def turn(text,enabled):
                await page.locator('#model-settings').evaluate('e=>e.open=true')
                await page.locator('#quick-replies').set_checked(enabled)
                await page.wait_for_timeout(100)
                mark=await page.evaluate('({event:qEvents.length,pcm:qPCM.length,at:performance.now(),underflows:robo.metrics.underflows||0})')
                await page.fill('#message',text);await page.locator('#form').evaluate('f=>f.requestSubmit()')
                await page.wait_for_function('(n)=>qEvents.slice(n).some(e=>e.type==="turn_done"||e.type==="error")',arg=mark['event'],timeout=90000)
                await page.wait_for_function('!robo.sources.length',timeout=90000)
                events=await page.evaluate('(n)=>qEvents.slice(n)',mark['event']);audio=await page.evaluate('(n)=>qPCM.slice(n)',mark['pcm'])
                replies=[e for e in events if e['type']=='reply'];metrics={}
                for e in events:
                    if e['type']=='latency_update':metrics.update(e.get('metrics',{}))
                    if e['type']=='pipeline_metrics':metrics.update(e.get('metrics',{}))
                reactions=[e for e in replies if e.get('sentence_role')=='reaction']
                row={'input':text,'enabled':enabled,'events':events,'audio':audio,'metrics':metrics,
                    'text':replies[-1]['text'] if replies else '',
                    'first_pcm_ms':round(audio[0]['at']-mark['at'],2) if audio else None,
                    'new_underflows':(audio[-1]['underflows']-mark['underflows']) if audio else None}
                if audio:
                    first=audio[0];row['scheduled_playback_ms']=round(row['first_pcm_ms']+max(0,first['next']-first['samples']/24000-first['now'])*1000,2)
                    row['max_gap_ms']=round(max([max(0,s['next']-s['samples']/24000-s['previous_end'])*1000 for s in audio[1:]]or[0]),2)
                row['passed']=bool(audio and replies) and not any(e['type']=='error' for e in events)
                if not enabled:row['passed'] &= not reactions
                for e in reactions:row['passed'] &= len(re.findall(r"[\w]+(?:['’\-][\w]+)*",e['sentence_text']))<=6
                report['turns'].append(row);return row
            for text in ['Warum ist der Himmel blau?', 'Erzähl mir einen kurzen Katzenwitz.']:
                row=await turn(text,True);assert row['passed']
                roles=[e.get('sentence_role') for e in row['events'] if e['type']=='reply']
                assert roles[:2]==['reaction','answer'] and 'informed_bridge' not in roles,roles
                assert row['metrics'].get('main_informed_by_reaction'),row['metrics']
                assert row['metrics'].get('tts_backend')=='continuous_batch'
                assert row['metrics'].get('tts_temperature')==.8
            row=await turn('Schreib ein Gedicht mit vier Zeilen über einen kleinen Roboter.',True)
            assert row['passed'] and len([x for x in row['text'].splitlines() if x.strip()])==4,row['text']
            assert not any(e.get('sentence_role')=='reaction' for e in row['events'])
            row=await turn('Wie entsteht ein Regenbogen?',False);assert row['passed']
            # Interrupt a real long response; retained playback IDs must not resume.
            await page.locator('#quick-replies').set_checked(True)
            mark=await page.evaluate('({event:qEvents.length,pcm:qPCM.length})')
            await page.fill('#message','Erzähle mir eine lange Geschichte mit zwölf Sätzen über einen kleinen Roboter.');await page.locator('#form').evaluate('f=>f.requestSubmit()')
            await page.wait_for_function('robo.sources.length>0',timeout=40000)
            await page.click('#interrupt');await page.wait_for_timeout(1800)
            ev=await page.evaluate('(n)=>qEvents.slice(n)',mark['event'])
            report['cancellation']={'events':ev,'sources_remaining':await page.evaluate('robo.sources.length'),
                'passed':any(e['type']=='reply_cancelled' for e in ev) and not await page.evaluate('robo.sources.length')}
            assert report['cancellation']['passed']
            await page.locator('#quick-replies').set_checked(False)
            await page.evaluate('robo.stop()');await page.reload();await page.wait_for_function('window.robo&&document.querySelector("#model-settings")')
            assert not await page.locator('#quick-replies').is_checked();report['off_setting_persists']=True
            await page.set_viewport_size({'width':390,'height':844})
            assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            await page.locator('#model-settings').evaluate('e=>e.open=true')
            assert await page.locator('#asr-model').is_visible() and await page.locator('#llm-model').is_visible()
            await page.screenshot(path=str(ROOT/'logs/two-stage-mobile.png'),full_page=True)
            report['mobile_controls_accessible']=True;report['passed']=not errors
        except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)[:1500]
        finally:
            try:await page.evaluate('window.robo?.stop()')
            except Exception:pass
            await browser.close();report['errors']=errors;report['source_unchanged']=report['source_sha256']==deployed_source_hash();report['passed'] &= report['source_unchanged']
            raw=json.dumps(report,indent=2,ensure_ascii=False);(ROOT/'logs'/('two-stage-browser-'+stamp+'.json')).write_text(raw);(ROOT/'static/two-stage-browser.json').write_text(raw)
    print(json.dumps({'passed':report['passed'],'error':report.get('error'),'turns':len(report['turns'])}),flush=True)
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com');args=parser.parse_args()
    lock=(ROOT/'logs/voice-assistant-runs/.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    asyncio.run(main(args))
