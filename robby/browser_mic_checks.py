"""Real browser capture/worklet/WebSocket/ASR/Luna/TTS with public synthetic speech.

No physical microphone or private recording is used. Unlike a typed-message
smoke test, this exercises getUserMedia and the browser's 48-kHz -> 16-kHz path.
Run sequentially (the pilot has one live slot):
uv run --no-project --with playwright --python /path/to/python browser_mic_checks.py
"""
import argparse,asyncio,json,subprocess,tempfile,time
from pathlib import Path
from playwright.async_api import async_playwright

ROOT=Path(__file__).resolve().parent
FIXTURES=Path('/mnt/nvme/robo-speaker-studies/speaker-live-fixtures')

async def run(args):
    results=[]
    suffix='desk' if '/desk-robot' in args.url else 'original'
    for target in ('browser-capture-checks.json',f'browser-capture-{suffix}.json'):
        (ROOT/'logs'/target).write_text(json.dumps({'synthetic_only':True,'url':args.url,'passed':False,'running':True,'results':[]}))
    with tempfile.TemporaryDirectory(prefix='robby-mic-browser-') as folder:
        wav=Path(folder)/'public-synthetic-introduction.wav'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(FIXTURES/'christoph-intro.mp3'),
            '-af','adelay=1280,apad=pad_dur=35','-ar','48000','-ac','1','-c:a','pcm_s16le',str(wav)],check=True)
        async with async_playwright() as p:
            browser=await p.chromium.launch(executable_path=args.chrome,headless=True,args=[
                '--no-sandbox','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream',
                '--use-file-for-fake-audio-capture='+str(wav),'--autoplay-policy=no-user-gesture-required'])
            try:
                for filtered in (False,True):
                    context=await browser.new_context(viewport={'width':1280,'height':900})
                    page=await context.new_page();errors=[];events=[];identities=[]
                    await page.add_init_script("localStorage.setItem('robby.cameraAuto','false')")
                    page.on('pageerror',lambda e:errors.append(str(e)))
                    def frame(payload):
                        if isinstance(payload,str):
                            item=json.loads(payload)
                            if item.get('type')=='identity':identities.append(item)
                            if item.get('type') not in ('state','identity','input_status'):
                                events.append(item)
                    page.on('websocket',lambda ws:ws.on('framereceived',frame))
                    try:
                        await page.goto(args.url)
                        await page.wait_for_function('window.robo && !document.querySelector("#call").disabled',timeout=90000)
                        await page.locator('#recognition-settings > summary').click()
                        await page.locator('.input-details > summary').click()
                        # Reproduce the reported upgrade from a typed-capable
                        # connection, including a suspended existing context.
                        await page.evaluate('async()=>{await window.robo.start(false);await window.robo.ctx.suspend();}')
                        await page.click('#call' if filtered else '#unfiltered-talk')
                        deadline=time.monotonic()+90
                        while time.monotonic()<deadline:
                            if any(e.get('type')=='endpoint' for e in events) and any(e.get('type')=='reply' for e in events) and await page.evaluate('window.robo.metrics.reply_chunks>0'):break
                            await page.wait_for_timeout(250)
                        report=await page.evaluate('''()=>({metrics:window.robo.metrics,
                            context_state:window.robo.ctx?.state,
                            mic_status:document.querySelector('#mic-status').textContent,
                            asr_status:document.querySelector('#asr-status').textContent,
                            asr_preview:document.querySelector('#asr-preview').textContent,
                            filter:document.querySelector('#identity-gate').checked,
                            mobile_no_overflow:document.documentElement.scrollWidth<=innerWidth})''')
                        report.update(requested_filter=filtered,events=events,page_errors=errors)
                        if await page.evaluate('document.body.dataset.robotView==="desk3d"'):
                            await page.wait_for_function('document.querySelector("#acting-status").textContent.includes("Luna SCRIPT")',timeout=15000)
                            report['desk_robot_playback']=await page.evaluate('({expression:robotFace.rig.snapshot().expression,plans:robotFace.timeline.plans.size,cue:document.querySelector("#acting-status").textContent})')
                        results.append(report)
                        (ROOT/'logs/browser-capture-checks.json').write_text(json.dumps({'synthetic_only':True,'results':results},indent=2))
                        await page.screenshot(path=str(ROOT/'logs'/('mic-filtered.png' if filtered else 'mic-unfiltered.png')))
                        endpoints=[e for e in events if e.get('type')=='endpoint']
                        assert not errors,errors
                        assert report['context_state']=='running',report
                        assert report['metrics']['mic_blocks']>100,report
                        assert endpoints and any(e.get('type')=='reply' for e in events) and report['metrics']['reply_chunks']>0,'No actual voice-to-voice turn: '+json.dumps(report)
                        assert any('Christoph' in e.get('transcript','') for e in endpoints),endpoints
                        assert report['filter']==filtered,report
                        if filtered:assert any(e.get('speaker_name')=='Christoph' for e in endpoints),endpoints
                        if filtered:
                            async def confirmed(predicate):
                                for _ in range(80):
                                    if any(predicate(item) for item in identities):return
                                    await page.wait_for_timeout(100)
                                raise AssertionError('UI setting received no matching backend acknowledgment')
                            await page.click('#interrupt')
                            identities.clear();await page.get_by_role('button',name='Forget enrolled users',exact=True).click()
                            await confirmed(lambda m:not m.get('users'))
                            assert await page.locator('#speaker-chips').text_content()==''
                            checked=['forget_users_clears_server_and_chips']
                            for mode in ('v4-long3','v4','v4-long'):
                                identities.clear();await page.select_option('#identity-model',mode)
                                await confirmed(lambda m:m.get('mode')==mode and m.get('enabled'))
                                checked.append('select_'+mode)
                            identities.clear();await page.get_by_label('Forget inactive users after').select_option('300')
                            await confirmed(lambda m:m.get('timeout_s')==300);checked.append('inactivity_timeout_acknowledged')
                            # Range inputs are not text-fillable. Set their
                            # native value and exercise input/change handlers.
                            identities.clear();await page.locator('#utterance-strictness').evaluate(
                                "el=>{el.value='0.56';el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}));}")
                            await confirmed(lambda m:m.get('utterance_threshold')==.56)
                            assert await page.locator('#utterance-threshold-value').text_content()=='0.56'
                            checked.append('full_sentence_threshold_acknowledged')
                            identities.clear();await page.select_option('#identity-model','v4-long3')
                            await confirmed(lambda m:m.get('mode')=='v4-long3' and m.get('utterance_threshold')==.56)
                            checked.append('threshold_retained_after_model_switch')
                            identities.clear();await page.locator('#speaker-strictness').evaluate(
                                "el=>{el.value='0.67';el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}));}")
                            await confirmed(lambda m:m.get('threshold')==.67);checked.append('fast_threshold_acknowledged')
                            assert await page.evaluate('window.robo.ctx.state==="running"&&window.robo.stream.getAudioTracks()[0].readyState==="live"')
                            report['speaker_ui_checks']=checked
                        report['passed']=True
                        payload={'synthetic_only':True,'url':args.url,'results':results,'passed':len(results)==2 and all(r.get('passed') for r in results)}
                        (ROOT/'logs/browser-capture-checks.json').write_text(json.dumps(payload,indent=2))
                        suffix='desk' if '/desk-robot' in args.url else 'original'
                        (ROOT/'logs'/f'browser-capture-{suffix}.json').write_text(json.dumps(payload,indent=2))
                        print(json.dumps({'passed':True,'filter':filtered,'metrics':report['metrics'],
                            'endpoints':endpoints,'page_errors':errors}),flush=True)
                    finally:
                        await page.evaluate('window.robo?.stop()');await context.close()
                        await asyncio.sleep(1)
            finally:await browser.close()
    print('Both actual browser voice-input modes passed.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com/')
    p.add_argument('--chrome',default='/usr/bin/google-chrome');asyncio.run(run(p.parse_args()))
