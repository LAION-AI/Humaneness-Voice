"""HTTPS Chrome fake-microphone integration: wake-free ON and strict OFF.

Public synthetic voices only. Exercises capture, actual ASR, diarization,
speaker memory, LLM and TTS. Does not measure real-room false-accept rates.
"""
import asyncio,json,subprocess,tempfile,time
from pathlib import Path
from playwright.async_api import async_playwright
from final_verify import runtime_signature

async def main():
    report=dict(started=time.time(),runtime_source_sha256=runtime_signature(),synthetic_only=True,passed=False,cases=[])
    target=Path('logs/open-speakers-browser.json')
    fixtures=Path('/mnt/nvme/robo-speaker-studies/speaker-live-fixtures')
    async with async_playwright() as p:
        for stem,enabled in [('christoph-followup',True),('selma-followup-clear',True),('robert-followup',False)]:
            with tempfile.TemporaryDirectory(prefix='robby-open-speaker-') as folder:
                wav=Path(folder)/'public.wav'
                subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(fixtures/(stem+'.mp3')),'-af','adelay=1600,apad=pad_dur=80','-ar','48000','-ac','1','-c:a','pcm_s16le',str(wav)],check=True)
                browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream','--use-file-for-fake-audio-capture='+str(wav),'--autoplay-policy=no-user-gesture-required'])
                page=await browser.new_page();events=[];errors=[]
                page.on('pageerror',lambda e:errors.append(str(e)))
                def frame(raw):
                    if isinstance(raw,str):
                        m=json.loads(raw)
                        if m.get('type') not in ('state','input_status'):events.append({**m,'received_at':time.time()})
                page.on('websocket',lambda ws:ws.on('framereceived',frame))
                await page.add_init_script("localStorage.setItem('robby.cameraAuto','false');localStorage.setItem('robby.openSpeakers',"+json.dumps(str(enabled).lower())+");")
                row=dict(fixture=stem,open_enrollment=enabled,passed=False)
                try:
                    await page.goto('https://diane-asylum-understood-invoice.trycloudflare.com/desk-robot/')
                    await page.wait_for_function('window.robo && !document.querySelector("#call").disabled',timeout=90000)
                    row['start']=time.time();await page.click('#call')
                    deadline=time.monotonic()+(55 if enabled else 20)
                    while time.monotonic()<deadline:
                        if enabled and any(e.get('type')=='reply' for e in events) and await page.evaluate('robo.metrics.reply_chunks>0'):break
                        await page.wait_for_timeout(200)
                    row['metrics']=await page.evaluate('robo.metrics');row['events']=events;row['page_errors']=errors
                    endpoints=[e for e in events if e.get('type')=='endpoint'];replies=[e for e in events if e.get('type')=='reply']
                    settings=[e for e in events if e.get('type')=='identity' and e.get('open_enrollment')==enabled]
                    row['passed']=bool(not errors and settings and ((endpoints and replies and row['metrics']['reply_chunks']>0) if enabled else not endpoints and not replies))
                    if enabled:row['asked_name']=any('?' in e.get('text','') and any(w in e.get('text','').lower() for w in ('heiß','heiss','name','nennen')) for e in replies)
                except Exception as e:row['error']=type(e).__name__+': '+str(e)
                finally:
                    await page.evaluate('window.robo?.stop()');await browser.close();await asyncio.sleep(.6)
                    report['cases'].append(row);target.write_text(json.dumps(report,indent=2,ensure_ascii=False))
                    print(json.dumps({k:row[k] for k in ('fixture','open_enrollment','passed')},ensure_ascii=False),flush=True)
    report.update(finished=time.time(),passed=all(r['passed'] and r.get('asked_name',True) for r in report['cases']) and runtime_signature()==report['runtime_source_sha256'])
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit('Inspect retained open-speaker failures.')

if __name__=='__main__':asyncio.run(main())
