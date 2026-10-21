"""Real HTTPS UI, controlled synthetic MediaStream, actual ASR/TTS/ack playback.

No transcript injection, fake server events or physical recordings. Run alone:
the working demo has one ASR slot. Delayed bot echo is mixed into the fake mic.
"""
import argparse,asyncio,base64,json,struct,time
from pathlib import Path
from playwright.async_api import async_playwright
from speaker_live.live_test import audio
from final_verify import runtime_signature


async def main(url='https://diane-asylum-understood-invoice.trycloudflare.com/desk-robot/',command='christoph-stop'):
    report={'passed':False,'synthetic_only':True,'runtime_source_sha256':runtime_signature(),
            'echo_delay_ms':120,'echo_gain':.05,'command':command,'url':url,'events':[]}
    events=report['events'];played=[];errors=[];samples={}
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,
            args=['--no-sandbox','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page()
        if url.startswith('http://127.0.0.1:8793/'):
            await page.add_init_script('''const WS=window.WebSocket;window.WebSocket=class extends WS{constructor(url,...rest){const u=new URL(url);u.searchParams.set('synthetic_diagnostics','1');super(u.toString(),...rest);}};''')
        await page.add_init_script('''
            localStorage.setItem('robby.cameraAuto','false');
            navigator.mediaDevices.getUserMedia=async()=>{
                const ctx=window.robo.ctx,d=ctx.createMediaStreamDestination();
                window.testMic=d;
                const delay=ctx.createDelay(1),gain=ctx.createGain();
                delay.delayTime.value=.12;gain.gain.value=.05;
                robo.output.connect(delay);delay.connect(gain);gain.connect(d);
                return d.stream;
            };
            window.testSpeak=async(encoded)=>{
                const bytes=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));
                const x=new Float32Array(bytes.buffer),ctx=robo.ctx;
                const b=ctx.createBuffer(1,x.length,16000);b.getChannelData(0).set(x);
                const s=ctx.createBufferSource();s.buffer=b;s.connect(testMic);
                await new Promise(resolve=>{s.onended=resolve;s.start();});s.disconnect();
            };
        ''')
        def received(raw):
            if isinstance(raw,str):
                e=json.loads(raw)
                if e.get('type') in ('endpoint','reply','barge','reply_cancelled','stop_command','stop_feedback','turn_done','error','warning','ignored','asr_preview','synthetic_stop_decode','synthetic_barge_verification'):
                    events.append({'wall':time.monotonic(),**e})
            elif len(raw)>4:
                rid=struct.unpack_from('<i',raw)[0]
                samples[rid]=samples.get(rid,0)+(len(raw)-4)//2
        def sent(raw):
            if isinstance(raw,str) and raw.startswith('played:'):played.append(raw)
        def socket(ws):ws.on('framereceived',received);ws.on('framesent',sent)
        page.on('websocket',socket);page.on('pageerror',lambda e:errors.append(str(e)))
        async def wait(predicate,seconds=70):
            until=time.monotonic()+seconds
            while time.monotonic()<until:
                if predicate():return
                await page.wait_for_timeout(100)
            raise TimeoutError('Browser spoken-stop stage did not complete')
        async def speak(name):
            await page.evaluate('encoded=>testSpeak(encoded)',base64.b64encode(audio(name).astype('<f4').tobytes()).decode())
        try:
            await page.goto(url)
            await page.wait_for_function('window.robo&&!document.querySelector("#call").disabled',timeout=90000)
            await page.locator('#stop-feedback').check();await page.locator('#call').click()
            await page.wait_for_function('window.testMic&&robo.metrics.mic_blocks>30')
            await speak('christoph-intro')
            await wait(lambda:any(e['type']=='turn_done' for e in events))
            await page.wait_for_function('robo.sources.length===0');await page.wait_for_timeout(1500)
            await speak('christoph-long-reading')
            await wait(lambda:len([e for e in events if e['type']=='reply'])>=2)
            reply=[e for e in events if e['type']=='reply'][-1];rid=reply['reply_id']
            assert reply.get('speaker_name')=='Christoph' and len(reply['text'].split())>=70,'Invalid long-reading setup'
            await page.wait_for_function('rid=>robo.sources.some(s=>s.roboId===rid)',arg=rid)
            await page.wait_for_timeout(6000)
            assert await page.evaluate('rid=>robo.sources.some(s=>s.roboId===rid)',rid),'Long answer ended before Stop'
            start=time.monotonic();report['command_wall']=start;await speak(command)
            await wait(lambda:any(e['type']=='barge' and e['wall']>=start for e in events),10)
            barge=next(e for e in events if e['type']=='barge' and e['wall']>=start)
            report['command_to_barge_s']=barge['wall']-start
            assert not await page.evaluate('rid=>robo.sources.some(s=>s.roboId===rid)',rid)
            await wait(lambda:any(e['type']=='stop_feedback' for e in events),15)
            phrase=next(e['text'] for e in events if e['type']=='stop_feedback')
            await wait(lambda:any(e['type']=='reply' and e['text']==phrase for e in events),20)
            ack=next(e for e in events if e['type']=='reply' and e['text']==phrase);ack_id=ack['reply_id']
            await wait(lambda:any(v.startswith(f'played:{ack_id}:') for v in played),25)
            assert samples.get(ack_id,0)>0,'No acknowledgment PCM'
            assert await page.evaluate('rid=>robo.queued[rid]>0',ack_id),'Browser never scheduled acknowledgment'
            assert await page.evaluate('robo.stream.getAudioTracks()[0].readyState==="live"')
            assert not errors and not any(e['type']=='error' for e in events)
            report.update(passed=runtime_signature()==report['runtime_source_sha256'],acknowledgment_text=phrase,
                acknowledgment_reply_id=ack_id,acknowledgment_samples=samples[ack_id],
                acknowledgment_played=True,old_audio_cancelled=True,microphone_remained_active=True)
        except Exception as exc:
            report['error']=str(exc)
        finally:
            report.update(page_errors=errors,played_notifications=played)
            await page.evaluate('window.robo?.stop()');await browser.close()
            Path('logs/browser-spoken-stop.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in report.items() if k!='events'},ensure_ascii=False))
    if not report['passed']:raise SystemExit('Browser spoken-stop failed; receipt retained')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com/desk-robot/')
    p.add_argument('--command',choices=('christoph-stop','christoph-stop-speaking'),default='christoph-stop')
    args=p.parse_args();asyncio.run(main(args.url,args.command))
