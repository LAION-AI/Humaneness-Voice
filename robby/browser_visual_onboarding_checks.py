"""Actual browser microphone -> ASR -> identity -> TTS, with modest SYNTHETIC
landmarks processed by the production visual detector. No physical lip-sync
accuracy claim. A held mouth must block, with a voice-only recovery UI.
"""
import argparse,asyncio,json,subprocess,tempfile,time
from pathlib import Path
from playwright.async_api import async_playwright
from final_verify import runtime_signature

ROOT=Path(__file__).parent
FIXTURE=Path('/mnt/nvme/robo-speaker-studies/speaker-live-fixtures/christoph-intro.mp3')

async def main(url):
    report={'started':time.time(),'passed':False,'runtime_source_sha256':runtime_signature(),
        'physical_devices_used':False,'visual_input':'synthetic 10-Hz landmarks, not forced speaking flags',
        'audio_path':'Chrome fake microphone, real worklet/resampling/WebSocket/ASR/speaker enrollment/Luna/TTS','cases':[]}
    try:
        with tempfile.TemporaryDirectory(prefix='robby-visual-onboarding-') as folder:
            wav=Path(folder)/'intro.wav'
            subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(FIXTURE),'-af','adelay=1280,apad=pad_dur=35',
                '-ar','48000','-ac','1','-c:a','pcm_s16le',str(wav)],check=True)
            async with async_playwright() as p:
                browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=[
                    '--no-sandbox','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream',
                    '--use-file-for-fake-audio-capture='+str(wav),'--autoplay-policy=no-user-gesture-required'])
                try:
                    for mode in ('audio_default','modest','static'):
                        context=await browser.new_context();page=await context.new_page();events=[];errors=[]
                        record={'mode':mode,'passed':False};report['cases'].append(record)
                        try:
                            await page.add_init_script("localStorage.setItem('robby.cameraAuto','false')")
                            page.on('pageerror',lambda e:errors.append(str(e)))
                            def frame(payload):
                                if isinstance(payload,str):
                                    m=json.loads(payload)
                                    if m.get('type') in ('endpoint','reply','ignored','visual_person','visual_warning','error'):events.append(m)
                            page.on('websocket',lambda ws:ws.on('framereceived',frame))
                            await page.goto(url);await page.wait_for_function('window.robo && !document.querySelector("#call").disabled',timeout=90000)
                            # Open the session before starting the fake device:
                            # Chrome starts the file immediately on getUserMedia,
                            # unlike a human waiting for the Listening prompt.
                            await page.evaluate('async()=>{await robo.start(false);}')
                            await page.evaluate('''async mode=>{
                              const {testFace}=await import('/static/multimodal-test-cases.js');
                              const m=robotFace.multimodal;m.setCamera('on');window.visualProbe=[];window.micProbe=[];window.eventProbe=[];
                              if(mode!=='audio_default')for(const key of ['ignoreOffscreenSpeakers','faceIdEnabled']){const input=document.getElementById(key);input.checked=true;input.onchange();}
                              const originalMic=robo.mic.bind(robo),originalMessage=robo.on.message;
                              window.lastVoicedAt=-Infinity;
                              robo.mic=(t,x)=>{let energy=0;for(const value of x)energy+=value*value;
                                const rms=Math.sqrt(energy/x.length);if(rms>.004)window.lastVoicedAt=performance.now();
                                if(rms>.004)micProbe.push((robo.sentMicSamples||0)/16000);originalMic(t,x);};
                              robo.on.message=m=>{if(['ignored','asr_preview','endpoint'].includes(m.type))eventProbe.push({audio:robo.audioTimeAt(),...m});originalMessage(m);};
                              window.visualTimer=setInterval(()=>{
                                const at=performance.now(),audio=robo.audioTimeAt(at),active=at-window.lastVoicedAt<300;
                                const open=mode==='modest'&&active ? .10+.03*Math.sin(audio*4*Math.PI) : .10;
                                // A zero-size video prevents unrelated descriptor extraction from these artificial landmarks.
                                m.frame({faceLandmarks:[testFace(.3,open)]},at,{videoWidth:0,videoHeight:0},0);
                                if(audio!==null)visualProbe.push({audio,speaking:m.tracks[0]?.speaking,confidence:m.tracks[0]?.confidence});
                              },100);
                            }''',mode)
                            await page.click('#call');deadline=time.monotonic()+35
                            while time.monotonic()<deadline:
                                if mode!='static' and any(e['type']=='endpoint' for e in events) and await page.evaluate('robo.metrics.reply_chunks>0'):break
                                if mode=='static' and any(e['type']=='ignored' and e.get('source')=='visual_filter' for e in events):break
                                await page.wait_for_timeout(200)
                            record.update(events=events,page_errors=errors,state=await page.evaluate('''()=>({
                              metrics:robo.metrics,sent_audio_s:robo.sentMicSamples/16000,
                              voice_filter:document.querySelector('#identity-gate').checked,
                              visual_filter:robotFace.multimodal.settings.ignoreOffscreenSpeakers,
                              face_id:robotFace.multimodal.settings.faceIdEnabled,face_id_worker:!!robotFace.multimodal.worker,
                              moving_frames:visualProbe.filter(f=>f.speaking).length,
                              voiced_samples:micProbe,visual_samples:visualProbe,events:eventProbe,
                              status:document.querySelector('#asr-status').textContent})'''))
                            assert not errors,errors
                            assert not any(e['type'] in ('error','visual_warning') for e in events),events
                            assert record['state']['voice_filter'],record
                            assert record['state']['visual_filter']==(mode!='audio_default'),record
                            if mode!='static':
                                assert any(e['type']=='endpoint' and e.get('speaker_name')=='Christoph' for e in events),record
                                if mode=='modest':assert any(e['type']=='visual_person' and e.get('name')=='Christoph' for e in events),record
                                else:
                                    assert not record['state']['face_id'] and not record['state']['face_id_worker'],record
                                    assert not any(e['type']=='visual_person' for e in events),record
                                assert record['state']['metrics']['reply_chunks']>0,record
                            else:
                                assert not any(e['type']=='endpoint' for e in events),record
                                assert any(e.get('code')=='visual_no_motion' for e in events),record
                                await page.locator('#visual-filter-recovery').click()
                                record['recovery']=await page.evaluate('''()=>({
                                  voice_filter:document.querySelector('#identity-gate').checked,
                                  visual_filter:robotFace.multimodal.settings.ignoreOffscreenSpeakers,
                                  camera:robotFace.multimodal.camera,
                                  persisted:JSON.parse(localStorage.getItem('robby.multimodal.settings.v2')).ignoreOffscreenSpeakers})''')
                                assert record['recovery']=={'voice_filter':True,'visual_filter':False,'camera':'on','persisted':False},record
                            record['passed']=True
                            print(json.dumps({'mode':mode,'passed':True,'state':{k:v for k,v in record['state'].items() if k not in ('voiced_samples','visual_samples','events')}}),flush=True)
                        finally:
                            await page.evaluate('()=>{clearInterval(window.visualTimer);window.robo?.stop();}')
                            await context.close();await asyncio.sleep(2)
                finally:await browser.close()
        report['passed']=report['runtime_source_sha256']==runtime_signature()
    finally:
        report['finished']=time.time();(ROOT/'logs/visual-onboarding-checks.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8793/desk-robot/');asyncio.run(main(parser.parse_args().url))
