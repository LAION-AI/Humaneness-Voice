"""HTTPS auto-start / browser permissions / synthetic Hi Robbie end-to-end.

Uses a fake camera and authored synthetic WAV, never a physical device. No
voiceprints, audio or credentials are written to the public receipt.
"""
import argparse,asyncio,datetime,json,re
from pathlib import Path
import numpy as np,soundfile as sf
from playwright.async_api import async_playwright
from generate_adversarial_fixtures import decode
from run_voice_assistant_checks import deployed_source_hash
ROOT=Path(__file__).resolve().parent
INIT='''window.autoEvents=[];window.autoPCM=0;window.mediaRequests=[];
const Socket=window.WebSocket;window.WebSocket=class extends Socket{constructor(...args){super(...args);this.addEventListener('message',e=>{
 if(typeof e.data==='string'){const m=JSON.parse(e.data);if(['ready','endpoint','reply','turn_done','error','asr_settings'].includes(m.type))autoEvents.push(m);}
 else if(e.data instanceof ArrayBuffer)autoPCM++;
});}};
const get=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
navigator.mediaDevices.getUserMedia=constraints=>{mediaRequests.push({audio:!!constraints.audio,video:!!constraints.video});return get(constraints);};
localStorage.setItem('robby.avatar-idle-enabled','false');localStorage.setItem('robby.avatar-pet-invites-enabled','false');
localStorage.setItem('robby.llmSelection',JSON.stringify({model:'local',provider:'local'}));'''
async def main(url):
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');report={'scope':__doc__,'source_sha256':deployed_source_hash(),'passed':False,'cases':[]}
 voice=decode(ROOT/'logs/auto-wake-hi-robbie.mp3');wav=ROOT/'logs/auto-wake-fake-microphone.wav'
 sf.write(wav,np.r_[np.zeros(6*16000,np.float32),voice,np.zeros(20*16000,np.float32)],16000,subtype='PCM_16')
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream','--use-file-for-fake-audio-capture='+str(wav),'--autoplay-policy=no-user-gesture-required'])
  async def case(mode):
   page=await browser.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)));await page.add_init_script(INIT)
   if mode=='legacy':
    await page.add_init_script('''if(!sessionStorage.getItem('autoWakeTestInitialized')){sessionStorage.setItem('autoWakeTestInitialized','1');
      localStorage.setItem('robby.autoStart','false');localStorage.setItem('robby.cameraAuto','false');localStorage.setItem('robby.cameraAutoVersion','2');localStorage.setItem('robby.followPosition','false');localStorage.setItem('robby.memorySettings',JSON.stringify({enabled:false}));}''')
   if mode=='camera-denied':await page.add_init_script('''{const get=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
     navigator.mediaDevices.getUserMedia=x=>x.video?Promise.reject(new DOMException('Camera denied','NotAllowedError')):get(x);}''')
   if mode=='microphone-denied':await page.add_init_script('''{const get=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
     navigator.mediaDevices.getUserMedia=x=>x.audio?Promise.reject(new DOMException('Microphone denied','NotAllowedError')):get(x);}''')
   row={'case':mode,'passed':False}
   try:
    response=await page.goto(url.rstrip('/')+'/desk-robot/');assert response.status==200
    await page.wait_for_function('window.robo&&window.robotFace?.environmentControls&&document.querySelector("#wake-mode")',timeout=60000)
    assert await page.locator('#robby-auto-start').is_checked();assert await page.locator('#camera-auto').is_checked();assert await page.locator('#mirror').is_checked();assert await page.locator('#wake-mode').is_checked()
    assert await page.locator('#memory-enabled').is_checked();row['persistent_memory_default_on']=True
    assert not await page.locator('#faceIdEnabled').is_checked();assert not await page.locator('#ignoreOffscreenSpeakers').is_checked()
    if mode=='microphone-denied':
     await page.locator('#enable-robby').wait_for(state='visible',timeout=30000)
     row['notice']=await page.locator('#auto-start-status').inner_text();assert 'permission' in row['notice'].lower() or 'allow' in row['notice'].lower()
     assert not await page.evaluate('!!robo.stream');row['microphone_denied_fallback']=True
    else:
     await page.wait_for_function('robo.stream&&robo.ws?.readyState===1&&robo.metrics.mic_blocks>20',timeout=60000)
     row['microphone_packets']=await page.evaluate('robo.metrics.mic_blocks');row['actual_microphone_was_fake']=True
     if mode=='camera-denied':
      await page.wait_for_function('document.querySelector("#camera-status").textContent.includes("unavailable")',timeout=60000)
      assert await page.evaluate('!!robo.stream&&!robotFace.cameraControls.active');row['camera_denied_audio_continues']=True
     else:
      await page.wait_for_function('robotFace.cameraControls.active&&robotFace.cameraStats.inferences>=3',timeout=60000)
      row['camera_frames']=await page.evaluate('robotFace.cameraStats.inferences')
      # Real UI gesture path, using synthetic landmarks: head/eye translation
      # produces opposite pan targets, with no dependence on facial roll.
      row['gaze']=await page.evaluate('''async()=>{
       const {eyeObservation,predictGaze}=await import('/static/gaze.js');
       function face(u){const pts=Array.from({length:478},()=>({x:u,y:.5,z:0}));
        for(const [i,x] of [[33,u-.06],[133,u-.02],[362,u+.02],[263,u+.06]])pts[i]={x,y:.5,z:0};return {faceLandmarks:[pts]};}
       const left=predictGaze(eyeObservation(face(.3),640,480)),right=predictGaze(eyeObservation(face(.7),640,480));
       return {left_pan:left.pan,right_pan:right.pan,translation_tracked:left.pan>0&&right.pan<0};}''')
      assert row['gaze']['translation_tracked']
    if mode=='legacy':
     await page.wait_for_function('autoEvents.some(e=>e.type==="endpoint")',timeout=90000)
     await page.wait_for_function('autoEvents.some(e=>e.type==="turn_done")&&autoPCM>0',timeout=90000)
     events=await page.evaluate('autoEvents');ends=[e for e in events if e['type']=='endpoint'];settings=[e for e in events if e['type']=='asr_settings']
     row['transcripts']=[e.get('transcript') for e in ends];row['pcm_chunks']=await page.evaluate('autoPCM')
     row['actual_reply_model']=await page.locator('#recent-reply-model').inner_text()
     assert any(re.search(r'\b(?:hi|hey|hallo)\W+r[oa]b{1,2}(?:y|ie|i|o)\b',x or '',re.I) for x in row['transcripts']),row['transcripts']
     assert settings and settings[-1]['wake_mode']['remaining_s']>=8,settings[-1]['wake_mode']
     row['ten_seconds_after_playback']=True
     await page.evaluate('window.stoppedMic=robo.stream;window.stoppedVideo=document.querySelector("#video").srcObject')
     await page.locator('#disable-robby').click();await page.wait_for_timeout(500)
     assert await page.evaluate('!robo.stream&&!robo.ws&&!robotFace.cameraControls.active&&stoppedMic.getTracks().every(t=>t.readyState==="ended")&&stoppedVideo.getTracks().every(t=>t.readyState==="ended")')
     await page.reload();await page.wait_for_function('window.robotFace?.environmentControls');await page.wait_for_timeout(1600)
     assert not await page.locator('#robby-auto-start').is_checked();assert not await page.locator('#camera-auto').is_checked();assert await page.evaluate('!robo.stream&&!robotFace.cameraControls.active')
     row['explicit_off_stops_and_persists']=True
    row['page_errors']=errors;row['passed']=not errors
   except Exception as exc:
    row['error']=type(exc).__name__+': '+str(exc)[:600]
    row['events']=await page.evaluate('window.autoEvents||[]')
   finally:
    await page.evaluate('window.robo?.stop();window.robotFace?.cameraControls?.stop()');await page.close();await asyncio.sleep(.6)
   report['cases'].append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
  try:
   for mode in ('legacy','camera-denied','microphone-denied'):await case(mode)
   report['passed']=all(r['passed'] for r in report['cases'])
  finally:await browser.close()
 report['source_unchanged']=report['source_sha256']==deployed_source_hash();report['passed'] &= report['source_unchanged']
 path=ROOT/'static'/('auto-wake-browser-'+stamp+'.json');path.write_text(json.dumps(report,ensure_ascii=False,indent=2));print({'passed':report['passed'],'receipt':path.name},flush=True)
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com');a=parser.parse_args();asyncio.run(main(a.url))
