"""Real HTTPS, real Qwen/TTS, both model choices; no private microphone input."""
import asyncio,json,time,os
from pathlib import Path
from playwright.async_api import async_playwright
ROOT=Path(__file__).resolve().parent
URL=os.environ.get('ROBO_TEST_URL','https://diane-asylum-understood-invoice.trycloudflare.com').rstrip('/')

async def main():
 report={'passed':False,'scope':'Typed synthetic prompts over HTTPS, real selected TTS model, PCM arrival and avatar actions. No physical microphone or listening-quality rating.','turns':[],'switches':[]};errors=[]
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,
    args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
  page=await browser.new_page(viewport={'width':1440,'height':1050});page.on('pageerror',lambda e:errors.append(str(e)))
  await page.add_init_script("localStorage.setItem('robby.ttsModel','small_rl');localStorage.setItem('robby.llmSelection',JSON.stringify({model:'qwen',provider:'cerebras/fp16'}));localStorage.setItem('robby.autoStartDefaultVersion','1');localStorage.setItem('robby.autoStart','false');localStorage.setItem('robby.cameraAutoVersion','3');localStorage.setItem('robby.cameraAuto','false');localStorage.setItem('robby.avatar-idle-enabled','false');localStorage.setItem('robby.avatar-actions-enabled','true');localStorage.setItem('robby.memoryDefaultVersion','1');localStorage.setItem('robby.memorySettings',JSON.stringify({enabled:false}));")
  try:
   await page.goto(URL+'/desk-robot/')
   await page.wait_for_function('window.robo&&window.robotFace&&!document.querySelector("#call").disabled',timeout=120000)
   await page.locator('#tts-model').wait_for(state='visible');assert await page.locator('#tts-model').bounding_box()
   await page.evaluate('''()=>{window.testEvents=[];window.testAudio=[];window.testFrames=[];
    const old=robo.on.message,play=robo.play.bind(robo),frame=robotFace.actions.frame.bind(robotFace.actions);
    robo.on.message=m=>{if(['reply','pipeline_metrics','turn_done','error','busy','avatar_action','barge'].includes(m.type))testEvents.push({at:performance.now(),...m});old(m)};
    robo.play=raw=>{testAudio.push({at:performance.now(),bytes:raw.byteLength});play(raw)};
    robotFace.actions.frame=(...args)=>{const result=frame(...args),a=robotFace.actions.active;if(a)testFrames.push({kind:a.kind,source:a.source});return result;};}''')
   for model,prompt,action in [('small_rl','Sage bitte genau: Hallo, ich freue mich auf unser Gespräch.',None),
       ('large17','Sage bitte genau: Hallo, ich freue mich auf unser Gespräch.',None),
       ('large17','Dreh dich einmal im Kreis und sage dabei, dass dir das Spaß macht.','spin'),
       ('small_rl','Sage bitte genau: Schön, dass wir uns wiedersehen.',None)]:
    await page.wait_for_function('!document.querySelector("#tts-model").disabled',timeout=900000)
    if await page.locator('#tts-model').input_value()!=model:
     begin=time.monotonic();await page.locator('#tts-model').select_option(model)
     await page.wait_for_function('(id)=>document.querySelector("#tts-model").value===id&&!document.querySelector("#tts-model").disabled&&document.querySelector("#tts-model-status").textContent.includes("ready")',arg=model,timeout=900000)
     report['switches'].append({'model':model,'wall_seconds':time.monotonic()-begin})
    before=await page.evaluate('testEvents.length');audio_before=await page.evaluate('testAudio.length');frames_before=await page.evaluate('testFrames.length');start=await page.evaluate('performance.now()')
    await page.locator('#message').fill(prompt);await page.locator('#form').evaluate('f=>f.requestSubmit()')
    await page.wait_for_function('(n)=>testEvents.slice(n).some(e=>e.type==="turn_done"||e.type==="error")',arg=before,timeout=180000)
    events=await page.evaluate('(n)=>testEvents.slice(n)',before);audio=await page.evaluate('(n)=>testAudio.slice(n)',audio_before)
    bad=[e for e in events if e['type'] in ('error','busy')];assert not bad,bad
    replies=[e for e in events if e['type']=='reply'];assert replies and audio,events
    metrics=next(e['metrics'] for e in events if e['type']=='pipeline_metrics');assert metrics['tts_model']==model and metrics['tts_actual_model']==model,metrics
    frames=await page.evaluate('(n)=>testFrames.slice(n)',frames_before)
    if action:assert any(f['kind']==action for f in frames),frames[:20]
    row={'model':model,'prompt':prompt,'reply':replies[-1]['text'],'first_pcm_ms':audio[0]['at']-start,'audio_chunks':len(audio),'metrics':metrics,'action':action}
    report['turns'].append(row);(ROOT/'browser-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(row,ensure_ascii=False),flush=True)
    # Finish actual playout before switching; a generated turn_done is earlier.
    await page.wait_for_timeout(1000)
    await page.locator('#interrupt').click();await page.wait_for_timeout(300)
   await page.screenshot(path=str(ROOT/'browser-model-choice.png'),full_page=False)
   report['passed']=not errors
  except Exception as exc:report['error']=repr(exc);print('BROWSER FAIL',repr(exc),flush=True)
  finally:
   report['page_errors']=errors
   try:await page.evaluate('window.robo?.stop()')
   except Exception:pass
   await browser.close()
 (ROOT/'browser-check.json').write_text(json.dumps(report,indent=2));print('BROWSER PASS',report['passed'],flush=True)
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':asyncio.run(main())
