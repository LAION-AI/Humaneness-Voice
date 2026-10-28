import asyncio,json
from playwright.async_api import async_playwright
async def main():
 async with async_playwright() as p:
  b=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream','--autoplay-policy=no-user-gesture-required'])
  page=await b.new_page(viewport={'width':1440,'height':1050});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  await page.add_init_script("localStorage.setItem('robby.cameraAuto','false');localStorage.setItem('robby.avatar-idle-enabled','false');localStorage.setItem('robby.avatar-pet-invites-enabled','false')")
  await page.goto('https://diane-asylum-understood-invoice.trycloudflare.com/desk-robot/');await page.wait_for_function('window.robotFace?.llmControls?.catalog&&!document.querySelector("#call").disabled',timeout=90000)
  assert await page.locator('#llm-model').is_visible();print('models',await page.locator('#llm-model option').all_text_contents(),flush=True)
  await page.select_option('#llm-model','gemma12');await page.wait_for_function('robotFace.llmControls.current?.model==="gemma12"',timeout=30000)
  await page.evaluate('''()=>{window.gemmaEvents=[];const old=robo.on.message;robo.on.message=m=>{if(['reply','pipeline_metrics','turn_done','error'].includes(m.type))gemmaEvents.push(m);old(m)}}''')
  await page.locator('#message').fill('Sag mir einen kurzen netten Satz.');await page.locator('#form button').click()
  await page.wait_for_function('gemmaEvents.some(m=>m.type==="turn_done")',timeout=90000)
  data=await page.evaluate('({events:gemmaEvents,reply:document.querySelector("#recent-robby").textContent,pcm:robo.metrics.chunks})');print(json.dumps(data,ensure_ascii=False),flush=True)
  assert data['pcm']>0 and data['reply'] and any(x.get('metrics',{}).get('llm_model')=='gemma-4-12b-q4' for x in data['events'])
  assert not errors,errors
  await page.select_option('#llm-model','qwen');await page.wait_for_function('robotFace.llmControls.current?.model==="qwen"',timeout=30000)
  print('qwen switching works',flush=True);await page.evaluate('robo.stop()');await b.close()
asyncio.run(main())
