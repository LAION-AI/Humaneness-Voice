"""Real HTTPS desktop/mobile UI, fake microphone, all component cards and keys."""
import asyncio,json,tempfile,time
from pathlib import Path
import numpy as np,soundfile as sf
from scipy.signal import resample_poly
from playwright.async_api import async_playwright
from credential_store import store
ROOT=Path(__file__).resolve().parent
URL='https://arrive-locations-parade-research.trycloudflare.com'

async def main():
    report={'passed':False,'started':time.time(),'physical_microphone':False,'turns':[]};errors=[];responses=[]
    wave=np.r_[np.zeros(4*16000),np.load(ROOT/'logs/asr-models/german-0.npy'),np.zeros(240*16000)].astype(np.float32)
    with tempfile.TemporaryDirectory(prefix='robby-cloud-browser-') as temp:
        wav=Path(temp)/'synthetic-mic.wav';sf.write(wav,resample_poly(wave,3,1),48000,subtype='PCM_16')
        async with async_playwright() as p:
            browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=[
                '--no-sandbox','--disable-gpu','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream',
                '--use-file-for-fake-audio-capture='+str(wav),'--autoplay-policy=no-user-gesture-required'])
            page=await browser.new_page(viewport={'width':1440,'height':1050});page.on('pageerror',lambda e:errors.append(str(e)))
            async def response_seen(r):
                if r.url.startswith(URL) and any(t in r.headers.get('content-type','') for t in ('json','javascript','html')):
                    try:responses.append(await r.text())
                    except Exception:pass
            page.on('response',response_seen)
            await page.add_init_script("localStorage.setItem('robby.cameraAuto','false');localStorage.setItem('robby.avatar-idle-enabled','false');localStorage.setItem('robby.avatar-pet-invites-enabled','false')")
            try:
                await page.goto(URL+'/desk-robot/')
                await page.wait_for_timeout(1000);assert not errors,errors
                await page.wait_for_function('window.robotFace?.llmControls?.catalog&&!document.querySelector("#call").disabled',timeout=60000)
                assert await page.locator('#llm-model').input_value()=='qwen'
                assert await page.locator('#llm-provider').input_value()=='cerebras/fp16'
                assert await page.locator('#llm-personal-key').input_value()==''
                await page.evaluate('''()=>{window.cloudEvents=[];const old=robo.on.message;robo.on.message=m=>{if(['latency_start','latency_update','pipeline_metrics','reply','asr_sentence','endpoint','turn_done','llm_settings','interaction','error'].includes(m.type))cloudEvents.push(m);old(m)}}''')
                async def selection(model,provider):
                    await page.select_option('#llm-model',model)
                    if model!='local':await page.select_option('#llm-provider',provider)
                    await page.wait_for_function('(v)=>robotFace.llmControls.current?.model===v[0]&&robotFace.llmControls.current?.provider===v[1]',arg=[model,provider],timeout=15000)
                async def finish(name):
                    await page.wait_for_function('cloudEvents.some(m=>m.type==="turn_done")&&robo.metrics.chunks>0',timeout=90000)
                    data=await page.evaluate('({events:cloudEvents,values:robotFace.overview.values,asr:document.querySelector("#latency-asr").textContent,speaker:document.querySelector("#latency-speaker").textContent,user:document.querySelector("#recent-user").textContent,robot:document.querySelector("#recent-robby").textContent,pcm:robo.metrics.chunks})')
                    assert not [m for m in data['events'] if m['type']=='error'],data['events']
                    assert data['robot'].strip() and data['pcm']>0
                    for key in ('llm','tts','reply','buffer'):assert isinstance(data['values'].get(key),(int,float)) and data['values'][key]>=0,(key,data['values'])
                    report['turns'].append({'case':name,'reply':data['robot'],'user':data['user'],'timings':data['values'],'asr_card':data['asr'],'speaker_card':data['speaker'],'pcm_total':data['pcm']})
                    return data
                await page.select_option('#asr-model','whisper_turbo')
                await page.wait_for_function('robotFace.asrControls.active==="whisper_turbo"',timeout=15000)
                await page.click('#call')
                data=await finish('fake microphone · Whisper → Qwen/Cerebras → TTS')
                assert isinstance(data['values'].get('asr'),(float,int)) and isinstance(data['values'].get('speaker'),(float,int)),data['values']
                assert 'grimass' in data['user'].casefold()
                await page.evaluate('cloudEvents=[]')
                await page.locator('#message').fill('Schreibe ein fröhliches Gedicht mit vier Zeilen.');await page.locator('#form button').click()
                data=await finish('four-line poem · Qwen/Cerebras')
                assert data['asr']=='Typed' and len(data['robot'].splitlines())==4,data
                await page.locator('#avatar').scroll_into_view_if_needed()
                avatar=await page.locator('#avatar').bounding_box();recent=await page.locator('#live-conversation').bounding_box();latency=await page.locator('#latency-dashboard').bounding_box()
                assert recent['x']>avatar['x']+avatar['width']*.7 and min(avatar['y']+avatar['height'],recent['y']+recent['height'])>max(avatar['y'],recent['y']),[avatar,recent]
                assert latency['y']>=avatar['y']+avatar['height']-20,[avatar,latency]
                await page.screenshot(path=str(ROOT/'logs/openrouter-desktop.png'))
                report['desktop_context_beside_avatar']=True;report['timings_below_avatar']=True
                await selection('local','local');await page.evaluate('cloudEvents=[]')
                await page.locator('#message').fill('Schlage ein lustiges Spiel vor.');await page.locator('#form button').click()
                data=await finish('local Nemotron switch');assert data['values']['provider']=='local',data['values']
                await selection('qwen','deepinfra/bf16');await page.evaluate('cloudEvents=[]')
                await page.locator('#message').fill('Sag nur: Guten Morgen.');await page.locator('#form button').click()
                data=await finish('Qwen alternative endpoint · DeepInfra');assert 'DeepInfra' in data['values']['provider'],data['values']
                await selection('qwen','cerebras/fp16');await page.evaluate('cloudEvents=[]')
                await page.locator('#message').fill('Wenn ich über Bananen spreche, unterbrich mich und sage Stopp, stopp!');await page.locator('#form button').click()
                await finish('real interruption-rule tool · Cerebras')
                assert await page.evaluate('cloudEvents.some(m=>m.type==="interaction"&&m.rules?.length)'),await page.evaluate('cloudEvents')
                report['real_rule_installed']=True
                await selection('local','local');await page.locator('#llm-model-controls details').evaluate('(e)=>e.open=true')
                fake='sk-or-v1-'+'browser-fixture-'*4
                await page.locator('#llm-personal-key').fill(fake);await page.click('#llm-key-use')
                await page.wait_for_function('document.querySelector("#llm-personal-key").value===""&&cloudEvents.some(m=>m.type==="llm_settings"&&m.key_mode==="personal_session")',timeout=10000)
                assert fake not in await page.evaluate('JSON.stringify(Object.fromEntries(Object.entries(localStorage)))')
                await page.click('#llm-key-clear');report['personal_key_not_persisted']=True
                await selection('qwen','cerebras/fp16');await page.evaluate('robo.stop()')
                await page.reload();await page.wait_for_function('window.robotFace?.llmControls?.catalog&&!document.querySelector("#call").disabled',timeout=60000)
                assert await page.locator('#llm-model').input_value()=='qwen' and await page.locator('#llm-provider').input_value()=='cerebras/fp16'
                assert await page.locator('#llm-personal-key').input_value()=='';report['selection_persists_after_reload']=True
                await page.set_viewport_size({'width':390,'height':844})
                assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
                report['mobile_no_overflow']=True
                assert not errors,errors
                owner=store().owner();assert all(owner not in text for text in responses)
                report['owner_key_not_in_browser_responses']=True;report['passed']=True
            except Exception as exc:
                report['error']=str(exc)
                try:report['last_events']=await page.evaluate('window.cloudEvents||[]')
                except Exception:pass
                await page.screenshot(path=str(ROOT/'logs/openrouter-browser-failure.png'))
            finally:
                try:await page.evaluate('window.robo?.stop()')
                except Exception:pass
                await browser.close()
    report.update(errors=errors,finished=time.time())
    output=json.dumps(report,indent=2,ensure_ascii=False);assert store().owner() not in output
    (ROOT/'static/openrouter-browser.json').write_text(output);print(output)
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':asyncio.run(main())
