"""Real HTTPS CPU Chrome: all five ASR choices, fake microphone, LLM and TTS."""
import asyncio,json,tempfile,time
from pathlib import Path
import numpy as np,soundfile as sf
from scipy.signal import resample_poly
from playwright.async_api import async_playwright
from asr_catalog import MODELS

ROOT=Path(__file__).resolve().parent
URL='https://arrive-locations-parade-research.trycloudflare.com'

async def main():
    report={'passed':False,'physical_microphone':False,'url':URL,'rows':[],'started':time.time()}
    wave=np.r_[np.zeros(4*16000),np.load(ROOT/'logs/asr-models/german-0.npy'),np.zeros(14*16000)].astype(np.float32)
    with tempfile.TemporaryDirectory(prefix='robby-asr-browser-') as temp:
        path=Path(temp)/'synthetic-german.wav';sf.write(path,resample_poly(wave,3,1),48000,subtype='PCM_16')
        async with async_playwright() as p:
            for key in MODELS:
                row={'model':key,'passed':False};errors=[]
                browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=[
                    '--no-sandbox','--disable-gpu','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream',
                    '--use-file-for-fake-audio-capture='+str(path),'--autoplay-policy=no-user-gesture-required'])
                page=await browser.new_page(viewport={'width':1280,'height':1000});page.on('pageerror',lambda e:errors.append(str(e)))
                await page.add_init_script("localStorage.setItem('robby.llmSelection',JSON.stringify({model:'local',provider:'local'}));localStorage.setItem('robby.cameraAuto','false');localStorage.setItem('robby.avatar-idle-enabled','false');localStorage.setItem('robby.avatar-pet-invites-enabled','false')")
                try:
                    await page.goto(URL+'/desk-robot/')
                    await page.wait_for_function('window.robotFace?.asrControls && !document.querySelector("#call").disabled',timeout=60000)
                    assert await page.locator('#asr-model-controls').is_visible()
                    assert await page.locator('#asr-model option').count()==5
                    await page.evaluate('''()=>{window.asrEvents=[];const old=robo.on.message;robo.on.message=m=>{if(['asr_settings','asr_sentence','endpoint','reply','turn_done','error','warning'].includes(m.type))asrEvents.push({...m,at:performance.now()});old(m)}}''')
                    await page.select_option('#asr-model',key)
                    await page.wait_for_function('(key)=>asrEvents.some(m=>m.type==="asr_settings"&&m.selected_model===key&&!m.loading_model)',arg=key,timeout=30000)
                    await page.click('#call')
                    await page.wait_for_function('asrEvents.some(m=>m.type==="turn_done")&&robo.metrics.chunks>0',timeout=60000)
                    events=await page.evaluate('asrEvents');sent=[m for m in events if m['type']=='asr_sentence'];end=[m for m in events if m['type']=='endpoint']
                    assert len(sent)==1 and sent[0]['asr']['model']==key and 'grimass' in sent[0]['text'].casefold() and 'machen' in sent[0]['text'].casefold(),events
                    assert end and end[0]['asr']['model']==key,events
                    assert not [m for m in events if m['type']=='error'] and not errors,events
                    row.update(passed=True,transcript=sent[0]['text'],selected_final_asr=sent[0]['asr'],
                        speech_pause_ms=sent[0]['speech_pause_ms'])
                except Exception as exc:
                    row['error']=str(exc)
                    try:row['events']=await page.evaluate('window.asrEvents||[]')
                    except Exception:pass
                finally:
                    try:
                        row['pcm_chunks']=await page.evaluate('robo.metrics.chunks');row['saved_choice']=await page.evaluate("localStorage.getItem('robby.asrModel')")
                        row['reply']=await page.evaluate("(window.asrEvents||[]).filter(m=>m.type==='reply').at(-1)?.text||''")
                        await page.evaluate('robo.stop()')
                    except Exception:pass
                    row['browser_errors']=errors;await browser.close()
                report['rows'].append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    report.update(passed=all(r['passed'] for r in report['rows']),finished=time.time())
    (ROOT/'static/asr-model-browser.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':asyncio.run(main())
