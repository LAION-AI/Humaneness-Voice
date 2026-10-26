"""Actual session-owned history: six typed turns, real browser and streamed TTS.

No caller-supplied history. Compliments/offers cannot substitute for the task.
Digital playback timing only; no physical microphone or human humor scoring.
"""
import asyncio,json,os,re,time
from pathlib import Path
from playwright.async_api import async_playwright
from validate_conversation_continuity import SCENARIO,TOPICS

async def main():
    root=Path(__file__).parent;errors=[]
    report={'passed':False,'started':time.time(),'scope':'Actual WebSocket session memory, typed prompts, real Qwen/Cerebras and sentence TTS, CPU browser; no supplied history or physical mic.','turns':[]}
    url=os.environ.get('ROBBY_TEST_URL','http://127.0.0.1:8793').rstrip('/')
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=os.environ.get('ROBBY_TEST_CHROME','/usr/bin/google-chrome'),headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script('localStorage.setItem("robby.cameraAuto","false");localStorage.setItem("robby.avatar-idle-enabled","false");localStorage.setItem("robby.avatar-pet-invites-enabled","false");')
        try:
            await page.goto(url+'/desk-robot/');await page.wait_for_function('window.robo&&robotFace.llmControls?.catalog&&!document.querySelector("#call").disabled',timeout=60000)
            await page.evaluate('''()=>{window.dialogEvents=[];window.pcmCount=0;const old=robo.on.message,play=robo.play.bind(robo);robo.on.message=m=>{dialogEvents.push(m);old(m)};robo.play=x=>{pcmCount++;play(x)}}''')
            await page.evaluate('robo.start(false)');await page.wait_for_function('dialogEvents.some(e=>e.type==="ready")',timeout=20000)
            previous=[]
            for index,(name,prompt,min_words,topic) in enumerate(SCENARIO):
                if index==0:prompt='Erzähl mir einen Witz über einen kleinen Roboter.'
                baseline=await page.evaluate('({event:dialogEvents.length,pcm:pcmCount})')
                await page.fill('#message',prompt);await page.locator('#form').evaluate('f=>f.requestSubmit()')
                await page.wait_for_function('(n)=>dialogEvents.slice(n).some(e=>e.type==="turn_done")&&!robo.sources.length',arg=baseline['event'],timeout=120000)
                events=await page.evaluate('(n)=>dialogEvents.slice(n)',baseline['event'])
                replies=[e for e in events if e['type']=='reply'];text=replies[-1].get('text','') if replies else ''
                fresh=await page.evaluate('pcmCount')-baseline['pcm'];normalized=re.sub(r'\W+',' ',text.casefold()).strip()
                checks={'task_content':len(text.split())>=min_words,'topic':bool(re.search(TOPICS[topic],text,re.I)),
                    'different':normalized not in previous,'real_audio':fresh>0,'no_service_errors':not any(e['type']=='error' for e in events)}
                if name=='longer':checks['different']=True
                report['turns'].append({'case':name,'input':prompt,'text':text,'word_count':len(text.split()),'pcm_chunks':fresh,'checks':checks,'passed':all(checks.values())})
                previous.append(normalized);print(json.dumps(report['turns'][-1],ensure_ascii=False),flush=True)
            report['passed']=all(row['passed'] for row in report['turns']) and not errors
        except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)[:400]
        finally:
            try:await page.evaluate('window.robo?.stop()')
            except Exception:pass
            await browser.close()
    report.update(errors=errors,finished=time.time());(root/'static/conversation-continuity-browser.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':asyncio.run(main())
