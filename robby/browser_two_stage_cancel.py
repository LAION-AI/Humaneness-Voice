"""HTTPS cancellation during the new reaction and informed-transition paths.

Typed authored text and the existing Stop button, not acoustic interruption.
"""
import asyncio,datetime,json,time
from pathlib import Path
import httpx
from playwright.async_api import async_playwright
from run_voice_assistant_checks import deployed_source_hash
ROOT=Path(__file__).resolve().parent
URL='https://diane-asylum-understood-invoice.trycloudflare.com'
async def main():
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report={'scope':__doc__,'passed':False,'source_sha256':deployed_source_hash(),'trials':[]};errors=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script("localStorage.setItem('robby.autoStart','false');localStorage.setItem('robby.environment','false');")
        await page.add_init_script("localStorage.setItem('robby.quickReplies','true');localStorage.setItem('robby.avatar-idle-enabled','false');localStorage.setItem('robby.avatar-pet-invites-enabled','false');")
        try:
            await page.goto(URL+'/desk-robot/');await page.wait_for_function('window.robo&&!document.querySelector("#call").disabled',timeout=60000)
            await page.evaluate('''()=>{window.ev=[];const old=robo.on.message;robo.on.message=m=>{ev.push({...m,at:performance.now()});old(m)};}''')
            await page.evaluate('robo.start(false)');await page.click('#message');await page.evaluate('robo.ctx.resume()')
            for role in ('reaction','informed_bridge'):
                mark=await page.evaluate('ev.length')
                await page.fill('#message','Erkläre mir ausführlich, warum der Himmel blau ist und wie ein Regenbogen entsteht.')
                await page.locator('#form').evaluate('f=>f.requestSubmit()')
                await page.wait_for_function('({n,role})=>ev.slice(n).some(e=>e.type==="reply"&&e.sentence_role===role)&&robo.sources.length>0',arg={'n':mark,'role':role},timeout=60000)
                before=await page.evaluate('(n)=>ev.slice(n).filter(e=>e.type==="reply").at(-1)',mark)
                assert before['timing']['tts_backend']=='continuous_batch'
                await page.click('#interrupt');started=time.perf_counter()
                await page.wait_for_function('robo.sources.length===0',timeout=3000)
                async with httpx.AsyncClient() as c:
                    for _ in range(100):
                        state=(await c.get('http://127.0.0.1:8792/api/state')).json()
                        if not state['busy']:break
                        await asyncio.sleep(.1)
                    assert not state['busy'],'CUDA owner did not release'
                idle_ms=round((time.perf_counter()-started)*1000,2)
                await page.wait_for_timeout(1000)
                events=await page.evaluate('(n)=>ev.slice(n)',mark)
                assert any(e['type']=='reply_cancelled' for e in events)
                assert await page.evaluate('robo.sources.length')==0
                report['trials'].append({'role':role,'events':events,'worker_idle_ms':idle_ms,'no_stale_playback':True})
            report['passed']=not errors
        except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)[:1000]
        finally:
            try:await page.evaluate('window.robo?.stop()')
            except Exception:pass
            await browser.close();report['errors']=errors;report['source_unchanged']=report['source_sha256']==deployed_source_hash();report['passed'] &= report['source_unchanged']
            raw=json.dumps(report,ensure_ascii=False,indent=2);(ROOT/'logs'/('two-stage-cancel-'+stamp+'.json')).write_text(raw);(ROOT/'static/two-stage-cancel.json').write_text(raw)
    print(json.dumps({k:report.get(k) for k in ('passed','error')}),flush=True)
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':asyncio.run(main())
