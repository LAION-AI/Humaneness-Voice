"""HTTPS defaults + real Parakeet, parallel GPU captions, LLM and TTS.

Synthetic public audio enters the actual browser transport. No live microphone
or face image is recorded. UI migration and caption-off controls are tested too.
"""
import argparse,asyncio,datetime,json,fcntl
from pathlib import Path
from types import SimpleNamespace
from playwright.async_api import async_playwright
from browser_adversarial_check import BrowserSession
from generate_adversarial_fixtures import decode,fixture_root
from run_voice_assistant_checks import deployed_source_hash,public_state

ROOT=Path(__file__).resolve().parent
async def run(args):
    report={'passed':False,'scope':'HTTPS browser + synthetic microphone; real GPU models and TTS; not a room test.',
        'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':deployed_source_hash(),
        'server':public_state(args.url),'injections':[]};errors=[]
    out=ROOT/'logs'/('voice-captions-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script('''window.captureCalls=[];navigator.mediaDevices.getUserMedia=async constraints=>{captureCalls.push(constraints);throw Error('Synthetic test: no real camera');};''')
        session=None
        try:
            await page.goto(args.url+'/desk-robot/')
            await page.wait_for_function('window.robotFace?.asrControls&&!document.querySelector("#call").disabled',timeout=60000)
            report['fresh_defaults']={'asr':await page.locator('#asr-model').input_value(),'camera':await page.locator('#camera-auto').is_checked(),
                'captions':await page.locator('#voice-captions').is_checked(),'capture_calls':await page.evaluate('captureCalls.length')}
            assert report['fresh_defaults']=={'asr':'parakeet_v3','camera':False,'captions':True,'capture_calls':0}
            await page.evaluate('''()=>{localStorage.removeItem('robby.asrDefaultVersion');localStorage.setItem('robby.asrModel','nemotron');localStorage.removeItem('robby.cameraAutoVersion');localStorage.setItem('robby.cameraAuto','true');}''')
            await page.reload();await page.wait_for_function('window.robotFace?.asrControls',timeout=30000)
            assert await page.locator('#asr-model').input_value()=='parakeet_v3'
            assert not await page.locator('#camera-auto').is_checked()
            assert await page.evaluate('captureCalls.length')==0
            report['legacy_defaults_migrated']=True
            await page.evaluate("localStorage.setItem('robby.asrModel','nemotron')")
            await page.reload();await page.wait_for_function('window.robotFace?.asrControls',timeout=30000)
            assert await page.locator('#asr-model').input_value()=='nemotron';report['explicit_later_choice_preserved']=True
            await page.close();page=await browser.new_page()
            manifest=json.loads((fixture_root()/'manifest.json').read_text());rows={r['name']:r for r in manifest['cases']}
            clips={n:decode(fixture_root()/r['file']) for n,r in rows.items()}
            session=BrowserSession(page,clips,rows,report,SimpleNamespace(url=args.url,threshold=.28,pause_ms=700,seed=56,asr_model='parakeet_v3'))
            await session.open()
            row=await session.speak({'id':'caption-on','clip':'christoph-intro','max_submissions':2},{'speaker':'Christoph'},setup=True)
            report['caption_on']=row
            caps=[e for e in row['events'] if e['type']=='voice_caption']
            assert row['passed'],row['checks']
            assert any(e.get('status')=='available' and e.get('device','').startswith('cuda') for e in caps),caps
            report['caption_visible']=await page.locator('#voice-caption-status').text_content()
            assert 'uncertain estimate' in report['caption_visible']
            assert all(e.get('asr',{}).get('model')=='parakeet_v3' for e in row['events'] if e['type']=='endpoint')
            await page.locator('#conversation-options').evaluate('e=>e.open=true')
            await page.uncheck('#voice-captions');await page.wait_for_timeout(100)
            row=await session.speak({'id':'caption-off','clip':'christoph-followup'},{'speaker':'Christoph'})
            report['caption_off']=row
            caps=[e for e in row['events'] if e['type']=='voice_caption']
            assert row['passed'],row['checks']
            assert caps and all(e.get('status')=='disabled' for e in caps),caps
            await page.evaluate('robo.stop()');await page.reload()
            await page.wait_for_function('window.robotFace?.asrControls',timeout=30000)
            assert not await page.locator('#voice-captions').is_checked()
            report['caption_setting_persists']=True
            assert not errors+session.errors,errors+session.errors
            report['passed']=True
        except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)[:1500]
        finally:
            try:await page.evaluate('window.robo?.stop()')
            except Exception:pass
            await browser.close()
            report['source_unchanged']=report['source_sha256']==deployed_source_hash();report['passed'] &= report['source_unchanged']
            report['errors']=errors+(session.errors if session else [])
            text=json.dumps(report,indent=2,ensure_ascii=False);out.write_text(text);(ROOT/'static/voice-caption-browser.json').write_text(text)
    print(json.dumps({'passed':report['passed'],'error':report.get('error'),'receipt':str(out)}),flush=True)
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com');args=p.parse_args()
    lock=(ROOT/'logs/voice-assistant-runs/.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    asyncio.run(run(args))
