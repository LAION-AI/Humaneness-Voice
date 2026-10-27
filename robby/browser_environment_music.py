"""Real HTTPS fake-microphone music regression; authored CC0 melody and drums.

Quiet music must not trigger; prominent music must produce a spoken response.
A second onset after the detector cooldown must not repeat the commentary.
"""
import argparse,asyncio,datetime,json
from pathlib import Path
import numpy as np
import soundfile as sf
from playwright.async_api import async_playwright
from browser_environment_check import INIT
from environment_fixtures import music_fixture
from run_voice_assistant_checks import deployed_source_hash,public_state

ROOT=Path(__file__).resolve().parent
async def main(url):
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    wave=np.random.default_rng(912).normal(0,.0003,72*16000).astype(np.float32)
    for at,duration,gain in ((3,8,.005),(18,12,1.),(56,10,1.)):
        clip=music_fixture(duration)*gain;start=at*16000;wave[start:start+len(clip)]+=clip
    fixture=ROOT/'logs/environment-music-microphone.wav';sf.write(fixture,wave,16000,subtype='PCM_16')
    report={'passed':False,'scope':__doc__,'source_sha256':deployed_source_hash(),'server':public_state(url)}
    errors=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=[
            '--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required',
            '--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream','--use-file-for-fake-audio-capture='+str(fixture)])
        page=await browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script(INIT)
        try:
            response=await page.goto(url+'/desk-robot/');assert response.status==200
            await page.wait_for_function('window.robo&&envEvents.some(e=>e.type==="ready")',timeout=30000)
            await page.wait_for_function('envEvents.filter(e=>e.type==="input_status").at(-1)?.seconds>=15',timeout=30000)
            assert not await page.evaluate('envEvents.some(e=>e.type==="sound_caption")'), 'Quiet music should not be captioned'
            report['quiet_music_ignored']=True
            await page.wait_for_function('envEvents.some(e=>e.type==="environment_observation"&&e.kind==="music"&&e.status==="responding")',timeout=20000)
            await page.wait_for_function('envEvents.filter(e=>e.type==="turn_done").length>=2',timeout=30000)
            report['caption_visible']=await page.locator('#environment-caption').text_content()
            assert 'music' in report['caption_visible'].lower() or 'melod' in report['caption_visible'].lower()
            report['latency_card']=await page.locator('#latency-sound').inner_text();assert 'ms' in report['latency_card']
            await page.wait_for_function('envEvents.filter(e=>e.type==="input_status").at(-1)?.seconds>=66',timeout=50000)
            events=await page.evaluate('envEvents');pcm=await page.evaluate('envPCM')
            captions=[e for e in events if e['type']=='sound_caption']
            music=[e for e in events if e['type']=='environment_observation' and e.get('kind')=='music' and e.get('status')=='responding']
            assert len(music)==1,music
            assert any(e.get('suppression_reason')=='music_cooldown' for e in captions),captions
            replies=[e for e in events if e['type']=='reply' and e['at']>=music[0]['at']]
            assert replies and any(x['id']==replies[0]['reply_id'] for x in pcm)
            report['reply']=replies[-1]['text'];assert any(w in report['reply'].lower() for w in ('musik','melodie','klav','song','stück','lied'))
            assert not [e for e in events if e['type']=='endpoint'],'Music must not create a user utterance'
            assert not [e for e in events if e['type']=='error']
            report['music_cooldown_verified']=True;report['no_false_user_turn']=True
            report['captions']=[{k:e.get(k) for k in ('caption','kind','eligible','request_ms','suppression_reason')} for e in captions]
            report['passed']=not errors
        except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)[:1800]
        finally:
            report['events']=await page.evaluate('window.envEvents||[]');report['pcm']=await page.evaluate('window.envPCM||[]');report['page_errors']=errors
            await page.evaluate('window.robo?.stop()');await browser.close()
    path=ROOT/'static'/f'environment-music-{stamp}.json';path.write_text(json.dumps(report,ensure_ascii=False,separators=(',',':')))
    print(json.dumps({k:v for k,v in report.items() if k not in ('events','pcm','server')},ensure_ascii=False,indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com')
    asyncio.run(main(p.parse_args().url))
