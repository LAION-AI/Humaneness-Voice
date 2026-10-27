"""Digital browser microphone, real ASR/ECAPA/Ears/LLM/TTS, synthetic voices.

Enroll one foreground speaker; quiet TV must remain ignored. Two clear but
uninvited different-voice segments may elicit one cautious question, without
enrolling them or changing the focused partner. No physical microphone test.
"""
import asyncio,json,datetime,argparse
from pathlib import Path
from types import SimpleNamespace
from playwright.async_api import async_playwright
from browser_adversarial_check import BrowserSession
from generate_adversarial_fixtures import decode,fixture_root
from run_voice_assistant_checks import deployed_source_hash

ROOT=Path(__file__).resolve().parent
async def main(url):
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report={'passed':False,'scope':__doc__,'source_sha256':deployed_source_hash(),'injections':[],'cases':[]}
    args=SimpleNamespace(url=url,threshold=.28,pause_ms=700,asr_model='parakeet_v3',seed=42)
    manifest=json.loads((fixture_root()/'manifest.json').read_text());rows={r['name']:r for r in manifest['cases']}
    names=['christoph-intro','background','selma-followup-clear','tv-spoof']
    clips={name:decode(fixture_root()/rows[name]['file']) for name in names}
    async with async_playwright() as p:
        b=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
        page=await b.new_page();s=BrowserSession(page,clips,rows,report,args)
        try:
            await s.open();await page.evaluate('robo.send(\'environment:{"enabled":true,"voices":true,"impressions":true}\')')
            setup=await s.speak({'id':'setup','clip':'christoph-intro','expect':{'min_words':1}}, {'speaker':'Christoph'},setup=True)
            report['setup']=setup;assert setup['passed'],setup['checks']
            for name,gain,repeats in [('tv-spoof',.12,2),('selma-followup-clear',1.,2)]:
                mark=await s.mark()
                for _ in range(repeats):await s.feed(clips[name]*gain,record=True);await s.idle_audio(3.)
                await s.idle_audio(1.);await s.settle()
                events=await s.events(mark);focus=[e for e in events if e['type']=='speaker_focus']
                observations=[e for e in events if e['type']=='environment_observation']
                endpoints=[e for e in events if e['type']=='endpoint'];replies=[e for e in events if e['type']=='reply']
                row={'fixture':name,'gain':gain,'events':events,'speaker_checks':focus,'observations':observations,
                    'reply':replies[-1]['text'] if replies else '', 'no_unauthorized_enrollment':not endpoints}
                if name=='tv-spoof':row['passed']=not replies and not endpoints and not observations
                else:row['passed']=not endpoints and any(e.get('kind')=='unfamiliar_voice' and e.get('status')=='responding' for e in observations) and bool(replies)
                report['cases'].append(row)
            # Ask the controller for its unchanged participant list.
            await page.evaluate('robo.send("identity:focus:enabled:on")');await s.idle_audio(.5)
            report['final_identity']=await page.evaluate('stressEvents.filter(e=>e.type==="identity").at(-1)')
            report['passed']=all(c['passed'] for c in report['cases']) and len(report['final_identity']['speakers'])==1 and not s.errors
        except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)[:1200]
        finally:
            report['events']=await page.evaluate('window.stressEvents||[]');report['page_errors']=s.errors
            await page.evaluate('window.robo?.stop()');await b.close()
    path=ROOT/'static'/f'environment-voices-{stamp}.json';path.write_text(json.dumps(report,ensure_ascii=False,separators=(',',':')))
    print(json.dumps({'report':str(path),'passed':report['passed'],'error':report.get('error'),'cases':[{k:v for k,v in c.items() if k not in ('events','speaker_checks')} for c in report['cases']]},ensure_ascii=False,indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com')
    asyncio.run(main(parser.parse_args().url))
