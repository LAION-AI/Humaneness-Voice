"""Real microphone-transport checks for closing/blunt stop phrases and note leaks.

Reuses the adversarial browser/audio harness. Public synthetic inputs only;
real ASR, speaker gate, LLM, TTS and playback, no physical microphone claim.
"""
import argparse,asyncio,datetime,fcntl,hashlib,json,os,time
from pathlib import Path
from types import SimpleNamespace
import httpx
from playwright.async_api import async_playwright
from browser_adversarial_check import BrowserSession
from generate_adversarial_fixtures import decode,fixture_root
from run_voice_assistant_checks import deployed_source_hash,public_state

ROOT=Path(__file__).resolve().parent
CASES=[('thanks-enough','Danke, das reicht.'),('shut-up','Halt die Klappe!'),
       ('not-enough','Danke, das reicht nicht. Erzähl bitte weiter.')]

def prepare(folder):
    folder.mkdir(parents=True,exist_ok=True)
    with httpx.Client(base_url='http://127.0.0.1:8792',timeout=180) as c:
        for name,text in CASES:
            audio=folder/(name+'.mp3');meta=folder/(name+'.json')
            if audio.exists() and meta.exists():
                assert hashlib.sha256(audio.read_bytes()).hexdigest()==json.loads(meta.read_text())['sha256'];continue
            if audio.exists() or meta.exists():raise RuntimeError('Incomplete fixture; no silent replacement')
            state=c.get('/api/state').json()
            if state.get('busy'):raise RuntimeError('TTS occupied; no user request was interrupted')
            body={'model':'small_rl','input_mode':'manual','format':'structured',
                'message':'GENERAL: Clear natural conversational German. Preserve the reference speaker.\nSCRIPT:\n'+text,
                'generation':'single','n':1,'streaming':False,'seconds':12,'temperature':.7,'guidance':2.,
                'voice_mode':'profile','voice_profile':'anime_088','voice_conversion':False,'align':True,'seed':20261028}
            candidate=None
            with c.stream('POST','/api/turn',json=body) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.startswith('data: '):continue
                    e=json.loads(line[6:])
                    if e.get('event')=='error':raise RuntimeError('Fixture generation failed')
                    if e.get('event')=='result':candidate=e['candidates'][0]
            if candidate is None:raise RuntimeError('No fixture audio')
            r=c.get(candidate['url']);r.raise_for_status();audio.write_bytes(r.content)
            meta.write_text(json.dumps({'name':name,'text':text,'request':body,'checkpoint':state.get('checkpoint'),
                'sha256':hashlib.sha256(r.content).hexdigest(),'attempt':1,'public_synthetic_only':True},indent=2,ensure_ascii=False))
            print('Prepared '+name,flush=True)

async def run(args):
    rows={r['name']:r for r in json.loads((fixture_root()/'manifest.json').read_text())['cases']}
    clips={name:decode(fixture_root()/row['file']) for name,row in rows.items()}
    for name,text in CASES:
        p=args.fixtures/(name+'.mp3');row=json.loads(p.with_suffix('.json').read_text())
        assert row['sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
        rows[name]=row;clips[name]=decode(p)
    report={'passed':False,'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Synthetic microphone injection; real ASR/identity/LLM/TTS. No physical room or arbitrary-phase reliability claim.',
        'source_sha256':deployed_source_hash(),'server_before':public_state(args.url),'cases':[],
        'final_asr_model':'parakeet_v3','live_preview_and_interruptions':'nemotron',
        'fixtures':{n:rows[n] for n,_ in CASES},'injections':[]}
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=ROOT/'logs'/('completion-stop-'+stamp+'.json')
    def save():
        raw=json.dumps(report,indent=2,ensure_ascii=False);out.write_text(raw);(ROOT/'static/completion-stop-browser.json').write_text(raw)
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page();session=BrowserSession(page,clips,rows,report,SimpleNamespace(url=args.url,threshold=.28,pause_ms=700,seed=45,asr_model='parakeet_v3'))
        try:
            await session.open()
            setup=await session.speak({'id':'enroll','clip':'christoph-intro','max_submissions':2},{'speaker':'Christoph'},setup=True)
            report['setup']=setup;save()
            if not setup['passed']:raise RuntimeError('Enrollment failed; stop checks blocked')
            for name in ('thanks-enough','shut-up'):
                for phase in (0,160):
                    row=await session.control({'id':name+'-'+str(phase),'kind':'stop','clip':name,'phase_ms':phase,'snr_db':12})
                    report['cases'].append(row);save();print(json.dumps({'id':row['id'],'passed':row['passed'],'checks':row['checks'],'energy_end_to_cancel_ms':row.get('energy_end_to_cancel_ms')},ensure_ascii=False),flush=True)
            row=await session.control({'id':'negated-enough','kind':'negative_control','clip':'not-enough','seconds_after':1.2})
            previews=[e.get('text','') for e in row['events'] if e['type'] in ('asr_preview','partial')]
            row['checks']['negation_was_recognized']=any('nicht' in t.casefold() for t in previews)
            row['passed']=all(row['checks'].values());report['cases'].append(row);save()
            mark=await session.mark();await page.fill('#message','Ja, dreh dich doch mal!');await page.locator('#form').evaluate('f=>f.requestSubmit()')
            await session.wait(f'stressEvents.slice({mark["event"]}).some(e=>e.type==="turn_done")',45);await session.settle()
            events=await session.events(mark);texts=[e.get('text','') for e in events if e['type']=='reply']
            visible=await page.locator('#recent-robby').text_content()
            report['spin_after_interrupt']={'texts':texts,'visible':visible,'passed':bool(texts) and all('preceding reply' not in t and 'may have been heard' not in t for t in texts+[visible])}
            # Explicitly reproduce the reported old-server payload at the UI
            # boundary. This is marked as injected, not a real model answer.
            legacy='[My preceding reply was interrupted. The following text was generated; only a prefix may have been heard. Do not assume it was all heard or repeat it automatically.]'
            await page.evaluate('(text)=>robo.on.message({type:"reply",reply_id:"synthetic-note-check",text})',legacy+'\nRobby-Tornado, los geht\'s!')
            recent=await page.locator('#recent-robby').text_content();chat=await page.locator('#chat').text_content()
            report['injected_ui_guard']={'passed':recent=="Robby-Tornado, los geht's!" and 'preceding reply' not in chat,'synthetic_message_only':True}
            report['passed']=all(r['passed'] for r in report['cases']) and report['spin_after_interrupt']['passed'] and report['injected_ui_guard']['passed'] and not session.errors
        except Exception as e:report['error']=type(e).__name__+': '+str(e)[:400]
        finally:
            try:await page.evaluate('robo.stop()')
            except Exception:pass
            await browser.close();report['finished']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            report['source_unchanged']=report['source_sha256']==deployed_source_hash();report['passed'] &= report['source_unchanged'];save()
    print(json.dumps({'receipt':str(out),'passed':report['passed'],'error':report.get('error')}),flush=True)
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--url',default='http://127.0.0.1:8793')
    p.add_argument('--fixtures',type=Path,default=Path(os.environ.get('ROBBY_COMPLETION_FIXTURES','/mnt/nvme/robo-speaker-studies/stop-completion-v1')));a=p.parse_args()
    lock=(ROOT/'logs/voice-assistant-runs/.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.prepare:prepare(a.fixtures)
    asyncio.run(run(a))
