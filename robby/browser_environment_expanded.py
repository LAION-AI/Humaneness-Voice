"""Real HTTPS / ASR / ECAPA / local+API LLM / TTS diagnostics.

Digital PCM injection uses the actual browser playback clock. Not a physical
microphone/acoustic echo certification. Every run is archived, including failures.
Public ESC-50 inputs stay local; the receipt includes provenance, never raw audio.
"""
import argparse,asyncio,datetime,json,re
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from playwright.async_api import async_playwright
from browser_adversarial_check import BrowserSession
from generate_adversarial_fixtures import decode,fixture_root
from check_environment_diversity import load_fixtures
from environment_fixtures import music_fixture
from run_voice_assistant_checks import deployed_source_hash,public_state
ROOT=Path(__file__).resolve().parent

async def main(url,group):
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 report={'passed':False,'scope':__doc__,'started':stamp,'group':group,'source_sha256':deployed_source_hash(),'injections':[],'cases':[]}
 args=SimpleNamespace(url=url,threshold=.28,pause_ms=700,asr_model='parakeet_v3',seed=42,focus=False)
 manifest=json.loads((fixture_root()/'manifest.json').read_text());meta={r['name']:r for r in manifest['cases']}
 clips={name:decode(fixture_root()/meta[name]['file']) for name in ('background','selma-followup-clear','christoph-intro','selma-intro')}
 sounds,provenance=load_fixtures();report['sound_provenance']=provenance
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
  async def open_session():
   page=await browser.new_page();session=BrowserSession(page,clips,meta,report,args)
   await session.open()
   assert not await page.locator('#quick-replies').is_checked()
   assert not await page.locator('#conversation-focus').is_checked()
   assert await page.locator('#sound-prominence').input_value()=='8'
   assert await page.locator('#preserve-sounds').is_checked()
   return page,session
  async def receipt(page,s,mark,name):
   await s.idle_audio(3.)
   try:
    await s.wait(f'stressEvents.slice({mark["event"]}).some(e=>e.type==="turn_done")',35)
    await s.settle(60)
   except TimeoutError:pass
   events=await s.events(mark);replies=[e for e in events if e['type']=='reply'];pcm=await page.evaluate('(n)=>stressPCM.slice(n)',mark['pcm'])
   row={'case':name,'events':events,'reply':replies[-1]['text'] if replies else '',
    'captions':[e for e in events if e['type']=='sound_caption'],'pcm_chunks':len(pcm),'page_errors':s.errors,
    'passed':bool(replies and pcm) and not any(e['type']=='error' for e in events) and not s.errors}
   report['cases'].append(row);return row
  try:
   if group in ('sounds','all'):
    for name,wave in [('music_onset',music_fixture(5.)),('music_already_playing',music_fixture(16.)),
       ('dog',sounds['dog-1']),('car_horn',sounds['car_horn-1']),('glass_breaking',sounds['glass_breaking-1']),('cat',sounds['cat-1'])]:
     page,s=await open_session();mark=await s.mark()
     if name!='music_already_playing':await s.idle_audio(2.)
     await s.feed(wave,record=True)
     if name=='music_onset':
      await s.wait('robo.sources.length>0',30)
      # Reproduce the old browser shortcut: VAD-positive noise + high RMS
      # must not send avatar:cancel or discard a sound reply before playback.
      await page.evaluate('''()=>{window.noiseCancelCommands=[];const send=robo.send.bind(robo);robo.send=t=>{if(t==='avatar:cancel')noiseCancelCommands.push(t);send(t)};
        robo.on.message({type:'speech_activity',foreground_speech:true});robo.on.input({rms:.2,state:'running',sent:0});}''')
      await s.idle_audio(.4)
      assert not await page.evaluate('noiseCancelCommands'), 'Raw noise cancelled the sound reply'
     row=await receipt(page,s,mark,name)
     row['passed'] &= any(e.get('eligible') for e in row['captions'])
     if 'music' in name:row['passed'] &= any(e.get('kind')=='music' for e in row['captions'])
     print(json.dumps({k:row[k] for k in ('case','passed','reply','pcm_chunks')},ensure_ascii=False),flush=True)
     await page.evaluate('robo.stop()');await page.close();await asyncio.sleep(.5)
   if group in ('questions','all'):
    for enabled in (False,True):
     page,s=await open_session();await s.idle_audio(2.)
     await page.locator('#model-settings').evaluate('e=>e.open=true');await page.locator('#quick-replies').set_checked(enabled)
     mark=await s.mark()
     await page.fill('#message','Was ist das hier?');await page.locator('#form').evaluate('f=>f.requestSubmit()')
     await s.idle_audio(.2);await s.feed(music_fixture(3.),record=True)
     row=await receipt(page,s,mark,'after_question_music_bridge_'+str(enabled))
     row['passed'] &= any(e.get('requested') and e.get('kind')=='music' for e in row['captions'])
     roles=[e.get('sentence_role') for e in row['events'] if e['type']=='reply']
     if not enabled:row['passed'] &= 'reaction' not in roles
     row['passed'] &= bool(re.search('musik|melodie|stück|lied',row['reply'],re.I))
     print(json.dumps({k:row[k] for k in ('case','passed','reply','pcm_chunks')},ensure_ascii=False),flush=True)
     await page.evaluate('robo.stop()');await page.close();await asyncio.sleep(.5)
   if group in ('questions','all'):
    # Spoken question followed immediately by music: real ASR, no typed shortcut.
    question=ROOT/'logs/environment-question.mp3'
    if not question.exists():raise FileNotFoundError('Run generate_environment_question.py first')
    page,s=await open_session();await s.idle_audio(2.);mark=await s.mark()
    await s.feed(decode(question),record=True);await s.idle_audio(.35);await s.feed(music_fixture(5.),record=True)
    row=await receipt(page,s,mark,'spoken_question_then_music')
    ends=[e for e in row['events'] if e['type']=='endpoint'];row['transcripts']=[e.get('transcript') for e in ends]
    row['passed'] &= bool(ends) and bool(re.search('musik|melodie|stück|lied|beat',row['reply'],re.I))
    print(json.dumps({k:row[k] for k in ('case','passed','reply','transcripts')},ensure_ascii=False),flush=True)
    await page.evaluate('robo.stop()');await page.close();await asyncio.sleep(.5)
   if group in ('speakers','all'):
    page,s=await open_session();await s.idle_audio(2.)
    for name in ('christoph-intro','selma-followup-clear','selma-intro','selma-followup-clear'):
     mark=await s.mark();await s.feed(clips[name],record=True);row=await receipt(page,s,mark,name)
     ends=[e for e in row['events'] if e['type']=='endpoint'];row['speakers']=[e.get('speaker_name') for e in ends]
     row['passed'] &= bool(ends)
     print(json.dumps({k:row[k] for k in ('case','passed','reply','speakers')},ensure_ascii=False),flush=True)
     if name=='christoph-intro':
      quiet_mark=await s.mark();await s.feed(clips['background']*.1,record=True);await s.idle_audio(4.)
      quiet_events=await s.events(quiet_mark)
      report['quiet_background']={'events':quiet_events,'passed':any(e['type']=='ignored' and e.get('source')=='background_salience' for e in quiet_events) and not any(e['type']=='endpoint' for e in quiet_events)}
      assert report['quiet_background']['passed'], 'Quiet undirected speech was admitted'
    state=await page.evaluate('stressEvents.filter(e=>e.type==="identity").at(-1)');report['final_identity']=state
    names=[u.get('name') for u in state.get('speakers',[])]
    report['named_speakers_present']='Christoph' in names and 'Selma' in names
    assert report['named_speakers_present'],names
    assert re.search(r'(heißt|heisst|nenn|name)',report['cases'][-3]['reply'],re.I),report['cases'][-3]['reply']
    await page.evaluate('robo.stop()');await page.close()
   report['passed']=bool(report['cases']) and all(r['passed'] for r in report['cases'])
  except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)[:1500]
  finally:
   await browser.close();report['source_unchanged']=report['source_sha256']==deployed_source_hash();report['passed'] &= report['source_unchanged']
 path=ROOT/'static'/f'environment-expanded-{group}-{stamp}.json';path.write_text(json.dumps(report,ensure_ascii=False,separators=(',',':')))
 print(json.dumps({'report':str(path),'passed':report['passed'],'error':report.get('error')}),flush=True)
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com');p.add_argument('--group',choices=['all','sounds','questions','speakers'],default='all');a=p.parse_args()
 asyncio.run(main(a.url,a.group))
