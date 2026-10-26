"""Full CPU-browser smoke test with authored synthetic microphone files.

Tests a wake-free sentence, a name, another voice, an explicit correction, and
the original voice returning. Reports actual browser playback scheduling.
"""
import argparse,asyncio,json,tempfile,time
from pathlib import Path
import numpy as np,soundfile as sf
from scipy.signal import resample_poly
from playwright.async_api import async_playwright

ROOT=Path(__file__).resolve().parent
FIXTURES=Path('/mnt/nvme/robo-speaker-studies/speaker-live-fixtures')

def fixture(name):
    wave,sr=sf.read(FIXTURES/(name+'.mp3'),dtype='float32',always_2d=True);wave=wave.mean(1)
    return resample_poly(wave,16000//np.gcd(sr,16000),sr//np.gcd(sr,16000))

async def main(args):
    report={'passed':False,'url':args.url,'physical_microphone':False,'synthetic_mic_file':True,
        'cpu_browser':True,'began':time.time()}
    clips=[(6.,'wake_free',np.load(ROOT/'logs/bolt-benchmark/input16.npy')),
           (22.,'name',fixture('christoph-intro')),
           (39.,'new_voice',fixture('robert-followup')),
           (56.,'voice_correction',np.load(ROOT/'logs/bolt-benchmark/voice-correction16.npy')),
           (75.,'returning_voice',fixture('robert-followup'))]
    # Fake-device capture may start reading the file before getUserMedia and
    # AudioWorklet initialization complete. Leave explicit setup pre-roll.
    wave=np.zeros(110*16000,np.float32)
    for at,name,clip in clips:
        start=round(at*16000);wave[start:start+len(clip)]=clip
    with tempfile.TemporaryDirectory(prefix='robby-bolt-browser-') as temp:
        wav=Path(temp)/'authored-synthetic-conversation.wav';sf.write(wav,resample_poly(wave,3,1),48000,subtype='PCM_16')
        async with async_playwright() as p:
            browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=[
                '--no-sandbox','--disable-gpu','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream',
                '--use-file-for-fake-audio-capture='+str(wav),'--autoplay-policy=no-user-gesture-required'])
            page=await browser.new_page(viewport={'width':1280,'height':1000});errors=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            await page.add_init_script("localStorage.setItem('robby.llmSelection',JSON.stringify({model:'local',provider:'local'}));localStorage.setItem('robby.cameraAuto','false');localStorage.setItem('robby.avatar-idle-enabled','false');localStorage.setItem('robby.avatar-pet-invites-enabled','false')")
            try:
                await page.goto(args.url+'/desk-robot/')
                await page.wait_for_function('window.robo && !document.querySelector("#call").disabled',timeout=90000)
                await page.evaluate('''()=>{
                 window.boltEvents=[];window.boltAudio=[];window.boltPlayback=[];window.boltInput=[];window.boltASR=[];window.boltMic=[];
                 const mic=robo.mic.bind(robo);robo.mic=(t,x)=>{if(boltMic.length<800)boltMic.push(Array.from(x));mic(t,x)};
                 const old=robo.on.message,play=robo.play.bind(robo),input=robo.on.input;
                 robo.on.message=m=>{if(m.type==='asr_preview')boltASR.push({...m,at:performance.now()});if(!['state','input_status','asr_preview'].includes(m.type))boltEvents.push({...m,at:performance.now()});old(m)};
                 robo.on.input=s=>{if(s.rms>.003)boltInput.push({at:performance.now(),rms:s.rms});input(s)};
                 robo.play=raw=>{const rid=new DataView(raw).getInt32(0,true),before=robo.next,gap=before?Math.max(0,robo.ctx.currentTime-before)*1000:0;play(raw);boltAudio.push({rid,at:performance.now(),samples:(raw.byteLength-4)/2,gap_ms:gap,start_buffer_ms:robo.metrics.startup_buffer_ms})};
                 const seen=new Set();setInterval(()=>{const p=robo.ctx?.state==='running'?robo.clock.position():null;if(p&&!seen.has(p.id)){seen.add(p.id);boltPlayback.push({...p,at:performance.now()})}},10);
                }''')
                await page.click('#call')
                await page.wait_for_function('boltEvents.some(m=>m.type==="endpoint")',timeout=25000)
                first=await page.evaluate('boltEvents.find(m=>m.type==="endpoint")')
                assert 'was können wir' in first['transcript'].casefold() and 'gemeinsam machen' in first['transcript'].casefold(),first
                await page.wait_for_function('boltAudio.length>0&&boltPlayback.length>0',timeout=20000)
                timing=await page.evaluate('''()=>{const a=boltAudio[0],p=boltPlayback[0],i=boltInput.filter(v=>v.at<a.at).at(-1);return {pcm_after_last_capture_ms:a.at-i.at,playback_after_last_capture_ms:p.at-i.at,playback_reserve_ms:p.at-a.at,configured_reserve_ms:a.start_buffer_ms}}''')
                report['first_turn']=timing
                if not args.first_only:
                    await page.wait_for_function('boltEvents.some(m=>m.type==="endpoint"&&m.speaker_name==="Christoph")',timeout=35000)
                    await page.wait_for_function('boltEvents.some(m=>m.type==="endpoint"&&m.at>boltEvents.find(m=>m.type==="endpoint").at+29000&&m.speaker_name!=="Christoph")',timeout=30000)
                    await page.wait_for_function('boltEvents.some(m=>m.type==="identity"&&m.speakers?.some(s=>s.name==="Christoph"&&s.voice_variants>=2)&&m.at>boltEvents.find(m=>m.type==="endpoint").at+46000)',timeout=40000)
                    await page.wait_for_function('boltEvents.some(m=>m.type==="endpoint"&&m.speaker_name==="Christoph"&&m.at>boltEvents.find(m=>m.type==="endpoint").at+66000)',timeout=35000)
                    await page.wait_for_function('boltEvents.filter(m=>m.type==="turn_done").length>=5',timeout=25000)
                report['events']=await page.evaluate('boltEvents')
                report['audio']=await page.evaluate('boltAudio')
                report['playback']=await page.evaluate('boltPlayback')
                report['client_metrics']=await page.evaluate('robo.metrics')
                report['errors']=errors;assert not errors,errors
                assert not [m for m in report['events'] if m['type']=='error'],'Server error'
                report['passed']=True
            except Exception as exc:
                report.update(error=str(exc),errors=errors)
                await page.wait_for_timeout(2000)
                try:report['events']=await page.evaluate('window.boltEvents||[]')
                except Exception:pass
                try:report['asr_previews']=await page.evaluate('window.boltASR||[]')
                except Exception:pass
            finally:
                try:
                    capture=await page.evaluate('window.boltMic||[]')
                    if capture:np.save(ROOT/'logs/bolt-benchmark/browser-capture16.npy',np.asarray(capture,np.float32).reshape(-1))
                except Exception:pass
                await page.evaluate('window.robo?.stop()');await browser.close()
    report['finished']=time.time();report['first_only']=args.first_only
    (ROOT/'static'/('bolt-browser-first-turn.json' if args.first_only else 'bolt-browser-check.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ('events','audio','playback')},indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='https://arrive-locations-parade-research.trycloudflare.com');p.add_argument('--first-only',action='store_true');asyncio.run(main(p.parse_args()))
