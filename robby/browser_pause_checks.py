"""HTTPS real-browser capture + continuously talking background + 800-ms pause.

The input is generated solely from public synthetic fixtures, not a person's
microphone. Both 2-s and 3.04-s identity contexts are exercised with actual
ASR/Luna/TTS. Captured test PCM stays in RAM for clock alignment, never a file.
"""
import argparse,asyncio,json,struct,tempfile,time,re
import aiohttp
from pathlib import Path
from urllib.parse import urljoin
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from playwright.async_api import async_playwright
from speaker_live.live_test import audio
from speaker_turns import word_distance,words

ROOT=Path(__file__).resolve().parent

def clock_alignment(captured,template):
    segment=captured[:min(len(captured),12*16000)]
    values=fftconvolve(segment,template[::-1],mode='valid')
    at=int(np.argmax(values));a=segment[at:at+len(template)]
    correlation=float(a@template/max(np.linalg.norm(a)*np.linalg.norm(template),1e-10))
    return at/16000,correlation

async def main(args):
    async with aiohttp.ClientSession() as client:
        for _ in range(180):
            try:
                async with client.get(urljoin(args.url,'/api/state'),timeout=aiohttp.ClientTimeout(total=3)) as response:
                    if response.status==200 and (await response.json()).get('ready'):break
            except (aiohttp.ClientError,asyncio.TimeoutError):pass
            await asyncio.sleep(.5)
        else:raise TimeoutError('Robby not ready; no browser/audio case was started')
    receipt={'public_synthetic_only':True,'physical_microphone_used':False,'passed':False,'cases':[]}
    if args.label and not re.fullmatch(r'[a-z0-9-]+',args.label):raise ValueError('Use an alphanumeric receipt label')
    target=ROOT/'logs'/('browser-pause-'+args.label+'.json' if args.label else 'browser-pause-checks.json');target.write_text(json.dumps(receipt))
    intro=audio('christoph-intro');follow=audio('christoph-followup');bg=audio('background')
    # Locate a low-energy boundary inside the sentence, not an arbitrary cut
    # through a loud phoneme. Insert a hesitation strictly shorter than 1 s.
    hop=160;energy=np.mean(follow[:len(follow)//hop*hop].reshape(-1,hop)**2,axis=1)
    lo,hi=round(.9*100),min(round(1.9*100),len(energy)-1)
    cut=(lo+int(np.argmin(energy[lo:hi])))*hop
    question=np.r_[follow[:cut],np.zeros(12800,np.float32),follow[cut:]]
    length=round(48*16000);wave=np.resize(bg,length)*args.background_gain
    # This latency case targets an ALREADY recognized speaker. Enroll first
    # in clean audio, then keep background speech running throughout the
    # later question, its hesitation and the post-answer negative interval.
    # Noisy enrollment remains a separate explicit condition/multi-user QA.
    if not args.noisy_enrollment:wave[:8*16000]=0
    wave[32000:32000+len(intro)]+=intro;start=22*16000
    wave[start:start+len(question)]+=question
    with tempfile.TemporaryDirectory(prefix='robby-public-pause-') as directory:
        path=Path(directory)/'conversation.wav';sf.write(path,wave,16000,subtype='PCM_16')
        async with async_playwright() as p:
            for mode in args.modes:
                browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,
                    args=['--no-sandbox','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream',
                        '--use-file-for-fake-audio-capture='+str(path),'--autoplay-policy=no-user-gesture-required'])
                page=await browser.new_page();events=[];pcm=[];errors=[];audio_first={};capture_wall=None
                if args.diagnostics:
                    if not args.url.startswith('http://127.0.0.1:8793'):raise ValueError('Synthetic diagnostics require loopback URL')
                    await page.add_init_script('''const WS=window.WebSocket;window.WebSocket=class extends WS{constructor(url,...rest){const value=new URL(url);value.searchParams.set('synthetic_diagnostics','1');super(value.toString(),...rest);}};''')
                def sent(payload):
                    nonlocal capture_wall
                    if isinstance(payload,bytes) and len(payload)>8:
                        if capture_wall is None:capture_wall=time.monotonic()
                        pcm.append(np.frombuffer(payload[8:],'<i2').astype(np.float32)/32768)
                def received(payload):
                    now=time.monotonic()
                    if isinstance(payload,str):
                        item=json.loads(payload)
                        if item.get('type') not in ('state','input_status'):events.append({'wall':now,**item})
                    elif isinstance(payload,bytes) and len(payload)>=4:
                        rid=struct.unpack_from('<i',payload)[0]
                        if rid:audio_first.setdefault(rid,now)
                def socket(ws):ws.on('framesent',sent);ws.on('framereceived',received)
                page.on('websocket',socket);page.on('pageerror',lambda e:errors.append(str(e)))
                try:
                    await page.goto(args.url)
                    await page.wait_for_function('window.robo&&!document.querySelector("#call").disabled',timeout=90000)
                    await page.evaluate('window.robo.start(false)')
                    await page.evaluate('(mode)=>window.robo.send("identity:mode:"+mode)',mode)
                    for _ in range(100):
                        if any(e.get('type')=='identity' and e.get('mode')==mode for e in events):break
                        await page.wait_for_timeout(50)
                    else:raise AssertionError('Requested speaker context received no acknowledgment')
                    await page.click('#call')
                    deadline=time.monotonic()+46
                    while time.monotonic()<deadline:
                        endpoints=[e for e in events if e.get('type')=='endpoint']
                        replies=[e for e in events if e.get('type')=='reply']
                        if len(endpoints)>=2 and len(audio_first)>=2 and time.monotonic()-endpoints[1]['wall']>=8:break
                        await page.wait_for_timeout(100)
                    captured=np.concatenate(pcm) if pcm else np.zeros(0,np.float32)
                    offset,corr=clock_alignment(captured,intro)
                    question_end=offset+(22-2)+len(question)/16000
                    endpoints=[e for e in events if e.get('type')=='endpoint'];replies=[e for e in events if e.get('type')=='reply']
                    result={'mode':mode,'background_gain':args.background_gain,'noisy_enrollment':args.noisy_enrollment,'internal_pause_ms':800,
                        'alignment_correlation':corr,'aligned_question_end_capture_s':question_end,
                        'capture_seconds':len(captured)/16000,'page_errors':errors,
                        'endpoints':[{k:v for k,v in e.items() if k!='wall'} for e in endpoints],
                        'replies':[e.get('text') for e in replies],'actual_audio_reply_ids':list(audio_first),'passed':False}
                    result['diagnostic_events']=[{k:v for k,v in e.items() if k!='wall'} for e in events
                        if e.get('type') in ('warning','error','ignored','asr_preview','synthetic_activity','synthetic_asr_views')]
                    result['identity_checks']=[{k:v for k,v in e.items() if k in ('decision','verification','continuity')}
                        for e in events if e.get('type')=='identity' and any(k in e for k in ('decision','verification','continuity'))]
                    receipt['cases'].append(result)
                    assert corr>.35,'Synthetic PCM clock could not be aligned; no latency claim is valid'
                    assert len(endpoints)==2 and len(replies)>=2 and len(audio_first)>=2,'Missing/duplicate/background turn or no actual TTS audio'
                    assert all(e.get('speaker_name')=='Christoph' for e in endpoints),'Wrong speaker'
                    assert endpoints[0]['speaker_id']==endpoints[1]['speaker_id'],'Returning identity changed'
                    assert 'Geschichte' in endpoints[1]['transcript'],'Question lost across the 800-ms hesitation'
                    expected=words('Kannst du mir eine kleine Geschichte erzählen?')
                    wer=word_distance(expected,words(endpoints[1]['transcript']))/len(expected)
                    result['question_wer']=round(wer,4)
                    assert wer<=.25,'Question prefix/words lost across speaker recovery or internal hesitation'
                    assert not any(p in e['transcript'].casefold() for e in endpoints for p in ('nachrichten','morgen wird','bericht','wetter')),'Background text submitted'
                    latency=endpoints[1]['wall']-capture_wall-question_end
                    result['pause_to_accepted_turn_s']=round(latency,3)
                    rid=replies[1].get('reply_id');result['pause_to_first_audio_s']=round(audio_first[rid]-capture_wall-question_end,3)
                    result['endpoint_metrics']=endpoints[1].get('timing')
                    assert .65<=latency<=args.max_endpoint_delay,f'Premature/late endpoint after target silence: {latency:.3f}s'
                    assert not errors,errors
                    assert not any(e.get('type')=='error' for e in events),'Runtime voice error'
                    result['passed']=True;print(json.dumps(result),flush=True)
                    receipt['passed']=len(receipt['cases'])==len(args.modes) and all(c['passed'] for c in receipt['cases'])
                    target.write_text(json.dumps(receipt,indent=2))
                finally:
                    target.write_text(json.dumps(receipt,indent=2))
                    await page.evaluate('window.robo?.stop()');await browser.close();await asyncio.sleep(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com/desk-robot/')
    p.add_argument('--background-gain',type=float,default=.16);p.add_argument('--max-endpoint-delay',type=float,default=2.)
    p.add_argument('--modes',nargs='+',default=['v4-long','v4-long3'],choices=['v4-long','v4-long3'])
    p.add_argument('--label',default=None,help='Separate receipt for an explicit context comparison, including failed cases')
    p.add_argument('--noisy-enrollment',action='store_true',help='Also mix background into the initial introduction; separate from the already-recognized-speaker latency condition')
    p.add_argument('--diagnostics',action='store_true',help='Loopback-only synthetic activity diagnostics; never physical microphone input')
    asyncio.run(main(p.parse_args()))
