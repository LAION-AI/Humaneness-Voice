"""Reproducible browser clock checks; optional actual API/voice and camera QA.

uv run --no-project --with playwright --with numpy==1.26.4 --python /path/to/python browser_checks.py [--live] [--camera]
The live fixture is synthetic text; microphone input is generated silence.
"""
import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from playwright.async_api import async_playwright


async def run(args):
    logs=Path(__file__).parent/'logs';logs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='robby-browser-') as temporary:
        silence=Path(temporary)/'synthetic-silence.wav'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','anullsrc=r=48000:cl=mono',
            '-t','30',str(silence)],check=True)
        async with async_playwright() as playwright:
            browser=await playwright.chromium.launch(executable_path=args.chrome,headless=True,
                args=['--no-sandbox','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream',
                    '--use-file-for-fake-audio-capture='+str(silence),'--autoplay-policy=no-user-gesture-required'])
            try:
                page=await browser.new_page(viewport={'width':1440,'height':1000})
                errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                await page.goto(args.url);await page.wait_for_function('window.robo')
                clock=await page.evaluate('''async()=>{
                  const {PlaybackClock}=await import('/static/clock.js');let before=[],after=[];
                  for(const output of [.005,.02,.055,.15,.25])for(const input of [.005,.04,.12])for(const network of [0,.1,.5,.9]){
                    const context={currentTime:10.1,getOutputTimestamp:()=>({contextTime:10.1-output,performanceTime:10000})};
                    const c=new PlaybackClock(context);c.inputLatency=input;c.add(1,8,240000,0);
                    const actual=Math.floor((10-output-input-8)*24000),tag=c.tag(10,10000)[1];
                    after.push(Math.abs(tag-actual)/24);before.push(Math.abs(output+input+network-.15)*1000);
                  }
                  const ctx={currentTime:9,baseLatency:0,outputLatency:0};const c=new PlaybackClock(ctx);c.inputLatency=0;
                  c.add(1,9.005,240,0);const parts=c.pieces(9,320,16000,10000);
                  if(JSON.stringify(parts.map(p=>[p.first,p.end,p.id]))!==JSON.stringify([[0,80,0],[80,240,1],[240,320,0]]))throw Error('Playback-edge split failed');
                  const d=new PlaybackClock({currentTime:9,baseLatency:.1,outputLatency:0});d.inputLatency=.02;d.add(2,8,48000,0);d.stop(9);
                  const kept=d.tag(9.05,10000)[0]===2,cut=d.tag(9.5,10000)[1]===-1;if(!kept||!cut)throw Error('Cancel history failed');
                  return {before,after,playback_edge_split:true,cancel_keeps_history:kept,cancel_removes_future:cut};
                }''')
                result={'protocol':'clock-v3 explicit latency grid','scenarios':len(clock['before']),
                    'before_p50_ms':float(np.median(clock['before'])),'before_p95_ms':float(np.percentile(clock['before'],95)),
                    'after_p50_ms':float(np.median(clock['after'])),'after_max_ms':max(clock['after']),
                    **{k:v for k,v in clock.items() if k not in ('before','after')},
                    'hardware_accuracy':'Ideal-clock simulation, not physical audio-device measurement'}
                await page.screenshot(path=str(logs/'robo-desktop.png'))
                await page.set_viewport_size({'width':390,'height':844});await page.screenshot(path=str(logs/'robo-mobile.png'))
                result['mobile_no_overflow']=await page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                if args.camera:
                    await page.click('.settings summary');await page.click('#camera')
                    await page.wait_for_function('''()=>{let t=document.querySelector('#camera-status').textContent;return t.startsWith('Local CPU worker')||t.startsWith('No face visible')||t.includes('unavailable')}''',timeout=60000)
                    await page.wait_for_timeout(1000)
                    result['camera_status']=await page.locator('#camera-status').text_content()
                    result['camera_video_running']=await page.evaluate('document.querySelector("#video").readyState>=2')
                    result['camera_capture_fps']=await page.evaluate('document.querySelector("#video").srcObject?.getVideoTracks()[0]?.getSettings().frameRate')
                    assert result['camera_video_running'] and 'unavailable' not in result['camera_status'].lower(),result
                    assert result['camera_capture_fps']<=10.1,result
                    await page.click('#camera')
                if args.live:
                    await page.click('#call');await page.wait_for_function('window.robo.ws?.readyState===1')
                    await page.fill('#message','Say hello in just four words.');await page.click('button[type=submit]')
                    await page.wait_for_function('window.robo.metrics.tagged_blocks>0',timeout=90000)
                    await page.wait_for_function('document.querySelector("#phase-text").textContent==="Listening to you"',timeout=30000)
                    result['live']=await page.evaluate('({...window.robo.metrics,clock:window.robo.clock.report()})')
                    await page.evaluate('window.robo.stop()')
                result['page_errors']=errors
                assert not errors and result['mobile_no_overflow'] and result['after_max_ms']<.05,result
                (logs/'browser-checks.json').write_text(json.dumps(result,indent=2))
                print(json.dumps(result,indent=2))
            finally:await browser.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8793')
    parser.add_argument('--chrome',default='/usr/bin/google-chrome');parser.add_argument('--live',action='store_true')
    parser.add_argument('--camera',action='store_true');asyncio.run(run(parser.parse_args()))
