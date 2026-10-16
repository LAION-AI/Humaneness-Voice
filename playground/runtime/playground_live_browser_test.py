"""Real HTTPS optional single-take Luna/TTS/live playback smoke; not a quality study."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',default='https://switch-surprised-clock-sparc.trycloudflare.com')
    args=parser.parse_args()
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,
            args=['--no-sandbox','--disable-dev-shm-usage'])
        context=browser.new_context(viewport={'width':1440,'height':1050})
        page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(args.base).status==200
        page.wait_for_function('ready && !running && !!smallLoras',timeout=240000)
        body=page.evaluate('buildTurnRequest("default check")')
        assert body['generation']=='best' and body['n']==3 and body['streaming']
        assert body['chunk_frames']==20 and body['guidance']==1.5 and body['temperature']==1.
        assert body['format']=='caption' and body['voice_profile']=='emolia_c0542' and body['align']
        assert body['adapters']=={'id':None,'weight':1,'ref15':False,'ref15_weight':1}
        assert body['speaker']=={'method':'off','weight':0}
        assert body['ranking']=={'mode':'ears_mix','model':'medium','metric':'ce','weight':.5}
        page.locator('#generation').select_option('single')
        body=page.evaluate('buildTurnRequest("single-live check")')
        assert body['generation']=='single' and body['n']==1
        page.evaluate('''() => {
            window.streamEvents=[];window.actualResult=null;window.directionResult=null;
            const oldChunk=LivePlayer.prototype.chunk;
            LivePlayer.prototype.chunk=async function(event){
                await oldChunk.call(this,event);
                window.streamEvents.push({...event,receivedAt:performance.now(),beforeResult:!window.actualResult});
            };
            const oldTakes=takes;
            takes=function(box,result){window.resultAt=performance.now();window.actualResult=result;return oldTakes(box,result);};
        }''')
        page.locator('#q').fill('Spiel eine Szene mit zurückgehaltener Wut: Jemand hat dein Vertrauen gebrochen. '
            'Antworte direkt dieser Person in zwei kurzen Sätzen. Am Ende lässt die Wut nach, aber die Enttäuschung bleibt.')
        page.locator('#send').click()
        page.wait_for_function('actualResult!==null || !!document.querySelector(".error")',timeout=240000)
        assert not page.locator('.error').count(),page.locator('.error').all_text_contents()
        page.wait_for_function('!running',timeout=10000)
        result=page.evaluate('actualResult')
        chunks=page.evaluate('streamEvents')
        assert chunks and chunks[0]['beforeResult']
        assert page.evaluate('livePlayback.played && livePlayback.buffers.length>0')
        assert page.evaluate('livePlayback.buffers.every(b=>b.numberOfChannels===1)')
        assert result['generation']=='single' and len(result['candidates'])==1
        assert result['adapters']['selection']==body['adapters']
        assert result['reference']['mode']=='profile' and result['reference']['dynamic']
        assert 37 < result['reference']['tts_context']['codec_frames'] <=188
        assert result['live_stream']['enabled'] and result['live_stream']['first_playable_seconds']>0
        assert result['performance']['streaming'] and result['performance']['chunk_frames']==20
        assert result['performance']['batches'][0]['live_callback_seconds']>0
        # Capture actual direction from the rendered model-prompt disclosure.
        content=page.locator('#chat').text_content()
        identity="Use the same speaker as the reference audio. Preserve that speaker's voice identity."
        prompts=page.locator('#chat pre').all_text_contents()
        assert any(text.startswith('CAPTION: '+identity) for text in prompts)
        assert any(text.startswith('CFG-Vergleich:\nCAPTION: '+identity) for text in prompts)
        print(json.dumps({'live_chunks':len(chunks),'first_before_result':chunks[0]['beforeResult'],
            'wer':result['candidates'][0]['wer'],'on_device_rtf':result['performance']['aggregate_rtf'],
            'first_playable_seconds':result['live_stream']['first_playable_seconds']}),flush=True)
        for label,url in [('chunk',chunks[0]['url']),('take',result['candidates'][0]['url'])]:
            audio=context.request.get(args.base+url)
            assert audio.ok and audio.headers['content-type'].startswith('audio/mpeg')
            output=ROOT/'test-results'/f'playground-single-{label}.mp3';output.write_bytes(audio.body())
            stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(output)]))['streams'][0]
            assert stream['channels']==1 and int(stream['bit_rate'])==96000
            foreign=browser.new_context()
            assert foreign.request.get(args.base+url).status==404
            foreign.close()
        for width in [390,320]:
            page.set_viewport_size({'width':width,'height':844});page.wait_for_timeout(200)
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        assert not errors,errors
        report={'verified_at':datetime.now(timezone.utc).isoformat(),'base':args.base,
            'verification':'Real HTTPS Chromium, optional single-live, Luna, GPU TTS, Parakeet fades, mono 96k MP3',
            'defaults':body,'chunks':chunks,'result':result,'javascript_errors':errors,
            'quality_claim':'Integration only; identity-preserving neutral CFG was not quality-evaluated.'}
        (ROOT/'test-results/playground-single-live.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
        browser.close()


if __name__=='__main__':main()
