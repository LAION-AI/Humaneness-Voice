"""Actual HTTPS search generation and gapless scheduled MP3-segment playback."""
import json
from pathlib import Path
import subprocess
import time

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent
URL='https://switch-surprised-clock-sparc.trycloudflare.com'


def main():
    output=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/snap/bin/chromium',headless=True,args=['--no-sandbox'])
        context=browser.new_context(viewport={'width':1440,'height':1050})
        page=context.new_page()
        errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(URL).status==200
        page.wait_for_function('!document.getElementById("send").disabled',timeout=240000)
        assert page.locator('#format').input_value()=='caption'
        assert page.locator('#temperature').input_value()=='1'
        page.locator('#ref-toggle').click()
        page.locator('#fileA').set_input_files(str(ROOT/'performance/benchmark-reference.wav'))
        page.wait_for_function('document.getElementById("reference").value.length>0')
        page.locator('#inputmode').select_option('manual')
        page.evaluate('''() => {
            const originalChunk=LivePlayer.prototype.chunk;
            LivePlayer.prototype.chunk=async function(event){await originalChunk.call(this,event);window.chunkEvents.push({...event,at:performance.now(),decodedSeconds:this.buffers.at(-1).duration,contextState:this.ctx.state});};
            const originalTakes=takes;
            takes=function(box,result){window.actualResult=result;window.resultAt=performance.now();return originalTakes(box,result);};
        }''')
        assert page.locator('#searchn option').evaluate_all('(items)=>items.map(e=>Number(e.value))')==list(range(1,17))
        assert page.locator('#searchchunk option').evaluate_all('(items)=>items.map(e=>Number(e.value))')==[10,20]
        for n,width,chunk in [(1,2,10),(3,2,10),(4,1,10),(8,2,20),(16,1,20)]:
            page.evaluate('window.chunkEvents=[];window.actualResult=null;window.resultAt=null;')
            page.locator('#generation').select_option('search_winner' if width==1 else 'search_beam')
            assert page.locator('#searchn').is_visible()
            page.locator('#searchn').select_option(str(n))
            page.locator('#searchchunk').select_option(str(chunk))
            page.locator('#searchseconds').fill('6.6')
            page.locator('#seed').fill(str(1777+n))
            page.locator('#q').fill('CAPTION: Warm natural conversational speech, clear unhurried words.\nTRANSCRIPT: "You came back. I thought I would never see you again."')
            page.locator('#send').click()
            page.wait_for_function('window.actualResult!==null || document.querySelector(".error")',timeout=240000)
            assert not page.locator('.error').count(),page.locator('.error').all_text_contents()
            data=page.evaluate('({result:window.actualResult,chunks:window.chunkEvents,resultAt:window.resultAt,played:livePlayback.played})')
            perf=data['result']['search_performance']
            assert perf['n']==n and perf['beam_width']==min(n,width) and perf['chunk_ms']==chunk*80
            assert perf['requested_beam_width']==width
            assert data['played'] and data['chunks']
            assert data['chunks'][0]['at']<data['resultAt'], 'Audio must arrive before the completed result'
            assert all(c['contextState']=='running' for c in data['chunks'])
            playback_seconds=perf.get('live_alignment',{}).get('emitted_seconds',perf['actual_seconds'])
            assert abs(sum(c['decodedSeconds'] for c in data['chunks'])-playback_seconds)<.002, 'Independent MP3 chunks must decode without encoder-padding gaps'
            assert abs(sum(c['seconds'] for c in data['chunks'])-playback_seconds)<.002
            mp3=context.request.get(URL+data['chunks'][0]['url'])
            assert mp3.ok and mp3.headers['content-type'].startswith('audio/mpeg')
            sample=ROOT/'test-results'/f'search-{n}-{width}-{chunk}.mp3'
            sample.write_bytes(mp3.body())
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(sample)]))['streams'][0]
            assert probe['channels']==1 and int(probe['bit_rate'])==96000
            foreign=browser.new_context()
            assert foreign.request.get(URL+data['chunks'][0]['url']).status==404
            foreign.close()
            assert 'Suche + Scoring' in page.locator('#m_rtf').inner_text()
            output.append({'n':n,'beam_width':perf['beam_width'],'requested_beam_width':width,
                'chunk_ms':chunk*80,'chunk_count':len(data['chunks']),
                'rtf':perf['search_rtf'],'first_commit_seconds':perf['first_commit_seconds'],
                'final_wer':data['result']['candidates'][0]['wer'],
                'decoded_segment_seconds':sum(c['decodedSeconds'] for c in data['chunks']),
                'raw_pcm_seconds':perf['actual_seconds'],'playback_seconds':playback_seconds,
                'live_alignment':perf.get('live_alignment'),'audio_before_final_result':True})
            page.wait_for_function('!document.getElementById("send").disabled')
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.screenshot(path=str(ROOT/'test-results/search-ui-mobile.png'))
        page.locator('#deleteA').click()
        page.wait_for_function('document.getElementById("reference").options.length===1')
        assert not errors,errors
        browser.close()
    (ROOT/'test-results/search-browser.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output),flush=True)
    print('PASS: BS 1–16 options, real HTTPS N1/N3 top-2 and N4/N16 winner/N8 top-2, both chunk sizes, audio before final result, gapless decoded MP3 chunks, mono 96k, private-session isolation, mobile layout',flush=True)


if __name__=='__main__':main()
