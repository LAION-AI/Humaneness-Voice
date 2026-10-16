"""Own long-upload target, streaming and removal over HTTPS; no user data touched."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent
URL='https://switch-surprised-clock-sparc.trycloudflare.com'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',required=True,type=Path)
    args=parser.parse_args()
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,
            args=['--no-sandbox','--disable-dev-shm-usage'])
        context=browser.new_context(viewport={'width':1440,'height':1050})
        page=context.new_page();errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(URL).status==200
        page.wait_for_function('ready && !running && !!smallLoras',timeout=240000)
        page.evaluate('document.querySelectorAll(".settings-card").forEach(el=>el.open=true)')
        page.locator('#ref-toggle').click()
        page.locator('#fileA').set_input_files(str(args.reference))
        page.wait_for_function('document.getElementById("reference").value.length>0')
        owned_id=page.locator('#reference').input_value()
        try:
            assert page.locator('#voice-choice').input_value()=='custom'
            page.locator('#vc').check()
            page.locator('#inputmode').select_option('manual')
            page.locator('#language').select_option('de')
            page.evaluate('''() => {
                window.actualResult=null;window.chunkEvents=[];
                const oldChunk=LivePlayer.prototype.chunk;
                LivePlayer.prototype.chunk=async function(event){await oldChunk.call(this,event);
                    window.chunkEvents.push({...event,beforeResult:!window.actualResult});};
                const oldTakes=takes;
                takes=function(box,result){window.actualResult=result;return oldTakes(box,result);};
            }''')
            page.locator('#q').fill('CAPTION: Use the same speaker as the reference audio. Preserve that speaker\'s voice identity. '
                'Warm, natural conversational speech.\nTRANSCRIPT: "Da bist du ja. Ich habe auf dich gewartet."')
            page.locator('#send').click()
            page.wait_for_function('actualResult!==null || !!document.querySelector(".error")',timeout=300000)
            assert not page.locator('.error').count(),page.locator('.error').all_text_contents()
            page.wait_for_function('!running',timeout=10000)
            result=page.evaluate('actualResult');chunks=page.evaluate('chunkEvents')
            assert result['reference']['mode']=='custom'
            assert result['reference']['tts_context']['used_seconds']==15
            assert result['reference']['tts_context']['codec_frames']==187
            assert result['voice_conversion']['reference_source_seconds']==15
            assert result['voice_conversion']['reference_used_seconds']==10
            assert chunks and chunks[0]['beforeResult'] and page.evaluate('livePlayback.played')
            assert page.locator('#reference').input_value()==owned_id
            audio=context.request.get(URL+result['candidates'][0]['url'])
            assert audio.ok and audio.headers['content-type'].startswith('audio/mpeg')
            (ROOT/'test-results/chatterbox-custom-take.mp3').write_bytes(audio.body())
            assert not errors,errors
            report={'verified_at':datetime.now(timezone.utc).isoformat(),'base':URL,
                'verification':'Real HTTPS own long catalog-voice upload; TTS15s/187 frames, VC10s, live converted PCM',
                'passed':True,'result':result,'chunks':chunks,'javascript_errors':errors}
            (ROOT/'test-results/chatterbox-custom-live.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({'passed':True,'tts_reference_seconds':15,'vc_reference_seconds':10,
                'chunks_before_result':len(chunks),'wer':result['candidates'][0]['wer'],
                'total_seconds':result['total_ms']/1000,'first_audio_seconds':result['live_stream']['first_playable_seconds']}),flush=True)
        finally:
            # Exact ID belongs to this fresh test context. Never remove another
            # session's reference or stop an in-flight job to force cleanup.
            response=context.request.delete(URL+'/api/reference/'+owned_id)
            assert response.ok,response.text()
        page.wait_for_function('ready && !running',timeout=300000)
        page.locator('#voice-choice').select_option('none')
        assert page.locator('#vc').is_disabled() and not page.locator('#vc').is_checked()
        browser.close()


if __name__=='__main__':main()
