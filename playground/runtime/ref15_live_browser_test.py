"""Real HTTPS Chromium clicks/upload/GPU generation, separate from mock UI tests."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',default='https://switch-surprised-clock-sparc.trycloudflare.com')
    parser.add_argument('--reference',required=True,type=Path)
    args=parser.parse_args()
    rows=[]
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,
            args=['--no-sandbox','--disable-dev-shm-usage'])
        context=browser.new_context(viewport={'width':1440,'height':1050})
        page=context.new_page();errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        assert page.goto(args.base).status==200
        page.wait_for_function('ready && !!globalThis.smallLoras',timeout=240000)
        page.locator('#ranking-mode').select_option('legacy')
        assert page.locator('#voice-choice').input_value()=='emolia_c0542'
        assert page.locator('#speaker-method').input_value()=='ecapa'
        assert page.locator('#speaker-weight').input_value()=='0.6'
        assert page.locator('#searchn').input_value()=='12'
        assert page.locator('#searchchunk').input_value()=='20'
        assert page.locator('#searchseconds').input_value()=='30'
        assert page.locator('#reference-limit').inner_text()=='15'
        assert page.evaluate('smallLoras.selection()')=={'id':'subset_pitch_quality_p2_rank016','weight':1,'ref15':False,'ref15_weight':1}
        page.locator('#ref-toggle').click()
        page.locator('#fileA').set_input_files(str(args.reference))
        page.wait_for_function('document.getElementById("reference").value.length>0')
        reference_id=page.locator('#reference').input_value()
        try:
            assert page.locator('#voice-choice').input_value()=='custom'
            page.locator('#inputmode').select_option('manual')
            page.locator('#language').select_option('en')
            page.locator('#speaker-method').select_option('off')
            page.locator('#generation').select_option('single')
            page.locator('#seconds').fill('5')
            page.locator('#autoplay').uncheck()
            page.evaluate('''() => {
                const original=takes;
                takes=function(box,result){window.actualResult=result;return original(box,result);};
            }''')
            for preset,id_,ref15 in [('base',None,False),('dpo','subset_pitch_quality_p2_rank016',False),
                                     ('ref15',None,True),('both','subset_pitch_quality_p2_rank016',True)]:
                page.wait_for_function('ready && !running',timeout=240000)
                page.evaluate('window.actualResult=null')
                page.locator(f'[data-lora-preset="{preset}"]').click()
                page.locator('#q').fill('CAPTION: Warm natural conversational delivery.\n'
                                       'TRANSCRIPT: "I kept the letter. Would you like to read it?"')
                expected={'id':id_,'weight':1,'ref15':ref15,'ref15_weight':1}
                assert page.evaluate('smallLoras.selection()')==expected
                page.locator('#send').click()
                page.wait_for_function('window.actualResult!==null || document.querySelector(".error")',timeout=240000)
                assert not page.locator('.error').count(),page.locator('.error').all_text_contents()
                result=page.evaluate('window.actualResult')
                assert result['adapters']['selection']==expected
                assert result['reference']['tts_context']['used_seconds']==15
                assert result['reference']['tts_context']['codec_frames']==187
                assert page.get_by_text('TTS-Referenz-Kontext: 15.00 s · 187 Frames · Limit 15 s').count()>0
                audio=context.request.get(args.base+result['candidates'][0]['url'])
                assert audio.ok and audio.headers['content-type'].startswith('audio/mpeg')
                output=ROOT/'test-results'/f'ref15-https-{preset}.mp3'
                output.write_bytes(audio.body())
                stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(output)]))['streams'][0]
                assert stream['channels']==1 and int(stream['bit_rate'])==96000
                foreign=browser.new_context()
                assert foreign.request.get(args.base+result['candidates'][0]['url']).status==404
                foreign.close()
                rows.append({'preset':preset,'selection':expected,'context':result['reference']['tts_context'],
                    'channels':stream['channels'],'bitrate':int(stream['bit_rate']),
                    'alignment':result['candidates'][0].get('alignment'),'private_session_verified':True})
                print(json.dumps({'passed':preset,'https':args.base,'context':rows[-1]['context']}),flush=True)
            page.wait_for_function('ready && !running',timeout=240000)
            page.locator('#lora-open').click()
            assert page.locator('#lora-dialog').is_visible()
            assert page.locator('#ref15-enable').is_checked()
            page.locator('#dpo-weight').fill('0.5')
            page.locator('#ref15-weight').fill('0.25')
            page.locator('#lora-apply').click()
            assert page.evaluate('smallLoras.selection()')=={'id':'subset_pitch_quality_p2_rank016',
                'weight':.5,'ref15':True,'ref15_weight':.25}
            for width,height in [(1440,1050),(390,844),(320,740)]:
                page.set_viewport_size({'width':width,'height':height})
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                page.screenshot(path=str(ROOT/'test-results'/f'ref15-https-{width}.png'))
            page.locator('[data-lora-preset="base"]').click()
            page.reload()
            page.wait_for_function('ready && !!globalThis.smallLoras',timeout=240000)
            assert page.evaluate('smallLoras.selection()')=={'id':None,'weight':1,'ref15':False,'ref15_weight':1}
            assert page.locator('#voice-choice').input_value()=='custom'
            assert page.locator('#reference').input_value()==reference_id
            assert not errors,errors
        finally:
            deleted=context.request.delete(args.base+'/api/reference/'+reference_id)
            assert deleted.ok
        browser.close()
    report={'timestamp':datetime.now(timezone.utc).isoformat(),'url':args.base,'actual_https_gpu_generations':len(rows),
        'default_rank16_verified':True,'reference_15_seconds_verified':True,'preset_and_modal_controls_verified':True,
        'reference_persists_reload':True,'viewport_widths':[1440,390,320],'javascript_errors':errors,'cases':rows}
    (ROOT/'test-results/ref15-https-browser.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'completed':len(rows),'errors':errors}),flush=True)


if __name__=='__main__':main()
