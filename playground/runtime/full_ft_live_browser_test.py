"""Real HTTPS/GPU Full-FT defaults and opt-in DPO smoke; not a quality study."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent
URL='https://switch-surprised-clock-sparc.trycloudflare.com'
STAGE='S3-Ref15-FullFT'
SHA='e1b2c4613545d1db21c6f60a5ab2c27b5320aa05c177717ae268b1ff870d406e'


def main():
    report={'verified_at':datetime.now(timezone.utc).isoformat(),'base':URL,
        'scope':'Real HTTPS Chromium + GPU Full FT/ASR/Ears; integration only','cases':[]}
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,
            args=['--no-sandbox','--disable-dev-shm-usage'])
        context=browser.new_context(viewport={'width':1440,'height':1050})
        page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(URL).status==200
        page.wait_for_function('ready && !running && !!smallLoras',timeout=300000)
        state=context.request.get(URL+'/api/state').json()
        assert state['stage']==STAGE and state['checkpoint']['sha256']==SHA
        assert state['checkpoint']['step']==26774 and state['ears_loaded']==['base','medium']
        assert state['generation_default']=='best' and state['n_default']==3
        body=page.evaluate('buildTurnRequest("check")');report['defaults']=body
        assert body['generation']=='best' and body['n']==3
        assert body['adapters']=={'id':None,'weight':1,'ref15':False,'ref15_weight':1}
        assert body['voice_profile']=='emolia_c0542' and body['chunk_frames']==20 and body['streaming']
        assert body['guidance']==2 and body['temperature']==1 and body['align']
        assert body['ranking']=={'mode':'ears_mix','model':'medium','metric':'ce','weight':.5}
        page.evaluate('document.querySelectorAll(".settings-card").forEach(el=>el.open=true)')
        page.locator('#autoplay').uncheck()
        page.evaluate('''() => {const old=takes;takes=function(box,result){
            window.actualResult=result;return old(box,result);};}''')

        def turn(label,prompt):
            page.wait_for_function('ready && !running',timeout=300000)
            page.evaluate('window.actualResult=null')
            page.locator('#q').fill(prompt);page.locator('#send').click()
            page.wait_for_function('actualResult!==null || !!document.querySelector(".error")',timeout=300000)
            assert not page.locator('.error').count(),page.locator('.error').all_text_contents()
            page.wait_for_function('!running',timeout=10000)
            result=page.evaluate('actualResult')
            assert result['checkpoint']['sha256']==SHA and result['checkpoint']['step']==26774
            info=result['reference']['tts_context']
            # Dynamic clips can genuinely be short; only the explicit 15s
            # upload below must exceed the historical 37-frame ceiling.
            assert 0<info['codec_frames']<=188
            assert abs(info['used_seconds']-min(info['source_seconds'],15))<.01
            for candidate in result['candidates']:
                assert candidate['ears']['available'] and candidate['ears']['model']=='medium'
            selected=max(result['candidates'],key=lambda c:c['reward'])
            assert selected['index']==result['winner']
            response=context.request.get(URL+selected['url']);assert response.ok
            assert response.headers['content-type'].startswith('audio/mpeg')
            audio=response.body();output=ROOT/'test-results'/f'full-ft-{label}.mp3';output.write_bytes(audio)
            streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(output)]))['streams']
            assert streams[0]['channels']==1 and int(streams[0]['bit_rate'])==96000
            foreign=browser.new_context();assert foreign.request.get(URL+selected['url']).status==404;foreign.close()
            report['cases'].append({'case':label,'passed':True,'result':result,'audio_sha256':hashlib.sha256(audio).hexdigest()})
            (ROOT/'test-results/full-ft-live.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
            print(json.dumps({'case':label,'takes':len(result['candidates']),'wer':selected['wer'],
                'on_device_aggregate_rtf':result['performance']['aggregate_rtf'],
                'adapters':result['adapters']['active'],'seconds':result['total_ms']/1000}),flush=True)
            return result

        first=turn('default-best3','Answer in English in exactly two short sentences, with warm relief. '
            'Speak directly to a friend who has finally returned. Keep it brief.')
        assert first['generation']=='best' and len(first['candidates'])==3
        assert not first['adapters']['active'] and not first['reference']['dynamic']
        assert first['reference']['policy']=='fixed'
        assert first['performance']['streaming'] and first['performance']['chunk_frames']==20
        page.locator('#inputmode').select_option('manual');page.locator('#language').select_option('en')
        prompt=('CAPTION: Use the same speaker as the reference audio. Preserve that speaker\'s voice identity. '
            'Warm relief, natural conversation.\nTRANSCRIPT: "You came back. I am so glad to see you."')
        page.locator('[data-lora-preset="dpo"]').click()
        dpo=turn('optional-dpo',prompt)
        assert len(dpo['candidates'])==3 and len(dpo['adapters']['active'])==1
        assert dpo['adapters']['selection']['id']=='subset_pitch_quality_p2_rank016'
        assert dpo['adapters']['cross_base'] and dpo['adapters']['experimental_mix']
        page.locator('[data-lora-preset="base"]').click()
        restored=turn('restored-full-ft',prompt)
        assert len(restored['candidates'])==3 and not restored['adapters']['active']
        assert restored['adapters']['base_sha256']==SHA and not restored['adapters']['cross_base']
        # Only a fresh private test-session upload is created/removed here.
        upload=context.request.post(URL+'/api/reference?slot=A',multipart={'file':{
            'name':'long-selma-test.mp3','mimeType':'audio/mpeg',
            'buffer':Path('/mnt/nvme/moss-15-v2-assets/refs3/audio/emolia_c0542/emolia_c0542__V__VALN__moderately_high__de.c000.mp3').read_bytes()}})
        assert upload.ok,upload.text();item=upload.json()
        try:
            page.reload();page.wait_for_function('ready && refs.length>0 && !running',timeout=300000)
            page.evaluate('document.querySelectorAll(".settings-card").forEach(el=>el.open=true)')
            page.locator('#voice-choice').select_option('custom')
            page.locator('#inputmode').select_option('manual');page.locator('#language').select_option('en')
            page.locator('#autoplay').uncheck()
            page.evaluate('''() => {const old=takes;takes=function(box,result){window.actualResult=result;return old(box,result);};}''')
            long=turn('15s-reference-best3',prompt)
            assert long['reference']['mode']=='custom' and not long['adapters']['active']
            info=long['reference']['tts_context']
            assert info['used_seconds']>14.9 and 187<=info['codec_frames']<=188
        finally:
            assert context.request.delete(URL+'/api/reference/'+item['id']).ok
        for width in [390,320]:
            page.set_viewport_size({'width':width,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        assert not errors,errors
        report['passed']=True;report['javascript_errors']=errors
        (ROOT/'test-results/full-ft-live.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
        browser.close()


if __name__=='__main__':main()
