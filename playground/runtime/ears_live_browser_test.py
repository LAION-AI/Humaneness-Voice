"""Actual HTTPS/GPU ranking contracts, both Ears variants and converted output."""
from datetime import datetime, timezone
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent
URL='https://switch-surprised-clock-sparc.trycloudflare.com'


def main():
    report={'verified_at':datetime.now(timezone.utc).isoformat(),'base':URL,
        'scope':'Real HTTPS Chromium + GPU Ears/TTS/ASR/VoiceCLAP; ranking integration, not quality study','cases':[]}
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,
            args=['--no-sandbox','--disable-dev-shm-usage'])
        context=browser.new_context(viewport={'width':1440,'height':1050})
        page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(URL).status==200
        page.wait_for_function('ready && !running && !!smallLoras',timeout=300000)
        state=context.request.get(URL+'/api/state').json()
        assert state['ears_loaded']==['base','medium']
        assert state['ranking_default']=={'mode':'ears_mix','model':'medium','metric':'ce','weight':.5}
        assert page.locator('#ranking-mode').input_value()=='ears_mix'
        assert page.locator('#ranking-controls').is_visible()
        page.evaluate('document.querySelectorAll(".settings-card").forEach(el=>el.open=true)')
        page.locator('#inputmode').select_option('manual')
        page.locator('#language').select_option('en')
        page.locator('#autoplay').uncheck()
        page.evaluate('''() => {
            const oldChunk=LivePlayer.prototype.chunk;
            LivePlayer.prototype.chunk=async function(event){await oldChunk.call(this,event);
                window.chunks.push({...event,beforeResult:!window.actualResult});};
            const oldTakes=takes;
            takes=function(box,result){window.actualResult=result;return oldTakes(box,result);};
        }''')
        cases=[('mixed-medium-ce','all','ears_mix','medium','ce',.5,False),
            ('pure-base-pq','all','ears_only','base','pq',.5,False),
            ('pure-medium-mos','best','ears_only','medium','mos',.5,False),
            ('converted-mixed-base-ce','best','ears_mix','base','ce',.75,True),
            ('winner-mixed-medium-ce','search_winner','ears_mix','medium','ce',.5,False),
            ('top-two-pure-medium-ce','search_beam','ears_only','medium','ce',.5,False)]
        prompt=('CAPTION: Use the same speaker as the reference audio. Preserve that speaker\'s voice identity. '
            'Warm natural conversational delivery.\nTRANSCRIPT: "You came back. I thought I would never see you again."')
        for label,generation,mode,model,metric,weight,convert in cases:
            page.wait_for_function('ready && !running',timeout=300000)
            page.evaluate('window.actualResult=null;window.chunks=[]')
            page.locator('#ranking-mode').select_option(mode)
            page.locator('#ears-model').select_option(model)
            page.locator('#ears-metric').select_option(metric)
            if mode=='ears_mix':page.locator('#ears-weight').evaluate('(el,w)=>{el.value=w;el.dispatchEvent(new Event("input",{bubbles:true}));}',str(weight))
            page.locator('#vc').set_checked(convert)
            page.locator('#generation').select_option(generation)
            if generation.startswith('search_'):
                page.locator('#searchn').select_option('2')
                page.locator('#searchchunk').select_option('10' if generation=='search_winner' else '20')
                page.locator('#searchseconds').fill('6.4')
            else:
                page.locator('#streaming').select_option('offline');page.locator('#n').fill('3')
            page.locator('#q').fill(prompt);page.locator('#send').click()
            page.wait_for_function('actualResult!==null || !!document.querySelector(".error")',timeout=300000)
            assert not page.locator('.error').count(),page.locator('.error').all_text_contents()
            page.wait_for_function('!running',timeout=10000)
            result=page.evaluate('actualResult');chunks=page.evaluate('window.chunks')
            assert result['ranking']['mode']==mode and result['ranking']['model']==model and result['ranking']['metric']==metric
            assert not result['speaker_request']['active']
            assert result['voice_conversion']['enabled']==convert
            if convert:assert result['voice_conversion']['scoring_audio']=='converted output, never raw TTS'
            for item in result['candidates']:
                assert item['ears']['available'] and item['ears']['model']==model and item['ears']['metric']==metric
                assert item['reward_model']==mode
                if mode=='ears_only':
                    assert item['reward']==item['ears']['raw'] and item['ranking_gate']==1
                else:
                    expected=((1-weight)*item['n_clap']+weight*item['ears']['normalized'])*item['gate']
                    assert abs(item['reward']-expected)<1e-7
            winner=max(result['candidates'],key=lambda c:c['reward'])
            assert result['winner']==winner['index'] and winner['rank']==0
            if generation=='all':assert 'Rang 1' in page.locator('.take').first.inner_text()
            if generation.startswith('search_'):
                perf=result['search_performance']
                assert chunks and chunks[0]['beforeResult']
                assert all('ears_seconds' in step and all('ears' in s for s in step['scores']) for step in perf['timeline'])
            assert page.get_by_text('Modellvorhersage',exact=False).count()>0
            report['cases'].append({'case':label,'passed':True,'result':result,'chunks':chunks})
            (ROOT/'test-results/ears-live.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({'case':label,'passed':True,'selected_raw':winner['ears']['raw'],
                'reward':winner['reward'],'wer':winner['wer'],'seconds':result['total_ms']/1000}),flush=True)
        for width in [390,320]:
            page.set_viewport_size({'width':width,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        assert not errors,errors
        report['passed']=True;report['javascript_errors']=errors
        (ROOT/'test-results/ears-live.json').write_text(json.dumps(report,indent=2))
        browser.close()


if __name__=='__main__':main()
