"""Real HTTPS optional burst reward/UI regression checks with private uploads."""
import json
import math
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent
URL='https://switch-surprised-clock-sparc.trycloudflare.com'


def main():
    results=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/snap/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1050})
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(URL).status==200
        page.wait_for_function('!document.getElementById("send").disabled',timeout=300000)
        page.wait_for_function('dimensionCatalog.length===112')
        page.locator('#ranking-mode').select_option('legacy')
        assert page.locator('#burstmode').input_value()=='off'
        page.locator('#ref-toggle').click()
        page.locator('#fileA').set_input_files(str(ROOT/'performance/benchmark-reference.wav'))
        page.wait_for_function('document.getElementById("reference").value.length>0')
        page.locator('#inputmode').select_option('manual')
        page.locator('#align').uncheck();page.locator('#autoplay').uncheck()
        page.evaluate('''() => {const original=takes;takes=function(box,result){window.actualResult=result;return original(box,result);};}''')
        caption='CAPTION: Warm gently amused natural conversational speech.\nTRANSCRIPT: "You came back. I thought I would never see you again."'
        script='GENERAL: Warm amused conversation, perform a short audible chuckle between the two sentences.\nSCRIPT:\n[0.2 seconds pause] [1.4 seconds duration] You came back. (chuckle, 0.3 seconds) [2.8 seconds duration] I thought I would never see you again.'
        leading='GENERAL: A brief exasperated sigh, followed by clear restrained dramatic speech.\nSCRIPT:\n[0.2 seconds pause] (exasperated sigh, 0.3 seconds) [4.5 seconds duration] You came back. I thought I would never see you again.'
        cases=[
            ('manual_location_best4','best','clap','location','manual','caption',caption,10),
            ('word_typed_winner4_800','search_winner','clap','typed','script','structured',script,10),
            ('adaptive_typed_beam4_1600','search_beam','adaptive_manual','typed','script','structured',leading,20),
            ('no_target_unchanged','single','clap','typed','script','caption',caption,10),
            ('off_legacy','search_winner','clap','off','script','caption',caption,20),
        ]
        for number,(label,generation,reward,mode,source,fmt,prompt,chunk) in enumerate(cases):
            page.locator('#rewardmode').select_option(reward)
            if reward=='adaptive_manual':
                row=page.locator('.reward-row').first
                row.locator('.reward-dimension').select_option('vn:S_DRAM')
                row.locator('select').nth(1).select_option('maximize')
                row.locator('input').nth(1).fill('1')
            page.locator('#burstmode').select_option(mode)
            if mode!='off':
                page.locator('#burstsource').select_option(source)
                if source=='manual':
                    assert page.locator('#bursttiming').input_value()=='clock'
                    assert page.locator('#bursttiming').is_disabled()
                    row=page.locator('.burst-row').first
                    for index,value in enumerate(['chuckle','1.6','0.3']):row.locator('input').nth(index).fill(value)
                else:
                    page.locator('#bursttiming').select_option('words')
                if mode=='typed':page.locator('#burstmatch').select_option('family')
            page.locator('#format').select_option(fmt)
            page.locator('#generation').select_option(generation)
            if generation in ['best','all']:page.locator('#n').fill('4')
            if generation.startswith('search_'):
                page.locator('#searchn').select_option('4')
                page.locator('#searchchunk').select_option(str(chunk))
            page.locator('#seed').fill(str(5377+number));page.locator('#q').fill(prompt)
            page.evaluate('window.actualResult=null');page.locator('#send').click()
            page.wait_for_function('window.actualResult!==null || document.querySelector(".error")',timeout=300000)
            assert not page.locator('.error').count(),page.locator('.error').all_text_contents()
            result=page.evaluate('window.actualResult')
            request=result['burst_request']
            active=mode!='off' and label!='no_target_unchanged'
            if mode=='off':assert request is None
            else:assert request['active']==active
            for candidate in result['candidates']:
                if active:
                    report=candidate['burst_alignment'];value=report['score']
                    assert value is not None and math.isfinite(value) and 0<=value<=1
                    assert report['pending_targets']==0
                    assert abs(candidate['quality']-((1-report['weight'])*candidate['base_quality']+report['weight']*value))<1e-7
                    assert abs(candidate['reward']-candidate['quality']*candidate['gate'])<1e-7
                    for event in report['events']:
                        assert 0<=event['start']<=event['end']<=candidate['sec']+.0001
                        if mode=='typed':assert len(event['probabilities'])==17
                    if reward=='adaptive_manual':assert len(candidate['dimension_scores'])==1
                else:assert 'burst_alignment' not in candidate
            performance=result.get('search_performance')
            if performance and active:
                assert any(t['burst_seconds']>0 for t in performance['timeline'])
                assert 'burst locator/classifier' in performance['includes']
            best=min(result['candidates'],key=lambda c:c['rank'])
            if active:assert best['reward']==max(c['reward'] for c in result['candidates'])
            results.append({'case':label,'generation':generation,'reward_mode':reward,'burst_request':request,
                'search_rtf':performance['search_rtf'] if performance else None,
                'burst_scoring_seconds':sum(t.get('burst_seconds',0) for t in performance['timeline']) if performance else None,
                'reward_loading_ms':result['reward_loading_ms'],'scores':result['candidates']})
            print(json.dumps({'case':label,'success':True,'search_rtf':results[-1]['search_rtf'],
                'winner_wer':best['wer'],'burst_score':(best.get('burst_alignment') or {}).get('score'),
                'detected_events':len((best.get('burst_alignment') or {}).get('events',[]))}),flush=True)
            page.wait_for_function('!document.getElementById("send").disabled')
        page.locator('#burstmode').select_option('typed');page.locator('#burstsource').select_option('manual')
        for _ in range(7):page.locator('#burstadd').click()
        assert page.locator('.burst-row').count()==8 and page.locator('#burstadd').is_disabled()
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.screenshot(path=str(ROOT/'test-results/burst-reward-mobile.png'))
        page.locator('#deleteA').click();assert not errors,errors
        browser.close()
    (ROOT/'test-results/burst-browser.json').write_text(json.dumps(results,indent=2))
    print('PASS: off default, manual windows, word anchors, locator/type families, adaptive combination, Best-of-4, winner/Top-2 800/1600, no-target legacy, 8-target limit and mobile UI',flush=True)


if __name__=='__main__':main()
