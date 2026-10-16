"""Real HTTPS checks: native UI editing, mixed heads, automatic Luna selection."""
import json
import math
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent
URL='https://switch-surprised-clock-sparc.trycloudflare.com'


def edit_terms(page,terms):
    page.locator('#rewardmode').select_option('adaptive_manual')
    while page.locator('.reward-row').count()>1:
        page.locator('.reward-row button').last.click()
    for index,term in enumerate(terms):
        if index:page.locator('#rewardadd').click()
        row=page.locator('.reward-row').nth(index)
        row.locator('.reward-dimension').select_option(term['dimension'])
        row.locator('select').nth(1).select_option(term.get('objective','target'))
        if term.get('objective','target')=='target':row.locator('input').nth(0).fill(str(term['target']))
        row.locator('input').nth(1).fill(str(term.get('weight',1)))


def main():
    results=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/snap/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1050})
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(URL).status==200
        page.wait_for_function('!document.getElementById("send").disabled',timeout=240000)
        page.wait_for_function('dimensionCatalog.length===112')
        page.locator('#ranking-mode').select_option('legacy')
        page.locator('#rewardmode').select_option('adaptive_auto')
        page.locator('#q').fill('Whisper a secret softly, with restrained fear, low arousal, never ranting.')
        page.locator('#rewardpreview').click()
        page.wait_for_function('!running',timeout=90000)
        assert page.locator('#rewardmode').input_value()=='adaptive_manual'
        assert 1<=page.locator('.reward-row').count()<=4
        assert 'vn:S_WHIS' in page.evaluate('rewardRows.map(t=>t.dimension)')
        page.locator('#ref-toggle').click()
        page.locator('#fileA').set_input_files(str(ROOT/'performance/benchmark-reference.wav'))
        page.wait_for_function('document.getElementById("reference").value.length>0')
        page.locator('#inputmode').select_option('manual')
        page.locator('#align').uncheck()
        page.locator('#autoplay').uncheck()
        page.evaluate('''() => {const original=takes;takes=function(box,result){window.actualResult=result;return original(box,result);};}''')
        cases=[
            ('vn_search','search_winner','adaptive_manual',10,[
                {'dimension':'vn:S_WHIS','target':5,'weight':2},
                {'dimension':'vn:AROU','target':1.5}],
             'Quiet whispered secret, very calm, soft and intimate.'),
            ('mixed_best','best','adaptive_manual',10,[
                {'dimension':'vn:S_RANT','target':5,'weight':2},
                {'dimension':'vn:AROU','target':4.5},
                {'dimension':'eiv:Anger','target':3},
                {'dimension':'eiv:score_speech_quality','objective':'maximize'}],
             'Angry dramatic rant, openly furious, energetic and intense.'),
            ('auto_beam','search_beam','adaptive_auto',20,[],
             'Warm affectionate conversation, tender gratitude, gently amused, not angry.'),
            ('mixed_search16','search_winner','adaptive_manual',20,[
                {'dimension':'vn:AROU','target':4.5},
                {'dimension':'eiv:Anger','target':3},
                {'dimension':'eiv:score_speech_quality','objective':'maximize'}],
             'Angry dramatic speech, controlled intensity, clear words.'),
            ('legacy_search','search_winner','clap',10,[],
             'Warm natural conversational speech, clear unhurried words.'),
        ]
        for number,(label,generation,mode,chunk,terms,caption) in enumerate(cases):
            page.locator('#rewardmode').select_option(mode)
            if terms:edit_terms(page,terms)
            page.locator('#generation').select_option(generation)
            if generation in ['best','all']:page.locator('#n').fill('4')
            if generation.startswith('search_'):
                page.locator('#searchn').select_option('16' if label=='mixed_search16' else '4')
                page.locator('#searchchunk').select_option(str(chunk))
            page.locator('#seed').fill(str(4277+number))
            page.locator('#q').fill('CAPTION: '+caption+'\nTRANSCRIPT: "You came back. I thought I would never see you again."')
            page.evaluate('window.actualResult=null')
            page.locator('#send').click()
            page.wait_for_function('window.actualResult!==null || document.querySelector(".error")',timeout=240000)
            assert not page.locator('.error').count(),page.locator('.error').all_text_contents()
            result=page.evaluate('window.actualResult')
            assert result['reward_mode']==mode
            if mode!='clap':
                assert 1<=len(result['reward_plan']['terms'])<=4
                if terms:assert result['reward_plan']['reason']=='Manuell gewählte Reward-Dimensionen.'
                if mode=='adaptive_auto':
                    assert not any('endpoint direction not fully specified' in next(d['description'] for d in
                        page.evaluate('dimensionCatalog') if d['id']==t['dimension']) for t in result['reward_plan']['terms'])
                for candidate in result['candidates']:
                    dims=candidate['dimension_scores']
                    assert len(dims)==len(result['reward_plan']['terms'])
                    assert all(math.isfinite(s['raw']) and 0<=s['normalized']<=1 and 0<=s['utility']<=1 for s in dims)
                    assert abs(sum(s['normalized_weight'] for s in dims)-1)<1e-7
                    assert abs(candidate['reward']-candidate['quality']*candidate['gate'])<1e-7
                    if terms:assert {s['dimension'] for s in dims}=={t['dimension'] for t in terms}
                assert page.get_by_text('Reward-Dimensionen · Rohwerte & normalisierte Ziel-Erfüllung',exact=True).count()>0
            else:assert result['reward_plan'] is None and 'dimension_scores' not in result['candidates'][0]
            best=min(result['candidates'],key=lambda c:c['rank'])
            assert best['reward']==max(c['reward'] for c in result['candidates'])
            results.append({'case':label,'reward_mode':mode,'plan':result['reward_plan'],
                'generation':generation,'search_rtf':(result.get('search_performance') or {}).get('search_rtf'),
                'reward_planning_ms':result['reward_planning_ms'],'reward_loading_ms':result['reward_loading_ms'],
                'scores':result['candidates']})
            print(json.dumps({'case':label,'success':True,'search_rtf':results[-1]['search_rtf'],
                'terms':result['reward_plan'],'winner_wer':best['wer']}),flush=True)
            page.wait_for_function('!document.getElementById("send").disabled')
        page.locator('#rewardmode').select_option('adaptive_manual')
        edit_terms(page,cases[1][4])
        assert page.locator('#rewardadd').is_disabled()
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.screenshot(path=str(ROOT/'test-results/adaptive-reward-mobile.png'))
        page.locator('#deleteA').click()
        assert not errors,errors
        browser.close()
    (ROOT/'test-results/adaptive-browser.json').write_text(json.dumps(results,indent=2))
    print('PASS: 112 catalog entries, real Luna preview/auto, manual native-scale targets/weights, 2/4-head VoiceNet/EIV+ mixtures, BS 4/16 chunk search/beam/Best-of-4, legacy mode, score details and mobile UI',flush=True)

if __name__=='__main__':main()
