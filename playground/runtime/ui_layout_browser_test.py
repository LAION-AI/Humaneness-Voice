"""Offline real-Chromium UI checks; mocked API, no GPU/performance claims."""
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent
ORIGIN='https://demo.test'


def main():
    html=(ROOT/'index.html').read_text()
    css=re.search(r'<style>(.*?)</style>',(ROOT.parent/'index.html').read_text(),re.S).group(1)
    catalog=json.loads((ROOT/'reward_catalog.json').read_text())
    adapter_catalog=json.loads((ROOT/'test-results/dpo-catalog.json').read_text())
    adapter_catalog.update(default=None,base_stage='S3-Ref15-FullFT',cross_base=True,preferred_dpo='subset_pitch_quality_p2_rank016',
        ref15={'id':'ref15_rank128','rank':128})
    selection=json.loads((ROOT/'fixed-references/selection.json').read_text())
    voice_catalog={'default':selection['default'],'voices':[
        {'id':x['voice'],'name':x['name'],'reference_seconds':x['seconds'],
         'content_enjoyment':x['ears']['raw'],'recommended':x['voice'] in selection['selected'],
         'preview_url':'/api/voices/reference/'+x['voice']} for x in selection['candidates']]}
    payloads=[];posts=[];refs=[]
    plan={'terms':[{'dimension':'vn:AROU','objective':'target','target':4.5,'weight':2},
                   {'dimension':'eiv:Anger','objective':'maximize','weight':1}],
          'reason':'Mocked planner response for UI wiring verification only.'}
    def handler(route):
        request=route.request;path=urlsplit(request.url).path
        if path=='/':route.fulfill(content_type='text/html',body=html)
        elif path=='/legacy.css':route.fulfill(content_type='text/css',body=css)
        elif path=='/lora-ui.js':route.fulfill(content_type='text/javascript',body=(ROOT/'lora_ui.js').read_text())
        elif path=='/api/adapters/catalog':route.fulfill(json=adapter_catalog)
        elif path=='/api/voices/catalog':route.fulfill(json=voice_catalog)
        elif path=='/api/reward/catalog':route.fulfill(json=catalog)
        elif path=='/api/reward/plan':route.fulfill(json=plan)
        elif path=='/api/state':route.fulfill(json={'ready':True,'busy':False,'error':None,
            'stage':'UI fixture','director':'Luna','director_ready':True,'references':refs,
            'alignment_available':True,'phase':'ready','voice_conversion':{'available':True},
            'ranking_default':{'mode':'ears_mix','model':'medium','metric':'ce','weight':.5}})
        elif path=='/api/reference':
            refs.append({'id':'mock-ref','slot':'A','name':'Test reference','url':'/api/audio/mock-ref'})
            route.fulfill(json={'id':'mock-ref','seconds':2.88,'references':refs})
        elif path.startswith('/api/reference/'):
            refs.clear();route.fulfill(json={'references':[]})
        elif path=='/api/turn':
            body=request.post_data_json;posts.append(body)
            candidate={'index':0,'rank':0,'sec':4.,'wer':0.,'hyp':'Ready.',
                'genuineness':3.,'blend':5.,'clap':.2,'reward':.5,'url':'/api/audio/mock-take'}
            search=body['generation'].startswith('search_')
            performance={'n':body['search_n'],'beam_width':min(body['search_n'],
                2 if body['generation']=='search_beam' else 1),'chunk_ms':body['search_chunk_frames']*80,
                'tts_codec_seconds':1.,'search_rtf':.5,'first_commit_seconds':.3,
                'hit_frame_limit':False,'timeline':[]} if search else None
            result={'event':'result','candidates':[candidate],'winner':0,'generation':body['generation'],
                'guidance':body['guidance'],'seed':body['seed'],'cfg_effective':body['guidance']!=1,
                'llm_ms':0,'generation_ms':1,'alignment_ms':0,'scoring_ms':1,'total_ms':2,
                'reward_mode':body['reward_mode'],'reward_plan':body.get('reward_plan'),
                'reward_planning_ms':0,'reward_loading_ms':0,'burst_request':None,
                'performance':None,'search_performance':performance}
            direction={'event':'direction','text':'Ready.','lang':'en','format':body['format'],
                'frames':50,'reference':None,'prompt':body['message'],'unconditional':'neutral'}
            events=[direction,result]
            route.fulfill(content_type='text/event-stream',body=''.join(
                'data: '+json.dumps(event)+'\n\n' for event in events))
        elif path.startswith('/api/audio/'):
            route.fulfill(content_type='audio/mpeg',body=(ROOT/'test-results/browser-take.mp3').read_bytes())
        elif path.startswith('/api/voices/reference/'):
            route.fulfill(content_type='audio/mpeg',body=(ROOT/'test-results/browser-take.mp3').read_bytes())
        else:route.fulfill(status=404,body='Offline UI fixture: route unavailable')

    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,
            args=['--no-sandbox','--disable-dev-shm-usage','--disable-background-networking'])
        context=browser.new_context(viewport={'width':1440,'height':1050})
        context.route('**/*',handler)
        page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        assert page.goto(ORIGIN).status==200
        page.wait_for_function('ready && dimensionCatalog.length===112')
        page.wait_for_function('!!globalThis.smallLoras')
        default_ranking=page.evaluate('buildTurnRequest("Default ranking")')
        assert default_ranking['ranking']=={'mode':'ears_mix','model':'medium','metric':'ce','weight':.5}
        assert default_ranking['speaker']['method']=='off'
        assert page.locator('#ranking-controls').is_visible() and not page.locator('#speaker-controls').is_visible()
        page.locator('#ears-weight').evaluate('(el)=>{el.value=.75;el.dispatchEvent(new Event("input",{bubbles:true}));}')
        assert '75 %' in page.locator('#ears-mix').inner_text()
        assert page.evaluate('buildTurnRequest("Mixture").ranking.weight')==.75
        for model in ['base','medium']:
            page.locator('#ears-model').select_option(model)
            for metric in ['ce','pq','mos','ovrl','cu','pc']:
                page.locator('#ears-metric').select_option(metric)
                page.locator('#ranking-mode').select_option('ears_only')
                assert page.locator('#ears-weight').is_disabled()
                payloads.append(page.evaluate('buildTurnRequest("Pure quality")'))
        page.locator('#ranking-mode').select_option('legacy')
        assert page.locator('#speaker-controls').is_visible() and page.locator('#rewardcontrols').is_visible()
        assert page.locator('#voice-choice').input_value()=='emolia_c0542'
        assert page.locator('#speaker-method').input_value()=='ecapa'
        assert page.locator('#speaker-weight').input_value()=='0.6'
        assert page.locator('.settings-card').first.get_attribute('id')=='reference-settings'
        assert '40 %' in page.locator('#speaker-mix').inner_text()
        assert page.locator('#searchchunk').input_value()=='20'
        assert page.locator('#chunkframes').input_value()=='20'
        assert not page.locator('#vc').is_checked() and page.locator('#vc').is_enabled()
        page.locator('#vc').check();assert page.evaluate('buildTurnRequest("VC fixture").voice_conversion')
        page.locator('#vc').uncheck()
        initial=page.evaluate('buildTurnRequest("Default fixture")')
        assert initial['generation']=='best' and initial['streaming'] and initial['n']==3
        assert initial['guidance']==2 and initial['temperature']==1 and initial['seconds'] is None
        assert page.locator('#voice-choice optgroup').first.locator('option').count()==6
        assert page.locator('#catalog-preview').is_visible()
        assert initial['adapters']=={'id':None,'weight':1,'ref15':False,'ref15_weight':1}
        page.locator('#speaker-weight').evaluate('(el)=>{el.value=.7;el.dispatchEvent(new Event("input",{bubbles:true}));}')
        body=page.evaluate('buildTurnRequest("Identity fixture")');payloads.append(body)
        assert body['speaker']=={'method':'ecapa','weight':.7}
        assert '70 %' in page.locator('#speaker-mix').inner_text()
        page.locator('#speaker-method').select_option('ecapa')
        payloads.append(page.evaluate('buildTurnRequest("ECAPA fixture")'))
        page.locator('#speaker-method').select_option('off');assert page.locator('#speaker-weight').is_disabled()
        page.locator('#speaker-method').select_option('voicenet')
        page.locator('#speaker-weight').evaluate('(el)=>{el.value=.34;el.dispatchEvent(new Event("input",{bubbles:true}));}')
        for preset,id_,ref15 in [('base',None,False),('dpo','subset_pitch_quality_p2_rank016',False),
                                 ('ref15',None,True),('both','subset_pitch_quality_p2_rank016',True)]:
            page.locator(f'[data-lora-preset="{preset}"]').click()
            pick=page.evaluate('buildTurnRequest("Preset fixture").adapters');payloads.append(page.evaluate('buildTurnRequest("Preset fixture")'))
            assert pick=={'id':id_,'weight':1,'ref15':ref15,'ref15_weight':1}
        page.locator('[data-lora-preset="base"]').click()
        page.locator('#lora-open').click()
        assert page.locator('#lora-dialog').is_visible()
        assert page.evaluate('smallLoras.selection().id') is None
        assert page.locator('#searchseconds').input_value()=='30'
        page.locator('#lora-search').fill('rebalanced_rank064')
        page.locator('#lora-list input[value="rebalanced_rank064"]').check()
        page.locator('#dpo-weight').fill('0.5')
        page.locator('#ref15-enable').check()
        page.locator('#ref15-weight').fill('0.25')
        page.locator('#lora-apply').click()
        assert not page.locator('#lora-dialog').is_visible()
        chosen=page.evaluate('buildTurnRequest("LoRA fixture")')['adapters']
        assert chosen=={'id':'rebalanced_rank064','weight':.5,'ref15':True,'ref15_weight':.25}
        page.locator('#lora-open-sidebar').click()
        page.locator('#lora-clear').click()
        page.locator('#lora-cancel').click()
        assert page.evaluate('smallLoras.selection()')==chosen,'Cancel must not alter next-turn selection'
        page.locator('#lora-open').click();page.locator('#lora-clear').click();page.locator('#lora-apply').click()
        assert page.evaluate('smallLoras.selection().id') is None
        page.locator('#lora-open').click();page.locator('#lora-default').click();page.locator('#lora-apply').click()
        assert page.evaluate('smallLoras.selection()')=={'id':None,'weight':1,'ref15':False,'ref15_weight':1}
        page.locator('#lora-open').click();page.keyboard.press('Escape')
        assert not page.locator('#lora-dialog').is_visible()
        for id in ['generation','searchn','searchchunk','searchseconds','n','streaming','chunkframes',
            'inputmode','format','language','reference','cfg','g','temperature','seed','seconds',
            'align','autoplay','rewardmode','rewardshare','burstmode','burstsource','bursttiming']:
            assert page.locator('#'+id).count()==1,id
        assert page.locator('#searchoptions').is_visible(), 'Search must be discoverable before activation'
        assert page.locator('#searchn').is_disabled()
        assert page.locator('#searchn').input_value()=='12'
        assert page.locator('#generation').input_value()=='best'
        assert page.get_by_role('button',name='Beam Search aktivieren').is_visible()
        assert page.locator('#searchn option').evaluate_all('(xs)=>xs.map(x=>Number(x.value))')==list(range(1,17))
        assert page.locator('#searchchunk option').evaluate_all('(xs)=>xs.map(x=>Number(x.value))')==[10,20]
        assert page.locator('#temperature').get_attribute('min')=='0'
        assert page.locator('#g').input_value()=='2'
        assert page.locator('#burstmode').input_value()=='off'
        assert page.locator('#n').is_enabled() and page.locator('#n').input_value()=='3'
        assert page.locator('#chunkframes').is_enabled()
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert page.evaluate('document.querySelector("footer").scrollHeight<=document.querySelector("footer").clientHeight')
        for generation in ['search_winner','search_beam']:
            page.locator(f'[data-generation="{generation}"]').click()
            assert page.locator('#generation').input_value()==generation
            assert page.locator('#searchn').is_enabled() and page.locator('#seconds').is_disabled()
            assert page.locator('#streaming').is_disabled()
            for n in range(1,17):
                page.locator('#searchn').select_option(str(n))
                for frames in [10,20]:
                    page.locator('#searchchunk').select_option(str(frames))
                    body=page.evaluate('buildTurnRequest("Test request")')
                    assert body['generation']==generation and body['search_n']==n
                    assert body['search_chunk_frames']==frames and body['seconds'] is None
                    assert not body['streaming']
                    payloads.append(body)
                if generation=='search_beam' and n==7:assert '4 + 3' in page.locator('#beam-expansion').inner_text()
                if n==1:assert 'ein Suchpfad' in page.locator('#beam-expansion').inner_text()
        page.locator('#searchn').select_option('8')
        assert '4 + 4' in page.locator('#beam-expansion').inner_text()
        page.locator('#autoplay').uncheck()
        page.evaluate('''() => {const original=takes;takes=function(box,result){window.actualResult=result;return original(box,result);};}''')
        for generation in ['single','best','all','search_winner','search_beam']:
            page.locator('#generation').select_option(generation)
            if generation in ['best','all']:page.locator('#n').fill('4')
            if not generation.startswith('search_'):
                page.locator('#streaming').select_option('streaming')
                assert page.locator('#chunkframes').is_enabled()
                page.locator('#chunkframes').select_option('5');page.locator('#seconds').fill('3.2')
            page.locator('#inputmode').select_option('manual')
            assert page.locator('#manualhelp').is_visible()
            page.locator('#format').select_option('caption');page.locator('#template').click()
            assert 'TRANSCRIPT:' in page.locator('#q').input_value()
            page.evaluate('window.actualResult=null');page.locator('#send').click()
            page.wait_for_function('!running && window.actualResult!==null')
            assert posts[-1]['generation']==generation
            if generation.startswith('search_'):
                assert posts[-1]['seconds'] is None and not posts[-1]['streaming']
                assert 'Suche + Scoring' in page.locator('#m_rtf').inner_text()
                assert page.get_by_text('Technische Messwerte & Suchschritte (JSON)',exact=True).count()>0
            else:assert posts[-1]['seconds']==3.2 and posts[-1]['streaming']
        page.locator('#rewardmode').select_option('adaptive_auto')
        page.locator('#q').fill('Angry dramatic speaking, high arousal.')
        page.locator('#rewardpreview').click();page.wait_for_function('!running && rewardRows.length===2')
        assert page.locator('#rewardmode').input_value()=='adaptive_manual'
        assert page.locator('.reward-row').count()==2
        body=page.evaluate('buildTurnRequest("Test request")');payloads.append(body)
        assert {t['dimension'] for t in body['reward_plan']['terms']}=={'vn:AROU','eiv:Anger'}
        page.locator('#rewardmode').select_option('adaptive_auto')
        payloads.append(page.evaluate('buildTurnRequest("Test request")'))
        page.locator('#burstmode').select_option('typed');page.locator('#burstsource').select_option('manual')
        assert page.locator('#burstmanual').is_visible()
        assert page.locator('#bursttiming').input_value()=='clock' and page.locator('#bursttiming').is_disabled()
        for _ in range(7):page.locator('#burstadd').click()
        assert page.locator('#burstadd').is_disabled() and page.locator('.burst-row').count()==8
        body=page.evaluate('buildTurnRequest("Test request")');payloads.append(body)
        assert len(body['burst']['targets'])==8
        page.locator('#burstsource').select_option('script')
        assert page.locator('#bursttiming').input_value()=='words' and page.locator('#bursttiming').is_enabled()
        page.locator('#temperature').evaluate('(el)=>{el.value=0;el.dispatchEvent(new Event("input",{bubbles:true}));}')
        assert page.locator('#tv').inner_text()=='0.00'
        page.locator('#cfg').uncheck();body=page.evaluate('buildTurnRequest("Test request")');payloads.append(body)
        assert body['temperature']==0 and body['guidance']==1
        page.locator('#ref-toggle').click()
        page.locator('#fileA').set_input_files(str(ROOT/'performance/benchmark-reference.wav'))
        page.wait_for_function('document.getElementById("reference").value==="mock-ref"')
        assert page.locator('#voice-choice').input_value()=='custom'
        assert 'eigene Referenz' in page.locator('#configuration-summary').inner_text()
        assert page.evaluate('buildTurnRequest("Test").reference_id')=='mock-ref'
        assert page.locator('#vc').is_enabled()
        page.locator('#vc').check()
        assert page.evaluate('buildTurnRequest("Custom VC").voice_conversion')
        page.locator('#voice-choice').select_option('emolia_c1699')
        voice_request=page.evaluate('buildTurnRequest("Test")');payloads.append(voice_request)
        assert voice_request['voice_mode']=='profile' and voice_request['reference_id'] is None
        page.locator('#voice-choice').select_option('none')
        assert page.evaluate('buildTurnRequest("Test").reference_id') is None
        assert page.locator('#vc').is_disabled() and not page.locator('#vc').is_checked()
        page.locator('#voice-choice').select_option('custom')
        assert page.locator('#vc').is_enabled()
        page.locator('#vc').check()
        page.locator('#deleteA').click();page.wait_for_function('document.getElementById("reference").value===""')
        assert page.locator('#vc').is_disabled() and not page.locator('#vc').is_checked()
        page.locator('#cfg').check();page.locator('#recommended').click()
        assert page.locator('#temperature').input_value()=='1' and page.locator('#g').input_value()=='2'
        assert page.evaluate('buildTurnRequest("Recommended ranking").ranking')=={'mode':'ears_mix','model':'medium','metric':'ce','weight':.5}
        recommended=page.evaluate('buildTurnRequest("Recommended defaults")')
        assert recommended['generation']=='best' and recommended['n']==3 and recommended['streaming']
        assert recommended['adapters']=={'id':None,'weight':1,'ref15':False,'ref15_weight':1}
        assert recommended['chunk_frames']==20
        page.locator('#reset').click();assert page.locator('.take').count()==0
        page.locator('#settings-panel').evaluate('(el)=>el.scrollTop=0')
        page.screenshot(path=str(ROOT/'test-results/ui-layout-desktop.png'))
        for viewport in [{'width':1024,'height':768},{'width':390,'height':844},{'width':320,'height':700}]:
            page.set_viewport_size(viewport)
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),viewport
            assert page.locator('#q').is_visible()
        mobile=context.new_page()
        mobile.set_viewport_size({'width':390,'height':844});mobile.goto(ORIGIN)
        mobile.wait_for_function('ready && dimensionCatalog.length===112')
        assert mobile.locator('#reference-settings').bounding_box()['y']<mobile.locator('#conversation-panel').bounding_box()['y']
        assert mobile.locator('#voice-choice').bounding_box()['y']<844
        mobile.locator('#lora-open').click()
        assert mobile.locator('#lora-dialog').is_visible()
        mobile.locator('#lora-list input[value="original_rank128"]').check()
        mobile.screenshot(path=str(ROOT/'test-results/lora-overlay-mobile.png'))
        assert mobile.evaluate('document.getElementById("lora-dialog").scrollWidth<=document.getElementById("lora-dialog").clientWidth')
        mobile.locator('#lora-cancel').click()
        assert mobile.locator('#generation-settings').get_attribute('open') is not None
        assert mobile.locator('#reward-settings').get_attribute('open') is None
        mobile.locator('.header-nav a[href="#generation-settings"]').click()
        mobile.get_by_role('button',name='Beam Search aktivieren').click()
        assert mobile.locator('#searchn').is_enabled()
        mobile.locator('.settings-nav a[href="#reward-settings"]').click()
        mobile.locator('#ranking-mode').select_option('legacy')
        assert mobile.locator('#rewardmode').is_visible()
        mobile.locator('.settings-nav a[href="#burst-settings"]').click()
        assert mobile.locator('#burstmode').is_visible()
        mobile.locator('.header-nav a[href="#generation-settings"]').click()
        mobile.screenshot(path=str(ROOT/'test-results/ui-layout-mobile.png'))
        assert not errors,errors
        # API failure must not hide the selectable search UI or pretend ready.
        failed=browser.new_context(viewport={'width':1440,'height':1050})
        def unavailable(route):
            path=urlsplit(route.request.url).path
            if path.startswith('/api/'):route.fulfill(status=503,json={'detail':'Unavailable fixture'})
            else:handler(route)
        failed.route('**/*',unavailable);offline=failed.new_page();offline.goto(ORIGIN)
        offline.wait_for_function('document.getElementById("catalog-status").textContent.includes("nicht erreichbar")')
        assert offline.locator('#send').is_disabled()
        offline.get_by_role('button',name='Beam Search aktivieren').click()
        assert offline.locator('#searchn').is_enabled()
        failed.close();browser.close()
    result={'verification':'Real Chromium with fully mocked API; no live GPU generation or benchmark',
        'search_combinations':64,'posted_modes':[r['generation'] for r in posts],
        'catalog_dimensions':len(catalog['dimensions']),'lora_overlay':{'dpo_adapters':6,'ref15_adapters':1,'presets':4,'reference_voices':10,'selection':chosen,'cancel_clear_escape_mobile':'passed'},'viewport_widths':[1440,1024,390,320],
        'payloads':payloads+posts,'javascript_errors':errors}
    (ROOT/'test-results/ui-layout.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='payloads'}),flush=True)


if __name__=='__main__':main()
