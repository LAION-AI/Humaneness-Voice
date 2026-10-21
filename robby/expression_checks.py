"""Real Chromium geometry, interpolation, playback cue and live Luna checks.

uv run --with playwright --python /path/to/python expression_checks.py [--live]
Synthetic typed requests only, no private microphone/camera recordings.
"""
import argparse,asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright

async def run(args):
    logs=Path(__file__).parent/'logs';logs.mkdir(exist_ok=True)
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,
            args=['--no-sandbox','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page(viewport={'width':1440,'height':1100});errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        try:
            await page.goto(args.url+'/expressions/');await page.wait_for_function('window.robotGallery')
            result=await page.evaluate('''async()=>{
             const {profiles,pose,mix,geometry,KEYPOINT_IDS}=window.robotGallery;
             const assert=(yes,msg)=>{if(!yes)throw Error(msg)};
             assert(profiles.length===40,'Forty profiles missing');assert(KEYPOINT_IDS.length===101,'Anchor count');
             assert(new Set(KEYPOINT_IDS).size===101,'Duplicate anchors');
             const points=profiles.map(p=>geometry(pose(p.slug,1,0)).points.flat());
             assert(points.every(p=>p.length===202&&p.every(Number.isFinite)),'Invalid coordinates');
             assert(new Set(points.map(p=>JSON.stringify(p))).size===40,'Identical archetypes');
             let pairs=0,minDistance=Infinity;
             for(let i=0;i<40;i++)for(let j=i+1;j<40;j++){
              const distance=Math.sqrt(points[i].reduce((a,v,k)=>a+(v-points[j][k])**2,0)/202);minDistance=Math.min(minDistance,distance);
              for(const t of [.25,.5,.75])assert(geometry(mix(pose(profiles[i].slug),pose(profiles[j].slug),t)).points.flat().every(Number.isFinite),'Invalid mixture');
              pairs++;
             }
             const {ActingTimeline}=await import('/static/emotions.js');const tl=new ActingTimeline();
             tl.set(7,{source:'test',sentences:[{emotion:'anger',start_s:0,end_s:1},{emotion:'relief',start_s:1,end_s:2}]});
             assert(tl.at({id:7,sample:12000}).emotion==='anger','Early cue');
             assert(tl.at({id:7,sample:36000}).emotion==='relief','Late cue');
             assert(tl.at(null)===null,'Buffer gap advanced plan');tl.enabled=false;assert(tl.at({id:7,sample:12000})===null,'Toggle ignored');
             tl.enabled=true;tl.clear();assert(tl.at({id:7,sample:12000})===null,'Interrupted cue survived');
             return {profiles:40,unique_archetype_geometries:40,keypoints:101,mixture_pairs:pairs,mixture_positions:3,
                min_pair_rms_pixels:minDistance,playback_cues:true,buffer_gap_holds:true,cancellation_clears:true,
                human_recognition_accuracy:null};}''')
            await page.screenshot(path=str(logs/'expressions-gallery.png'),full_page=True)
            await page.select_option('#a','anger');await page.select_option('#b','surprise');await page.fill('#blend','0.5')
            await page.check('#points');await page.wait_for_timeout(700)
            result['interactive_blend_and_anchors']=await page.evaluate('''({blend:document.querySelector('#blend-value').textContent,
              head:document.querySelectorAll('#keypoints circle').length,body:document.querySelectorAll('#body-keypoints circle').length,
              visible:document.querySelector('#body-keypoints').getAttribute('opacity')==='1'})''')
            assert result['interactive_blend_and_anchors']=={'blend':'50%','head':89,'body':12,'visible':True}
            await page.locator('.expression-lab').screenshot(path=str(logs/'expression-blend-keypoints.png'));await page.uncheck('#points')
            for emotion in ['anger','sadness','surprise','teasing','disgust','contentment']:
                await page.select_option('#a',emotion);await page.fill('#blend','0');await page.wait_for_timeout(700)
                await page.locator('.expression-lab').screenshot(path=str(logs/('expression-'+emotion+'.png')))
            await page.set_viewport_size({'width':390,'height':844})
            result['gallery_mobile_no_overflow']=await page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            await page.goto(args.url+'/');await page.wait_for_function('window.robotFace&&window.robo')
            result['no_heart_or_cross_in_robot']=await page.evaluate('''!document.querySelector('#avatar svg').outerHTML.match(/heart|cross|M535 565/)''')
            await page.select_option('#expression','anger');await page.wait_for_timeout(500)
            result['manual_preview']=await page.locator('#acting-status').text_content()
            await page.select_option('#expression','auto')
            await page.screenshot(path=str(logs/'robby-expressive-mobile.png'),full_page=True)
            if args.live:
                await page.fill('#message','Give me two very short sentences: first very angry, then relieved. Make the change clear in your acting directions.');await page.click('button[type=submit]')
                await page.wait_for_function('window.robotFace.timeline.plans.size>0',timeout=90000)
                result['luna_plans']=await page.evaluate('Array.from(window.robotFace.timeline.plans.entries())')
                await page.wait_for_function('window.robo.metrics.chunks>0',timeout=90000)
                await page.wait_for_function('window.robo.clock.position()!==null&&document.querySelector("#acting-status").textContent.includes("Luna SCRIPT")',timeout=30000)
                result['live_first_playback']=await page.evaluate('({chunks:robo.metrics.chunks,position:robo.clock.position(),face:document.querySelector("#acting-status").textContent})')
                await page.wait_for_function('document.querySelector("#acting-status").textContent.includes("sentence 2")',timeout=30000)
                result['live_second_sentence']=await page.locator('#acting-status').text_content()
                await page.screenshot(path=str(logs/'robby-luna-relief.png'))
                await page.click('#interrupt');await page.wait_for_timeout(150)
                result['live_interruption_clears']=await page.evaluate('robotFace.timeline.plans.size===0&&robo.clock.position()===null')
                await page.evaluate('robo.stop()')
            result['page_errors']=errors
            assert not errors and result['gallery_mobile_no_overflow'] and result['no_heart_or_cross_in_robot'],result
            (logs/('expression-checks-live.json' if args.live else 'expression-checks.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
        finally:await browser.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8793');parser.add_argument('--live',action='store_true');asyncio.run(run(parser.parse_args()))
