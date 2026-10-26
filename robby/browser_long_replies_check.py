"""HTTPS CPU Chrome: a complete eight-line spoken poem, then a continuation."""
import asyncio,json,time
from pathlib import Path
from playwright.async_api import async_playwright

ROOT=Path(__file__).resolve().parent
URL='https://arrive-locations-parade-research.trycloudflare.com'

def compact_receipt(report):
    """This is a content regression receipt; GPU profiling lives elsewhere."""
    result=json.loads(json.dumps(report))
    for turn in result.get('turns',[]):turn.get('metrics',{}).pop('tts_server_stages',None)
    result['events']=[{k:e[k] for k in ('type','at','text','sentence_index','sentence_text','reply_id','error') if k in e} for e in result.get('events',[])]
    return result

async def main():
    report={'passed':False,'url':URL,'physical_microphone':False,'scope':'Typed requests, real local LLM/TTS and AudioContext playback','started':time.time()}
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=[
            '--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page(viewport={'width':1280,'height':1000});errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script("localStorage.setItem('robby.llmSelection',JSON.stringify({model:'local',provider:'local'}));localStorage.setItem('robby.cameraAuto','false');localStorage.setItem('robby.avatar-idle-enabled','false');localStorage.setItem('robby.avatar-pet-invites-enabled','false')")
        try:
            await page.goto(URL+'/desk-robot/')
            await page.wait_for_function('window.robo && !document.querySelector("#call").disabled',timeout=60000)
            await page.evaluate('''()=>{window.longEvents=[];window.longAudio=[];
              const old=robo.on.message,play=robo.play.bind(robo);
              robo.on.message=m=>{if(['reply','pipeline_metrics','turn_done','error','busy','barge'].includes(m.type))longEvents.push({...m,at:performance.now()});old(m)};
              robo.play=raw=>{longAudio.push({at:performance.now(),samples:(raw.byteLength-4)/2});play(raw)};
            }''')
            for index,(text,count) in enumerate([
                ('Schreibe ein lustiges Gedicht mit acht Zeilen über einen kleinen Roboter. Keine Einleitung.',8),
                ('Schreib noch eine zusätzliche Strophe mit vier Zeilen dazu. Bleibe bei unserem kleinen Roboter.',4)]):
                await page.locator('#message').fill(text);await page.locator('#form button').click()
                await page.wait_for_function(f'longEvents.filter(e=>e.type==="turn_done").length>={index+1}',timeout=90000)
                await page.wait_for_timeout(200)
                events=await page.evaluate('longEvents');previous=[i for i,e in enumerate(events) if e['type']=='turn_done']
                begin=0 if index==0 else previous[index-1]+1
                current=events[begin:previous[index]+1];replies=[e for e in current if e['type']=='reply']
                assert len(replies)==count,[(r.get('sentence_index'),r.get('text')) for r in replies]
                assert len([line for line in replies[-1]['text'].splitlines() if line.strip()])==count,replies[-1]['text']
                metrics=next(e['metrics'] for e in current if e['type']=='pipeline_metrics')
                assert metrics['response_complete'] and metrics['sentences']==count,metrics
                assert not [e for e in current if e['type'] in ('error','busy','barge')],current
                bubbles=await page.locator('.bubble.assistant').all_text_contents()
                assert bubbles[-1]==replies[-1]['text'],bubbles
                report.setdefault('turns',[]).append({'request':text,'reply':replies[-1]['text'],
                    'streamed_parts':len(replies),'metrics':metrics})
            audio=await page.evaluate('longAudio')
            report['audio_chunks']=len(audio);report['total_audio_s']=sum(c['samples'] for c in audio)/24000
            report['errors']=errors;report['events']=await page.evaluate('longEvents')
            assert len(audio)>=12 and not errors
            report['passed']=True
        except Exception as exc:
            report['error']=str(exc);report['errors']=errors
            try:report['events']=await page.evaluate('window.longEvents||[]')
            except Exception:pass
        finally:
            await page.evaluate('window.robo?.stop()');await browser.close()
    report['finished']=time.time();(ROOT/'static/long-replies-browser.json').write_text(json.dumps(compact_receipt(report),indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in report.items() if k!='events'},indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':asyncio.run(main())
