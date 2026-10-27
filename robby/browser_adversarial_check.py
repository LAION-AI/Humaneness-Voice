"""Real browser/ASR/identity/LLM/tools/TTS stateful stress tests, no hidden retries.

Digital 16-kHz microphone injection uses the real client's playback accounting.
The suite does not test hardware capture/acoustic echo. Every attempt gets a
new directory, full synthetic events, input hashes, source/model inventory and
explicit failures. A failed setup blocks its dependent steps, never passes them.
"""
import argparse, asyncio, base64, datetime, fcntl, hashlib, json, os, re, time, uuid
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
from playwright.async_api import async_playwright
from adversarial_audio import SR, activity, judge_text, make_bed, normalize, rms, sha_audio, soundscape, wilson, word_error_rate, zero_event_rate_bound
from adversarial_scenarios import scenarios, validate_plan
from generate_adversarial_fixtures import decode, fixture_root
from run_voice_assistant_checks import deployed_source_hash, model_inventory, public_state, source_files, digest

ROOT=Path(__file__).resolve().parent
STORY='Erzähle bitte eine lange Geschichte mit zwanzig Sätzen über das Kätzchen Lotti, das einen roten Drachen sucht. Erzähle direkt die ganze Geschichte.'


class BrowserSession:
    def __init__(self,page,clips,manifest,report,args):
        self.page=page;self.clips=clips;self.manifest=manifest;self.report=report;self.args=args
        self.samples=0;self.bed_rms=0.;self.backgrounds=[clips['background'],clips['selma-followup-clear']]
        self.previous='';self.owner=None;self.errors=[]
        self.wait_bed=None;self.bed_offset=0

    async def open(self):
        await self.page.add_init_script('''localStorage.setItem('robby.autoStart','false');localStorage.setItem('robby.cameraAuto','false');
            localStorage.setItem('robby.avatar-idle-enabled','false');localStorage.setItem('robby.avatar-pet-invites-enabled','false');
            localStorage.setItem('robby.focusBalanceVersion','2');localStorage.setItem('robby.pauseBalanceVersion','2');'''
            +f'localStorage.setItem("robby.focusThreshold","{self.args.threshold}");localStorage.setItem("robby.speechPauseMs","{self.args.pause_ms}");')
        focus=getattr(self.args,'focus',True)
        await self.page.add_init_script('localStorage.setItem("robby.speakerModeVersion","3");localStorage.setItem("robby.conversationFocus",'+json.dumps(str(focus).lower())+');')
        model=getattr(self.args,'asr_model','nemotron')
        await self.page.add_init_script('localStorage.setItem("robby.asrDefaultVersion","2");localStorage.setItem("robby.asrModel",'+json.dumps(model)+');')
        self.page.on('pageerror',lambda e:self.errors.append(str(e)))
        await self.page.goto(self.args.url.rstrip('/')+'/desk-robot/')
        await self.page.wait_for_function('window.robo&&robotFace.llmControls?.catalog&&!document.querySelector("#call").disabled',timeout=60000)
        await self.page.evaluate('''()=>{
            window.stressEvents=[];window.stressPCM=[];window.stressBusy=false;
            const old=robo.on.message,play=robo.play.bind(robo);
            robo.on.message=m=>{if(['endpoint','latency_start'].includes(m.type))stressBusy=true;
                if(['turn_done','error','reply_cancelled','barge'].includes(m.type))stressBusy=false;
                if(!['state','speech_activity','input_status','sync'].includes(m.type))stressEvents.push({...m,at:performance.now()});old(m)};
            robo.play=x=>{stressPCM.push({at:performance.now(),bytes:x.byteLength});play(x)};
        }''')
        await self.page.evaluate('robo.start(false)')
        await self.wait('stressEvents.some(e=>e.type==="ready")',15)
        await self.wait('robotFace.asrControls.active==='+json.dumps(model)+'&&!document.querySelector("#asr-model-status").textContent.startsWith("Loading")',30)
        await self.page.evaluate('robo.send("interaction:helpful:off")')

    async def mark(self):
        return await self.page.evaluate('({event:stressEvents.length,pcm:stressPCM.length,at:performance.now()})')

    async def events(self,mark):
        return await self.page.evaluate('(n)=>stressEvents.slice(n)',mark['event'])

    async def feed(self,x,record=False):
        x=np.asarray(x,dtype='<f4');start_sample=self.samples;self.samples+=len(x)
        encoded=base64.b64encode(x.tobytes()).decode()
        timing=await self.page.evaluate('''async encoded=>{
            const x=new Float32Array(Uint8Array.from(atob(encoded),c=>c.charCodeAt(0)).buffer);
            const start=performance.now();let maxLag=0,lag=0,packets=0;
            for(let at=0;at<x.length;at+=320){
                const late=Math.max(0,performance.now()-start-at/16);maxLag=Math.max(maxLag,late);lag+=late;packets++;
                robo.mic(robo.ctx.currentTime,x.subarray(at,at+320));
                await new Promise(r=>setTimeout(r,Math.max(0,start+(at+320)/16-performance.now())));
            }
            return {browser_start_ms:start,browser_end_ms:performance.now(),packets,max_schedule_lag_ms:maxLag,mean_schedule_lag_ms:lag/packets};
        }''',encoded)
        if record:self.report['injections'].append({**timing,'stream_start_sample':start_sample,'samples':len(x),'sha256':sha_audio(x)})
        return {**timing,'stream_start_sample':start_sample}

    async def idle_audio(self,seconds=.4):
        n=round(seconds*SR)
        if self.bed_rms:
            if self.wait_bed is None:
                bed,_=make_bed(self.backgrounds,60*SR,self.args.seed)
                self.wait_bed=normalize(bed,self.bed_rms)
            x=self.wait_bed[(np.arange(n)+self.bed_offset)%len(self.wait_bed)]
            self.bed_offset+=n
        else:x=np.zeros(n,np.float32)
        return await self.feed(x)

    async def wait(self,predicate,seconds=45):
        for _ in range(round(seconds/.4)):
            if await self.page.evaluate(predicate):return
            await self.idle_audio()
        raise TimeoutError('Timed out waiting for '+predicate[:120])

    async def settle(self,seconds=90):
        stable=0.
        for _ in range(round(seconds/.4)):
            await self.idle_audio()
            idle=await self.page.evaluate('!stressBusy&&!robo.sources.length')
            stable=stable+.4 if idle else 0.
            if stable>=1.2:return
        raise TimeoutError('Replies did not finish within the playback deadline')

    async def typed_story(self):
        await self.settle()
        mark=await self.mark()
        await self.page.fill('#message',STORY);await self.page.locator('#form').evaluate('f=>f.requestSubmit()')
        await self.wait('robo.sources.length>0',40)
        return mark

    async def cleanup_reply(self):
        # Called only AFTER observations have been archived. Never count this as a voice Stop.
        await self.page.locator('#interrupt').evaluate('e=>e.click()')
        self.bed_rms=0
        await self.idle_audio(1.)

    async def speak(self,spec,scenario,setup=False):
        await self.settle()
        mark=await self.mark();name=spec['clip'];clip=self.clips[name]
        mixed,meta,bed=soundscape(clip,self.backgrounds,seed=self.args.seed+len(self.report['injections']),
            snr_db=spec.get('snr_db',None),speed=spec.get('speed',1.),reverb_s=spec.get('reverb_s',0.))
        self.bed_rms=meta['bed_rms'];self.wait_bed=bed;self.bed_offset=len(bed)
        timing=await self.feed(mixed,record=True)
        error=None
        try:
            await self.wait(f'stressEvents.slice({mark["event"]}).some(e=>e.type==="endpoint")',15)
            await self.wait(f'stressEvents.slice({mark["event"]}).some(e=>e.type==="turn_done")',90)
            await self.settle()
        except TimeoutError as exc:error=str(exc)
        es=await self.events(mark);ends=[e for e in es if e['type']=='endpoint'];replies=[e for e in es if e['type']=='reply']
        # Final cumulative metadata per reply, never double-count sentence prefixes.
        latest={e.get('reply_id'):e for e in replies};text='\n'.join(e.get('text','') for e in latest.values())
        transcript=' '.join(e.get('transcript','') for e in ends if e.get('by')!='agreed_interruption')
        wer=word_error_rate(self.manifest[name]['text'],transcript)
        checks=judge_text(text,spec.get('expect',{}),self.previous)
        expected_speaker=spec.get('speaker',scenario['speaker'])
        # A two-sentence recording may contain a real pause exceeding the user's
        # endpoint setting. Count distinct complete submissions, not the file as
        # one mandatory turn. Still reject duplicated text and extra TV turns.
        texts=[re.sub(r'\W+',' ',e.get('transcript','').casefold()).strip() for e in ends]
        checks.update(submission_count=1<=len(ends)<=spec.get('max_submissions',2 if setup else 1),
            no_duplicate_submission=len(texts)==len(set(texts)),speaker=bool(ends) and all(e.get('speaker_name')==expected_speaker for e in ends),
            transcript=wer<=.35,completed=not error,real_audio=await self.page.evaluate('stressPCM.length')>mark['pcm'],
            no_service_errors=not any(e['type']=='error' for e in es),no_unwanted_cancel=not any(e['type']=='reply_cancelled' for e in es))
        expect=spec.get('expect',{})
        sources=[s for e in latest.values() for s in e.get('sources',[])]
        if expect.get('source_host'):
            host=expect['source_host']
            checks['verified_source_metadata']=any((urlparse(s.get('url','')).hostname or '') in (host,'www.'+host,'de.'+host,'en.'+host) for s in sources)
            checks['real_tool_route']=any(e.get('timing',{}).get('tool_director_ms',0)>0 for e in replies)
        if expect.get('min_poem_lines'):
            lines=[line.strip() for line in text.splitlines() if line.strip()]
            checks['requested_poem_lines']=len(lines)==expect['min_poem_lines']
        if expect.get('permission'):
            checks['permission_applied']=any(e['type']=='speaker_permission' and e.get('applied') and e.get('action')==expect['permission'] for e in es)
        states=[e.get('conversation_focus',{}) for e in es if e['type']=='identity']
        members=[p['name'] for p in states[-1].get('participants',[])] if states else []
        if 'participants' in expect:checks['participants']=set(expect['participants'])<=set(members)
        if 'participants_exact' in expect:checks['participants_exact']=set(expect['participants_exact'])==set(members)
        record={'id':spec['id'],'kind':'setup' if setup else 'speak','required':spec.get('required',True),
            'input_script':self.manifest[name]['text'],'expected_speaker':expected_speaker,'mixture':meta,'injection':timing,
            'transcript':transcript,'wer':wer,'reply':text,'sources':sources,'participants':members,'checks':checks,
            'passed':all(checks.values()),'error':error,'events':es}
        if ends:
            record['energy_end_to_submit_ms']=ends[0]['at']-timing['browser_start_ms']-meta['target_activity_s'][1]*1000
        pcm=await self.page.evaluate('(i)=>stressPCM.slice(i)',mark['pcm'])
        if pcm:record['energy_end_to_first_pcm_ms']=pcm[0]['at']-timing['browser_start_ms']-meta['target_activity_s'][1]*1000
        if error:await self.cleanup_reply()
        self.previous=text
        return record

    async def control(self,spec):
        kind=spec['kind'];story_mark=None
        if kind!='reject':story_mark=await self.typed_story()
        else:await self.settle()
        if kind=='stop':
            desired=spec['phase_ms']*16
            pad=(desired-self.samples%5120)%5120
            if pad:await self.idle_audio(pad/SR)
        mark=await self.mark();clip=self.clips[spec['clip']]*spec.get('gain',1.)
        if kind=='stop':
            wave,meta,bed=soundscape(clip,self.backgrounds,seed=self.args.seed+len(self.report['injections']),
                                  snr_db=spec.get('snr_db',12),onset_s=0.,tail_s=1.2)
            self.bed_rms=meta['bed_rms'];self.wait_bed=bed;self.bed_offset=len(bed)
        else:
            wave=np.concatenate((clip,np.zeros(round(spec.get('seconds_after',1.2)*SR),np.float32)))
            mask=np.flatnonzero(activity(clip))
            meta={'target_activity_s':[int(mask[0])/SR,(int(mask[-1])+1)/SR] if len(mask) else None,
                  'gain':spec.get('gain',1.),'duration_s':len(wave)/SR,'mixture_sha256':sha_audio(wave)}
            self.bed_rms=0.
        timing=await self.feed(wave,record=True);error=None
        if kind=='stop':
            try:await self.wait(f'stressEvents.slice({mark["event"]}).some(e=>e.type==="reply_cancelled")',5)
            except TimeoutError as exc:error=str(exc)
        es=await self.events(mark);cancel=[e for e in es if e['type']=='reply_cancelled'];ends=[e for e in es if e['type']=='endpoint']
        checks={'no_service_errors':not any(e['type']=='error' for e in es)}
        if kind=='stop':
            checks['cancelled']=bool(cancel)
            checks['actual_phase']=timing['stream_start_sample']%5120==spec['phase_ms']*16
            checks['playback_cleared']=not await self.page.evaluate('robo.sources.length>0')
        else:
            checks.update(no_abort=not cancel,no_conversational_reply=not ends,
                          no_permission_change=not any(e['type']=='speaker_permission' and e.get('applied') for e in es))
            if kind=='negative_control':
                checks['reply_was_still_active']=bool(await self.page.evaluate('stressBusy||robo.sources.length'))
        record={'id':spec['id'],'kind':kind,'required':True,'input_script':self.manifest[spec['clip']]['text'],
                'mixture':meta,'injection':timing,'events':es,'checks':checks,'passed':all(checks.values()),'error':error}
        if kind!='stop':record['negative_exposure_seconds']=len(wave)/SR
        if cancel:
            record['capture_to_cancel_ms']=cancel[0]['at']-timing['browser_start_ms']
            record['energy_end_to_cancel_ms']=record['capture_to_cancel_ms']-meta['target_activity_s'][1]*1000
            if kind=='stop':
                checks['cancel_within_1500ms_energy_end']=record['energy_end_to_cancel_ms']<=1500
                record['passed']=all(checks.values())
        if kind!='reject':await self.cleanup_reply()
        return record


async def run(args):
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    out=ROOT/'logs/adversarial-runs'/stamp;out.mkdir(parents=True)
    plan=validate_plan(scenarios());selected=args.scenarios.split(',') if args.scenarios!='all' else [s['id'] for s in plan]
    if set(selected)-{s['id'] for s in plan}:raise ValueError('Unknown scenario')
    plan=[s for s in plan if s['id'] in selected]
    (out/'plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False))
    manifest=json.loads((args.fixtures/'manifest.json').read_text());rows={r['name']:r for r in manifest['cases']}
    pin=json.loads((ROOT/'static/adversarial-fixtures.json').read_text())
    if manifest!=pin:raise RuntimeError('Fixture manifest differs from checked-in pin')
    clips={}
    for name,row in rows.items():
        path=args.fixtures/row['file']
        if digest(path)!=row['sha256']:raise RuntimeError('Changed fixture '+name)
        clips[name]=decode(path)
        if sha_audio(clips[name])!=row['decoded_normalized_sha256']:raise RuntimeError('Decoded fixture drift '+name)
    report={'schema_version':2,'run_id':stamp,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Public synthetic digital microphone injection in CPU Chrome; real ASR/ECAPA/LLM/tools/TTS. No physical capture, real-room echo or human accuracy certification.',
        'configuration':{'seed':args.seed,'threshold':args.threshold,'pause_ms':args.pause_ms,'asr_model':args.asr_model,'url':args.url,'scenarios':selected},
        'plan_sha256':digest(out/'plan.json'),'fixture_manifest_sha256':digest(args.fixtures/'manifest.json'),
        'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in source_files()},
        'server_before':public_state(args.url),'model_inventory_before':model_inventory(),
        'scenarios':[],'injections':[],'passed':False,'all_cases_passed':False}
    receipt=out/'result.json'
    def save():receipt.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    save()
    if report['server_before'].get('source_sha256')!=deployed_source_hash():raise RuntimeError('Deployed source differs; no restart attempted')
    try:
        async with async_playwright() as p:
            browser=await p.chromium.launch(executable_path=os.environ.get('ROBBY_TEST_CHROME','/usr/bin/google-chrome'),headless=True,
                args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
            report['browser_version']=browser.version
            for scenario in plan:
                entry={'id':scenario['id'],'description':scenario['description'],'steps':[],'passed':False};report['scenarios'].append(entry)
                context=await browser.new_context();page=await context.new_page()
                s=BrowserSession(page,clips,rows,report,args)
                if scenario['speaker']=='Selma':s.backgrounds=[clips['background'],clips['christoph-followup']]
                try:
                    await s.open()
                    setup=await s.speak({'id':'enroll','clip':scenario['intro'],'expect':{'min_words':2}},scenario,setup=True)
                    entry['setup']=setup;save()
                    if not setup['passed']:raise RuntimeError('Enrollment setup failed; dependent steps blocked')
                    for spec in scenario['steps']:
                        try:
                            if spec.get('kind','speak')=='speak':row=await s.speak(spec,scenario)
                            else:row=await s.control(spec)
                        except Exception as exc:
                            row={'id':spec['id'],'required':spec.get('required',True),'passed':False,'error':type(exc).__name__+': '+str(exc)[:350]}
                            await s.cleanup_reply()
                        entry['steps'].append(row);save()
                        print(json.dumps({'scenario':scenario['id'],'case':row['id'],'passed':row['passed'],'checks':row.get('checks'),
                                          'transcript':row.get('transcript'),'reply':row.get('reply'),'error':row.get('error')},ensure_ascii=False),flush=True)
                    entry['passed']=all(x['passed'] for x in entry['steps'] if x.get('required',True)) and not s.errors
                except Exception as exc:
                    entry['error']=type(exc).__name__+': '+str(exc)[:400]
                    completed={x['id'] for x in entry['steps']}
                    entry['blocked_steps']=[x['id'] for x in scenario['steps'] if x['id'] not in completed]
                finally:
                    entry['page_errors']=s.errors
                    try:await page.evaluate('window.robo?.stop()')
                    except Exception:pass
                    await context.close();save()
                await asyncio.sleep(1.)
            await browser.close()
        report['server_after']=public_state(args.url);report['model_inventory_after']=model_inventory()
        report['source_unchanged']=report['source_sha256']=={str(p.relative_to(ROOT)):digest(p) for p in source_files()}
        report['runtime_unchanged']=report['server_after']['source_sha256']==report['server_before']['source_sha256']
        report['models_unchanged']=report['model_inventory_after']==report['model_inventory_before']
        report['passed']=all(s['passed'] for s in report['scenarios']) and all(report[k] for k in ('source_unchanged','runtime_unchanged','models_unchanged'))
        steps=[x for s in report['scenarios'] for x in s['steps']]
        report['all_cases_passed']=report['passed'] and all(x['passed'] for x in steps)
        negative=[x for x in steps if 'negative_exposure_seconds' in x]
        exposure=sum(x['negative_exposure_seconds'] for x in negative)
        false_events=sum(any(not v for k,v in x['checks'].items() if k in ('no_abort','no_conversational_reply','no_permission_change')) for x in negative)
        report['summary']={'executed_steps':len(steps),'passed_steps':sum(x['passed'] for x in steps),
            'blocked_steps':sum(len(s.get('blocked_steps',[])) for s in report['scenarios']),
            'step_pass_wilson95_descriptive_only':wilson(sum(x['passed'] for x in steps),len(steps)),
            'negative_exposure_seconds':exposure,'negative_windows_with_false_events':false_events,
            'zero_event_poisson_upper95_per_hour':zero_event_rate_bound(exposure) if not false_events else None,
            'statistics_limit':'Correlated synthetic development cases, not independent held-out trials. Do not infer field accuracy from these intervals.'}
    except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)[:400]
    finally:
        report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
        (ROOT/'static/adversarial-check.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'run':str(out),'passed':report['passed'],'summary':report.get('summary')}),flush=True)
    return report['passed']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default=os.environ.get('ROBBY_TEST_URL','http://127.0.0.1:8793'))
    parser.add_argument('--fixtures',type=Path,default=fixture_root())
    parser.add_argument('--scenarios',default='all');parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threshold',type=float,default=float(os.environ.get('ROBBY_TEST_THRESHOLD','.28')))
    parser.add_argument('--pause-ms',type=int,default=int(os.environ.get('ROBBY_TEST_PAUSE_MS','700')))
    from asr_catalog import MODELS
    parser.add_argument('--asr-model',choices=list(MODELS),default=os.environ.get('ROBBY_TEST_ASR_MODEL','nemotron'))
    args=parser.parse_args()
    if not .2<=args.threshold<=.65 or not 200<=args.pause_ms<=2000:parser.error('Settings outside UI range')
    lock=None
    if os.environ.get('ROBBY_PROTOCOL_LOCK_HELD')!='1':
        lock_path=ROOT/'logs/voice-assistant-runs/.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
        lock=lock_path.open('w')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another protocol runner is active; no session was interrupted')
    if not asyncio.run(run(args)):raise SystemExit(1)
