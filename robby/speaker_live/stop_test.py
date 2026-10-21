"""Actual streamed replies, simulated DAC/echo, authorized and denied stop voices.

Only public synthetic fixtures. No private mic, no replacement ASR, no model
reload. Each case has a new live session; no retry is hidden in a pass.
"""
import argparse,asyncio,datetime,json,struct,time
from pathlib import Path
import aiohttp,numpy as np
from scipy.signal import resample_poly
from .live_test import audio
from speaker_v4.common import atomic
from final_verify import runtime_signature

FOLDER=Path('/mnt/nvme/robo-speaker-studies/speaker-live-results')

async def case(command,authorized=True,enroll_other=False,long_reading=False,feedback=False):
    events=[];segments=[];received={};stopped=set();done=[];first={};sample=0;reply_end=0;phase='intro';ready=None
    async with aiohttp.ClientSession() as client:
        async with client.ws_connect('http://127.0.0.1:8793/ws?synthetic_diagnostics=1') as ws:
            t0=time.monotonic()
            async def read():
                nonlocal reply_end,ready
                async for msg in ws:
                    if msg.type==aiohttp.WSMsgType.BINARY:
                        rid=struct.unpack_from('<i',msg.data)[0]
                        if not rid:continue
                        x=np.frombuffer(msg.data,'<i2',offset=4).astype(np.float32)/32768
                        pos=received.get(rid,0);received[rid]=pos+len(x)
                        wave=resample_poly(x,2,3).astype(np.float32)
                        start=max(sample+3840,reply_end);segments.append((start,start+len(wave),rid,pos,wave));reply_end=start+len(wave)
                        first.setdefault(rid,start/16000)
                    elif msg.type==aiohttp.WSMsgType.TEXT:
                        item=json.loads(msg.data)
                        if item.get('type')=='ready':ready=item
                        if item.get('type') in ('reply','endpoint','barge','turn_done','stop_command','stop_feedback','warning','ignored','error','synthetic_barge_proposal','synthetic_barge_verification','synthetic_stop_probe','synthetic_stop_decode','synthetic_semantic_decision'):
                            events.append({'at_s':round(sample/16000,3),'phase':phase,**item})
                        if item.get('type')=='barge':stopped.update(received)
                        if item.get('type')=='turn_done':done.append(item)
            reader=asyncio.create_task(read())
            async def feed(wave=None,seconds=None):
                nonlocal sample
                n=len(wave) if wave is not None else round(seconds*16000)
                for at in range(0,n,320):
                    count=min(320,n-at);mic=np.zeros(count,np.float32)
                    if wave is not None:mic+=wave[at:at+count]
                    left=sample-1920;right=left+count;cuts={0,count};active=[]
                    for a,b,rid,pos,x in segments:
                        if rid in stopped or b<=left or a>=right:continue
                        lo=max(a,left);hi=min(b,right);i=lo-left;j=hi-left
                        mic[i:j]+=x[lo-a:hi-a]*.05;active.append((i,j,rid,pos+round((lo-a)*1.5)));cuts.update((i,j))
                    edges=sorted(cuts)
                    for a,b in zip(edges,edges[1:]):
                        match=next(((rid,pos+round((a-i)*1.5)) for i,j,rid,pos in active if i<=a<j),None)
                        rid,pos=match if match else (0,-1)
                        pcm=np.clip(np.rint(mic[a:b]*32767),-32768,32767).astype('<i2')
                        await ws.send_bytes(struct.pack('<ii',rid,pos)+pcm.tobytes())
                    sample+=count
                    for rid,total in received.items():
                        if rid in stopped:continue
                        played=max((pos+round((min(sample,b)-a)*1.5) for a,b,r,pos,x in segments if r==rid and sample>=a),default=0)
                        if played:await ws.send_str(f'played:{rid}:{min(total,played)}')
                    await asyncio.sleep(max(0,t0+sample/16000-time.monotonic()))
            async def until(predicate,limit=40):
                for _ in range(round(limit/.32)):
                    if predicate():return
                    await feed(seconds=.32)
                raise TimeoutError('Synthetic stop test failed to advance')
            async def settle():
                # A real >1-s pause inside an intro can produce two legitimate
                # queued turns. Do not start a different user's test while
                # that second reply is still being prepared or played.
                stable=0.
                for _ in range(188):
                    await feed(seconds=.32)
                    pending=sum(e['type']=='endpoint' for e in events)>len(done)
                    stable=0. if pending else stable+.32
                    if stable>=1.92:return
                raise TimeoutError('Introduction replies did not settle')
            result={'command':command,'authorized':authorized,'other_enrolled':enroll_other,'long_reading':long_reading,'feedback_enabled':feedback,'passed':False}
            try:
                await until(lambda:ready is not None,limit=5)
                await ws.send_str('interaction:stop-feedback:'+('on' if feedback else 'off'))
                assert ready.get('identity_enabled') and ready.get('identity_mode')=='v4-long3','Expected hybrid speaker gate was not active; invalid test setup'
                await feed(seconds=1.28);await feed(audio('christoph-intro'));await until(lambda:len(done)>=1)
                await settle()
                assert not any(e['type']=='barge' for e in events),'Echo caused an early barge'
                if enroll_other:
                    await feed(seconds=1.2);await feed(audio('robert-intro'));await until(lambda:len(done)>=2)
                    await settle()
                    if long_reading:
                        # Re-establish the actual intended addressee before a
                        # long request. Retain failed split-onset attempts;
                        # never mistake a reply to Robert for a Christoph test.
                        before=len(done);await feed(audio('christoph-followup'));await until(lambda:len(done)>before);await settle()
                previous=len(first);phase='request';await feed(seconds=1.2);await feed(audio('christoph-long-reading' if long_reading else 'christoph-followup'))
                await until(lambda:len(first)>previous);rid=max(first)
                await until(lambda:sample/16000>=first[rid]+(6. if long_reading else .8),limit=15)
                reply=next((e for e in reversed(events) if e['type']=='reply'),{})
                assert reply.get('speaker_name')=='Christoph','Invalid setup: reply is not addressed to the expected current speaker'
                if long_reading:assert len(reply.get('text','').split())>=70,'Invalid setup: expected genuinely long reading'
                result['reading_words']=len(reply.get('text','').split());result['playback_before_command_s']=sample/16000-first[rid]
                phase='stop';begin=sample/16000;reply_count=sum(e['type']=='reply' for e in events)
                wave=audio(command);await feed(wave);await feed(seconds=8. if feedback else 5.)
                barges=[e for e in events if e['type']=='barge' and e['phase']=='stop']
                stop_events=[e for e in events if e['type']=='stop_command' and e['phase']=='stop']
                new_replies=sum(e['type']=='reply' for e in events)-reply_count
                result.update(command_start_s=begin,command_duration_s=len(wave)/16000,
                    interruption_latency_s=(barges[0]['at_s']-begin) if barges else None,
                    interrupted=bool(barges),stop_acknowledged=bool(stop_events),new_spoken_replies=new_replies,
                    actual_streamed_tts=True,feedback_events=[e for e in events if e['type']=='stop_feedback'],events=events)
                assert bool(barges)==authorized,'Allowed stop was missed or another person interrupted'
                if authorized:assert new_replies<=(1 if feedback else 0),'Stop triggered an unrequested spoken answer'
                if not authorized:
                    assert not stop_events,'Another person triggered stop-command acknowledgment'
                    assert new_replies==0,'Background stop or stale request triggered another spoken reply'
                assert not any(e['type']=='error' for e in events),'Runtime error'
                result['passed']=True
            except Exception as exc:result.update(error=str(exc),events=events)
            finally:
                await ws.close();reader.cancel();await asyncio.gather(reader,return_exceptions=True)
            return result

async def main(commands,keep_going=False,long_reading=False,feedback=False):
    root=Path(__file__).resolve().parent.parent
    record=json.loads((root/'.runtime/server.json').read_text())
    started=datetime.datetime.fromisoformat(record['started_utc'].replace('Z','+00:00')).timestamp()
    changed=[name for name in ('server.py','speaker_session.py','stop_commands.py','audio_sync.py','live_decisions.py','interaction_policy.py','conversation_director.py','tts_bridge.py') if (root/name).stat().st_mtime>started+1]
    if changed:raise RuntimeError('Invalid setup: running backend predates source edits: '+', '.join(changed))
    report={'started':time.time(),'physical_microphone_used':False,'public_synthetic_only':True,
        'owned_server_pid':record['pid'],'owned_server_started_utc':record['started_utc'],
        'runtime_source_sha256':runtime_signature(),
        'simulated_echo_delay_ms':120,'simulated_playback_buffer_ms':240,'echo_gain':.05,'passed':False,'cases':[]}
    target=FOLDER/'stop-commands-live.json'
    if target.exists():atomic(FOLDER/('stop-attempt-'+str(int(time.time()))+'.json'),json.loads(target.read_text()))
    atomic(target,report)
    async with aiohttp.ClientSession() as client:
        for _ in range(180):
            try:
                async with client.get('http://127.0.0.1:8793/api/state') as response:
                    if (await response.json()).get('ready'):break
            except aiohttp.ClientError:pass
            await asyncio.sleep(.5)
        else:raise RuntimeError('Owned Robo service did not become ready')
    for command in commands:
        authorized=command.startswith('christoph-');enrolled=command.startswith('robert-')
        result=await case(command,authorized,enrolled,long_reading,feedback);report['cases'].append(result);atomic(target,report)
        print(json.dumps({k:v for k,v in result.items() if k!='events'}),flush=True)
        if not result['passed'] and not keep_going:raise SystemExit('Stop case failed; retained receipt, no hidden retry')
        await asyncio.sleep(.5)
    unchanged=report['runtime_source_sha256']==runtime_signature()
    report.update(passed=unchanged and all(c['passed'] for c in report['cases']),source_unchanged=unchanged,finished=time.time());atomic(target,report)
    if not report['passed']:raise SystemExit('One or more stop cases failed; see retained receipts')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--commands',nargs='+',default=['christoph-stop','christoph-halt','christoph-quiet','christoph-stop-speaking','robert-stop','background-stop','robert-stop-short','background-stop-short'])
    p.add_argument('--continue-on-failure',action='store_true');p.add_argument('--long-reading',action='store_true');p.add_argument('--feedback',action='store_true')
    args=p.parse_args();asyncio.run(main(args.commands,args.continue_on_failure,args.long_reading,args.feedback))
