"""Real ASR/tool-call/TTS test: speak BEFORE an ongoing user fixture ends.

Public synthetic voices only, real-time microphone packets, virtual 240-ms DAC
and 120-ms/5% echo. No test-only rule injection or replacement model decisions.
"""
import argparse,asyncio,datetime,json,struct,time
from pathlib import Path
import aiohttp,numpy as np
from scipy.signal import resample_poly
from .live_test import audio
from speaker_v4.common import atomic
from final_verify import runtime_signature

FOLDER=Path('/mnt/nvme/robo-speaker-studies/speaker-live-results')

async def case(condition,visual=False):
    events=[];segments=[];received={};stopped=set();done=[];first={}
    sample=0;reply_end=0;phase='intro';ready=None;rules=[];enabled=True;visual_person=None
    async with aiohttp.ClientSession() as client:
        async with client.ws_connect('http://127.0.0.1:8793/ws?synthetic_diagnostics=1') as ws:
            t0=time.monotonic()
            async def read():
                nonlocal reply_end,ready,rules,enabled,visual_person
                async for msg in ws:
                    if msg.type==aiohttp.WSMsgType.BINARY:
                        rid=struct.unpack_from('<i',msg.data)[0]
                        if not rid:continue
                        x=np.frombuffer(msg.data,'<i2',offset=4).astype(np.float32)/32768
                        pos=received.get(rid,0);received[rid]=pos+len(x)
                        wave=resample_poly(x,2,3).astype(np.float32)
                        start=max(sample+3840,reply_end)
                        segments.append((start,start+len(wave),rid,pos,wave));reply_end=start+len(wave)
                        first.setdefault(rid,start/16000)
                    elif msg.type==aiohttp.WSMsgType.TEXT:
                        item=json.loads(msg.data);kind=item.get('type')
                        if kind=='ready':ready=item
                        if kind=='visual_person' and item.get('track_id')=='face-1':visual_person=item['person_id']
                        if kind=='interaction':rules=item.get('rules',[]);enabled=item.get('enabled',enabled)
                        if kind not in ('audio','partial'):
                            events.append({'at_s':round(sample/16000,3),'phase':phase,**item})
                        if kind=='barge':stopped.update(received)
                        if kind=='turn_done':done.append(item)
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
                    if visual and sample%1600<count:
                        face_id='face-2' if condition=='unknown' and phase=='probe' else 'face-1'
                        await ws.send_str('visual:'+json.dumps({'type':'frame','camera':'on','audio_s':sample/16000,
                            'faces':[{'id':face_id,'speaking':wave is not None,'confidence':.9 if wave is not None else 0}]}))
                        if visual_person and face_id=='face-1':await ws.send_str('visual:'+json.dumps({'type':'identify','track_id':'face-1','person_id':visual_person,'embedding_age_ms':0}))
                    for rid,total in received.items():
                        if rid in stopped:continue
                        played=max((pos+round((min(sample,b)-a)*1.5) for a,b,r,pos,x in segments if r==rid and sample>=a),default=0)
                        if played:await ws.send_str(f'played:{rid}:{min(total,played)}')
                    await asyncio.sleep(max(0,t0+sample/16000-time.monotonic()))
            async def until(predicate,limit=40):
                for _ in range(round(limit/.32)):
                    if predicate():return
                    await feed(seconds=.32)
                raise TimeoutError('Live proactive test failed to advance')
            result={'condition':condition,'passed':False}
            try:
                await until(lambda:ready is not None,5)
                assert ready.get('identity_enabled') and ready.get('identity_mode')=='v4-long3'
                assert ready.get('interaction',{}).get('enabled') is True,'Default proactive setting missing'
                await feed(seconds=1.28);await feed(audio('christoph-intro'));await until(lambda:len(done)>=1)
                phase='setup';await feed(seconds=1.2);await feed(audio('christoph-game'))
                await until(lambda:bool(rules) and len(done)>=2,50)
                installed=list(rules)
                if condition=='disabled':
                    await ws.send_str('interaction:off');await until(lambda:not enabled,5)
                phase='probe';await feed(seconds=1.2)
                fixture={'positive':'christoph-bananas','repeated':'christoph-bananas','disabled':'christoph-bananas',
                         'unrelated':'christoph-apples','unknown':'background-bananas'}[condition]
                wave=audio(fixture)
                if condition=='repeated':wave=np.tile(wave,3)
                begin=sample/16000;before=set(first)
                await feed(wave);end=sample/16000;await feed(seconds=7.)
                interrupts=[e for e in events if e['type']=='endpoint' and e.get('by')=='agreed_interruption' and e['phase']=='probe']
                replies=[e for e in events if e['type']=='reply' and e['phase']=='probe']
                proactive_replies=[e for e in replies if 'stop' in (e.get('text') or '').lower()]
                new_audio=[at for rid,at in first.items() if rid not in before]
                result.update(user_start_s=begin,user_end_s=end,fixture=fixture,rules=installed,
                    interrupted=bool(interrupts),decision_at_s=interrupts[0]['at_s'] if interrupts else None,
                    first_new_audio_s=min(new_audio) if new_audio else None,
                    spoken_before_user_end=bool(new_audio and min(new_audio)<end),
                    actual_streamed_tts=True,events=events)
                if condition in ('positive','repeated'):
                    assert interrupts,'No live rule interruption'
                    assert interrupts[0]['at_s']<end,'Decision waited for the user to finish'
                    assert result['spoken_before_user_end'],'No spoken audio before the ongoing utterance ended'
                    assert proactive_replies,'Missing exact requested spoken interjection'
                    assert len(replies)==len(interrupts)==len(proactive_replies),'Consumed monologue generated an unrequested ordinary reply'
                    if condition=='repeated':assert len(interrupts)>=2,'Fresh repeated topics stayed latched until silence'
                    assert not [e for e in events if e['type']=='barge' and e['phase']=='probe'],'Continuing speech cancelled the interjection'
                else:assert not interrupts,'Disabled/unrelated/unknown voice triggered a rule'
                assert not any(e['type']=='error' for e in events),'Runtime error'
                result['passed']=True
            except Exception as exc:result.update(error=str(exc),events=events)
            finally:
                await ws.close();reader.cancel();await asyncio.gather(reader,return_exceptions=True)
            return result

async def main(conditions,visual=False):
    root=Path(__file__).resolve().parent.parent
    record=json.loads((root/'.runtime/server.json').read_text())
    started=datetime.datetime.fromisoformat(record['started_utc'].replace('Z','+00:00')).timestamp()
    names=('server.py','speaker_session.py','stop_commands.py','audio_sync.py','live_decisions.py','interaction_policy.py','conversation_director.py','tts_bridge.py')
    changed=[name for name in names if (root/name).stat().st_mtime>started+1]
    if changed:raise RuntimeError('Running backend predates source edits: '+', '.join(changed))
    report={'runtime_source_sha256':runtime_signature(),'owned_server_pid':record['pid'],
        'physical_microphone_used':False,'public_synthetic_only':True,'simulated_echo_delay_ms':120,
        'simulated_playback_buffer_ms':240,'echo_gain':.05,'started':time.time(),'passed':False,'cases':[]}
    report['synthetic_visual_identity']=visual
    target=FOLDER/('proactive-visual-live.json' if visual else 'proactive-live.json')
    if target.exists():atomic(FOLDER/('proactive-attempt-'+str(int(time.time()))+'.json'),json.loads(target.read_text()))
    atomic(target,report)
    async with aiohttp.ClientSession() as client:
        for _ in range(180):
            try:
                async with client.get('http://127.0.0.1:8793/api/state') as response:
                    if (await response.json()).get('ready'):break
            except aiohttp.ClientError:pass
            await asyncio.sleep(.5)
        else:raise RuntimeError('Owned Robo service did not become ready')
    for condition in conditions:
        result=await case(condition,visual);report['cases'].append(result);atomic(target,report)
        print(json.dumps({k:v for k,v in result.items() if k!='events'}),flush=True)
        if not result['passed']:raise SystemExit('Live proactive case failed; retained receipt, no hidden retry')
    report.update(passed=all(r['passed'] for r in report['cases']) and report['runtime_source_sha256']==runtime_signature(),finished=time.time())
    atomic(target,report)
    if not report['passed']:raise SystemExit('Source changed during verification')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--conditions',nargs='+',default=['positive','disabled','unrelated','unknown'])
    parser.add_argument('--visual',action='store_true',help='Timestamped synthetic camera/Face-ID signal; not physical recognition accuracy')
    args=parser.parse_args();asyncio.run(main(args.conditions,args.visual))
