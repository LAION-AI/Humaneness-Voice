"""Two named synthetic users, return turns, background speech and an interruption."""
import argparse,asyncio,json,struct,time,base64
from pathlib import Path
import aiohttp
import numpy as np
from speaker_v4.common import atomic
from .live_test import audio
from .playback import SyntheticPlayback

async def main(extended=False,background_gain=.16,background_shift=0.,max_repeats=0,label=None,identity_mode=None):
    if max_repeats not in (0,1):raise ValueError('QA permits either strict first pass or one explicit repeat per turn')
    protocol=4 if max_repeats else 3
    atomic(Path('/mnt/nvme/robo-speaker-studies/speaker-live-results')/('three-user-live.json' if extended else 'multi-user-live.json'),
        {'public_synthetic_only':True,'protocol_version':protocol,'passed':False,'running':True})
    bg=audio('background');a=audio('christoph-intro');b=audio('robert-intro');follow=audio('christoph-followup')
    c=audio('selma-intro') if extended else None
    follow_b=audio('robert-followup') if extended else None;follow_c=audio('selma-followup-clear') if extended else None
    events=[];identities=[];counts={};sample=0;turn_done=asyncio.Event();first_chunk=asyncio.Event();t0=time.monotonic();boundaries=[];negative_rejected=None
    completed=[];last_reply_name=None;repeats=[];first_audio=[];played_ids=set()
    dac=SyntheticPlayback()
    trace=Path('/mnt/nvme/robo-speaker-studies/speaker-live-results')/f'inflight-{int(time.time())}.jsonl'
    def journal(item):
        with trace.open('a') as output:output.write(json.dumps({'at_s':round(time.monotonic()-t0,3),**item},ensure_ascii=False)+'\n')
    journal({'type':'protocol','public_synthetic_only':True,'extended':extended,'background_shift_s':background_shift,
        'protocol_version':protocol,'max_repeats_per_turn':max_repeats})
    async with aiohttp.ClientSession() as client:
        for _ in range(90):
            try:
                async with client.get('http://127.0.0.1:8796/state') as response:
                    diar_ready=(await response.json()).get('ready')
                async with client.get('http://127.0.0.1:8793/api/state') as response:
                    if diar_ready and (await response.json()).get('ready'):break
            except aiohttp.ClientError:pass
            await asyncio.sleep(.5)
        else:raise TimeoutError('Robby/diarizer did not become ready; conversation test was not started')
        async with client.ws_connect('http://127.0.0.1:8793/ws?synthetic_diagnostics=1') as ws:
            t0=time.monotonic()
            async def receive():
                nonlocal last_reply_name
                async for message in ws:
                    if message.type==aiohttp.WSMsgType.BINARY:
                        rid=struct.unpack_from('<i',message.data)[0]
                        if rid:
                            if rid not in counts:first_audio.append({'at_s':round(time.monotonic()-t0,3),'reply_id':rid,'name':last_reply_name})
                            count=(len(message.data)-4)//2;counts[rid]=counts.get(rid,0)+count
                            dac.add(rid,count,sample)
                    elif message.type==aiohttp.WSMsgType.TEXT:
                        item=json.loads(message.data)
                        if item.get('type')=='synthetic_audio_debug':
                            # Only this loopback client's public synthetic
                            # fixtures; ordinary/browser sessions cannot emit it.
                            import soundfile as sf
                            folder=Path('/mnt/nvme/robo-speaker-studies/speaker-live-results/synthetic-debug');folder.mkdir(exist_ok=True)
                            path=folder/f'{int(t0)}-{item["start"]:.3f}.wav'
                            sf.write(path,np.frombuffer(base64.b64decode(item.pop('audio')),'<f2').astype(np.float32),16000)
                            events.append({**item,'file':str(path)})
                        if item.get('type') in ('endpoint','reply','barge','turn_done','warning','error','ignored','asr_preview','synthetic_asr_views','synthetic_barge_proposal','synthetic_barge_verification','synthetic_activity'):
                            journal(item)
                        elif item.get('type')=='identity' and any(k in item for k in ('decision','verification','continuity')):
                            journal({k:v for k,v in item.items() if k in ('type','decision','verification','continuity','speakers','users')})
                        if item.get('type') in ('endpoint','reply','barge','turn_done','warning','error','asr_preview'):events.append({'at_s':round(time.monotonic()-t0,3),**item})
                        if item.get('type') in ('ignored','warning','error'):
                            events.append(item);print('Synthetic diagnostic:',json.dumps(item),flush=True)
                        if item.get('type')=='identity' and item.get('speakers'):identities.append(item['speakers'])
                        if item.get('type')=='identity' and item.get('verification'):
                            events.append({'at_s':round(time.monotonic()-t0,3),'type':'verification',**item['verification']})
                        if item.get('type')=='identity' and item.get('continuity'):
                            events.append({'at_s':round(time.monotonic()-t0,3),'type':'continuity',**item['continuity']})
                        if item.get('type')=='synthetic_asr_views':events.append(item)
                        if item.get('type') in ('synthetic_barge_proposal','synthetic_barge_verification'):events.append({'at_s':round(time.monotonic()-t0,3),**item})
                        if item.get('type')=='reply':last_reply_name=item.get('speaker_name')
                        if item.get('type')=='barge':dac.cancel(sample)
                        if item.get('type')=='turn_done' and not item.get('barged'):
                            completed.append({'at_s':time.monotonic()-t0,'name':last_reply_name});turn_done.set()
            task=asyncio.create_task(receive())
            if identity_mode:await ws.send_str('identity:mode:'+identity_mode)
            async def feed(wave=None,seconds=None,fixture=None):
                nonlocal sample
                n=len(wave) if wave is not None else round(seconds*16000)
                if wave is not None:boundaries.append({'fixture':fixture or ('christoph-intro' if wave is a else 'robert-intro' if wave is b else 'selma-intro' if wave is c else 'robert-followup' if wave is follow_b else 'selma-followup-clear' if wave is follow_c else 'christoph-followup'),'start_s':sample/16000,'end_s':(sample+n)/16000})
                for at in range(0,n,320):
                    length=min(320,n-at);indices=(np.arange(length)+sample+round(background_shift*16000))%len(bg);x=bg[indices]*background_gain
                    if wave is not None:x=x+wave[at:at+length]
                    pcm=np.clip(np.rint(x*32767),-32768,32767).astype('<i2')
                    if ws.closed:raise RuntimeError('WebSocket closed during synthetic conversation')
                    for lo,hi,rid,pos in dac.pieces(sample,length):
                        await ws.send_bytes(struct.pack('<ii',rid,pos)+pcm[lo:hi].tobytes())
                        if rid:first_chunk.set();played_ids.add(rid)
                    sample+=length
                    if sample//320%5==0:
                        for rid,played in dac.acknowledgments(sample).items():await ws.send_str(f'played:{rid}:{played}')
                    await asyncio.sleep(max(0,t0+sample/16000-time.monotonic()))
            async def wait_reply(name):
                target=boundaries[-1].copy();repeated=0
                for _ in range(260):
                    if any(e['name']==name and e['at_s']>=target['start_s'] for e in completed):return
                    # Optional recovery test, NOT a strict first-pass claim.
                    # Repeat the same public utterance once, only if no turn
                    # was accepted after nine seconds. Never repeat an already
                    # accepted turn just because shared TTS is queued/busy.
                    accepted=any(e.get('type')=='endpoint' and e.get('speaker_name')==name
                        and e['at_s']>=target['start_s'] for e in events)
                    if repeated<max_repeats and not accepted and sample/16000>=target['end_s']+9.:
                        reasons=[e.get('reason',e.get('text','')) for e in events if e.get('type') in ('ignored','warning')
                            and e.get('at_s',0)>=target['start_s']]
                        repeats.append({'name':name,'fixture':target['fixture'],'first_start_s':target['start_s'],
                            'repeat_start_s':sample/16000,'reason':reasons[-1] if reasons else 'No accepted turn within nine seconds'})
                        journal({'type':'explicit_test_repeat',**repeats[-1]})
                        repeated+=1;await feed(audio(target['fixture']),fixture=target['fixture'])
                    await feed(seconds=.32)
                atomic(Path('/mnt/nvme/robo-speaker-studies/speaker-live-results')/('multi-timeout-'+str(int(time.time()))+'.json'),{'events':events,'identities':identities})
                raise TimeoutError('Reply did not finish while background continued')
            try:
                await feed(seconds=1.28)
                await feed(a);await wait_reply('Christoph');turn_done.clear();first_chunk.clear()
                await feed(b);await wait_reply('Robert');turn_done.clear();first_chunk.clear()
                if extended:
                    await feed(c);await wait_reply('Selma');turn_done.clear();first_chunk.clear()
                    # The original Selma follow-up failed clean ASR (nonverbal
                    # synthesis instead of its intended words). Retain it as
                    # a negative: do not submit background TV as Selma's text.
                    before=sum(e.get('type')=='endpoint' for e in events)
                    await feed(audio('selma-followup'),fixture='selma-followup-nonverbal');await feed(seconds=3.84)
                    negative_rejected=sum(e.get('type')=='endpoint' for e in events)==before
                    if not negative_rejected:
                        atomic(Path('/mnt/nvme/robo-speaker-studies/speaker-live-results')/('negative-failure-'+str(int(time.time()))+'.json'),
                            {'public_synthetic_only':True,'events':events,'identities':identities,'fixture_boundaries':boundaries})
                    assert negative_rejected,'Nonverbal foreground submitted background words'
                    await feed(follow);await wait_reply('Christoph');turn_done.clear();first_chunk.clear()
                    await feed(follow_b);await wait_reply('Robert');turn_done.clear();first_chunk.clear()
                last=follow_c if extended else follow
                await feed(last)
                target_start=boundaries[-1]['start_s'];target_name='Selma' if extended else 'Christoph'
                target_reply=None
                for _ in range(100):
                    target_reply=next((e for e in events if e.get('type')=='reply' and
                        e.get('speaker_name')==target_name and e['at_s']>=target_start),None)
                    if target_reply and target_reply['reply_id'] in played_ids:break
                    await feed(seconds=.32)
                assert target_reply and target_reply['reply_id'] in played_ids,'The CURRENT reply never reached the simulated DAC; no voice interruption test is valid'
                # The same enrolled person talks over the actual response.
                await feed(last)
                await feed(seconds=2.56)
                natural_barges=[e for e in events if e.get('type')=='barge']
                await ws.send_str('interrupt')
                named={'Christoph','Robert','Selma'}
                all_endpoints=[e for e in events if e.get('type')=='endpoint'];all_replies=[e for e in events if e.get('type')=='reply']
                endpoints=[e for e in all_endpoints if e.get('speaker_name') in named];replies=[e for e in all_replies if e.get('speaker_name') in named]
                report={'protocol_version':protocol,'strict_content_checks':True,'max_repeats_per_turn':max_repeats,
                    'explicit_repeats':repeats,'all_first_attempts_accepted':not repeats,
                    'public_synthetic_only':True,'background_continues':True,'background_gain':background_gain,'background_shift_s':background_shift,'extended':extended,'endpoints':endpoints,'replies':replies,
                    'provisional_wake_endpoints':[e for e in all_endpoints if e not in endpoints],
                    'provisional_replies':[e for e in all_replies if e not in replies],'completed_replies':completed,
                    'barge_events':natural_barges,'last_enrolled_users':identities[-1] if identities else [],
                    'interruption_playback_reply_id':target_reply['reply_id'],
                    'nonverbal_foreground_rejected':negative_rejected,
                    'warnings':[e for e in events if e.get('type') in ('warning','error')],
                    'asr_previews':[e for e in events if e.get('type')=='asr_preview'],
                    'verification':[e for e in events if e.get('type')=='verification'],
                    'continuity':[e for e in events if e.get('type')=='continuity'],
                    'synthetic_audio_debug':[e for e in events if e.get('type')=='synthetic_audio_debug'],
                    'asr_views':[e for e in events if e.get('type')=='synthetic_asr_views'],'fixture_boundaries':boundaries,
                    'barge_verification':[e for e in events if e.get('type')=='synthetic_barge_verification'],
                    'first_reply_audio':first_audio,'requested_identity_mode':identity_mode or 'server default',
                    'playback':'Real-time synthetic DAC position tags and acknowledgments, including chunk gaps/cancellation; no physical microphone/DAC'}
                folder=Path('/mnt/nvme/robo-speaker-studies/speaker-live-results')
                atomic(folder/('multi-attempt-'+str(int(time.time()))+'.json'),report)
                print('Synthetic multi-user results:',json.dumps({'endpoints':[(e.get('speaker_name'),e.get('speaker_id')) for e in endpoints],
                    'reply_count':len(replies),'barge_count':len(report['barge_events']),'warnings':report['warnings']}),flush=True)
                expected=['Christoph','Robert','Selma','Christoph','Robert','Selma'] if extended else ['Christoph','Robert','Christoph']
                assert len(endpoints)>=len(expected),'Did not submit all named users'
                assert len(replies)>=len(expected),'An expected user received no reply (a partial zip is not a pass)'
                assert [e.get('speaker_name') for e in endpoints[:len(expected)]]==expected,'Names not bound to returning voice'
                bound={}
                for name,item in zip(expected,endpoints):
                    if name in bound:assert bound[name]==item['speaker_id'],'Returning speaker ID changed'
                    else:bound[name]=item['speaker_id']
                assert len(set(bound.values()))==len(bound),'Distinct people merged'
                required=['Christoph','Robert','Selma','Geschichte','Witz','zusammen'] if extended else ['Christoph','Robert','Geschichte']
                for word,item in zip(required,endpoints):assert word.casefold() in item['transcript'].casefold(),'Wrong/background words submitted in place of a target turn'
                assert not any(phrase in item['transcript'].casefold() for item in endpoints
                    for phrase in ('nachrichten','morgen wird','bericht','über das wetter')),'Background-TV words were appended to an accepted user turn'
                assert len(report['last_enrolled_users'])==len(set(expected)),'Background speaker enrolled without wake'
                assert [reply.get('speaker_name') for reply in replies[:len(expected)]]==expected,'Reply metadata lost its current speaker; spoken names are optional'
                assert report['barge_events'],'Enrolled interruption was lost'
                last_user=boundaries[-1]
                assert any(last_user['start_s']<=e['at_s']<=last_user['end_s']+1 for e in natural_barges),'No actual voice interruption during target fixture; cleanup Stop does not count'
                report['passed']=True
                atomic(folder/('three-user-live.json' if extended else 'multi-user-live.json'),report)
                if label:
                    import re
                    if not re.fullmatch(r'[a-z0-9-]{1,50}',label):raise ValueError('Unsafe receipt label')
                    atomic(folder/(label+'.json'),report)
                print(json.dumps({'passed':True,'extended':extended,'users':list(bound),'turns':len(endpoints),'replies':len(replies),'barge_events':len(report['barge_events'])}),flush=True)
            finally:
                await ws.close();task.cancel();await asyncio.gather(task,return_exceptions=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--extended',action='store_true');p.add_argument('--background-gain',type=float,default=.16);p.add_argument('--background-shift',type=float,default=0.)
    p.add_argument('--max-repeats',type=int,choices=(0,1),default=0,help='Default 0: strict first pass. 1: separately labelled bounded-recovery test.')
    p.add_argument('--label');p.add_argument('--identity-mode',choices=('v4-long','v4-long3'))
    args=p.parse_args();asyncio.run(main(args.extended,args.background_gain,args.background_shift,args.max_repeats,args.label,args.identity_mode))
