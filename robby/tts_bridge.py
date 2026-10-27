"""TYGER MOUTH wire -> existing Small RL/Luna HTTP service, no second TTS load."""
import asyncio
import json
import logging
import os
import io

import httpx
import numpy as np
import soundfile as sf

import paths
from performance import plan
from conversation_prompt import message as conversation_message,speaker_update as validated_speaker_update
from conversation_director import CompactDirector
from shared_turn import admitted_turn
from app.wire import async_recv_msg,async_send_msg
from app.runtime.audio import StreamingAudioResampler

log=logging.getLogger('robo.bridge')
TTS_CHUNK_FRAMES=20  # 80 ms per semantic frame: 1.6 s per normal streaming chunk.


def trim_ack(audio):
    """Remove generated leading/trailing silence, not the vocal gesture."""
    hop=480;n=len(audio)//hop
    if not n:return audio
    energy=np.sqrt(np.mean(audio[:n*hop].reshape(n,hop)**2,axis=1))
    active=np.flatnonzero(energy>max(.001,float(energy.max())*.05))
    if not len(active):return audio
    result=audio[max(0,(active[0]-1)*hop):min(len(audio),(active[-1]+2)*hop)].copy()
    fade=min(240,len(result)//2)
    if fade:result[:fade]*=np.linspace(0,1,fade);result[-fade:]*=np.linspace(1,0,fade)
    return result


async def decode_mp3_ffmpeg(raw):
    process=await asyncio.create_subprocess_exec('ffmpeg','-nostdin','-v','error','-i','pipe:0',
        '-f','f32le','-ac','1','-ar','48000','pipe:1',stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    try:pcm,error=await process.communicate(raw)
    except BaseException:
        if process.returncode is None:process.kill()
        await process.wait()
        raise
    if process.returncode:raise RuntimeError('Audio transport decoding failed')
    return np.frombuffer(pcm,dtype='<f4').copy()


def _decode_transport(raw):
    pcm,rate=sf.read(io.BytesIO(raw),dtype='float32',always_2d=True)
    # The actual shared service sends mono 48-kHz MP3. Preserve the old
    # resampling/downmix path for different rates/layouts or unsupported files.
    if rate!=48000 or pcm.shape[1]!=1:raise ValueError('Nonstandard audio transport')
    if not np.isfinite(pcm).all():raise ValueError('Invalid audio samples')
    return pcm[:,0].copy()


async def decode_mp3(raw):
    try:return await asyncio.to_thread(_decode_transport,raw)
    except (sf.LibsndfileError,ValueError):return await decode_mp3_ffmpeg(raw)


class MouthBridge:
    def __init__(self,base=None):
        self.base=base or os.environ.get('ROBO_TTS_BASE','http://127.0.0.1:8792')
        self.active=0
        self.ack=None
        self.director_mode=os.environ.get('ROBO_DIRECTOR_MODE','compact')
        if self.director_mode not in ('compact','shared','bolt'):raise ValueError('Invalid Robby director mode')

    async def prepare_ack(self):
        """One short synthetic acknowledgment, cached in RAM, never during a turn."""
        try:
            async with httpx.AsyncClient(base_url=self.base,timeout=90) as client:
                state=(await client.get('/api/state')).json()
                if state.get('busy'):return
                body={'model':'small_rl','input_mode':'manual','format':'structured',
                    'message':'GENERAL: Gentle natural affirmative acknowledgement.\nSCRIPT:\nMm-hm.',
                    'generation':'single','n':1,'streaming':False,'seconds':1.6,
                    'temperature':.7,'guidance':2.,'voice_mode':'profile','voice_profile':'emolia_c1699',
                    'voice_conversion':False,'align':False}
                async with client.stream('POST','/api/turn',json=body) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith('data: '):
                            event=json.loads(line[6:])
                            if event.get('event')=='result':
                                candidate=event['candidates'][0]
                                audio=await client.get(candidate['url']);audio.raise_for_status()
                                self.ack=trim_ack(StreamingAudioResampler(48000,24000).process(await decode_mp3(audio.content),final=True))
        except Exception as exc:log.warning('Acknowledgment cache unavailable: %s',type(exc).__name__)

    async def accept(self,reader,writer):
        task=None;warm_task=None;history=[];voice='emolia_c1699';latest=None;literal_cache={}
        async with httpx.AsyncClient(base_url=self.base,timeout=180,
                limits=httpx.Limits(max_connections=12,max_keepalive_connections=8,keepalive_expiry=120)) as client:
            lock=asyncio.Lock()
            async def send(kind,payload):
                async with lock:await async_send_msg(writer,kind,payload)
            from memory_session import MemorySession
            async def memory_notice(event):await send(b'X',{'memory_event':event})
            memory=MemorySession(memory_notice)

            def remember(record,interrupted=False):
                if record is None:return
                if record.get('avatar_context',{}).get('event') and not record.get('reply'):return
                from conversation_tasks import memory_entry
                entry=memory_entry(record,interrupted)
                if record.get('index') is None:
                    speaker=record.get('speaker') or {}
                    prefix=('Current session speaker: '+json.dumps({'speaker_id':speaker.get('speaker_id'),'display_name':speaker.get('display_name')})+'\n'
                        if speaker.get('enrolled') else '')
                    # A mouse gesture is not a fabricated spoken user turn.
                    if not record.get('avatar_context',{}).get('event') and not record.get('avatar_context',{}).get('environment_proactive'):
                        history.append({'role':'user','content':prefix+record['text']})
                    record['index']=len(history)
                    history.append(entry)
                else:history[record['index']].update(entry)
                memory.enqueue(record,history)

            async def warm_literals(replies,language,pending):
                try:
                    if pending and not pending.done():await asyncio.shield(pending)
                    for literal in replies[:4]:
                        if not isinstance(literal,str) or not 1<=len(literal)<=180 or len(literal.split())>8:continue
                        prepared_voice=voice;key=(prepared_voice,language,literal)
                        if key in literal_cache:continue
                        state=(await client.get('/api/state')).json()
                        if state.get('busy'):
                            log.info('Optional interjection preparation skipped: shared speech service busy');return
                        from conversation_director import validate
                        directed=validate({'general':'Brief, firm, playful conversational interruption.',
                            'script':'(determination, firm and playful) '+literal,'language':language})
                        body={'model':'small_rl','message':directed['manual_prompt'],'input_mode':'manual','format':'structured',
                            'generation':'single','n':1,'streaming':False,'seconds':6.,'temperature':.7,'guidance':2.,
                            'voice_mode':'profile','voice_profile':prepared_voice,'voice_conversion':False,'align':True}
                        async with client.stream('POST','/api/turn',json=body,timeout=30.) as response:
                            if response.status_code in (409,503):return
                            response.raise_for_status()
                            async for line in response.aiter_lines():
                                if not line.startswith('data: '):continue
                                event=json.loads(line[6:])
                                if event.get('event')=='result':
                                    candidate=event['candidates'][0];audio=await client.get(candidate['url']);audio.raise_for_status()
                                    pcm=StreamingAudioResampler(48000,24000).process(await decode_mp3(audio.content),final=True)
                                    trimmed=trim_ack(pcm)
                                    if 0<len(pcm)<5.8*24000 and 0<len(trimmed)<4.*24000:
                                        if len(literal_cache)>=4:literal_cache.pop(next(iter(literal_cache)))
                                        literal_cache[key]=trimmed
                                        await send(b'X',{'status_only':True,'stage':'interjection_ready','text':'Agreed interruption audio is ready in session RAM.'})
                                    else:log.info('Optional interjection preparation skipped: bounded duration check (%.2fs / trimmed %.2fs)',len(pcm)/24000,len(trimmed)/24000)
                except asyncio.CancelledError:raise
                except Exception as exc:log.info('Optional interjection preparation unavailable: %s',type(exc).__name__)

            async def turn(record):
                text=record['text']
                turn_voice=voice
                self.active+=1
                played=False;reply='';resampler=StreamingAudioResampler(48000,24000)
                stage='planning'
                try:
                    identity=record.get('speaker') or {};speaker_update=validated_speaker_update(text,identity)
                    if self.director_mode=='bolt' and not record.get('interaction_reply'):
                        from bolt_pipeline import pipeline
                        async for event,value in pipeline(client,text,identity,history,voice=turn_voice,
                                avatar_context=record.get('avatar_context'),interaction_state=record.get('interaction_state'),
                                llm_selection=record.get('llm_selection'),credential=record.get('llm_credential'),quick_replies=record.get('quick_replies',False),memory=memory):
                            if event=='sentence':
                                stage='speech'
                                reply=value['text'];record['reply']=reply
                                await send(b'X',{**value,'speaker_update':speaker_update})
                            elif event=='audio':await send(b'A',value.astype(np.float32).tobytes());played=True
                            elif event=='avatar_action':await send(b'X',{'avatar_action':value})
                            elif event=='latency':await send(b'X',{'component_metrics':value})
                            elif event=='done':
                                record['bolt_metrics']=value
                                await send(b'X',{'pipeline_metrics':value})
                        remember(record);await send(b'D',{});return
                    from conversation_tasks import history_for_model
                    shared_history,playback_note=history_for_model(history[-16:])
                    from voice_captioning import prompt_hint
                    body={'model':'small_rl','message':conversation_message(text,{k:v for k,v in identity.items() if k!='voice_delivery'})+playback_note+prompt_hint(identity),'input_mode':'chat','format':'structured',
                        'generation':'single','n':1,'streaming':True,'chunk_frames':TTS_CHUNK_FRAMES,
                        'cancel_on_disconnect':True,
                        'temperature':.7,'guidance':2.,'voice_mode':'profile','voice_profile':turn_voice,
                        'voice_conversion':False,'history':shared_history}
                    async def waiting(text,stage):await send(b'X',{'text':text,'status_only':True,'stage':stage})
                    director_ms=None
                    if self.director_mode in ('compact','bolt'):
                        await waiting('Preparing a useful spoken answer…','planning')
                        if record.get('interaction_reply'):
                            from conversation_director import validate
                            literal=record['interaction_reply']
                            if not isinstance(literal,str) or len(literal)>180:raise ValueError('Invalid interjection')
                            feedback=record.get('stop_feedback',False)
                            directed=validate({'general':'Brief, gentle, calm acknowledgment. Unhurried natural delivery.' if feedback else 'Brief, firm, playful conversational interruption.',
                                'script':('(interest, gentle and unhurried) ' if feedback else '(determination, firm and playful) ')+literal,'language':record.get('interaction_language','de')})
                            directed.update(director_ms=0.,interaction_actions=[])
                        else:
                            request=CompactDirector().generate(client,text,identity,history,interaction_state=record.get('interaction_state'),avatar_context=record.get('avatar_context'))
                            directed=(await asyncio.wait_for(request,8.) if record.get('avatar_context',{}).get('event') else await request)
                        if directed.get('silent'):
                            await send(b'D',{});return
                        if record.get('avatar_context',{}).get('event') and (await client.get('/api/state',timeout=3.)).json().get('busy'):
                            await send(b'D',{});return  # optional affection never queues behind someone else's TTS
                        if record.get('avatar_context',{}).get('event'):
                            from avatar_personality import event_steps
                            directed['avatar_actions']=event_steps(record['avatar_context'])
                        director_ms=directed['director_ms'];reply=directed['text'];record['reply']=reply
                        await send(b'X',{'text':reply,'performance':plan(directed),'speaker':identity,
                            'avatar_event':record.get('avatar_context',{}).get('event'),
                            'language':directed['language'],
                            'avatar_actions':directed.get('avatar_actions',[]),
                            'interaction_actions':directed.get('interaction_actions',[]),
                            'sources':directed.get('sources',[]),'lookups':directed.get('lookups',[]),
                            'speaker_update':speaker_update,'timing':{'director_ms':director_ms,
                                'director_mode':'compact','completion_tokens':directed.get('completion_tokens'),
                                'director_model':directed.get('director_model'),
                                'interjection_cache_hit':(turn_voice,directed['language'],record.get('interaction_reply')) in literal_cache}})
                        body.update(message=directed['manual_prompt'],input_mode='manual',history=[],language=directed['language'])
                        cached=literal_cache.get((turn_voice,directed['language'],record.get('interaction_reply')))
                        if cached is not None:
                            for at in range(0,len(cached),9600):await send(b'A',cached[at:at+9600].astype(np.float32).tobytes())
                            remember(record);await send(b'D',{});return
                    stage='speech'
                    async with admitted_turn(client,body,waiting,timeout=.25 if record.get('avatar_context',{}).get('event') else 45.) as response:
                        async for line in response.aiter_lines():
                            if not line.startswith('data: '):continue
                            event=json.loads(line[6:]);kind=event.get('event')
                            if kind=='error':raise RuntimeError('Shared speech service generation failed')
                            if kind in ('llm','spec','prompt','direction'):
                                candidate=event.get('spec') or event.get('result') or event
                                value=candidate.get('text')
                                if value and value!=reply:
                                    reply=value;record['reply']=reply;await send(b'X',{'text':reply,'performance':plan(candidate),'speaker_update':speaker_update,'speaker':identity,
                                        'timing':{'director_mode':'shared','director_ms':event.get('llm_ms')}})
                            if kind=='audio_chunk':
                                url=event.get('url')
                                if url:
                                    audio=await client.get(url);audio.raise_for_status()
                                    pcm=resampler.process(await decode_mp3(audio.content))
                                    if len(pcm):await send(b'A',pcm.astype(np.float32).tobytes());played=True
                            if kind=='result':
                                # Some backend versions expose text only with the final result.
                                if not reply:
                                    value=(event.get('spec') or {}).get('text')
                                    if value:reply=value;record['reply']=reply;await send(b'X',{'text':reply,'speaker_update':speaker_update,'speaker':identity})
                                if not played:
                                    candidates=event['candidates'];winner=event.get('winner',0)
                                    candidate=next((c for c in candidates if c.get('index')==winner),candidates[0])
                                    audio=await client.get(candidate['url']);audio.raise_for_status()
                                    pcm=resampler.process(await decode_mp3(audio.content))
                                    if len(pcm):await send(b'A',pcm.astype(np.float32).tobytes())
                    tail=resampler.process(np.zeros(0,np.float32),final=True)
                    if len(tail):await send(b'A',tail.astype(np.float32).tobytes())
                    remember(record)
                    await send(b'D',{})
                except asyncio.CancelledError:
                    remember(record,interrupted=True)
                    # This request opted into cooperative disconnect cancellation.
                    raise
                except Exception as exc:
                    status=getattr(getattr(exc,'response',None),'status_code',None)
                    log.warning('Shared service error: stage=%s type=%s status=%s',stage,type(exc).__name__,status)
                    if record.get('avatar_context',{}).get('event'):
                        await send(b'D',{});return  # optional comments fail quietly, never a canned emotional fallback
                    await send(b'X',{'text':'The '+stage+' service could not complete this reply. Please try again.',
                        'error_type':type(exc).__name__,'error_stage':stage,'http_status':status})
                    await send(b'D',{})
                finally:self.active-=1

            try:
                while True:
                    kind,raw=await async_recv_msg(reader)
                    payload=json.loads(raw) if kind in (b'T',b'B',b'V',b'C',b'H',b'W',b'M') and raw else {}
                    if kind==b'V':voice=payload.get('voice',voice)
                    elif kind==b'M':
                        try:
                            if payload.get('action')=='reset':
                                if task and not task.done():task.cancel();await asyncio.gather(task,return_exceptions=True)
                                latest=None
                            await memory.command(payload,history)
                        except (ValueError,TypeError,OSError,json.JSONDecodeError) as exc:
                            await memory.notice(state='error',error=type(exc).__name__)
                    elif kind==b'T':
                        if warm_task and not warm_task.done():warm_task.cancel();await asyncio.gather(warm_task,return_exceptions=True)
                        if task and not task.done():task.cancel();await asyncio.gather(task,return_exceptions=True)
                        latest={'text':payload['transcript'],'speaker':payload.get('speaker'),'reply':'','index':None,
                            'llm_selection':payload.get('llm_selection'),'llm_credential':payload.get('llm_credential'),'quick_replies':payload.get('quick_replies',False),
                            'interaction_state':payload.get('interaction_state'),'interaction_reply':payload.get('interaction_reply'),
                            'interaction_language':payload.get('interaction_language','de'),'stop_feedback':payload.get('stop_feedback',False),
                            'avatar_context':payload.get('avatar_context') or {}}
                        task=asyncio.create_task(turn(latest))
                    elif kind==b'W':
                        if warm_task and not warm_task.done():warm_task.cancel();await asyncio.gather(warm_task,return_exceptions=True)
                        warm_task=asyncio.create_task(warm_literals(payload.get('replies',[]),payload.get('language','de'),task))
                    elif kind==b'B':
                        if task and not task.done():task.cancel();await asyncio.gather(task,return_exceptions=True)
                        remember(latest,interrupted=True)
                        # A task cancelled before its first step never enters
                        # its CancelledError handler. Acknowledge exactly once
                        # here, after cancellation, so EARS cannot stay halting.
                        await send(b'D',{'barged':True})
                    elif kind==b'H':remember(latest,interrupted=True)
                    elif kind==b'C':
                        # Original TYGER decides when; reuse the same synthetic
                        # voice without starting another GPU/hosted job.
                        if self.ack is not None:
                            await send(b'K',self.ack.astype(np.float32).tobytes())
                        await send(b'D',{'bc':True,'aborted':self.ack is None})
            except (asyncio.IncompleteReadError,ConnectionError):pass
            finally:
                if warm_task and not warm_task.done():warm_task.cancel();await asyncio.gather(warm_task,return_exceptions=True)
                if task and not task.done():task.cancel();await asyncio.gather(task,return_exceptions=True)
                await memory.close()
                writer.close()
