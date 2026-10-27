"""Loopback-only multiplexed PCM session for Robby, one existing TTS engine."""
import asyncio,base64,json,queue,threading,time
from typing import Literal
import numpy as np
from fastapi import WebSocket,WebSocketDisconnect
from pydantic import BaseModel,Field,ConfigDict

class Start(BaseModel):
    model_config=ConfigDict(extra='forbid')
    type:Literal['start']='start'
    model:Literal['small_rl']='small_rl'
    voice_profile:str=Field(default='emolia_c1699',max_length=80)
    guidance:float=Field(default=2.,ge=1,le=5,allow_inf_nan=False)
    temperature:float=Field(default=.8,ge=0,le=1.5,allow_inf_nan=False)
    seed:int=Field(default=777,ge=0,le=2147483647)
    max_active:int=Field(default=3,ge=1,le=3)
    batch_decode:bool=True

class Submit(BaseModel):
    model_config=ConfigDict(extra='forbid')
    type:Literal['submit']='submit'
    id:str=Field(pattern=r'^[a-zA-Z0-9_-]{1,40}$')
    order:int=Field(ge=0,le=200)
    message:str=Field(min_length=1,max_length=5000)
    language:Literal['de','en']='de'
    seconds:float|None=Field(default=None,ge=1,le=60,allow_inf_nan=False)
    first:int=Field(default=2,ge=2,le=20)
    chunk:int=Field(default=5,ge=2,le=20)
    max_frames:int|None=Field(default=None,ge=2,le=810)
    align:bool=True

def install(app,state,switch_model):
    @app.websocket('/api/live-speech')
    async def live(ws:WebSocket):
        # This is an internal worker transport. Browser traffic continues to
        # use the existing authenticated/session-aware Robby routes.
        if ws.client.host not in ('127.0.0.1','::1') or ws.headers.get('x-forwarded-for') or ws.headers.get('origin'):
            await ws.close(code=1008);return
        await ws.accept()
        if state['busy'] or state['phase']!='ready':
            await ws.send_json({'event':'unavailable','retryable':True});await ws.close();return
        state['busy']=True;state['job_phase']='Continuous speech session';state['job_started']=time.monotonic()
        cancelled=threading.Event();commands=queue.Queue(maxsize=16);events=asyncio.Queue();worker=None;receiver=None;cutoffs={}
        loop=asyncio.get_running_loop()
        def emit(event):loop.call_soon_threadsafe(events.put_nowait,event)
        try:
            start=Start.model_validate(await asyncio.wait_for(ws.receive_json(),10))
            if getattr(state['engine'],'model_id',None)!=start.model:await asyncio.to_thread(switch_model,start.model)
            from dynamic_references import fixed_clip
            reference,info=fixed_clip(start.voice_profile)
            from live_batch import ContinuousBatch
            engine=state['engine'];engine.fast.graph_limit=max(engine.fast.graph_limit,8)
            batch=ContinuousBatch(engine,reference,commands,emit,cancelled,temperature=start.temperature,
                guidance=start.guidance,seed=start.seed,max_active=start.max_active,batch_decode=start.batch_decode,finisher=state['finisher'])
            def work():
                try:batch.run()
                except Exception as exc:
                    import logging
                    logging.getLogger('small').exception('Continuous TTS session failed')
                    emit({'event':'error','message':type(exc).__name__})
            worker=asyncio.create_task(asyncio.to_thread(work))
            import settings
            await ws.send_json({'event':'ready','backend':'continuous_batch','weight_instances':1,
                'voice_profile':start.voice_profile,'reference_seconds':min(float(info['seconds']),settings.REFERENCE_SECONDS),
                'model':start.model,'max_active':start.max_active})
            async def receive():
                from prompting import parse_literal,build
                while not cancelled.is_set():
                    raw=await asyncio.wait_for(ws.receive_json(),90)
                    if raw.get('type')=='submit':
                        sub=Submit.model_validate(raw)
                        directed=parse_literal(sub.message,'structured')
                        spec=build(directed,'structured',sub.language,sub.seconds,sub.message,reference_identity=True)
                        from alignment import LiveCutoff
                        cutoffs[sub.id]=LiveCutoff(spec,sub.align)
                        commands.put_nowait({'type':'submit','id':sub.id,'spec':spec,'order':sub.order,
                            'first':sub.first,'chunk':sub.chunk,'max_frames':sub.max_frames,'align':sub.align})
                    elif raw.get('type')=='cancel' and isinstance(raw.get('id'),str):commands.put_nowait({'type':'cancel','id':raw['id']})
                    elif raw.get('type')=='finish':commands.put_nowait({'type':'finish'});return
                    else:raise ValueError('Unknown session message')
            receiver=asyncio.create_task(receive())
            while True:
                if receiver.done() and receiver.exception():raise receiver.exception()
                get=asyncio.create_task(events.get())
                done,_=await asyncio.wait([get,receiver] if not receiver.done() else [get],timeout=90,return_when=asyncio.FIRST_COMPLETED)
                if not done:get.cancel();raise TimeoutError('TTS session stalled')
                if receiver in done and receiver.exception():get.cancel();raise receiver.exception()
                if get not in done:
                    # The input was cleanly sealed; finish draining generation.
                    event=await asyncio.wait_for(get,90)
                else:event=get.result()
                if '_wave' in event:
                    pcm=cutoffs[event['id']].feed(event.pop('_wave'),{})
                    if not len(pcm):continue
                    event['pcm_f32']=base64.b64encode(np.asarray(pcm,dtype='<f4').tobytes()).decode()
                if event.get('event')=='job_done':
                    aligned=event.pop('_aligned',None);cutoff=cutoffs.pop(event['id'],None)
                    if aligned is not None and cutoff is not None:
                        tail=cutoff.finish(aligned)
                        if len(tail):await ws.send_json({'event':'audio_chunk','id':event['id'],'sample_rate':48000,'channels':1,
                            'pcm_f32':base64.b64encode(np.asarray(tail,dtype='<f4').tobytes()).decode()})
                    event['live_alignment']=dict(cutoff.report) if cutoff else None
                await ws.send_json(event)
                if event.get('event') in ('session_done','error'):break
        except (WebSocketDisconnect,asyncio.CancelledError):pass
        except Exception as exc:
            try:await ws.send_json({'event':'error','message':type(exc).__name__})
            except Exception:pass
        finally:
            cancelled.set()
            if receiver:receiver.cancel();await asyncio.gather(receiver,return_exceptions=True)
            # CUDA owner must actually release caches before marking the shared
            # engine idle. Never detach an unfinished worker from its lock.
            if worker:await asyncio.shield(worker)
            state['busy']=False;state['job_phase']=None;state['job_started']=None
            try:await ws.close()
            except Exception:pass
