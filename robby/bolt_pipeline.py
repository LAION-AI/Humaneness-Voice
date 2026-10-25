"""Concurrent local LLM producer and sequential sentence TTS consumer.

No generated waveform needs to finish before the first PCM chunk is sent.
Cancellation closes both upstream streams and drops their bounded queues.
"""
import asyncio,os,time,json,uuid
import paths
import numpy as np
from bolt_director import BoltDirector,needs_tools
from bolt_text import Sentence,join_sentences
from avatar_actions import explicit_spin
from performance import plan
from shared_turn import admitted_turn

async def pipeline(client,text,identity,history,voice='emolia_c1699',avatar_context=None,
                   interaction_state=None,chunk_frames=None,tts=True,preset=None,reply_words=None):
    from app.runtime.audio import StreamingAudioResampler
    from tts_bridge import decode_mp3
    start=time.perf_counter();metrics={};queue=asyncio.Queue(maxsize=4)
    chunk_frames=chunk_frames or int(os.environ.get('ROBO_BOLT_CHUNK_FRAMES','20'))
    if chunk_frames not in (2,5,10,20):raise ValueError('Unsupported TTS chunk size')
    from avatar_commands import explicit_actions,silent_requested
    context=dict(avatar_context or {})
    requested=explicit_actions(text,sum(h.get('role')=='user' for h in history)) if preset is None and not context.get('event') else []
    avatar_steps=list(requested);actions=[];sources=[]
    silent=bool(requested and silent_requested(text))
    immediate=bool(requested and (silent or any(s['kind'] not in ('spin','carousel','afterglow') for s in requested)))
    action_id=uuid.uuid4().hex if immediate else None
    if requested:context['requested_actions']=requested
    if context.get('event'):
        state=await client.get('/api/state',timeout=3.)
        if state.json().get('busy'):
            yield 'done',{'silent':True,'reason':'optional event skipped while TTS is busy'};return
    if immediate:
        yield 'avatar_action',{'action_id':action_id,'steps':requested,'source':'validated_direct_request'}
    if silent:
        yield 'done',{'silent':True,'reason':'User explicitly requested a silent visual action'};return

    async def produce():
        try:
            if preset is not None:
                for sentence in preset:await queue.put(sentence)
            elif needs_tools(text,history,interaction_state) and not context.get('event'):
                # Preserve the existing real news/weather/rule tools through
                # the same local endpoint; their verified result precedes speech.
                from conversation_director import CompactDirector
                from bolt_director import local_config,response_plan
                from bolt_text import SentenceFramer
                result=await CompactDirector().generate(client,text,identity,history,
                    interaction_state=interaction_state,avatar_context=context,provider=local_config())
                avatar_steps[:]=requested or result.get('avatar_actions',[]);actions[:]=result.get('interaction_actions',[])
                sources[:]=result.get('sources',[])
                framer=SentenceFramer(lines_only=response_plan(text,history)['kind']=='poem')
                for sentence in framer.feed(result['script'],final=True):
                    sentence.language=result['language'];await queue.put(sentence)
                metrics['tool_director_ms']=result['director_ms']
            else:
                count=0
                for attempt in range(2 if requested else 1):
                    try:
                        async for sentence in BoltDirector().stream(client,text,identity,history,context,metrics,reply_words=reply_words):
                            await queue.put(sentence);count+=1
                    except ValueError:
                        # Retry only an unspoken malformed completion. The
                        # visual action was already dispatched and cannot be
                        # duplicated, and no spoken words are replayed.
                        if not requested or count or attempt:raise
                    if count or not requested:break
                    context['force_spoken_reaction']=True;metrics['spoken_reaction_retries']=attempt+1
                if requested and not count:raise RuntimeError('The language model did not provide a spoken reaction')
        except asyncio.CancelledError:raise
        except Exception as exc:await queue.put(exc)
        await queue.put(None)

    producer=asyncio.create_task(produce());spoken=[];scripts=[];cues=[];total_samples=0;index=0
    try:
        while True:
            sentence=await queue.get()
            if sentence is None:break
            if isinstance(sentence,Exception):raise sentence
            spec=sentence.spec();spoken.append(sentence);scripts.append(spec['script'])
            at=total_samples/24000
            current=plan(spec)['sentences']
            for p in current:cues.append({**p,'start_s':p['start_s']+at,'end_s':p['end_s']+at})
            if context.get('event'):
                from avatar_personality import event_steps
                avatar_steps=event_steps(context)
            metadata={'text':join_sentences(spoken),'script':' '.join(scripts),'language':spec['language'],
                'performance':{'sentences':list(cues),'source':'Local Nemotron sentence cues + fixed GENERAL catalogue',
                    'timing':'Sentence starts anchored to actual preceding PCM duration; within-sentence estimates'},
                'avatar_actions':avatar_steps,'avatar_action_id':action_id,'avatar_event':context.get('event'),
                'interaction_actions':actions if index==0 else [],'sources':sources,'speaker':identity,
                'template_id':spec['template_id'],'sentence_index':index,'sentence_text':sentence.text,
                'timing':{**metrics,'sentence_ready_ms':round((time.perf_counter()-start)*1000,2),'director_mode':'bolt',
                    'director_model':'nemotron-bolt','template_id':spec['template_id']}}
            yield 'sentence',metadata
            if not tts:index+=1;continue
            # Each completed sentence is admitted immediately; LLM generation
            # continues while this consumer performs the first TTS inference.
            body={'model':'small_rl','input_mode':'manual','format':'structured',
                'message':spec['manual_prompt'],'language':spec['language'],
                'generation':'single','n':1,'batch_size':1,'streaming':True,
                'chunk_frames':chunk_frames,'cancel_on_disconnect':True,
                'temperature':.7,'guidance':2.,'voice_mode':'profile','voice_profile':voice,
                'voice_conversion':False,'speaker_steering':False,'dynamic_reference':False,
                'align':True,'seed':int(os.environ.get('ROBO_BOLT_SEED','777'))+index}
            tts_start=time.perf_counter();resampler=StreamingAudioResampler(48000,24000);played=False
            async def wait_notice(text,stage):pass
            async with admitted_turn(client,body,wait_notice,timeout=.25 if context.get('event') else 45.) as response:
                async for line in response.aiter_lines():
                    if not line.startswith('data: '):continue
                    event=json.loads(line[6:]);kind=event.get('event')
                    if kind=='error':raise RuntimeError('TTS sentence generation failed')
                    if kind=='result':
                        metrics.setdefault('tts_server_stages',[]).append({k:event.get(k) for k in
                            ('generation_ms','alignment_ms','scoring_ms','speaker_preparation_ms','total_ms','performance')})
                    url=event.get('url') if kind=='audio_chunk' else None
                    if kind=='result' and not played:
                        candidates=event.get('candidates') or []
                        selected=next((c for c in candidates if c.get('index')==event.get('winner',0)),candidates[0] if candidates else {})
                        url=selected.get('url')
                    if not url:continue
                    response_audio=await client.get(url);response_audio.raise_for_status()
                    pcm=resampler.process(await decode_mp3(response_audio.content))
                    if not len(pcm):continue
                    if not played:
                        metrics.setdefault('tts_first_audio_ms',round((time.perf_counter()-tts_start)*1000,2))
                        metrics.setdefault('first_audio_ms',round((time.perf_counter()-start)*1000,2))
                    played=True;total_samples+=len(pcm);yield 'audio',pcm
            tail=resampler.process(np.zeros(0,np.float32),final=True)
            if len(tail):total_samples+=len(tail);yield 'audio',tail
            if not played:raise RuntimeError('TTS produced no audio')
            # Replace the estimated duration before appending the next cue.
            if cues:cues[-1]['end_s']=total_samples/24000
            metrics.setdefault('sentence_tts',[]).append({'sentence':index,'total_ms':round((time.perf_counter()-tts_start)*1000,2),
                'audio_s':round(total_samples/24000-at,3)})
            index+=1
        await producer
        metrics.update(total_ms=round((time.perf_counter()-start)*1000,2),audio_s=round(total_samples/24000,3),sentences=index)
        if total_samples:metrics['generation_rtf']=round(metrics['total_ms']/1000/(total_samples/24000),4)
        yield 'done',metrics
    finally:
        producer.cancel();await asyncio.gather(producer,return_exceptions=True)
