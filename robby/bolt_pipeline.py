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
                   interaction_state=None,chunk_frames=None,tts=True,preset=None,reply_words=None,llm_selection=None,credential=None,quick_replies=True,memory=None):
    from app.runtime.audio import StreamingAudioResampler
    from tts_bridge import decode_mp3
    start=time.perf_counter();metrics={};queue=asyncio.Queue(maxsize=4)
    chunk_frames=chunk_frames or int(os.environ.get('ROBO_BOLT_CHUNK_FRAMES','20'))
    if chunk_frames not in (2,5,10,20):raise ValueError('Unsupported TTS chunk size')
    from avatar_commands import explicit_actions,silent_requested
    context=dict(avatar_context or {})
    if memory is not None and not context.get('event'):
        try:
            history,remembered=await memory.prepare(text,identity,history,llm_selection)
            if remembered:
                context['knowledge_memory']=remembered
                metrics.update(memory_retrieval_ms=remembered['metrics']['retrieval_ms'],
                    memory_entries=len(remembered['entries']))
        except Exception as exc:
            await memory.notice(state='error',error=type(exc).__name__)
    if (interaction_state or {}).get('current_rule_applied'):
        context['applied_interruption_rules']=(interaction_state or {}).get('rules',[])
    requested=explicit_actions(text,sum(h.get('role')=='user' for h in history)) if preset is None and not context.get('event') else []
    avatar_steps=list(requested);actions=[];sources=[];lookups=[]
    silent=bool(requested and silent_requested(text))
    # A direct screen command must work even if LLM/TTS playback is delayed.
    # Dispatch once immediately; the matching ID prevents metadata replay.
    immediate=bool(requested)
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

    from quick_reaction import eligible,generate,duplicate_acknowledgment
    reaction_enabled=bool(quick_replies and tts and eligible(text,history,context,preset,reply_words,
        remote=(llm_selection or {}).get('model')=='qwen'))
    if reaction_enabled:context['parallel_quick_reaction']=True
    metrics['quick_reply_status']='racing' if reaction_enabled else 'disabled_or_ineligible'

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
                from llm_backends import config
                result=await CompactDirector().generate(client,text,identity,history,
                    interaction_state=interaction_state,avatar_context=context,
                    provider=config(llm_selection,credential) if llm_selection is not None else local_config())
                avatar_steps[:]=requested or result.get('avatar_actions',[]);actions[:]=result.get('interaction_actions',[])
                sources[:]=result.get('sources',[])
                lookups[:]=result.get('lookups',[])
                framer=SentenceFramer(lines_only=response_plan(text,history)['kind']=='poem')
                for sentence in framer.feed(result['script'],final=True):
                    sentence.language=result['language'];await queue.put(sentence)
                metrics['tool_director_ms']=result['director_ms']
                metrics.update({k:v for k,v in result.items() if k.startswith('llm_') or k=='director_model'})
            else:
                count=0
                for attempt in range(2 if requested else 1):
                    try:
                        async for sentence in BoltDirector().stream(client,text,identity,history,context,metrics,reply_words=reply_words,backend=llm_selection,credential=credential):
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
    from speech_language import reply_language
    reaction_task=asyncio.create_task(generate(client,text,identity,history,reply_language(text,history))) if reaction_enabled else None
    first_main=asyncio.create_task(queue.get());intro=None;initial=True;main_started=False;actions_sent=False
    try:
        if reaction_task:
            await asyncio.wait((first_main,reaction_task),return_when=asyncio.FIRST_COMPLETED)
            if first_main.done():
                metrics['quick_reply_status']='main_answer_won';reaction_task.cancel()
            else:
                reaction=await reaction_task
                metrics.update(quick_reply_generation_ms=reaction['ms'],quick_reply_model=reaction['model'],quick_reply_status=reaction['status'])
                if first_main.done():metrics['quick_reply_status']='main_answer_won'
                elif reaction.get('text'):
                    intro=Sentence(reaction['text'],reaction.get('direction','interest, warm, natural'),language='de' if reply_language(text,history)=='German' else 'en')
                    metrics.update(quick_reply_status='selected',quick_reply_words=reaction['word_count'])
        pending_intro=intro
        while True:
            role='reaction' if pending_intro else 'answer'
            if pending_intro:sentence=pending_intro;pending_intro=None
            elif initial:sentence=await first_main;initial=False
            else:sentence=await queue.get()
            if sentence is None:break
            if isinstance(sentence,Exception):raise sentence
            if role=='answer' and not main_started:
                main_started=True
                if intro and duplicate_acknowledgment(intro.text,sentence.text):
                    metrics['duplicate_first_ack_skipped']=True;continue
            spec=sentence.spec();spoken.append(sentence);scripts.append(spec['script'])
            at=total_samples/24000
            current=plan(spec)['sentences']
            for p in current:cues.append({**p,'start_s':p['start_s']+at,'end_s':p['end_s']+at})
            if context.get('event'):
                from avatar_personality import event_steps
                avatar_steps=event_steps(context)
            metadata={'text':join_sentences(spoken),'script':' '.join(scripts),'language':spec['language'],
                'performance':{'sentences':list(cues),'source':metrics.get('llm_model','Nemotron')+' sentence cues + fixed GENERAL catalogue',
                    'timing':'Sentence starts anchored to actual preceding PCM duration; within-sentence estimates'},
                'avatar_actions':avatar_steps,'avatar_action_id':action_id,'avatar_event':context.get('event'),
                'interaction_actions':actions if not actions_sent else [],'sources':sources,'lookups':lookups,'speaker':identity,
                'sentence_role':role,'template_id':spec['template_id'],'sentence_index':index,'sentence_text':sentence.text,
                'timing':{**metrics,'sentence_ready_ms':round((time.perf_counter()-start)*1000,2),'director_mode':'bolt',
                    'director_model':metrics.get('llm_model','nemotron-bolt'),'template_id':spec['template_id']}}
            yield 'sentence',metadata
            if actions:actions_sent=True
            if not tts:index+=1;continue
            # Each completed sentence is admitted immediately; LLM generation
            # continues while this consumer performs the first TTS inference.
            body={'model':'small_rl','input_mode':'manual','format':'structured',
                'message':spec['manual_prompt'],'language':spec['language'],
                'generation':'single','n':1,'batch_size':1,'streaming':True,'stream_transport':'pcm_f32','live_only':True,
                'chunk_frames':5 if role=='reaction' else chunk_frames,'cancel_on_disconnect':True,
                'temperature':.7,'guidance':2.,'voice_mode':'profile','voice_profile':voice,
                'voice_conversion':False,'speaker_steering':False,'dynamic_reference':False,
                'align':role!='reaction','seed':int(os.environ.get('ROBO_BOLT_SEED','777'))+index}
            if role=='reaction':
                # This is the model's duration CONDITION, not a hard ceiling.
                # A 6.4-second condition stretched four words over seven seconds.
                from quick_reaction import word_count
                body['seconds']=max(2.,word_count(sentence.text)*.4)
                body['first_chunk_frames']=2
            from quick_reaction import AudioEnvelope
            envelope=AudioEnvelope() if role=='reaction' else None
            tts_start=time.perf_counter();resampler=StreamingAudioResampler(48000,24000);played=False
            async def wait_notice(text,stage):pass
            async with admitted_turn(client,body,wait_notice,timeout=.25 if context.get('event') else 45.) as response:
                async for line in response.aiter_lines():
                    if not line.startswith('data: '):continue
                    event=json.loads(line[6:]);kind=event.get('event')
                    if kind=='error':
                        import logging
                        logging.getLogger('robo.pipeline').warning('TTS sentence failed: %s',str(event.get('error') or event.get('message') or event.get('text'))[:300])
                        raise RuntimeError('TTS sentence generation failed')
                    if kind=='result':
                        metrics.setdefault('tts_server_stages',[]).append({k:event.get(k) for k in
                            ('generation_ms','alignment_ms','scoring_ms','speaker_preparation_ms','total_ms','performance')})
                    url=event.get('url') if kind=='audio_chunk' else None
                    if kind=='result' and not played:
                        candidates=event.get('candidates') or []
                        selected=next((c for c in candidates if c.get('index')==event.get('winner',0)),candidates[0] if candidates else {})
                        url=selected.get('url')
                    if event.get('pcm_f32') is not None and kind=='audio_chunk':
                        import base64
                        if event.get('sample_rate')!=48000 or event.get('channels')!=1:raise ValueError('Unsupported PCM stream')
                        raw_pcm=np.frombuffer(base64.b64decode(event['pcm_f32'],validate=True),dtype='<f4')
                        if not np.isfinite(raw_pcm).all():raise ValueError('Invalid PCM stream')
                    elif url:
                        # Compatibility with older MP3-only Playground servers.
                        response_audio=await client.get(url);response_audio.raise_for_status()
                        raw_pcm=await decode_mp3(response_audio.content)
                    else:continue
                    pcm=resampler.process(raw_pcm)
                    if envelope is not None:pcm=envelope.feed(pcm)
                    if not len(pcm):continue
                    if not played:
                        if role=='answer':metrics.setdefault('answer_first_audio_ms',round((time.perf_counter()-start)*1000,2))
                        else:metrics['quick_reply_status']='played'
                        metrics.setdefault('tts_first_audio_ms',round((time.perf_counter()-tts_start)*1000,2))
                        metrics.setdefault('first_audio_ms',round((time.perf_counter()-start)*1000,2))
                        yield 'latency',dict(metrics)
                    played=True;total_samples+=len(pcm);yield 'audio',pcm
            tail=resampler.process(np.zeros(0,np.float32),final=True)
            if envelope is not None:tail=np.concatenate((envelope.feed(tail),envelope.finish()))
            if len(tail):total_samples+=len(tail);yield 'audio',tail
            if not played:raise RuntimeError('TTS produced no audio')
            # Replace the estimated duration before appending the next cue.
            if cues:cues[-1]['end_s']=total_samples/24000
            if role=='reaction':metrics.update(quick_reply_audio_s=round(total_samples/24000-at,3),quick_reply_tts_ms=round((time.perf_counter()-tts_start)*1000,2))
            metrics.setdefault('sentence_tts',[]).append({'sentence':index,'total_ms':round((time.perf_counter()-tts_start)*1000,2),
                'audio_s':round(total_samples/24000-at,3)})
            index+=1
        await producer
        metrics.update(total_ms=round((time.perf_counter()-start)*1000,2),audio_s=round(total_samples/24000,3),sentences=index)
        if total_samples:metrics['generation_rtf']=round(metrics['total_ms']/1000/(total_samples/24000),4)
        yield 'done',metrics
    finally:
        tasks=[t for t in (producer,first_main,reaction_task) if t is not None]
        for task in tasks:task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
