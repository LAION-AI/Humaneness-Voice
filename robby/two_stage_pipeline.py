"""Reaction -> informed transition -> unchanged main answer, with TTS lookahead."""
import asyncio,base64,time
import paths
import numpy as np
from bolt_text import Sentence,join_sentences
from quick_reaction import AudioEnvelope,duplicate_acknowledgment
from speech_language import reply_language
from performance import plan

async def render(client,live,*,text,identity,history,queue,first_main,reaction_task,
                 metrics,start,context,avatar_steps,action_id,actions,sources,lookups,chunk_frames):
    from app.runtime.audio import StreamingAudioResampler
    import informed_bridge
    stages=asyncio.Queue(maxsize=2);language=reply_language(text,history)
    spoken=[];scripts=[];cues=[];samples=0;index=0;actions_sent=False
    metrics.update(tts_backend='continuous_batch',tts_weight_instances=1,tts_temperature=.8,informed_bridge_status='pending')
    async def publish(role,sentence,order,job=None):
        job=job or await live.submit(sentence,role,order,chunk_frames)
        await stages.put((role,sentence,job))
    async def prepare():
        try:
            # The requested two-stage mode gives the tiny reaction its bounded
            # deadline even when Qwen is already ready. Both LLMs started earlier.
            reaction=await reaction_task;intro=None
            metrics.update(quick_reply_generation_ms=reaction['ms'],quick_reply_model=reaction.get('model'),quick_reply_status=reaction['status'])
            metrics.update(quick_reply_memory_entries=reaction.get('memory_entries',0),quick_reply_memory_person_matched=reaction.get('memory_person_matched',False))
            if reaction.get('text'):
                intro=Sentence(reaction['text'],reaction.get('direction','contentment, soft, warm'),language='de' if language=='German' else 'en')
                metrics['quick_reply_words']=reaction['word_count'];metrics['quick_reply_status']='selected'
                await publish('reaction',intro,0)
            main=await first_main
            if isinstance(main,Exception):raise main
            if main is None:
                await live.finish();await stages.put(None);return
            if intro and duplicate_acknowledgment(intro.text,main.text):
                metrics['duplicate_first_ack_skipped']=True;main=await queue.get()
                if isinstance(main,Exception):raise main
                if main is None:
                    await live.finish();await stages.put(None);return
            # Start the actual answer TTS NOW. It is buffered privately until
            # the informed transition has been played in the correct order.
            main_job=await live.submit(main,'answer',2,chunk_frames)
            following=[]
            if intro and not context.get('environment_proactive'):
                # A first sentence may be only a preamble. Peek at the next one
                # with a strict 60-ms budget, retaining every item in order.
                try:following.append(await asyncio.wait_for(queue.get(),.06))
                except TimeoutError:pass
                from bolt_director import response_plan
                # A joke's second sentence can be its entire punchline. Never
                # give that away to a model asked to write an earlier lead-in.
                visible=[] if response_plan(text,history)['kind']=='joke' else [s for s in following if isinstance(s,Sentence)]
                opening=main.text+' '+ ' '.join(s.text for s in visible)
                metrics['informed_bridge_context_sentences']=1+len(visible)
                bridge=await informed_bridge.generate(client,text,intro.text,opening.strip(),history,language)
                metrics.update(informed_bridge_generation_ms=bridge['ms'],informed_bridge_status=bridge['status'])
                if bridge.get('text'):
                    # Match the main utterance's delivery at the second seam.
                    link=Sentence(bridge['text'],main.direction,language=main.language)
                    metrics['informed_bridge_words']=bridge['word_count']
                    await publish('informed_bridge',link,1)
            else:metrics['informed_bridge_status']='short_environment_turn' if context.get('environment_proactive') else 'no_first_reaction'
            await publish('answer',main,2,main_job)
            order=3
            while True:
                sentence=following.pop(0) if following else await queue.get()
                if sentence is None:break
                if isinstance(sentence,Exception):raise sentence
                await publish('answer',sentence,order);order+=1
            await live.finish();await stages.put(None)
        except asyncio.CancelledError:raise
        except Exception as exc:await stages.put(exc)
    preparer=asyncio.create_task(prepare())
    try:
        while True:
            stage=await stages.get()
            if stage is None:break
            if isinstance(stage,Exception):raise stage
            role,sentence,job=stage;spec=sentence.spec();spoken.append(sentence);scripts.append(spec['script'])
            at=samples/24000;current=plan(spec)['sentences']
            for cue in current:cues.append({**cue,'start_s':cue['start_s']+at,'end_s':cue['end_s']+at})
            yield 'sentence',{'text':join_sentences(spoken),'script':' '.join(scripts),'language':spec['language'],
                'performance':{'sentences':list(cues),'source':'Two-stage local reaction + Qwen + shared TTS batch',
                    'timing':'Sentence starts anchored to preceding PCM; within-sentence estimates'},
                'avatar_actions':avatar_steps,'avatar_action_id':action_id,'avatar_event':context.get('event'),
                'interaction_actions':actions if role=='answer' and not actions_sent else [],'sources':sources,'lookups':lookups,'speaker':identity,
                'sentence_role':role,'template_id':spec['template_id'],'sentence_index':index,'sentence_text':sentence.text,
                'timing':{**metrics,'sentence_ready_ms':round((time.perf_counter()-start)*1000,2),'director_mode':'bolt',
                    'director_model':metrics.get('llm_model'),'template_id':spec['template_id']}}
            if role=='answer' and actions:actions_sent=True
            resampler=StreamingAudioResampler(48000,24000);envelope=AudioEnvelope() if role!='answer' else None;played=False
            async for event in live.events(job):
                if event['event']=='job_done':
                    metrics.setdefault('tts_server_stages',[]).append({k:event.get(k) for k in ['generation_ms','alignment_ms','alignment','frames','live_alignment']})
                    continue
                pcm=np.frombuffer(base64.b64decode(event['pcm_f32'],validate=True),dtype='<f4')
                if not np.isfinite(pcm).all():raise RuntimeError('Invalid live PCM')
                pcm=resampler.process(pcm)
                if envelope is not None:pcm=envelope.feed(pcm)
                if not len(pcm):continue
                if not played:
                    now=time.perf_counter();metrics.setdefault('first_audio_ms',round((now-start)*1000,2))
                    metrics.setdefault('tts_first_audio_ms',round(((job.first_pcm or now)-job.started)*1000,2))
                    if role=='answer':
                        metrics.setdefault('answer_first_audio_ms',round((now-start)*1000,2))
                        metrics.setdefault('answer_audio_ready_ms',round(((job.first_pcm or now)-start)*1000,2))
                    elif role=='informed_bridge':
                        metrics['informed_bridge_first_audio_ms']=round((now-start)*1000,2);metrics['informed_bridge_status']='played'
                    else:metrics['quick_reply_status']='played'
                    yield 'latency',dict(metrics)
                played=True;samples+=len(pcm);yield 'audio',pcm
            tail=resampler.process(np.zeros(0,np.float32),final=True)
            if envelope is not None:tail=np.concatenate((envelope.feed(tail),envelope.finish()))
            if len(tail):samples+=len(tail);yield 'audio',tail
            if not played:raise RuntimeError('Continuous TTS produced no audio')
            if cues:cues[-1]['end_s']=samples/24000
            if role=='reaction':metrics.update(quick_reply_audio_s=round(samples/24000-at,3),quick_reply_tts_ms=round((time.perf_counter()-job.started)*1000,2))
            if role=='informed_bridge':metrics['informed_bridge_audio_s']=round(samples/24000-at,3)
            metrics.setdefault('sentence_tts',[]).append({'sentence':index,'role':role,'audio_s':round(samples/24000-at,3),
                'total_ms':round((time.perf_counter()-job.started)*1000,2),'prefetched':True})
            index+=1
        await preparer
        await asyncio.wait_for(asyncio.shield(live.receiver),10)
        metrics.update(total_ms=round((time.perf_counter()-start)*1000,2),audio_s=round(samples/24000,3),sentences=index,
            continuous_tts=live.performance)
        if samples:metrics['generation_rtf']=round(metrics['total_ms']/1000/(samples/24000),4)
        yield 'done',metrics
    finally:
        preparer.cancel();await asyncio.gather(preparer,return_exceptions=True);await live.close()
