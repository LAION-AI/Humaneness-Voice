import asyncio,json,unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock,patch
import numpy as np
from bolt_text import SentenceFramer,ASRSentenceCommitter,select_template,CATALOGUE,join_sentences
from sentence_session import OpenVoiceMemory,name_claim,SentenceSession

class BoltTextTests(unittest.TestCase):
    def test_square_acting_keywords_are_not_literal_speech(self):
        source='[soft, warm] Danke. [interest, calm] Wie geht es dir?'
        for width in (1,2,7,len(source)):
            f=SentenceFramer();out=[]
            for at in range(0,len(source),width):out+=f.feed(source[at:at+width])
            out+=f.feed('',final=True)
            self.assertEqual([s.text for s in out],['Danke.','Wie geht es dir?'])
            self.assertEqual(out[0].direction,'soft, warm')
            self.assertNotIn('[soft',out[0].spec()['manual_prompt'])

    def test_internal_playback_note_never_leaks_across_stream_boundaries(self):
        note='[My preceding reply was interrupted. The following text was generated; only a prefix may have been heard. Do not assume it was all heard or repeat it automatically.]'
        for prefix in ('','(amusement) '):
            source=prefix+note+'\n(amusement) Robby-Tornado, los geht\'s!'
            for width in (1,2,7,31,len(source)):
                f=SentenceFramer();out=[]
                for at in range(0,len(source),width):
                    out+=f.feed(source[at:at+width]);self.assertFalse(any('preceding' in s.text or 'heard' in s.text for s in out))
                out+=f.feed('',final=True)
                self.assertEqual([s.text for s in out],["Robby-Tornado, los geht's!"])
                self.assertNotIn('interrupted',out[0].spec()['manual_prompt'])
    def test_requested_poem_and_story_structure_do_not_get_a_generic_short_reply(self):
        from bolt_director import response_guidance
        self.assertIn('exactly 8 actual poem lines',response_guidance('Ein Gedicht mit acht Zeilen.',[]))
        self.assertIn('exactly 8 actual poem lines',response_guidance('A poem in two stanzas of four lines each.',[]))
        self.assertIn('exactly 12 actual poem lines',response_guidance('Write a twelve-line poem.',[]))
        self.assertIn('exactly 4 actual poem lines',response_guidance('Schreib noch vier zusätzliche Zeilen.',[{'role':'user','content':'Schreib ein Gedicht.'}]))
        self.assertIn('exactly 5 complete prose sentences',response_guidance('Eine Geschichte in fünf Sätzen.',[]))
        self.assertNotIn('poetry',response_guidance('Bitte kein Gedicht, ich habe eine Frage.',[]))
    def test_ordinary_negation_does_not_route_poems_into_rule_tools(self):
        from bolt_director import needs_tools
        for text in ['Ein Gedicht mit acht Zeilen. Keine Einleitung.','Bitte kein Smalltalk, schreib das Gedicht.']:
            self.assertFalse(needs_tools(text,[],{'rules':[]}))
        self.assertTrue(needs_tools('Wenn ich Bananen sage, unterbrich mich.',[],{'rules':[]}))
        self.assertTrue(needs_tools('Unterbrich mich bitte nicht mehr.',[],{'rules':[]}))
    def test_incremental_cues_and_sentences_at_arbitrary_token_boundaries(self):
        text='(happy, soft) Schön, dich zu sehen! (interest, calm) Was machen wir heute?'
        for width in (1,2,7,100):
            f=SentenceFramer();out=[]
            for at in range(0,len(text),width):out+=f.feed(text[at:at+width])
            out+=f.feed('',final=True)
            self.assertEqual([x.text for x in out],['Schön, dich zu sehen!','Was machen wir heute?'])
            self.assertEqual(out[1].direction,'interest, calm')
    def test_catalogue_is_fixed_and_emotions_map_without_model(self):
        self.assertEqual(len(CATALOGUE),100)
        self.assertTrue(select_template('happy, soft')['id'].startswith('elation.'))
        self.assertTrue(select_template('pleasure_ecstasy, intense')['id'].startswith('pleasure_ecstasy.'))
        self.assertIn('unhurried',select_template('interest, slow')['general'])
    def test_decimal_and_abbreviation_not_prematurely_split(self):
        f=SentenceFramer();out=f.feed('(interest) Dr. Bob hat 3.5 Euro. Noch etwas?',final=True)
        self.assertEqual([s.text for s in out],['Dr. Bob hat 3.5 Euro.','Noch etwas?'])
    def test_poetry_without_full_stops_streams_all_lines_and_stanzas(self):
        source='(amusement, warm) Ein Roboter tanzt im Mondenschein\n(affection, soft) und lädt die kleinen Sterne ein\n\n(elation) Er lacht und dreht sich rundherum\n(amusement) und fällt vor Freude beinahe um'
        expected='Ein Roboter tanzt im Mondenschein\nund lädt die kleinen Sterne ein\n\nEr lacht und dreht sich rundherum\nund fällt vor Freude beinahe um'
        for width in (1,2,7,31,1000):
            f=SentenceFramer();out=[]
            for at in range(0,len(source),width):out+=f.feed(source[at:at+width])
            out+=f.feed('',final=True)
            self.assertEqual(len(out),4);self.assertEqual(join_sentences(out),expected)
            self.assertEqual(out[-1].direction,'amusement')
    def test_new_cue_can_close_an_unpunctuated_line(self):
        f=SentenceFramer();out=f.feed('(hope) First verse (elation) Second verse',final=True)
        self.assertEqual([s.text for s in out],['First verse','Second verse'])
    def test_long_unpunctuated_prose_is_bounded_without_word_loss(self):
        words=['wort'+str(i) for i in range(220)]
        source='(interest) '+' '.join(words);f=SentenceFramer();out=[]
        for at in range(0,len(source),3):out+=f.feed(source[at:at+3])
        out+=f.feed('',final=True)
        self.assertEqual(join_sentences(out).split(),words)
        self.assertTrue(all(len(s.text.split())<=64 for s in out))
    def test_stable_asr_prefix_commits_once_and_keeps_followup(self):
        c=ASRSentenceCommitter();text='Das ist ein Satz. Und noch einer.'
        self.assertEqual(c.update(text,0),[])
        self.assertEqual(c.update(text,.32),['Das ist ein Satz.'])
        self.assertEqual(c.update(text,.64),[])
        self.assertEqual(c.update(text,.96),['Und noch einer.'])
        self.assertEqual(c.update(text,1.3,final=True),[])
        self.assertEqual(c.update('Das war ganz anders.',2,final=True),[])
    def test_partial_sentence_flush_and_silent_event(self):
        c=ASRSentenceCommitter();self.assertEqual(c.update('Bye bye',0),[])
        self.assertEqual(c.update('Bye bye',.32,final=True),['Bye bye'])
        self.assertEqual(SentenceFramer().feed('[silent]',final=True),[])
    def test_standalone_visual_notes_never_become_empty_tts_requests(self):
        f=SentenceFramer();out=f.feed('(amusement) Schau mal!\n*[grimace]*',final=True)
        self.assertEqual([p.text for p in out],['Schau mal!'])
        self.assertEqual(SentenceFramer().feed('**[smiles]**',final=True),[])
        self.assertEqual(SentenceFramer().feed('(interest) Hier [0.25 seconds pause] bin ich.',final=True)[0].text,'Hier [0.25 seconds pause] bin ich.')

class BoltSpeakersTests(unittest.TestCase):
    def setUp(self):self.v=np.eye(8,dtype=np.float32);self.m=OpenVoiceMemory()
    def test_unknown_voices_are_answered_without_introduction(self):
        a=self.m.assign(self.v[0],'Wie geht es dir heute?')
        b=self.m.assign(self.v[1],'Was möchtest du heute machen?')
        self.assertNotEqual(a['speaker_id'],b['speaker_id']);self.assertTrue(a['needs_name'])
        self.assertEqual(self.m.assign(self.v[0],'Ich habe eine Frage.')['speaker_id'],a['speaker_id'])
    def test_explicit_voice_correction_links_multiple_prototypes(self):
        christoph=self.m.assign(self.v[0],'Ich bin Christoph.')
        guest=self.m.assign(self.v[1],'Guten Morgen, kleiner Roboter.')
        corrected=self.m.assign(self.v[1],'Nein, nein, ich bin auch Christoph. Du hast mich verwechselt.')
        self.assertEqual(corrected['speaker_id'],christoph['speaker_id'])
        self.assertNotIn(guest['speaker_id'],self.m.users)
        for v in self.v[:2]:self.assertEqual(self.m.assign(v,'Was machen wir?')['speaker_id'],christoph['speaker_id'])
        bob=self.m.assign(self.v[2],'My name is Bob.')
        self.assertNotEqual(bob['speaker_id'],christoph['speaker_id'])
    def test_name_claims_and_uncertain_audio_dont_force_identity(self):
        self.assertEqual(name_claim('Ich bin auch Christoph.'),'Christoph')
        for text in ('Christoph ist hier.','Ich bin müde.','Bob sagt hallo.'):
            self.assertIsNone(name_claim(text))
        result=self.m.assign(None,'Hallo.')
        self.assertFalse(result['enrolled']);self.assertEqual(len(self.m.users),0)
    def test_inactive_session_profiles_expire(self):
        now=[0.];m=OpenVoiceMemory(clock=lambda:now[0]);m.timeout_s=60
        m.assign(self.v[0],'Ich bin Christoph.');now[0]=61
        self.assertEqual(m.status(),[])
    def test_unique_name_spelling_is_repaired_only_on_explicit_correction(self):
        first=self.m.assign(self.v[0],'Ich bin Christoph.')
        self.m.assign(self.v[1],'Hallo, wie geht es dir?')
        fixed=self.m.assign(self.v[1],'Nein, ich bin auch Christos.')
        self.assertEqual(fixed['speaker_id'],first['speaker_id'])
        self.assertEqual(fixed['display_name'],'Christoph')
        self.assertEqual(fixed['name_resolution']['heard'],'Christos')
        other=OpenVoiceMemory();a=other.assign(self.v[0],'Ich bin Christoph.')
        b=other.assign(self.v[1],'Ich bin Christos.')
        self.assertNotEqual(a['speaker_id'],b['speaker_id'])

class BoltAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        optional=patch('voice_captioning.describe',new_callable=AsyncMock,return_value={'status':'disabled'})
        optional.start();self.addCleanup(optional.stop)

    async def test_acknowledgment_only_story_gets_one_bounded_real_continuation(self):
        from bolt_director import BoltDirector
        calls=[]
        class Response:
            def __init__(self,text):self.text=text
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            def raise_for_status(self):pass
            async def aiter_lines(self):
                yield 'data: '+json.dumps({'choices':[{'delta':{'content':self.text}}]})
                yield 'data: '+json.dumps({'choices':[{'delta':{},'finish_reason':'stop'}]})
        class Client:
            def stream(self,*args,**kwargs):
                calls.append(kwargs['json'])
                return Response('(interest) Das ist eine wunderbare Bitte, die ich sehr gerne erfülle.' if len(calls)==1 else
                    '(amusement) Ein Roboter fand einen Stern. (interest) Er brachte ihn nach Hause. (affection) Dort leuchtete er für alle Freunde.')
        metrics={};parts=[]
        async for part in BoltDirector().stream(Client(),'Erzähl mir eine kleine Geschichte.',{},[],metrics=metrics):parts.append(part)
        self.assertEqual(len(calls),2)
        self.assertIn('COMPLETE actual story',calls[1]['messages'][-1]['content'])
        self.assertIn('leuchtete',join_sentences(parts));self.assertTrue(metrics['response_complete'])
        self.assertEqual(metrics['completion_repairs'],1)

    async def test_premature_poem_end_continues_missing_lines_without_restarting(self):
        from bolt_director import BoltDirector
        calls=[]
        class Response:
            def __init__(self,text):self.text=text
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            def raise_for_status(self):pass
            async def aiter_lines(self):
                yield 'data: '+json.dumps({'choices':[{'delta':{'content':self.text}}]})
                yield 'data: '+json.dumps({'choices':[{'delta':{},'finish_reason':'stop'}]})
        class Client:
            def stream(self,*args,**kwargs):
                calls.append(kwargs['json']);start=1 if len(calls)==1 else 7;end=7 if len(calls)==1 else 9
                return Response('\n'.join('(amusement) Zeile '+str(i) for i in range(start,end)))
        metrics={};parts=[]
        async for part in BoltDirector().stream(Client(),'Schreib ein Gedicht mit acht Zeilen.',{},[],metrics=metrics):parts.append(part)
        self.assertEqual(join_sentences(parts),'\n'.join('Zeile '+str(i) for i in range(1,9)))
        self.assertFalse(calls[0]['chat_template_kwargs']['conversational_mode'])
        self.assertGreaterEqual(calls[0]['max_tokens'],2048)
        self.assertIn('exactly 2 additional verse lines',calls[1]['messages'][-1]['content'])
        self.assertTrue(metrics['response_complete']);self.assertEqual(metrics['completion_repairs'],1)

    async def test_joke_setup_is_continued_but_explicit_one_liner_stays_short(self):
        from bolt_director import BoltDirector
        class Response:
            def __init__(self,text):self.text=text
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            def raise_for_status(self):pass
            async def aiter_lines(self):
                yield 'data: '+json.dumps({'choices':[{'delta':{'content':self.text}}]})
                yield 'data: '+json.dumps({'choices':[{'delta':{},'finish_reason':'stop'}]})
        class Client:
            def __init__(self):self.calls=0
            def stream(self,*args,**kwargs):
                self.calls+=1
                return Response('(interest) Fips stand am Bahnhof und betrachtete die Anzeige.' if self.calls==1 else
                    '(amusement) Er suchte einen Anschluss und fand nur eine Steckdose.')
        client=Client();metrics={}
        parts=[p async for p in BoltDirector().stream(client,'Erzähl einen vollständigen Witz über Fips.',{},[],metrics=metrics)]
        self.assertEqual(client.calls,2);self.assertIn('Steckdose',join_sentences(parts));self.assertTrue(metrics['response_complete'])
        client=Client();metrics={}
        parts=[p async for p in BoltDirector().stream(client,'Erzähl einen Witz als Einzeiler.',{},[],metrics=metrics)]
        self.assertEqual(client.calls,1);self.assertEqual(metrics['task_minimum_units'],0)
    async def test_current_native_words_arrive_before_quiet_flush(self):
        import paths
        from server import RoboConversation
        from app.ears.conversation import Conversation
        order=[];c=RoboConversation.__new__(RoboConversation);c.ears=SimpleNamespace(sr=16000)
        c.avatar_personality=SimpleNamespace(last_asr=-100.)
        async def preview(text,at):order.append(('words',text))
        async def observe(event,end):order.append(('waveform',end))
        async def advance():order.append(('flush',None))
        c.sentence_stream=SimpleNamespace(preview_text=preview,observe=observe,advance=advance,foreground_last=.32,raw_loud_at=.32);c.ws=SimpleNamespace(send_json=AsyncMock())
        with patch.object(Conversation,'_on_events',new_callable=AsyncMock):
            await c._on_events([{'kind':'SPEAKER_FEATURES'},{'kind':'PARTIAL','text':'Complete sentence with final words.'}],5120)
        self.assertEqual(order,[('waveform',5120),('words','Complete sentence with final words.'),('flush',None)])
        self.assertGreater(c.avatar_personality.last_asr,0.,'Sentence mode must also suppress idle antics while listening')
    async def test_ready_embedding_is_reused_instead_of_recomputed_on_submit(self):
        c=SimpleNamespace(ws=SimpleNamespace(send_json=AsyncMock()),speaking=False,halting=False,
            _send_turn=AsyncMock(),current_speaker=None)
        s=SentenceSession(c);s.wake.configure(False);ready=asyncio.create_task(asyncio.sleep(0));await ready
        cached={'task':ready,'embedding':np.eye(192,dtype=np.float32)[0],'compute_ms':40.}
        await s.queue.put(('Wie geht es dir?',np.ones(32000,np.float32)*.01,2.,0.,cached))
        with patch('speaker_teacher.get_cached_teacher',side_effect=AssertionError('Must use prefetch')):
            worker=asyncio.create_task(s.deliver())
            for _ in range(50):
                if c._send_turn.await_count:break
                await asyncio.sleep(.001)
            worker.cancel();await asyncio.gather(worker,return_exceptions=True)
        c._send_turn.assert_awaited_once_with('Wie geht es dir?')
        endpoint=next(call.args[0] for call in c.ws.send_json.call_args_list if call.args[0]['type']=='endpoint')
        self.assertTrue(endpoint['timing']['speaker_prefetch_used'])

if __name__=='__main__':unittest.main()
