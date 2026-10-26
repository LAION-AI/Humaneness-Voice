import asyncio,json,threading,unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock,patch
import numpy as np
from sentence_session import SentenceSession,unit
from live_decisions import LiveDecisions,DecisionClient
from avatar_commands import explicit_actions


class ReliabilityTests(unittest.IsolatedAsyncioTestCase):
    def session(self):
        c=SimpleNamespace(ws=SimpleNamespace(closed=False,send_json=AsyncMock()),speaking=False,halting=False,
            identity_enabled=True,current_speaker=None,hop_pos=48000,
            session=SimpleNamespace(reset=Mock()),ears=SimpleNamespace(sr=16000),
            playback=SimpleNamespace(rid=1,heard_s=lambda at:.5,n16=16000,audio=np.zeros(0)),
            speaker_ext=SimpleNamespace(predictor=None),_on_barge=AsyncMock(),_send_turn=AsyncMock(),
            _send_interjection=AsyncMock(return_value=True))
        s=SentenceSession(c);c.sentence_stream=s;c.decisions=LiveDecisions(c)
        s.now=s.mic_now=s.last_speech=3.;s.begin=0.
        v=np.eye(192,dtype=np.float32);s.memory.assign(v[0],'Ich bin Christoph.');s.focus.accept('speaker-1')
        s.fast_anchors['speaker-1']=[v[0]];s.fast_records.append((3.,v[0]))
        s.records.append((0.,3.,np.ones(48000,np.float32)*.03))
        return s,c,v

    def test_common_elliptical_rotations_and_playful_actions(self):
        for text in ('Im Kreis drehen.','Bitte im Kreis drehen!','Dreh mal im Kreis!',
                     'Drehe dich einmal ganz herum.','Kannst du dich nochmal drehen?'):
            self.assertEqual(explicit_actions(text)[0]['kind'],'spin',text)
        for text in ('Mach mal Quatsch.','Mach ein Kunststück!','Spiel bitte Karussell.'):
            self.assertEqual([x['kind'] for x in explicit_actions(text)],['carousel','afterglow'],text)
        for text in ('Die Erde dreht sich im Kreis.','Warum soll ich mich im Kreis drehen?',
                     'Sag bitte: im Kreis drehen.','Dreh dich nicht im Kreis.'):
            self.assertEqual(explicit_actions(text),[],text)

    async def test_short_voice_rejection_cannot_crop_the_microphone(self):
        s,c,v=self.session();s.last_speech=1.;s.focus_activity={'accepted':False,'at':2.9,'last_owned':.5}
        s.vad=SimpleNamespace(feed=lambda x,end:3.1,active=True,probability=.9)
        await s._speech_audio(np.ones(320,np.float32),3.1)
        self.assertEqual(s.last_speech,3.1)

    async def test_embedding_prefetch_keeps_foreground_before_long_tv_tail(self):
        s,c,v=self.session()
        wave=np.r_[np.full(4*16000,.08,np.float32),np.full(3*16000,.005,np.float32)]
        s.records.clear();s.records.append((0.,7.,wave));s.now=s.mic_now=s.last_speech=7.
        captured=[]
        def embed(x):captured.append(x.copy());return v[0]
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=embed)):
            s.prepare_embedding(7.);await s.embedding_task
        self.assertEqual(len(captured),1)
        self.assertEqual(len(captured[0]),round(3.04*16000))
        self.assertGreater(float(captured[0].mean()),.075,'The final TV tail must not replace the owner request')

    async def test_single_word_stop_excludes_surrounding_background_from_voice_window(self):
        s,c,v=self.session();c.speaking=True;s.reply_started=0.;s.raw_loud_at=2.9
        s.stop_candidate={'at':2.6,'rid':1,'epoch':s.epoch}
        s.verify_barge=AsyncMock(return_value={'accepted':False})
        await s.assess_interrupt({'stop_text':'Stopp!'},3.,1)
        args=s.verify_barge.call_args
        self.assertEqual(args.kwargs,{'lexical':True,'brief':True})
        self.assertEqual(args.args[0],3.,'Fresh phonemes may extend the first partial cue')
        self.assertEqual(s.stop_candidate['at'],2.6,'Candidate expiry must not be refreshed by old ASR')

    async def test_long_background_preroll_cannot_dilute_foreground_level(self):
        s,c,v=self.session();s.memory.users['speaker-1']['level']=.08
        ready=asyncio.create_task(asyncio.sleep(0));await ready
        cached={'task':ready,'embedding':unit(.4*v[0]+np.sqrt(.84)*v[1]),'compute_ms':20.}
        wave=np.r_[np.full(15*16000,.003,np.float32),np.full(2*16000,.07,np.float32)]
        await s.queue.put(('Erzähl mir eine Geschichte.',wave,17.,0.,cached))
        worker=asyncio.create_task(s.deliver())
        for _ in range(100):
            await asyncio.sleep(.002)
            if c._send_turn.await_count:break
        worker.cancel();await worker
        c._send_turn.assert_awaited_once_with('Erzähl mir eine Geschichte.')

    async def test_quiet_impostor_never_teaches_fast_or_teacher_identity(self):
        s,c,v=self.session();s.memory.users['speaker-1']['level']=.1
        q=unit(.4*v[0]+np.sqrt(.84)*v[1])
        result=s.turn_decision(q,v[0],level=.01)
        self.assertFalse(result['accepted']);self.assertTrue(result['quiet_background'])
        # A strongly verified whisper can still match its registered person.
        self.assertTrue(s.turn_decision(v[0],v[0],level=.01)['accepted'])

    async def test_supported_intonation_dip_stays_with_owner_without_training_drift(self):
        s,c,v=self.session();q=unit(.21*v[0]+np.sqrt(1-.21**2)*v[2])
        result=s.turn_decision(q,v[0])
        self.assertTrue(result['accepted']);self.assertTrue(result['provisional'])
        self.assertEqual(len(s.memory.users['speaker-1']['vectors']),1)
        s.memory.assign(v[2],'Ich bin Robert.')
        self.assertFalse(s.turn_decision(v[2],v[0])['accepted'])

    async def test_weak_short_student_cannot_reject_good_whole_sentence(self):
        s,c,v=self.session()
        self.assertTrue(s.turn_decision(v[0],v[2])['accepted'])

    async def test_neural_tv_activity_can_finish_only_after_stable_recognized_words(self):
        s,c,v=self.session();s.vad=SimpleNamespace(active=True,probability=.95)
        s.preview='Kannst du bitte eine lustige Grimasse machen?'
        s.note_words(s.preview,3.);s.mic_now=3.7;self.assertFalse(s.quiet_ready())
        s.mic_now=4.7;self.assertTrue(s.quiet_ready())
        # A new word from the independent full-utterance pass postpones submit.
        s.note_words(s.preview+' Jetzt.',4.7);self.assertFalse(s.quiet_ready())
        # Decoder spelling differences alone do not keep the TV pause open.
        s.mic_now=6.4;s.note_words('Kannst du eine lustige Grimasse machen jetzt?',6.4)
        self.assertTrue(s.quiet_ready())

    async def test_quiet_tv_does_not_extend_foreground_clock_or_crop_ongoing_speech(self):
        s,c,v=self.session();s.memory.users['speaker-1']['level']=.08
        s.vad=SimpleNamespace(feed=lambda x,end:end,active=True,probability=.95)
        await s._speech_audio(np.full(320,.05,np.float32),3.)
        self.assertTrue(s.foreground_now)
        await s._speech_audio(np.full(320,.004,np.float32),3.8)
        self.assertTrue(s.quiet_ready());self.assertAlmostEqual(s.final_audio_end(3.8),3.32)
        await s._speech_audio(np.full(320,.05,np.float32),4.)
        self.assertFalse(s.quiet_ready())

    async def test_suffix_only_probe_retains_full_native_sentence(self):
        s,c,v=self.session();s.native_preview='Kannst du dich bitte einmal im Kreis drehen?'
        self.assertEqual(s.complete_text('im Kreis drehen'),s.native_preview)
        self.assertEqual(s.complete_text('Kannst du dich bitte einmal im Kreis drehen und dazu lachen?'),
                         'Kannst du dich bitte einmal im Kreis drehen und dazu lachen?')
        self.assertEqual(s.complete_text('Das ist eine andere Frage.'),'Das ist eine andere Frage.')

    async def test_handover_discards_old_queued_turns_and_accepts_new_voice_without_wake(self):
        s,c,v=self.session()
        await s.queue.put(('Alter Satz',np.ones(16000),3.,0.,None,{},None,0))
        s.invite_speaker()
        self.assertTrue(s.queue.empty());self.assertIsNone(s.focus.owner)
        self.assertEqual(s.segment_floor,3.);self.assertTrue(s.identity_state()['handover_pending'])
        worker=asyncio.create_task(s.deliver())
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=Mock(return_value=v[1]))):
            await s.queue.put(('Guten Morgen, wie geht es dir?',np.ones(32000,np.float32),5.,0.,None,{},None,s.admission_epoch))
            for _ in range(100):
                await asyncio.sleep(.002)
                if c._send_turn.await_count:break
        worker.cancel();await worker
        c._send_turn.assert_awaited_once();self.assertEqual(s.focus.owner,'speaker-2')
        self.assertFalse(s.identity_state()['handover_pending'])

    async def test_forget_clears_prototypes_invites_and_roster(self):
        s,c,v=self.session()
        s.permissions.command('Rede auch mit Alice.',{'speaker_id':'speaker-1'})
        self.assertTrue(s.permissions.pending)
        s.clear_identity()
        self.assertFalse(s.memory.users);self.assertFalse(s.fast_records);self.assertFalse(s.fast_anchors)
        self.assertFalse(s.focus.members);self.assertIsNone(s.permissions.pending);self.assertFalse(s.records)

    async def test_louder_named_human_replaces_unnamed_startup_background(self):
        s,c,v=self.session();s.memory.users['speaker-1'].update(named=False,name='Speaker 1',level=.005)
        worker=asyncio.create_task(s.deliver())
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=Mock(return_value=v[1]))):
            await s.queue.put(('Ich bin Christoph, schön dich kennenzulernen.',np.full(32000,.06,np.float32),4.,0.,None,{},None,s.admission_epoch))
            for _ in range(100):
                await asyncio.sleep(.002)
                if c._send_turn.await_count:break
        worker.cancel();await worker;c._send_turn.assert_awaited_once()
        self.assertEqual(s.focus.owner,'speaker-2');self.assertNotIn('speaker-1',s.focus.members)
        self.assertEqual(s.focus.primary,'speaker-2');self.assertEqual(s.memory.users['speaker-2']['name'],'Christoph')

    async def test_malformed_tool_ack_is_repaired_without_replaying_tool(self):
        from conversation_director import CompactDirector
        posts=[]
        class Client:
            async def post(self,url,**kwargs):
                posts.append(json.loads(json.dumps(kwargs['json'])))
                message=({'tool_calls':[{'id':'rule-1','type':'function','function':{'name':'set_interruption_rule',
                    'arguments':json.dumps({'condition':'Talks about bananas','trigger_terms':['Bananen'],'match_mode':'any_term','reply':'Stopp, stopp!'})}}]}
                    if len(posts)==1 else {'content':'(amusement) Alles klar!'} if len(posts)==2 else
                    {'content':json.dumps({'general':'Warm, clear and natural.','script':'(contentment) Alles klar!','language':'de'})})
                return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'choices':[{'message':message}]})
        settings=SimpleNamespace(LUNA_BASE='https://example.invalid',LUNA_MODEL='test',luna_key=lambda:'unused')
        with patch('conversation_director.shared_settings',return_value=settings):
            result=await CompactDirector().generate(Client(),'Wenn ich Bananen sage, unterbrich mich.',{},[],interaction_state={'enabled':True,'rules':[]})
        self.assertEqual(len(posts),3);self.assertEqual(posts[2]['tool_choice'],'none')
        self.assertEqual(len(result['interaction_actions']),1);self.assertEqual(result['text'],'Alles klar!')

    def test_explicit_banana_game_compiles_but_quotes_and_other_topics_do_not(self):
        from interaction_policy import explicit_literal_action
        text='Wenn ich über Bananen rede, unterbrich mich sofort und sage Stopp, stopp!'
        self.assertEqual(explicit_literal_action(text)['arguments']['reply'],'Stopp, stopp!')
        self.assertIsNone(explicit_literal_action('Bananen sind gesund.'))
        self.assertIsNone(explicit_literal_action('Wenn ich nicht Bananen sage, unterbrich mich und sage Stopp, stopp!'))
        self.assertIsNone(explicit_literal_action('Sag "Wenn ich Bananen sage, unterbrich mich und sage Stopp, stopp!"'))
        self.assertEqual(explicit_literal_action('Beende bitte das Bananen-Spiel und lösche die Unterbrechungsregel.')['name'],'clear_interruption_rules')

    async def test_handover_invalidates_teacher_already_running_for_old_turn(self):
        s,c,v=self.session();entered=asyncio.Event();release=threading.Event();loop=asyncio.get_running_loop()
        def embed(_):loop.call_soon_threadsafe(entered.set);release.wait(2);return v[0]
        worker=asyncio.create_task(s.deliver())
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=embed)):
            await s.queue.put(('Alter Satz',np.ones(16000),3.,0.,None,{},None,0))
            await asyncio.wait_for(entered.wait(),2);s.invite_speaker();release.set();await asyncio.sleep(.04)
        worker.cancel();await worker;c._send_turn.assert_not_awaited();self.assertIsNone(s.focus.owner)

    async def test_live_banana_rule_uses_sentence_identity_before_any_pause(self):
        s,c,v=self.session();d=c.decisions;d.helpful=False
        request='Wenn ich Bananen sage, unterbrich mich und sage Stopp, stopp!'
        d.book.apply([{'name':'set_interruption_rule','arguments':{
            'condition':'When I mention bananas','trigger_terms':['Bananen','bananas'],
            'match_mode':'any_term','reply':'Stopp, stopp!'}}],'speaker-1',request)
        await s.preview_text('Bananen sind gesund und ich rede einfach weiter über Bananen',3.)
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=Mock(return_value=v[0]))):
            d.tick(c.hop_pos);self.assertIsNotNone(d.task);await d.task
        c._send_interjection.assert_awaited_once()
        self.assertEqual(c._send_interjection.await_args.args[1]['speaker_id'],'speaker-1')

    async def test_semantic_stop_uses_same_guard_even_when_proactive_switch_off(self):
        s,c,v=self.session();d=c.decisions;d.enabled=False;c.speaking=True;d.last_agent='Eine längere Geschichte.'
        s.assess_interrupt=AsyncMock(return_value={'accepted':True})
        d.client.choose=AsyncMock(return_value={'action':'stop_reply','rule_id':None})
        await s.preview_text('Warte bitte, ich möchte etwas anderes fragen.',3.)
        d.tick(c.hop_pos);self.assertIsNotNone(d.task);await d.task
        d.client.choose.assert_awaited_once();s.assess_interrupt.assert_awaited_once();c._on_barge.assert_awaited_once()
        context=d.client.choose.await_args.args[0]
        self.assertEqual(context['agent_full_reply'],'Eine längere Geschichte.')
        self.assertEqual(context['agent_played_seconds'],.5)
        self.assertEqual(s.preview,'')

    async def test_semantic_stop_cannot_bypass_rejected_background(self):
        s,c,v=self.session();c.speaking=True
        s.assess_interrupt=AsyncMock(return_value={'accepted':False})
        c.decisions.client.choose=AsyncMock(return_value={'action':'stop_reply','rule_id':None})
        await s.preview_text('Warte bitte, ich möchte etwas fragen.',3.)
        c.decisions.tick(c.hop_pos);await c.decisions.task
        c._on_barge.assert_not_awaited()

    async def test_fresh_asr_stop_with_raw_voice_works_even_if_neural_vad_missed_it(self):
        s,c,v=self.session();c.speaking=True;s.last_speech=-100.;s.raw_loud_at=2.9
        s.reply_seen=1;s.reply_started=0.
        with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=Mock(return_value=v[0]))):
            await s.preview_text('Stopp!',3.);await s.barge_task
        c._on_barge.assert_awaited_once()
        receipt=next(call.args[0] for call in c.ws.send_json.await_args_list if call.args[0]['type']=='speaker_focus')
        self.assertTrue(receipt['lexical_audio_evidence'])

    async def test_stale_or_previous_reply_stop_cannot_bypass_speech_freshness(self):
        s,c,v=self.session();c.speaking=True;s.last_speech=-100.;s.raw_loud_at=1.
        s.stop_candidate={'at':1.,'rid':1,'epoch':s.epoch};s.mic_now=4.
        r=await s.assess_interrupt({'stop_text':'Stopp!'},4.,1)
        self.assertFalse(r['accepted']);self.assertFalse(r['lexical_audio_evidence'])
        s.mic_now=1.2
        r=await s.assess_interrupt({'stop_text':'Stopp!'},1.2,2)
        self.assertFalse(r['accepted'])

    async def test_bolt_mode_does_not_replace_configured_laya_stop_endpoint(self):
        seen=[]
        class Client:
            def __init__(self,**kw):pass
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            async def post(self,url,**kw):
                seen.append((url,kw['json']['model']))
                return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{
                    'choices':[{'message':{'content':json.dumps({'action':'stop_reply','rule_id':None})}}]})
        d=DecisionClient();d.base='http://127.0.0.1:8797';d.model='convaiinnovations/laya-multilingual'
        settings=SimpleNamespace(LUNA_BASE='https://example.invalid',LUNA_MODEL='test',luna_key=lambda:'unused')
        with patch.dict('os.environ',{'ROBO_DIRECTOR_MODE':'bolt'}),patch('live_decisions.httpx.AsyncClient',Client),patch('speaker_names.shared_settings',return_value=settings):
            await d.choose({'agent_speaking':True,'user_partial':'Stopp!'})
        self.assertEqual(seen,[('http://127.0.0.1:8797/v1/chat/completions','convaiinnovations/laya-multilingual')])

if __name__=='__main__':unittest.main()
