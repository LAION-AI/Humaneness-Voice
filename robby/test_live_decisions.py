import asyncio,unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from interaction_policy import RuleBook,validate_action,live_voice_verified,VoiceEvidence,face_corroborated_voice
from live_decisions import LiveDecisions

REQUEST='Wenn ich über Bananen rede, musst du mich unterbrechen und Stopp, stopp sagen.'
ACTION={'name':'set_interruption_rule','arguments':{'condition':'Talks about bananas','trigger_terms':['Banane','Bananen','bananas'],
    'match_mode':'any_term','reply':'Stopp, stopp!'}}

class RuleTests(unittest.TestCase):
    def test_face_evidence_uses_complete_context_not_unqualified_priming_audio(self):
        evidence=VoiceEvidence()
        evidence.update('a',.1,.5,complete=False);evidence.update('a',.2,1.,complete=False)
        self.assertIsNone(evidence.complete_mean)
        evidence.update('a',.51,2.5,complete=True)
        self.assertAlmostEqual(evidence.complete_mean,.51);self.assertLess(evidence.total/evidence.weight,.45)
        evidence.update('a',.38,3.2,quiet=True,complete=True)
        self.assertAlmostEqual(evidence.complete_mean,.38);self.assertEqual(evidence.initial_score,.38)
    def test_partial_keyword_requires_long_final_prefix_and_full_matching_stays_exact(self):
        book=RuleBook();book.apply([ACTION],'christoph',REQUEST);rule=book.for_owner('christoph')[0]
        self.assertTrue(rule.matches('Die banan',partial=True));self.assertFalse(rule.matches('Die banan'))
        self.assertFalse(rule.matches('Die bana',partial=True));self.assertFalse(rule.matches('Die banan sind',partial=True))
    def test_face_supported_voice_still_requires_both_audio_models_and_reference_evidence(self):
        f={'candidate_id':'a','candidate_similarity':.39};l={'candidate_id':'a','candidate_similarity':.56,'candidate_margin':1.}
        self.assertTrue(face_corroborated_voice(.49,1.,f,l,'a',.32,.48,.49))
        self.assertTrue(face_corroborated_voice(.40,1.,f,l,'a',.32,.40,.40))
        self.assertFalse(face_corroborated_voice(.29,1.,f,l,'a',.32,.48,.43))
        self.assertFalse(face_corroborated_voice(.40,1.,f,l,'a',.50,.40,.40))
        self.assertFalse(face_corroborated_voice(.49,1.,f,{**l,'candidate_id':'b'},'a',.32,.48,.49))
    def test_literal_rules_need_explicit_permission_and_are_speaker_scoped(self):
        book=RuleBook()
        with self.assertRaises(ValueError):book.apply([ACTION],'christoph','Bananen sind gesund.')
        self.assertFalse(book.rules)
        book.apply([ACTION],'christoph',REQUEST)
        self.assertEqual(len(book.for_owner('christoph')),1);self.assertFalse(book.for_owner('robert'))
        rule=book.for_owner('christoph')[0]
        for text in ('Bananen sind gesund.','BANANEN!','Bananenbananen, Bananen.'):
            self.assertTrue(rule.matches(text))
        self.assertFalse(rule.matches('Äpfel schmecken gut.'))
    def test_tool_bounds_and_cancellation_cannot_modify_other_people(self):
        book=RuleBook();book.apply([ACTION],'christoph',REQUEST);book.apply([ACTION],'robert',REQUEST)
        book.apply([{'name':'clear_interruption_rules','arguments':{}}],'christoph','Beende das Spiel.')
        self.assertEqual(len(book.rules),1);self.assertEqual(book.public()[0]['owner'],'robert')
        with self.assertRaises(ValueError):validate_action('shell',{'command':'anything'})
        with self.assertRaises(ValueError):validate_action('set_interruption_rule',{**ACTION['arguments'],'reply':'x '*25})
    def test_invalid_batch_is_atomic_and_negation_is_not_permission(self):
        book=RuleBook()
        with self.assertRaises(ValueError):book.apply([ACTION,{'name':'shell','arguments':{}}],'christoph',REQUEST)
        self.assertFalse(book.rules)
        for text in ('Unterbrich mich bitte nicht.', 'Do not interrupt me.'):
            with self.assertRaises(ValueError):book.apply([ACTION],'christoph',text)
        semantic=validate_action('set_semantic_interruption_rule',{'condition':'When I become repetitive','reply':'Let us move on.'})
        self.assertEqual(semantic['arguments']['match_mode'],'semantic')
    def test_strong_fast_cannot_rescue_unknown_live_teacher(self):
        vote={'candidate_id':'christoph','candidate_similarity':.80}
        self.assertFalse(live_voice_verified(.497,.5,vote,'christoph',.32,.62))
        self.assertTrue(live_voice_verified(.60,.5,{**vote,'candidate_similarity':.47},'christoph',.32,.62))
        self.assertFalse(live_voice_verified(.80,.5,vote,'robert',.32,.62))
        self.assertFalse(live_voice_verified(.80,.5,vote,'christoph',.32,.62,
            {'candidate_id':'christoph','candidate_similarity':.20,'candidate_margin':1.2},require_long=True))
        self.assertTrue(live_voice_verified(.80,.5,vote,'christoph',.32,.62,
            {'candidate_id':'christoph','candidate_similarity':.70,'candidate_margin':1.7},require_long=True))
    def test_weak_early_windows_are_not_erased_by_a_later_lucky_match(self):
        evidence=VoiceEvidence()
        for at,score in enumerate((.35,.38,.49,.54,.56)):
            mean=evidence.update('candidate',score,at*.96)
        self.assertLess(mean,.50)
        vote={'candidate_id':'candidate','candidate_similarity':.80}
        self.assertFalse(live_voice_verified(.56,1.,vote,'candidate',.32,.62,evidence_mean=mean))
        self.assertAlmostEqual(evidence.update('candidate',.8,6.,quiet=True),.8)
        sparse=VoiceEvidence();sparse.update('candidate',.377,1.)
        self.assertLess(sparse.update('candidate',.568,6.44),.50)
        sticky=VoiceEvidence();sticky.update('candidate',.43,1.);sticky.update('candidate',.85,9.)
        self.assertEqual(sticky.initial_score,.43)
        self.assertFalse(live_voice_verified(.85,1.,vote,'candidate',.32,.62,initial_score=sticky.initial_score,require_initial=True))
    def test_recovery_needs_four_separated_strong_windows_not_one_lucky_score(self):
        evidence=VoiceEvidence();evidence.update('candidate',.38,1.)
        for t in (2.,2.64,3.28):evidence.update('candidate',.7,t)
        self.assertFalse(evidence.recovered(.32));evidence.update('candidate',.7,3.92)
        self.assertTrue(evidence.recovered(.32));evidence.update('candidate',.49,4.56);self.assertFalse(evidence.recovered(.32))
    def test_recovery_remains_reachable_at_320ms_cadence(self):
        evidence=VoiceEvidence()
        for i in range(7):evidence.update('a',.7,i*.32)
        self.assertTrue(evidence.recovered(.32))

class LiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_task_survives_stale_proactive_phase_and_disabled_proactive_checkbox(self):
        d,c=self.make();c.speaking=True;d.enabled=False;d.task_phase='semantic'
        started=asyncio.Event();release=asyncio.Event()
        async def choose(_):started.set();await release.wait();return {'action':'stop_reply','rule_id':None}
        d.client.choose=choose;d.offer_agent(['Warte bitte'],c.hop_pos);d.tick(c.hop_pos)
        task=d.task;await started.wait();c.hop_pos+=5120;d.tick(c.hop_pos)
        self.assertIs(d.task,task);self.assertFalse(task.cancelled());release.set();await task
        c._on_barge.assert_awaited_once();self.assertIsNone(d.task_phase)
    async def test_late_semantic_stop_cannot_cancel_after_its_audio_is_stale(self):
        d,c=self.make();c.speaking=True
        async def choose(_):c.hop_pos+=64000;return {'action':'stop_reply','rule_id':None}
        d.client.choose=choose;d.offer_agent(['Bitte hör auf'],c.hop_pos);d.tick(c.hop_pos);await d.task
        c._on_barge.assert_not_awaited()
    async def test_weak_fast_cue_can_request_a_decision_but_not_bypass_final_identity(self):
        d,c=self.make();c.speaking=True
        c.speaker_ext.stop_fast_vote=lambda a,b:{'candidate_id':'christoph','candidate_similarity':.26}
        c.speaker_ext.confirm_interrupt.return_value=None
        d.client.choose=AsyncMock(return_value={'action':'stop_reply','rule_id':None})
        d.offer_agent(['Die ungefähr stop'],c.hop_pos);d.tick(c.hop_pos);await d.task
        d.client.choose.assert_awaited_once();c._on_barge.assert_not_awaited()
    async def test_new_literal_preempts_slow_semantic_request(self):
        d,c=self.make();d.book.apply([ACTION],'christoph',REQUEST);started=asyncio.Event()
        async def slow(_):started.set();await asyncio.Event().wait()
        d.client.choose=slow;d.offer('Jetzt werde ich dir noch viele andere Geschichten erzählen',c.hop_pos)
        d.tick(c.hop_pos);old=d.task;await started.wait()
        c.hop_pos+=5120;d.offer('Jetzt werde ich dir noch viele andere Geschichten erzählen Bananas',c.hop_pos)
        d.tick(c.hop_pos);await d.task;await asyncio.gather(old,return_exceptions=True)
        self.assertTrue(old.cancelled());c._send_interjection.assert_awaited_once()
    async def test_same_partial_retried_after_uncertain_voice(self):
        d,c=self.make();d.book.apply([ACTION],'christoph',REQUEST)
        c.speaker_ext.confirm_live_speaker.side_effect=[None,{'speaker_id':'christoph'}]
        d.offer('Bananen sind gesund',c.hop_pos);d.tick(c.hop_pos);await d.task
        self.assertFalse(d.last_text);self.assertFalse(d.consumed)
        c.hop_pos+=5120;d.tick(c.hop_pos);await d.task
        c._send_interjection.assert_awaited_once();self.assertEqual(d.attempts,2)
    async def test_repeated_topic_rearms_without_requiring_a_pause(self):
        d,c=self.make();d.book.apply([ACTION],'christoph',REQUEST);d.helpful=False
        d.offer('Bananen sind gesund',c.hop_pos);d.tick(c.hop_pos);await d.task
        d.finish_interjection(5*16000)
        c.hop_pos=8*16000
        c.speaker_ext.turns.tracks['christoph']=SimpleNamespace(sid='christoph',last_speech=8.,voiced_start=1.)
        d.offer('Bananen sind gesund und ich rede weiter über Bananas',c.hop_pos)
        d.tick(c.hop_pos);await d.task
        self.assertEqual(c._send_interjection.await_count,2)
    async def test_old_keyword_in_cumulative_transcript_does_not_retrigger(self):
        d,c=self.make();d.helpful=False;d.book.apply([ACTION],'christoph',REQUEST)
        d.offer('Bananen sind gesund',c.hop_pos);d.tick(c.hop_pos);await d.task;d.finish_interjection(5*16000)
        c.hop_pos=8*16000;c.speaker_ext.turns.tracks['christoph']=SimpleNamespace(sid='christoph',last_speech=8.,voiced_start=1.)
        d.offer('Bananen sind gesund und jetzt erzähle ich etwas über Äpfel',c.hop_pos);d.tick(c.hop_pos);await d.task
        self.assertEqual(c._send_interjection.await_count,1)
    async def test_usefulness_model_cannot_bypass_identity_or_disabled_option(self):
        d,c=self.make();d.client.choose=AsyncMock(return_value={'action':'interrupt_user','rule_id':'helpful-interjection'})
        d.offer('Warum unterbrichst du mich nicht bitte sag jetzt einmal etwas dazu',c.hop_pos)
        c.speaker_ext.confirm_live_speaker.return_value=None;d.tick(c.hop_pos);await d.task
        c._send_interjection.assert_not_awaited()
        c.hop_pos+=5120;c.speaker_ext.confirm_live_speaker.return_value={'speaker_id':'christoph'};d.tick(c.hop_pos);await d.task
        c._send_interjection.assert_awaited_once();self.assertEqual(d.last_decision['kind'],'helpful')
        await d.clear();d.helpful=False;d.task=None;d.offer('Warum unterbrichst du mich nicht bitte sag jetzt einmal etwas dazu',c.hop_pos+5120);d.tick(c.hop_pos+5120)
        self.assertIsNone(d.task)
    async def test_unchanged_stop_candidate_retries_after_failed_identity(self):
        d,c=self.make();c.speaking=True;d.client.choose=AsyncMock(return_value={'action':'stop_reply','rule_id':None})
        c.speaker_ext.confirm_interrupt.side_effect=[None,'christoph']
        d.offer_agent(['Bitte hör auf'],c.hop_pos);d.tick(c.hop_pos);await d.task
        c._on_barge.assert_not_awaited();c.hop_pos+=5120;d.tick(c.hop_pos);await d.task
        c._on_barge.assert_awaited_once()
    async def test_continuing_banana_sentence_cannot_cancel_interjection(self):
        d,c=self.make();c.speaking=True;d.proactive_active=True
        d.client.choose=AsyncMock(return_value={'action':'stop_reply','rule_id':None})
        d.offer_agent(['Man könnte den ganzen Tag','haben viele wunderbare Eigenschaften'],c.hop_pos)
        d.tick(c.hop_pos);self.assertIsNone(d.task);d.client.choose.assert_not_awaited()
    async def test_rejected_speech_admission_does_not_consume_utterance(self):
        d,c=self.make();d.book.apply([ACTION],'christoph',REQUEST);c._send_interjection.return_value=False
        d.offer('Bananen',c.hop_pos);d.tick(c.hop_pos);await d.task
        self.assertFalse(d.consumed);self.assertFalse(d.latched);self.assertFalse(d.last_text)
        c.hop_pos+=10240;c._send_interjection.return_value=True;d.tick(c.hop_pos);await d.task
        self.assertEqual(c._send_interjection.await_count,2);self.assertTrue(d.consumed)
    async def test_slow_asr_partial_is_checked_with_fresh_acoustics(self):
        d,c=self.make();d.book.apply([ACTION],'christoph',REQUEST)
        d.offer('Bananen',c.hop_pos);c.hop_pos+=32000;d.tick(c.hop_pos);await d.task
        c._send_interjection.assert_awaited_once()
    def make(self):
        ext=SimpleNamespace(confirm_live_speaker=AsyncMock(return_value={'speaker_id':'christoph','display_name':'Christoph','enrolled':True}),
            live_speech_recent=lambda at:True,
            turns=SimpleNamespace(tracks={}),memory=SimpleNamespace(users={'christoph':SimpleNamespace(name='Christoph')}),
            stop_fast_vote=lambda a,b:{'candidate_id':'christoph','candidate_similarity':.8},
            confirm_interrupt=AsyncMock(return_value='christoph'))
        c=SimpleNamespace(ears=SimpleNamespace(sr=16000),hop_pos=51200,speaker_ext=ext,identity_enabled=True,
            speaking=False,halting=False,current_speaker={'speaker_id':'christoph'},
            playback=SimpleNamespace(rid=7,n16=64000,heard_s=lambda at:1.2),
            ws=SimpleNamespace(closed=False,send_json=AsyncMock()),_send_interjection=AsyncMock(),_on_barge=AsyncMock())
        d=LiveDecisions(c);c.decisions=d;return d,c
    async def test_default_enabled_does_not_invent_unrequested_rules(self):
        d,c=self.make();self.assertTrue(d.enabled)
        d.offer('Bananen sind gesund.',c.hop_pos);d.tick(c.hop_pos)
        self.assertIsNone(d.task);c._send_interjection.assert_not_awaited()
    async def test_listening_gesture_is_silent_verified_cooled_down_and_epoch_scoped(self):
        d,c=self.make();d.partial_at=c.hop_pos/16000
        d.listening_gestures=False
        self.assertFalse(await d.emit_gesture('nod',c.hop_pos,d.epoch))
        c.speaker_ext.confirm_live_speaker.assert_not_awaited()
        d.listening_gestures=True
        self.assertTrue(await d.emit_gesture('nod',c.hop_pos,d.epoch))
        self.assertEqual(c.ws.send_json.call_args.args[0]['type'],'avatar_gesture')
        c._send_interjection.assert_not_awaited()
        self.assertFalse(await d.emit_gesture('shake',c.hop_pos,d.epoch))
        c.hop_pos+=160000;d.partial_at=c.hop_pos/16000
        c.speaker_ext.confirm_live_speaker.return_value=None
        self.assertFalse(await d.emit_gesture('nod',c.hop_pos,d.epoch))
        c.speaker_ext.confirm_live_speaker.return_value={'speaker_id':'christoph'}
        self.assertFalse(await d.emit_gesture('nod',c.hop_pos,d.epoch-1))
        c.speaking=True
        self.assertFalse(await d.emit_gesture('nod',c.hop_pos,d.epoch))
    async def test_rule_fires_from_partial_before_silence_and_not_twice_in_same_turn(self):
        d,c=self.make();d.book.apply([ACTION],'christoph',REQUEST)
        d.offer('Bananen sind',c.hop_pos);d.tick(c.hop_pos);await d.task
        c._send_interjection.assert_awaited_once()
        self.assertEqual(c._send_interjection.call_args.args[2],'Stopp, stopp!')
        self.assertTrue(d.suppresses(SimpleNamespace(sid='christoph',voiced_start=.2,start=.2,last_speech=3.2)))
        self.assertFalse(d.suppresses(SimpleNamespace(sid='robert',voiced_start=.2,start=.2,last_speech=3.2)))
        c.hop_pos+=5120;d.offer('Bananen sind gesund',c.hop_pos);d.tick(c.hop_pos)
        if d.task:await d.task
        self.assertEqual(c._send_interjection.await_count,1)
    async def test_disabled_or_other_speaker_never_fires_the_rule(self):
        d,c=self.make();d.book.apply([ACTION],'christoph',REQUEST)
        await d.toggle(False);d.offer('Bananen sind gesund',c.hop_pos);d.tick(c.hop_pos)
        self.assertIsNone(d.task)
        await d.toggle(True);c.speaker_ext.confirm_live_speaker.return_value={'speaker_id':'robert'}
        d.tick(c.hop_pos);await d.task;c._send_interjection.assert_not_awaited()
    async def test_interjection_playback_is_part_of_consumed_span_but_new_topics_are_not(self):
        d,c=self.make();d.book.apply([ACTION],'christoph',REQUEST)
        d.consumed.append(('christoph',1.,3.));d.proactive_active=True;d.finish_interjection(6*16000)
        track=SimpleNamespace(sid='christoph',voiced_start=5.,start=5.,last_speech=8.)
        self.assertTrue(d.suppresses(track,'Ich rede weiter über Bananen.'))
        self.assertTrue(d.suppresses(track,'und ihre gelbe Farbe sprechen.'))
        self.assertFalse(d.suppresses(track,'Erzähl mir etwas über Äpfel.'))
        self.assertFalse(d.suppresses(track,'Beende das Spiel mit Bananen.'))
        later=SimpleNamespace(sid='christoph',voiced_start=9.,start=9.,last_speech=11.)
        self.assertFalse(d.suppresses(later,'Das ist noch eine andere Sache.'))
    async def test_diarizer_split_tail_is_not_an_extra_reply(self):
        d,c=self.make();d.book.apply([ACTION],'christoph',REQUEST);d.consumed.append(('christoph',1.,5.))
        track=SimpleNamespace(sid='christoph',voiced_start=7.,start=7.,last_speech=10.)
        self.assertTrue(d.suppresses(track,'immer weiter über Bananen und ihre gelbe Farbe sprechen'))
        self.assertFalse(d.suppresses(track,'Erkläre mir etwas über Bananen'))
        self.assertFalse(d.suppresses(track,'Wie funktioniert ein Elektromotor?'))
        track.sid='robert';self.assertFalse(d.suppresses(track,'Bananen sind gesund'))
    async def test_disable_during_pending_verification_cannot_emit_late_interjection(self):
        d,c=self.make();d.book.apply([ACTION],'christoph',REQUEST)
        ready=asyncio.Event()
        async def slow(*_):await ready.wait();return {'speaker_id':'christoph'}
        c.speaker_ext.confirm_live_speaker.side_effect=slow
        d.offer('Bananen',c.hop_pos);d.tick(c.hop_pos);await asyncio.sleep(0)
        await d.toggle(False);ready.set();await asyncio.gather(d.task,return_exceptions=True)
        c._send_interjection.assert_not_awaited()
    async def test_cancellation_mentions_topic_without_proactive_interjection(self):
        d,c=self.make();d.book.apply([ACTION],'christoph',REQUEST)
        d.offer('Beende das Spiel, unterbrich mich nicht mehr wegen Bananen.',c.hop_pos);d.tick(c.hop_pos)
        self.assertIsNone(d.task);c._send_interjection.assert_not_awaited()
        d.last_decision={'action':'interrupt_user'};await d.clear()
        self.assertIsNone(d.last_decision)
    async def test_semantic_result_cannot_interject_after_user_has_gone_quiet(self):
        d,c=self.make();action={'name':ACTION['name'],'arguments':{**ACTION['arguments'],'match_mode':'semantic'}}
        d.book.apply([action],'christoph',REQUEST)
        async def slow_choice(context):
            c.speaker_ext.live_speech_recent=lambda at:False
            return {'action':'interrupt_user','rule_id':next(iter(d.book.rules))}
        d.client.choose=slow_choice;d.offer('Bananen sind gesund',c.hop_pos);d.tick(c.hop_pos);await d.task
        c._send_interjection.assert_not_awaited()
    async def test_semantic_stop_sees_agent_user_and_actual_playback_and_still_needs_voice(self):
        d,c=self.make();c.speaking=True;d.last_agent='A story about an apple tree.'
        d.client.choose=AsyncMock(return_value={'action':'stop_reply','rule_id':None})
        d.offer_agent(['Could you please stop for a second?'],c.hop_pos);d.tick(c.hop_pos);await d.task
        context=d.client.choose.call_args.args[0]
        self.assertEqual(context['agent_full_reply'],d.last_agent);self.assertEqual(context['agent_played_seconds'],1.2)
        self.assertIn('stop',context['user_partial']);c._on_barge.assert_awaited_once()
        c._on_barge.reset_mock();c.speaker_ext.confirm_interrupt.return_value=None
        d.offer_agent(['Please stop, I need to ask something.'],c.hop_pos+5120);d.tick(c.hop_pos+5120);await d.task
        c._on_barge.assert_not_awaited()

if __name__=='__main__':unittest.main()
