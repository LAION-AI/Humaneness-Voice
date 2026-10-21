import unittest
import numpy as np
from speaker_turns import SpeakerMemory,SpeakerTurns,Track,WAKE,self_name,wake_intent,choose_turn_text

class SpeakerTurnTests(unittest.TestCase):
    def test_reported_english_introduction(self):
        text='Hi Robbie, I am Christoph.'
        self.assertTrue(wake_intent(text));self.assertEqual(self_name(text),'Christoph')
    def setUp(self):
        self.a,self.b,self.noise=np.eye(192,dtype=np.float32)[:3]
        self.memory=SpeakerMemory(threshold=.55,clock=lambda:100.)
        self.turns=SpeakerTurns(self.memory);self.at=0.
    def hop(self,vector,kind='speech',speaking=False,level=.1):
        self.at+=.32
        p={'kind':kind,'embedding':vector,'confidence':.95 if kind=='speech' else 0}
        return self.turns.update(p,np.full(5120,level,np.float32),self.at,speaking=speaking)
    def enroll(self,vector,text):
        for _ in range(4):self.hop(vector)
        t=self.turns.decode_candidates()[0]
        d=self.turns.candidate_text(t.key,text)
        self.assertTrue(d['accepted'])
        # Most clock contracts below concern an already fully verified user.
        # Simulate the separate CPU-reference step with the unit fixture vector.
        if self.memory.users[d['id']].named:self.memory.users[d['id']].teacher=vector.copy()
        return d['id']
    def test_unverified_named_introduction_retains_registration_grace(self):
        for _ in range(4):self.hop(self.a)
        t=self.turns.decode_candidates()[0]
        self.turns.candidate_text(t.key,'Hey Robbie, ich bin Christoph.')
        for _ in range(4):self.hop(None,'silence')
        self.assertEqual(self.turns.due(),[])
        for _ in range(5):self.hop(None,'silence')
        self.assertEqual(len(self.turns.due()),1)
    def test_sparse_new_channel_gets_one_final_captured_context_probe(self):
        sid=self.enroll(self.a,'Hey Robbie, ich bin Christoph.')
        for _ in range(3):
            self.at+=.32
            self.turns.update({'kind':'speech','embedding':self.b,'confidence':.99,'diarization_channel':7},
                np.full(5120,.05,np.float32),self.at)
        t=self.turns.decode_candidates()[0]
        self.turns.candidate_text(t.key,'Please help me.')
        for _ in range(4):self.hop(None,'silence')
        self.assertEqual(self.turns.decode_candidates(),[t])
        self.turns.candidate_text(t.key,'Please help me now.')
        for _ in range(5):
            self.hop(None,'silence');self.assertEqual(self.turns.decode_candidates(),[])
        self.assertEqual(list(self.memory.users),[sid])
    def test_wake_variants_and_names(self):
        for text in ['Hey Robbie, ich bin Christoph.','Hey Robby, I am Robert.','Hallo Robbi, ich heiße Andreas.','hi Robo, my name is Selma.',
            'Hey Rabi, ich bin Christoph.','Hey Rabbi, ich bin Christoph.','He Roby, ich bin Robert.','هي Robby, ich bin Robert.']:
            self.assertIsNotNone(WAKE.search(text));self.assertIsNotNone(self_name(text))
        for text in ['Robbie erzählt etwas.','Ein Rabbi spricht.','Hey Robert, ich bin Christoph.']:
            self.assertIsNone(WAKE.search(text))
        for text in ['ich bin müde','I am happy','Robert says hello','Mein Freund ist Christoph','ich bin auch hier','Ich bin begeistert.','I am excited.','Ich bin glücklich.']:
            self.assertIsNone(self_name(text))
    def test_no_enrollment_without_candidate_owned_wake(self):
        for _ in range(5):self.hop(self.a)
        track=self.turns.decode_candidates()[0]
        self.assertIsNone(self.turns.candidate_text(track.key,'Ich bin Christoph.'))
        self.assertFalse(self.memory.users)
    def test_timestamped_same_name_wake_can_recover_clipped_greeting(self):
        for _ in range(5):self.hop(self.a)
        track=self.turns.decode_candidates()[0]
        self.assertIsNone(self.turns.candidate_text(track.key,'Hello there.',observed_wake=True))
        decision=self.turns.candidate_text(track.key,'Ich bin Selma.',observed_wake=True)
        self.assertEqual(decision['name'],'Selma')
    def test_followup_memory_stays_bounded_and_keeps_original_anchor(self):
        speaker,_=self.memory.enroll(self.a,name='Christoph');speaker.teacher=self.a.copy()
        for _ in range(20):speaker.adapt_teacher(self.b)
        self.assertEqual(len(speaker.teacher_examples),2)
        np.testing.assert_array_equal(speaker.teacher_anchor,self.a)
        self.assertGreater(float(speaker.teacher@self.a),.89)
    def test_postroll_background_suffix_is_not_appended_to_owned_sentence(self):
        owned='Was können wir heute zusammen machen?'
        text,which=choose_turn_text(['Was können wir heute zusammen machen und die Nachrichten?'],owned)
        self.assertEqual(text,owned);self.assertEqual(which,-1)
    def test_dropped_article_does_not_disable_background_suffix_guard(self):
        own='Kannst du mir kleine Geschichte erzählen?'
        text,which=choose_turn_text(['Kannst du mir eine kleine Geschichte erzählen? Morgen wird Sonne.'],own)
        self.assertEqual((text,which),(own,-1))
    def test_one_quiet_missing_final_word_can_still_be_recovered(self):
        text,which=choose_turn_text(['Erzähl mir bitte einen kurzen Witz.'],'Erzähl mir bitte einen kurzen')
        self.assertIn('Witz',text);self.assertEqual(which,0)
    def test_identical_words_prefer_latest_verified_crop_not_tv_preroll(self):
        text='Erzähl mir bitte einen kurzen Witz.'
        chosen,which=choose_turn_text(['Morgen wird '+text,text,'Morgen wird '+text,text],text)
        self.assertEqual(chosen,text);self.assertEqual(which,3)
        chosen,which=choose_turn_text([text,'Mir bitte einen kurzen Witz.'],text)
        self.assertEqual(chosen,text);self.assertEqual(which,0)
    def test_owned_wake_is_preserved_when_raw_room_asr_loses_greeting(self):
        owned='Hey Robby, ich bin Robert. Wie geht es dir heute?'
        text,which=choose_turn_text(['Im Hintergrund, ich bin Robert, wie geht es dir heute?','Ich bin Robert.'],owned)
        self.assertEqual(text,owned);self.assertEqual(which,-1)
    def test_complete_cropped_intro_is_not_replaced_by_corrupted_owned_words(self):
        clean='Hey Robbie, ich bin Christoph. Schön dich kennenzulernen.'
        text,which=choose_turn_text([clean],'Robbie, ich bin Christoph wündig kennenzulernen Morgen wir')
        self.assertEqual((text,which),(clean,0))
    def test_full_wake_intro_prefers_latest_identical_words_not_tv_prefix(self):
        clean='Hey Robbie, ich bin Robert. Wie geht es dir heute?'
        text,which=choose_turn_text(['Im Hintergrund läuft das Wetter. '+clean,'Über das Wetter. '+clean,'Ich bin Robert.'],clean)
        self.assertEqual((text,which),(clean,1))
    def test_full_greeting_is_preferred_over_bare_call_with_tv_prefix(self):
        clean='Hey Robbie, ich bin Robert. Wie geht es dir heute?'
        text,which=choose_turn_text(['Über das Wetter und die Robbie, ich bin Robert. Wie geht es dir heute?',clean],
            'Ich bin Robert.')
        self.assertEqual((text,which),(clean,1))
        text,which=choose_turn_text(['Über das Wetter und die Robbie, ich bin Robert.'],'Ich bin Robert.')
        self.assertEqual(text,'Robbie, ich bin Robert.')
    def test_track_snapshot_token_is_unique_across_new_turns(self):
        from speaker_turns import Track
        self.assertNotEqual(Track('a',0.,1.).token,Track('a',0.,1.).token)
    def test_delayed_diarizer_cannot_rewind_native_continuation_timestamp(self):
        sid=self.enroll(self.a,'Hey Robbie, ich bin Christoph.')
        self.turns.tracks[sid].last_speech=self.at+.32
        expected=self.turns.tracks[sid].last_speech
        self.turns.update({'kind':'speech','embedding':self.a,'confidence':.95},np.full(2560,.05,np.float32),self.at+.16)
        self.assertEqual(self.turns.tracks[sid].last_speech,expected)
    def test_multiduration_reference_does_not_change_fast_memory_or_user_count(self):
        sid=self.enroll(self.a,'Hey Robbie, ich bin Christoph.')
        user=self.memory.users[sid];user.teacher_templates=(self.b.copy(),)
        self.assertAlmostEqual(user.teacher_score(self.b),1.)
        self.assertIsNone(self.memory.match(self.b)['id'])
        self.assertEqual(len(self.memory.users),1)
    def test_background_does_not_extend_foreground_clock(self):
        sid=self.enroll(self.a,'Hey Robbie, ich bin Christoph.')
        for _ in range(4):self.hop(self.b)
        due=self.turns.due();self.assertEqual([t.sid for t in due],[sid])
        self.assertEqual(self.memory.users[sid].name,'Christoph')
        self.assertEqual(len(self.memory.users),1)
    def test_bound_diarizer_channel_cannot_turn_silence_into_speech(self):
        sid=self.enroll(self.a,'Hey Robbie, ich bin Christoph.')
        self.turns.channels[1]=sid
        for _ in range(4):
            self.at+=.32
            self.turns.update({'kind':'speech','encoder_activity_kind':'silence','confidence':.65,'raw_embedding':self.a,'diarization_channel':1},
                np.full(5120,.01,np.float32),self.at)
        self.assertEqual([t.sid for t in self.turns.due()],[sid])
    def test_matching_quiet_background_cannot_renew_known_speaker_clock(self):
        sid=self.enroll(self.a,'Hey Robbie, ich bin Christoph.');self.turns.channels[1]=sid
        for _ in range(4):
            self.at+=.32
            self.turns.update({'kind':'speech','raw_embedding':self.a,'confidence':.99,
                'diarization_channel':1,'foreground_energy':False},np.full(5120,.01,np.float32),self.at)
        self.assertEqual([t.sid for t in self.turns.due()],[sid])
    def test_quiet_background_cannot_reassign_an_existing_channel_either(self):
        a=self.enroll(self.a,'Hey Robbie, ich bin Christoph.')
        b=self.enroll(self.b,'Hey Robbie, ich bin Robert.');self.turns.channels[1]=a
        self.at+=.32
        self.turns.update({'kind':'speech','raw_embedding':self.b,'confidence':.99,
            'diarization_channel':1,'foreground_energy':False},np.full(5120,.01,np.float32),self.at)
        self.assertEqual(self.turns.channels[1],a)
    def test_new_audible_diarizer_channel_can_recover_student_miss(self):
        self.turns.update({'kind':'speech','encoder_activity_kind':'silence','confidence':.65,
            'raw_embedding':self.a,'diarization_channel':7},np.full(5120,.01,np.float32),.32)
        self.assertIn('channel-7',self.turns.tracks)
        self.assertIsNone(self.turns.tracks['channel-7'].sid)
    def test_finished_turn_keeps_session_channel_but_not_old_audio(self):
        sid=self.enroll(self.a,'Hey Robbie, ich bin Christoph.')
        self.turns.channels[3]=sid
        for _ in range(4):self.hop(None,'silence')
        self.assertEqual(len(self.turns.due()),1)
        self.assertEqual(self.turns.channels[3],sid)
        self.assertNotIn(sid,self.turns.tracks)
        self.assertIn(sid,self.memory.users)
    def test_known_channel_cannot_override_independent_voice_veto(self):
        sid=self.enroll(self.a,'Hey Robbie, ich bin Christoph.');self.turns.channels[3]=sid
        for _ in range(4):
            self.at+=.32
            self.turns.update({'kind':'speech','raw_embedding':self.a,'confidence':.99,
                'diarization_channel':3,'speaker_activity_confirmed':False},np.full(5120,.05,np.float32),self.at)
        self.assertEqual([t.sid for t in self.turns.due()],[sid])
    def test_known_channel_does_not_inherit_clearly_different_fast_voice(self):
        sid=self.enroll(self.a,'Hey Robbie, ich bin Christoph.');self.turns.channels[3]=sid
        for _ in range(4):
            self.at+=.32
            self.turns.update({'kind':'speech','raw_embedding':self.b,'confidence':.99,'diarization_channel':3},
                np.full(5120,.05,np.float32),self.at)
        self.assertEqual([t.sid for t in self.turns.due()],[sid])
    def test_loud_matching_voice_can_recover_short_activity_classifier_miss(self):
        sid=self.enroll(self.a,'Hey Robbie, ich bin Christoph.');self.turns.channels[3]=sid
        before=self.turns.tracks[sid].last_speech;self.at+=.32
        self.turns.update({'kind':'speech','encoder_activity_kind':'silence','raw_embedding':self.a,
            'confidence':.65,'diarization_channel':3},np.full(5120,.08,np.float32),self.at)
        self.assertGreater(self.turns.tracks[sid].last_speech,before)
    def test_strong_known_voice_match_supersedes_reused_other_person_channel(self):
        a=self.enroll(self.a,'Hey Robbie, ich bin Christoph.')
        b=self.enroll(self.b,'Hey Robbie, ich bin Robert.')
        self.turns.channels[3]=a;last=self.turns.tracks[a].last_speech;self.at+=.32
        match=self.turns.update({'kind':'speech','raw_embedding':self.b,'confidence':.99,'diarization_channel':3},
            np.full(5120,.05,np.float32),self.at)
        self.assertEqual(match['id'],b);self.assertEqual(self.turns.tracks[a].last_speech,last)
    def test_stale_long_context_cannot_veto_strong_current_known_voice(self):
        a=self.enroll(self.a,'Hey Robbie, ich bin Christoph.')
        b=self.enroll(self.b,'Hey Robbie, ich bin Robert.')
        self.memory.users[a].long=self.a;self.memory.users[b].long=self.b
        self.turns.channels[3]=a;last=self.turns.tracks[a].last_speech;self.at+=.32
        vector=.61*self.b+.38*self.a+np.sqrt(1-.61**2-.38**2)*self.noise
        match=self.turns.update({'kind':'speech','raw_embedding':vector,'long_embedding':self.a,
            'confidence':.99,'diarization_channel':3},np.full(5120,.05,np.float32),self.at)
        self.assertEqual(match['id'],b);self.assertEqual(self.turns.channels[3],b)
        self.assertEqual(self.turns.channels[3],b)
        # A less distinctive next syllable must not fall back to the stale
        # previous speaker and mask the rest of this known person's sentence.
        vector=.4*self.b+.2*self.a+np.sqrt(.8)*np.eye(192,dtype=np.float32)[2]
        self.at+=.32
        match=self.turns.update({'kind':'speech','raw_embedding':vector,'confidence':.99,'diarization_channel':3},
            np.full(5120,.05,np.float32),self.at)
        self.assertEqual(match['id'],b);self.assertEqual(self.turns.tracks[a].last_speech,last)
    def test_separate_wake_then_followup_name(self):
        sid=self.enroll(self.a,'Hey Robby, hörst du mich?')
        for _ in range(9):self.hop(None,'silence')
        self.turns.due()
        self.memory.users[sid].teacher=self.a.copy() # full CPU wake verification, separate from pure turn state
        for _ in range(5):self.hop(self.a)
        self.turns.tracks[sid].name_heard=True # incremental ASR heard the full introduction
        for _ in range(4):self.hop(None,'silence')
        t=self.turns.due()[0];context=self.turns.speaker_context(t,'Ja, ich bin Christoph.')
        self.assertEqual(context['speaker_id'],sid);self.assertEqual(context['display_name'],'Christoph')
    def test_two_names_remain_bound_to_acoustics(self):
        aid=self.enroll(self.a,'Hey Robbie, ich bin Christoph.')
        bid=self.enroll(self.b,'Hey Robby, ich bin Andreas.')
        self.assertNotEqual(aid,bid);self.assertEqual(self.memory.match(self.a)['name'],'Christoph')
        self.assertEqual(self.memory.match(self.b)['name'],'Andreas')
        self.assertIsNone(self.memory.match(self.noise)['id'])
        self.assertIsNone(self_name('Andreas hat mir erzählt, dass Christoph kommt.'))
    def test_noise_overlap_and_playback_cannot_enroll(self):
        for kind in ('silence','overlap','uncertain'):
            for _ in range(6):self.hop(self.noise,kind)
        for _ in range(6):self.hop(self.a,speaking=True)
        self.assertFalse(self.turns.tracks);self.assertFalse(self.memory.users)
    def test_no_repeated_decode_of_unchanged_background(self):
        for _ in range(4):self.hop(self.b)
        t=self.turns.decode_candidates()[0];self.turns.candidate_text(t.key,'Background television.')
        for _ in range(8):
            self.hop(None,'silence');self.assertEqual(self.turns.decode_candidates(),[])
    def test_separate_users_keep_separate_audio(self):
        aid=self.enroll(self.a,'Hey Robbie, ich bin Christoph.')
        bid=self.enroll(self.b,'Hey Robby, ich bin Robert.')
        a=self.turns.tracks[aid].waveform();b=self.turns.tracks[bid].waveform()
        self.assertAlmostEqual(float(a.mean()),.1,places=5);self.assertAlmostEqual(float(b.mean()),.1,places=5)
        # A different candidate must not import an unknown predecessor's wake.
        self.assertAlmostEqual(len(b)/16000,1.28,places=5)
    def test_buffers_and_number_of_candidates_are_bounded(self):
        for _ in range(200):self.hop(self.noise)
        self.assertLessEqual(sum(t.waveform().nbytes for t in self.turns.tracks.values()),600000)
        for track in self.turns.tracks.values():
            self.assertGreaterEqual(track.start,self.at-self.turns.candidate_s)
            self.assertGreaterEqual(track.voiced_start,track.start)
    def test_long_history_never_rescues_bad_short_match(self):
        a,_=self.memory.enroll(self.a,self.b,'Christoph');b,_=self.memory.enroll(self.b,self.a,'Robert')
        # Newest cue is clearly A but the lagging long cue strongly suggests B.
        match=self.memory.match(self.a,self.a)
        self.assertNotEqual(match['id'],b.id)
    def test_clear_forgets_session_voiceprints(self):
        self.enroll(self.a,'Hey Robbie, ich bin Christoph.');self.turns.clear()
        self.assertFalse(self.memory.users);self.assertFalse(self.turns.tracks)

    def test_diarization_keeps_enrolled_channel_through_weak_embedding(self):
        for _ in range(4):
            self.at+=.32;self.turns.update({'kind':'speech','embedding':self.a,'confidence':.95,'diarization_channel':1},np.ones(5120,np.float32),self.at)
        t=self.turns.decode_candidates()[0];decision=self.turns.candidate_text(t.key,'Hey Robbie, ich bin Christoph.')
        self.at+=.32
        # Weak but compatible continuity (0.4), not an orthogonal different
        # speaker. The latter is explicitly rejected in the new reuse test.
        weak=.4*self.a+np.sqrt(.84)*self.b
        match=self.turns.update({'kind':'speech','embedding':weak,'confidence':.7,'diarization_channel':1},np.ones(5120,np.float32),self.at)
        self.assertEqual(match['id'],decision['id']);self.assertEqual(match['name'],'Christoph')

    def test_different_diar_channel_cannot_impersonate_similar_voice(self):
        for _ in range(4):
            self.at+=.32;self.turns.update({'kind':'speech','embedding':self.a,'confidence':.95,'diarization_channel':1},np.ones(5120,np.float32),self.at)
        t=self.turns.decode_candidates()[0];decision=self.turns.candidate_text(t.key,'Hey Robbie, ich bin Christoph.')
        self.at+=.32
        match=self.turns.update({'kind':'speech','embedding':self.a,'confidence':.95,'diarization_channel':2},np.ones(5120,np.float32),self.at)
        self.assertIsNone(match['id']);self.assertIn('channel-2',self.turns.tracks)

    def test_unbound_background_channel_cannot_claim_displaced_user(self):
        a,_=self.memory.enroll(self.a,name='Christoph');b,_=self.memory.enroll(self.b,name='Robert')
        self.turns.channels[1]=b.id # a merged channel was reassigned to Robert
        self.at+=.32
        match=self.turns.update({'kind':'speech','embedding':self.a,'confidence':.95,'diarization_channel':0},np.ones(5120,np.float32),self.at)
        self.assertIsNone(match['id']);self.assertNotIn(a.id,self.turns.tracks)

    def test_out_of_order_preroll_cannot_overwrite_end_of_sentence(self):
        t=Track('test',0,1.28)
        t.chunks.extend([(.32,.64,np.full(5120,2,np.float32)),(0,.32,np.full(5120,1,np.float32)),(.64,1.28,np.full(10240,3,np.float32))])
        wave=t.waveform();self.assertEqual(len(wave),20480)
        np.testing.assert_array_equal(wave[:5120],1);np.testing.assert_array_equal(wave[5120:10240],2)
        np.testing.assert_array_equal(wave[10240:],3)

    def test_diarized_unknown_never_imports_other_channel_preroll(self):
        for channel,value in [(0,.1),(1,.8)]:
            self.at+=.32
            self.turns.update({'kind':'speech','embedding':self.a,'confidence':.95,'diarization_channel':channel},np.full(5120,value,np.float32),self.at)
        b=self.turns.tracks['channel-1']
        self.assertEqual(len(b.waveform()),5120);np.testing.assert_allclose(b.waveform(),.8)

    def test_nonverbal_foreground_cannot_submit_background_words(self):
        text,choice=choose_turn_text(['Bericht über das Wetter und die Nachrichten','über das Wetter und die Nachrichten'],'Sich nü')
        self.assertIsNone(text);self.assertIsNone(choice)

    def test_agreement_removes_unowned_background_prefix(self):
        text,choice=choose_turn_text(['Im Hintergrund kannst du mir eine Geschichte erzählen?','Kannst du mir eine Geschichte erzählen?'],'Kannst du mir eine Geschichte erzählen?')
        self.assertEqual(choice,1);self.assertTrue(text.startswith('Kannst du'))
        self.assertEqual(choose_turn_text(['Morgen wird es sonnig'],'Ja.'),('Ja.',-1))

    def test_addressed_introduction_survives_asr_dropped_greeting(self):
        for text in ['Robby, ich bin Robert.','Robbie, ich','Robbi.']:
            self.assertTrue(wake_intent(text))
        for text in ['Robby erzählt eine Geschichte.','Ich spreche über Robbie.']:
            self.assertFalse(wake_intent(text))

    def test_wake_only_has_brief_introduction_grace_not_an_infinite_wait(self):
        sid=self.enroll(self.a,'Hey Robbie.')
        for _ in range(4):self.hop(None,'silence')
        self.assertEqual(self.turns.due(),[])
        for _ in range(5):self.hop(None,'silence')
        self.assertEqual(self.turns.due()[0].sid,sid)

    def test_raw_preroll_is_not_owned_word_origin_evidence(self):
        t=Track('test',0,1.28,voiced_start=.64)
        t.chunks.extend([(0,.64,np.full(10240,.01,np.float32)),(.64,1.28,np.full(10240,.06,np.float32))])
        self.assertEqual(len(t.waveform()),20480);self.assertEqual(len(t.owned_waveform()),10240)
        np.testing.assert_allclose(t.owned_waveform(),.06)

    def test_bound_channel_cannot_fabricate_voice_from_silence(self):
        sid=self.enroll(self.a,'Hey Robbie, ich bin Christoph.')
        self.turns.channels[1]=sid;before=self.turns.tracks[sid].last_speech
        self.at+=.32
        result=self.turns.update({'kind':'silence','embedding':None,'confidence':0.,'diarization_channel':1},np.zeros(5120,np.float32),self.at)
        self.assertIsNone(result);self.assertEqual(self.turns.tracks[sid].last_speech,before)

if __name__=='__main__':unittest.main()
