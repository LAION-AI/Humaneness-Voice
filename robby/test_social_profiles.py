"""Social role, acoustic continuity, opt-in profiles and optional face contracts."""
import asyncio,unittest
from types import SimpleNamespace as NS
from unittest.mock import patch,AsyncMock
import numpy as np
from sentence_session import OpenVoiceMemory
from social_profiles import observe,describe,export_profiles,import_profiles,context_note

class SocialProfiles(unittest.TestCase):
 def setUp(self):self.now=[100.];self.m=OpenVoiceMemory(clock=lambda:self.now[0]);self.v=np.eye(192,dtype=np.float32)
 def voice(self,cosine):return self.v[0]*cosine+self.v[1]*np.sqrt(1-cosine*cosine)
 def test_recent_voice_variation_is_not_another_person_or_new_anchor(self):
  first=self.m.assign(self.v[0],'Ich bin Christoph.')
  for _ in range(4):
   self.now[0]+=3;row=self.m.assign(self.voice(.29),'Und wer bin ich?')
   self.assertEqual(row['speaker_id'],first['speaker_id']);self.assertFalse(row['needs_name']);self.assertTrue(row['identity_uncertain'])
  self.assertEqual(len(self.m.users[first['speaker_id']]['vectors']),1)
 def test_strong_different_voice_and_explicit_name_are_not_collapsed(self):
  a=self.m.assign(self.v[0],'Ich bin Christoph.');b=self.m.assign(self.v[2],'Ich bin Alice.')
  self.assertNotEqual(a['speaker_id'],b['speaker_id'])
  self.assertEqual(self.m.assign(self.v[0],'Und wer bin ich?')['display_name'],'Christoph')
 def test_name_prompt_cooldown_survives_unmatched_voice_variations(self):
  self.assertTrue(self.m.assign(self.v[0],'Hallo!')['needs_name'])
  for i in range(1,4):
   self.now[0]+=3;self.assertFalse(self.m.assign(self.v[i],'Was ist los?')['needs_name'])
 def test_explicit_correction_adds_second_anchor_to_same_name(self):
  a=self.m.assign(self.v[0],'Ich bin Christoph.');self.m.assign(self.v[2],'Hallo.')
  b=self.m.assign(self.v[2],'Nein, ich bin auch Christoph. Du hast mich verwechselt.')
  self.assertEqual(a['speaker_id'],b['speaker_id']);self.assertEqual(b['voice_variants'],2)
 def test_profile_cues_do_not_change_embedding_identity(self):
  a=self.m.assign(self.v[0],'Ich bin Alice.');u=self.m.users[a['speaker_id']]
  observe(u,{'status':'available','vocal_traits':{'GEND':{'score':4.2},'AGEV':{'score':3.1}}})
  self.assertEqual(self.m.assign(self.v[0],'Hallo.')['display_name'],'Alice')
  self.assertIn('masculine-sounding',str(describe(u)));self.assertIsNone(describe(u,False))
 def test_profile_roundtrip_requires_valid_ecapa_and_keeps_multiple_variants(self):
  a=self.m.assign(self.v[0],'Ich bin Christoph.');self.m.assign(self.v[1],'Ich bin auch Christoph.')
  rows=export_profiles(self.m);other=OpenVoiceMemory();import_profiles(other,rows)
  self.assertEqual(other.assign(self.v[1],'Und wer bin ich?')['display_name'],'Christoph')
  self.assertEqual(len(other.users),1)
  rows[0]['vectors'][0]=[1.]*128
  with self.assertRaises(ValueError):import_profiles(other,rows)
 def test_invalid_profiles_import_is_atomic(self):
  self.m.assign(self.v[0],'Ich bin Christoph.');rows=export_profiles(self.m);rows+=[{'name':'bad'}]
  other=OpenVoiceMemory()
  with self.assertRaises(ValueError):import_profiles(other,rows)
  self.assertFalse(other.users)
 def test_unnamed_early_visitors_do_not_hide_later_named_profile(self):
  for i in range(9):self.m.assign(self.v[i],'Hallo.')
  self.m.assign(self.v[10],'Ich bin Christoph.')
  self.assertEqual([p['name'] for p in export_profiles(self.m)],['Christoph'])
 def test_role_prompt_does_not_include_voiceprints(self):
  note=context_note({'named':True,'display_name':'Christoph','vectors':self.v[0].tolist()})
  self.assertIn('HUMAN USER',note);self.assertIn('Du bist Christoph',note);self.assertNotIn('vectors',note)
 def test_exact_name_question_uses_current_person_not_last_history_name(self):
  from social_profiles import identity_answer
  person={'named':True,'enrolled':True,'display_name':'Christoph'}
  self.assertEqual(identity_answer('Und wer bin ich?',person),'Du heißt Christoph.')
  self.assertIn('Stimmt das?',identity_answer('Wie heiße ich?',{**person,'identity_uncertain':True}))
  self.assertIsNone(identity_answer('Was weißt du über meine Freundin?',person))
  self.assertIsNone(identity_answer('Wer bin ich?',{'named':False}))

class Curiosity(unittest.IsolatedAsyncioTestCase):
 async def test_background_greeting_is_not_spoken_and_one_repair_can_recover(self):
  from bolt_director import BoltDirector
  from bolt_text import Sentence
  calls=[]
  async def draft(*args,**kwargs):
   calls.append(1)
   yield Sentence('Hallo Christoph!' if len(calls)==1 else 'Läuft da vielleicht ein Fernseher?','interest')
  metrics={};context={'environment_proactive':True,'environment':{'kind':'unfamiliar_voice'}}
  with patch.object(BoltDirector,'_stream',draft):
   parts=[s async for s in BoltDirector().stream(None,'',{},[],context,metrics)]
  self.assertEqual([s.text for s in parts],['Läuft da vielleicht ein Fernseher?']);self.assertEqual(metrics['background_question_attempts'],2)
 async def test_background_double_failure_stays_silent(self):
  from bolt_director import BoltDirector
  from bolt_text import Sentence
  async def draft(*args,**kwargs):yield Sentence('Hallo Christoph! Wie heißt du?','interest')
  metrics={};context={'environment_proactive':True,'environment':{'kind':'unfamiliar_voice'}}
  with patch.object(BoltDirector,'_stream',draft):
   parts=[s async for s in BoltDirector().stream(None,'',{},[],context,metrics)]
  self.assertFalse(parts);self.assertEqual(metrics['background_question_status'],'skipped_invalid')
 async def test_missing_or_invalid_noise_floor_does_not_prompt(self):
  from environment_session import EnvironmentSession
  e=EnvironmentSession(NS(speaking=False),clock=lambda:100.);e.enabled=True;e.describe_voice=AsyncMock()
  for prominence in (None,float('nan'),float('inf'),'loud'):
   e.notice_background(np.ones(32000,np.float32),{'estimated_level_above_floor_db':prominence},.02)
  self.assertFalse(e.background_notices);e.describe_voice.assert_not_awaited()
 async def test_repeated_audible_background_can_prompt_but_not_take_over(self):
  from environment_session import EnvironmentSession
  c=NS(speaking=False,sentence_stream=NS(permissions=NS(questions=True)))
  e=EnvironmentSession(c,clock=lambda:100.);e.enabled=True;e.describe_voice=AsyncMock()
  wave=np.ones(32000,np.float32)*.01;d={'known_voice_similarity':.1,'estimated_level_above_floor_db':12}
  e.notice_background(wave,d,.01);e.describe_voice.assert_not_awaited()
  e.notice_background(wave,d,.01);await e.job;e.describe_voice.assert_awaited_once()
  e.notice_background(wave,d,.01);e.notice_background(wave,d,.01);await asyncio.sleep(0);self.assertEqual(e.describe_voice.await_count,1)
 async def test_known_soft_voice_or_user_ignored_background_does_not_trigger(self):
  from environment_session import EnvironmentSession
  c=NS(speaking=False,sentence_stream=NS(permissions=NS(questions=False)))
  e=EnvironmentSession(c,clock=lambda:100.);e.enabled=True;e.describe_voice=AsyncMock()
  for _ in range(3):e.notice_background(np.ones(32000,np.float32),{'known_voice_similarity':.1,'estimated_level_above_floor_db':12},.02)
  await asyncio.sleep(0);e.describe_voice.assert_not_awaited()

class FaceSentenceIntegration(unittest.IsolatedAsyncioTestCase):
 async def test_optional_face_id_binds_in_sentence_mode_without_face_vad_gate(self):
  from test_sentence_reliability import ReliabilityTests
  from visual_identity import VisualIdentity
  s,c,v=ReliabilityTests().session();c.visual=VisualIdentity();c.visual.configure({'faceIdEnabled':True,'ignoreOffscreenSpeakers':False})
  for i in range(20):
   at=(i+1)*.1;c.visual.observe({'camera':'on','audio_s':at,'faces':[{'id':'face-1','confidence':.9,'speaking':True},{'id':'face-2','confidence':0,'speaking':False}]},at)
  ready=asyncio.create_task(asyncio.sleep(0));await ready
  await s.queue.put(('Ich bin Christoph.',np.ones(32000,np.float32)*.03,2.,0.,{'task':ready,'embedding':v[0],'compute_ms':10.},{},None,s.admission_epoch,None))
  task=asyncio.create_task(s.deliver())
  for _ in range(100):
   if c._send_turn.await_count:break
   await asyncio.sleep(.002)
  task.cancel();await task
  self.assertTrue(c._send_turn.await_count)
  self.assertEqual(c.visual.bindings['face-1']['name'],'Christoph')
  self.assertNotIn('face-2',c.visual.bindings)
  events=[x.args[0] for x in c.ws.send_json.call_args_list]
  self.assertTrue(any(e['type']=='visual_person' for e in events))
