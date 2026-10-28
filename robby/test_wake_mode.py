"""Wake expiry, speaker handover, background isolation and explicit sounds."""
import asyncio,unittest
from unittest.mock import AsyncMock,patch
import numpy as np
from wake_mode import WakeMode
from sentence_session import OpenVoiceMemory

class WakePolicy(unittest.TestCase):
 def setUp(self):
  self.t=[100.];self.w=WakeMode(clock=lambda:self.t[0]);self.m=OpenVoiceMemory(clock=lambda:self.t[0]);self.v=np.eye(192,dtype=np.float32)
 def test_only_addressed_start_then_same_voice_and_expiry(self):
  self.assertFalse(self.w.evaluate('Hallo, wie geht es dir?',self.v[0],self.m.users)['accepted'])
  self.assertTrue(self.w.evaluate('Hey Robbie, erzähl mir einen Witz.',self.v[0],self.m.users)['wake'])
  a=self.m.assign(self.v[0],'Hey Robbie, ich bin Christoph.');self.w.accept(a['speaker_id'],self.v[0])
  self.t[0]+=9.9;self.assertTrue(self.w.evaluate('Noch einen.',self.v[0],self.m.users)['accepted'])
  self.assertFalse(self.w.evaluate('Noch einen.',self.v[1],self.m.users)['accepted'])
  self.t[0]+=0.2;self.assertFalse(self.w.evaluate('Noch einen.',self.v[0],self.m.users)['accepted'])
  self.assertTrue(self.w.evaluate('Ein längerer Satz.',self.v[0],self.m.users,at=109.)['accepted'],'A sentence that began before expiry may finish later')
 def test_foreign_wake_handover_and_followup_refresh(self):
  a=self.m.assign(self.v[0],'Ich bin Christoph.');self.w.accept(a['speaker_id'],self.v[0]);self.t[0]+=8
  self.assertTrue(self.w.evaluate('Ja.',self.v[0],self.m.users)['accepted']);self.w.accept(a['speaker_id'],self.v[0]);self.t[0]+=8
  self.assertTrue(self.w.active());self.assertTrue(self.w.evaluate('Hallo Robby, ich bin Selma.',self.v[1],self.m.users)['wake'])
  b=self.m.assign(self.v[1],'Ich bin Selma.');self.w.accept(b['speaker_id'],self.v[1])
  self.assertFalse(self.w.evaluate('Hallo!',self.v[0],self.m.users)['accepted']);self.assertTrue(self.w.evaluate('Hallo!',self.v[1],self.m.users)['accepted'])
 def test_wake_must_be_at_start_and_no_identity_from_gender_or_missing_audio(self):
  self.assertFalse(self.w.evaluate('Im Fernsehen sagte jemand Hey Robby.',self.v[0],self.m.users)['accepted'])
  self.assertTrue(self.w.evaluate('Hey Robby, hallo.',None,self.m.users)['accepted']);self.w.accept('speaker-1',None)
  self.assertFalse(self.w.evaluate('Bin ich noch dran?',None,self.m.users)['accepted'])
 def test_open_mode_and_clear(self):
  self.w.configure(False);self.assertTrue(self.w.evaluate('Ohne Aufruf.',None,self.m.users)['accepted'])
  self.w.configure(True);self.assertFalse(self.w.active())
 def test_common_wake_spellings_and_greetings(self):
  for greeting in ('Hey','Hi','Hallo','Hello','hey'):
   for name in ('Robby','Robbie','Robbi','Robi'):
    with self.subTest(greeting=greeting,name=name):
     self.assertTrue(self.w.evaluate(f'{greeting}, {name}, wie geht es dir?',None,self.m.users)['wake'])
  for text in ('Jemand sagte Hey Robby.', 'Hallo, Robert.', 'Robby erzählt eine Geschichte.'):
   self.assertFalse(self.w.evaluate(text,None,self.m.users)['accepted'])
 def test_reply_keeps_same_speaker_available_then_renews_window(self):
  a=self.m.assign(self.v[0],'Ich bin Christoph.');self.w.accept(a['speaker_id'],self.v[0]);self.t[0]+=30
  self.assertFalse(self.w.active());self.assertTrue(self.w.evaluate('Weiter.',self.v[0],self.m.users,reply_active=True)['accepted'])
  self.assertFalse(self.w.evaluate('Fremd.',self.v[1],self.m.users,reply_active=True)['accepted'])
  self.w.finish_reply('wrong-speaker');self.assertFalse(self.w.active())
  self.w.finish_reply(a['speaker_id']);self.assertTrue(self.w.active());self.t[0]+=10.1;self.assertFalse(self.w.active())

class WakeIntegration(unittest.IsolatedAsyncioTestCase):
 async def test_bare_wake_waits_for_the_rest_of_the_request(self):
  from test_sentence_reliability import ReliabilityTests
  from types import SimpleNamespace
  s,c,v=ReliabilityTests().session();s.wake.configure(True);s.set_pause(300)
  s.vad=SimpleNamespace(active=False,probability=0.);s.preview='Hey Robby';s.last_speech=s.word_at=3.;s.mic_now=3.4
  self.assertFalse(s.quiet_ready())
  s.preview='Hey Robby erzähl mir einen Witz';self.assertTrue(s.quiet_ready())
  s.preview='Hey Robby';s.mic_now=4.3;self.assertTrue(s.quiet_ready())
  s.wake.configure(False);s.preview='Hoppy';s.mic_now=3.4;self.assertFalse(s.quiet_ready())
  s.preview='Was ist denn das hier?';self.assertTrue(s.quiet_ready())
 async def test_wake_interrupt_verifies_owner_and_rejects_other_voice(self):
  from test_sentence_reliability import ReliabilityTests
  from types import SimpleNamespace
  s,c,v=ReliabilityTests().session();s.wake.configure(True);s.focus.enabled=False;s.wake.accept('speaker-1',v[0])
  c.speaking=True;c.playback.n16=0;s.reply_started=0.;s.last_speech=3.
  with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda audio:v[0])):
   self.assertTrue((await s.verify_barge(3.,1,lexical=True))['accepted'])
  with patch('speaker_teacher.get_cached_teacher',return_value=SimpleNamespace(embed=lambda audio:v[1])):
   self.assertFalse((await s.verify_barge(3.,1,lexical=True))['accepted'])
 async def test_real_sentence_queue_only_sends_wake_then_same_speaker(self):
  from test_sentence_reliability import ReliabilityTests
  s,c,v=ReliabilityTests().session();s.wake.configure(True);s.focus.enabled=False
  async def send(text,vec):
   ready=asyncio.create_task(asyncio.sleep(0));await ready
   mark=c._send_turn.await_count
   await s.queue.put((text,np.ones(32000,np.float32)*.03,2.,0.,{'task':ready,'embedding':vec,'compute_ms':1.},{},None,s.admission_epoch,None))
   for _ in range(150):
    await asyncio.sleep(.002)
    if c._send_turn.await_count>mark or any(x.args[0].get('type')=='ignored' for x in c.ws.send_json.call_args_list):break
   await asyncio.sleep(.01)
   c.ws.send_json.reset_mock()
  task=asyncio.create_task(s.deliver())
  try:
   await send('Fremde Unterhaltung.',v[1]);self.assertEqual(c._send_turn.await_count,0)
   await send('Hey Robby, ich bin Christoph.',v[0]);self.assertEqual(c._send_turn.await_count,1)
   await send('Und noch eine Frage.',v[0]);self.assertEqual(c._send_turn.await_count,2)
   await send('Eine andere Person.',v[1]);self.assertEqual(c._send_turn.await_count,2)
   await send('Hey Robby, ich bin Selma.',v[1]);self.assertEqual(c._send_turn.await_count,3)
  finally:task.cancel();await task
 async def test_mixed_long_segment_checks_tail_without_learning_a_bad_anchor(self):
  from test_sentence_reliability import ReliabilityTests
  from types import SimpleNamespace
  s,c,v=ReliabilityTests().session();s.wake.configure(True);s.focus.enabled=False;s.wake.accept('speaker-1',v[0])
  ready=asyncio.create_task(asyncio.sleep(0));await ready
  cache={'task':ready,'embedding':v[1],'compute_ms':1.}
  await s.queue.put(('Was ist das hier?',np.ones(48000,np.float32)*.03,3.,0.,cache,{},None,s.admission_epoch,None))
  teacher=SimpleNamespace(embed=lambda audio:v[0]);task=asyncio.create_task(s.deliver())
  try:
   with patch('speaker_teacher.get_cached_teacher',return_value=teacher):
    for _ in range(150):
     await asyncio.sleep(.002)
     if c._send_turn.await_count:break
   c._send_turn.assert_awaited_once();self.assertEqual(len(s.memory.users['speaker-1']['vectors']),1)
  finally:task.cancel();await task
 async def test_tail_retry_does_not_admit_a_different_voice(self):
  from test_sentence_reliability import ReliabilityTests
  from types import SimpleNamespace
  s,c,v=ReliabilityTests().session();s.wake.configure(True);s.focus.enabled=False;s.wake.accept('speaker-1',v[0])
  ready=asyncio.create_task(asyncio.sleep(0));await ready
  cache={'task':ready,'embedding':v[1],'compute_ms':1.};await s.queue.put(('Fremde Stimme.',np.ones(48000,np.float32)*.03,3.,0.,cache,{},None,s.admission_epoch,None))
  teacher=SimpleNamespace(embed=lambda audio:v[1]);task=asyncio.create_task(s.deliver())
  try:
   with patch('speaker_teacher.get_cached_teacher',return_value=teacher):
    for _ in range(150):
     await asyncio.sleep(.002)
     if any(x.args[0].get('type')=='ignored' for x in c.ws.send_json.call_args_list):break
   c._send_turn.assert_not_awaited();self.assertTrue(any(x.args[0].get('source')=='wake_mode' for x in c.ws.send_json.call_args_list))
  finally:task.cancel();await task
 async def test_weak_match_needs_strong_independent_fast_voice_cue(self):
  from test_sentence_reliability import ReliabilityTests
  from sentence_session import unit
  s,c,v=ReliabilityTests().session();s.wake.configure(True);s.focus.enabled=False;s.wake.accept('speaker-1',v[0])
  q=unit(.28*v[0]+np.sqrt(1-.28**2)*v[1]);ready=asyncio.create_task(asyncio.sleep(0));await ready
  cache={'task':ready,'embedding':q,'compute_ms':1.};task=asyncio.create_task(s.deliver())
  try:
   await s.queue.put(('Fremd.',np.ones(32000,np.float32)*.03,3.,0.,cache,{},v[1],s.admission_epoch,None))
   for _ in range(100):
    await asyncio.sleep(.002)
    if any(x.args[0].get('type')=='ignored' for x in c.ws.send_json.call_args_list):break
   c._send_turn.assert_not_awaited()
   await s.queue.put(('Was ist das?',np.ones(32000,np.float32)*.03,3.,0.,cache,{},v[0],s.admission_epoch,None))
   for _ in range(100):
    await asyncio.sleep(.002)
    if c._send_turn.await_count:break
   c._send_turn.assert_awaited_once();self.assertEqual(len(s.memory.users['speaker-1']['vectors']),1)
  finally:task.cancel();await task
 async def test_sound_request_is_available_with_automatic_sensing_off(self):
  from test_environment import SessionTests
  e=SessionTests();await e.asyncSetUp()
  try:
   e.e.enabled=False;e.e.note_text('Was ist das hier?',5.);e.e.audio_end=7.5
   for i in range(200):e.e.audio_ring.append((3.5+(i+1)*.02,np.ones(320,np.float32)*.1))
   e.e.caption=AsyncMock(return_value={'kind':'music','requested':True})
   result=await e.e.context_for_turn('Was ist das hier?');self.assertEqual(result['kind'],'music')
   self.assertGreater(len(e.e.caption.call_args.args[0]['audio']),38000)
  finally:await e.asyncTearDown()
 async def test_sound_automatics_off_keeps_only_bounded_question_buffer(self):
  from test_environment import SessionTests
  e=SessionTests();await e.asyncSetUp()
  try:
   e.e.enabled=False
   with patch.object(e.e.detector,'feed') as detect:
    for n in range(900):e.e.feed(np.ones(320,np.float32)*.01,n*.02)
    detect.assert_not_called();self.assertEqual(len(e.e.audio_ring),750)
  finally:await e.asyncTearDown()
 async def test_sound_before_question_is_not_replaced_by_voice_only_afterward(self):
  from test_environment import SessionTests
  e=SessionTests();await e.asyncSetUp()
  try:
   e.e.enabled=False;e.e.note_text('Was ist das hier?',5.);e.e.audio_end=8.
   for i in range(225):e.e.audio_ring.append((3.5+(i+1)*.02,np.ones(320,np.float32)*.1))
   e.e.caption=AsyncMock(side_effect=[{'kind':'music','caption':'piano'},{'kind':'voice','caption':'speech'}])
   result=await e.e.context_for_turn('Was ist das hier?')
   self.assertEqual(result['kind'],'music');self.assertEqual(result['sound_window'],'before');self.assertEqual(e.e.caption.await_count,2)
  finally:await e.asyncTearDown()
 async def test_proactive_sound_waits_for_wake_and_foreign_voice_never_prompts(self):
  from test_environment import SessionTests
  from types import SimpleNamespace
  e=SessionTests();await e.asyncSetUp()
  try:
   e.e.enabled=True;e.s.wake=SimpleNamespace(enabled=True,active=lambda:False)
   e.e.pending={'kind':'sound','caption':'A knock','created':e.time,'epoch':e.e.epoch}
   task=asyncio.create_task(e.e.run());await asyncio.sleep(.15)
   self.assertIsNone(e.e.pending);e.c.to_mouth.assert_not_awaited()
   e.s.wake.active=lambda:True;e.e.pending={'kind':'unfamiliar_voice','caption':'Possible voice','created':e.time,'epoch':e.e.epoch}
   await asyncio.sleep(.15);self.assertIsNone(e.e.pending);e.c.to_mouth.assert_not_awaited()
   e.e.pending={'kind':'sound','caption':'A knock','created':e.time,'epoch':e.e.epoch}
   await asyncio.sleep(.15);e.c.to_mouth.assert_awaited_once();task.cancel();await asyncio.gather(task,return_exceptions=True)
  finally:await e.asyncTearDown()
