import asyncio,unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock,patch
from stop_feedback import StopFeedback,PHRASES,allows_ack

class FeedbackTests(unittest.IsolatedAsyncioTestCase):
    def make(self):
        c=SimpleNamespace(playback=SimpleNamespace(rid=3),current_speaker={'speaker_id':'a'},
            decisions=SimpleNamespace(language='de'),ws=SimpleNamespace(closed=False,send_json=AsyncMock()),
            speaking=False,halting=False,pending=None,hop_pos=160000,
            speaker_ext=SimpleNamespace(closed=False,memory=SimpleNamespace(users={'a':True}),pending_turns={},live_speech_recent=Mock(return_value=False)),
            session=SimpleNamespace(reset=Mock()),_start_speaking=Mock(),to_mouth=AsyncMock())
        return StopFeedback(c),c
    async def test_silent_default_and_wait_always_silent(self):
        f,c=self.make();f.schedule({'source':'verified_stop_command','stop_text':'Stopp'})
        self.assertIsNone(f.task);f.enabled=True
        for text in ('Warte bitte','Bitte sei leise','Pause','Be quiet','Wait please'):
            self.assertFalse(allows_ack(text));f.schedule({'source':'verified_stop_command','stop_text':text});self.assertIsNone(f.task)
    async def test_four_local_phrases_no_llm_and_no_prefix_invented(self):
        f,c=self.make();f.enabled=True
        with patch('stop_feedback.asyncio.sleep',new=AsyncMock()):
            for _ in range(4):f.schedule({'source':'verified_stop_command','stop_text':'Stopp'});await f.task
        self.assertEqual([call.args[1]['interaction_reply'] for call in c.to_mouth.await_args_list],list(PHRASES['de']))
        self.assertTrue(all(call.args[1]['stop_feedback'] for call in c.to_mouth.await_args_list))
    async def test_new_speech_new_reply_pending_turn_or_disconnect_suppress_ack(self):
        for reason in ('speech','reply','pending','disconnect','halting','disabled','other_owner'):
            f,c=self.make();f.enabled=True
            if reason=='speech':c.speaker_ext.live_speech_recent.return_value=True
            if reason=='reply':c.speaking=True
            if reason=='pending':c.speaker_ext.pending_turns={'turn':True}
            if reason=='disconnect':c.ws.closed=True
            if reason=='halting':c.halting=True
            with patch('stop_feedback.asyncio.sleep',new=AsyncMock()):
                f.schedule({'source':'verified_stop_command','stop_text':'Stopp'})
                if reason=='disabled':f.enabled=False
                if reason=='other_owner':c.current_speaker={'speaker_id':'b'}
                await f.task
            c.to_mouth.assert_not_awaited()
    async def test_cancel_on_new_turn_or_session_close(self):
        f,c=self.make();f.enabled=True;f.schedule({'source':'verified_stop_command','stop_text':'Stopp'})
        task=f.task;f.cancel();await asyncio.gather(task,return_exceptions=True)
        c.to_mouth.assert_not_awaited()
