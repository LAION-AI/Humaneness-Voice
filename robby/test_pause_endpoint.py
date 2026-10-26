import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock,Mock
import numpy as np
from pause_vad import SpeechClock
from sentence_session import SentenceSession

class Probabilities:
    def __init__(self,values):self.values=iter(values)
    def run(self,_,inputs):return np.array([[next(self.values)]],np.float32),inputs['state']

class SpeechClockTests(unittest.TestCase):
    def test_no_energy_gate_and_temporal_onset(self):
        # Loud sustained noise is deliberately assigned non-speech by the
        # neural model. One isolated high score is not a speech onset.
        clock=SpeechClock(Probabilities([.01,.02,.8,.02,.9,.9,.4,.1]))
        clock.feed(np.ones(8*512,np.float32)*.5,.256)
        self.assertAlmostEqual(clock.last_speech,.224)
        self.assertFalse(clock.active)
    def test_packet_fragmentation_preserves_32ms_timestamps(self):
        clock=SpeechClock(Probabilities([.9]*5))
        for index in range(8):clock.feed(np.ones(320,np.float32)*.02,(index+1)*.02)
        self.assertEqual(clock.calls,5);self.assertAlmostEqual(clock.last_speech,.16)

@asynccontextmanager
async def slot(_):yield

class PauseSubmissionTests(unittest.IsolatedAsyncioTestCase):
    def session(self):
        c=SimpleNamespace(ws=SimpleNamespace(send_json=AsyncMock()),speaking=False,halting=False,
            session=SimpleNamespace(reset=Mock()),ears=SimpleNamespace(sr=16000),_on_barge=AsyncMock())
        s=SentenceSession(c);s.begin=0.;s.last_speech=1.;s.mic_now=1.2;s.now=1.2
        s.records.append((0.,2.,np.zeros(32000,np.float32)))
        return s
    async def test_punctuation_and_native_endpoint_do_not_submit_while_speaking(self):
        s=self.session()
        await s.preview_text('That is one sentence. Here is another.',1.2,final=True)
        await s.native_endpoint('That is one sentence. Here is another.',1.2)
        self.assertTrue(s.queue.empty());self.assertIn('another',s.preview)
        s.c.session.reset.assert_not_called()
    async def test_pause_sends_all_sentences_once_in_one_turn(self):
        s=self.session();s.mic_now=1.4
        await s.native_endpoint('That is one sentence.',1.)
        await s.preview_text('Here is another.',1.2)
        text=s.native_text();epoch=s.epoch
        self.assertTrue(await s.commit_pause(text,1.4,epoch,1.))
        self.assertEqual(s.queue.get_nowait()[0],'That is one sentence. Here is another.')
        self.assertFalse(await s.commit_pause(text,1.4,epoch,1.))
    async def test_settings_are_bounded_and_apply_to_current_pause(self):
        s=self.session();s.mic_now=1.6;s.set_pause(900)
        self.assertFalse(await s.commit_pause('Bye bye.',1.6,s.epoch,1.))
        s.set_pause(300);self.assertTrue(await s.commit_pause('Bye bye.',1.6,s.epoch,1.))
        for value in [float('nan'),float('inf'),199,2001]:
            with self.assertRaises(ValueError):s.set_pause(value)
    async def test_speech_resuming_during_asr_probe_invalidates_flush(self):
        s=self.session();s.mic_now=s.now=1.5
        entered=asyncio.Event();release=__import__('threading').Event()
        loop=asyncio.get_running_loop()
        def transcribe(audio):
            loop.call_soon_threadsafe(entered.set);release.wait(2)
            return ['First sentence. Second sentence.']
        s.c.speaker_ext=SimpleNamespace(asr_slot=slot)
        s.c.ears.service=SimpleNamespace(engine=SimpleNamespace(asr=SimpleNamespace(transcribe=transcribe)))
        s._maybe_probe();await asyncio.wait_for(entered.wait(),2)
        s.last_speech=1.5;s.mic_now=1.6;release.set();await s.probe
        self.assertTrue(s.queue.empty())
    async def test_incipient_speech_also_blocks_a_pending_pause(self):
        s=self.session();s.mic_now=2.
        s.vad=SimpleNamespace(active=False,probability=.48)
        self.assertFalse(await s.commit_pause('This is not finished.',2.,s.epoch,1.))
    async def test_new_turn_preroll_never_reuses_previous_submitted_words(self):
        s=self.session();s.now=1.4;s.reset_segment()
        s.vad=SpeechClock(Probabilities([.9,.9]))
        await s._speech_audio(np.zeros(1024,np.float32),1.8)
        self.assertEqual(s.begin,1.4)
    async def test_stop_remains_immediate_while_reply_is_playing(self):
        s=self.session();s.c.speaking=True
        s.verify_barge=AsyncMock(return_value={'accepted':True})
        await s.preview_text('Stopp!',1.2)
        await s.barge_task
        s.c._on_barge.assert_awaited_once();self.assertTrue(s.queue.empty())

if __name__=='__main__':unittest.main()
