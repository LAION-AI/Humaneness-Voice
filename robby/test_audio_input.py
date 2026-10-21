"""Audio receipt and preview safety, without models or network services."""
import asyncio,json,struct,unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock,patch,Mock
import numpy as np
from aiohttp import WSMsgType
from server import RoboConversation,Conversation

class AudioInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_simultaneous_stop_paths_commit_cancellation_once(self):
        c=RoboConversation.__new__(RoboConversation)
        c.speaking=True;c.halting=False;c.speaker_ext=None;c.playback=SimpleNamespace(rid=9)
        c.ws=SimpleNamespace(send_json=AsyncMock());c.mouth_done=False;c.have_mouth=False
        c.identity_audio=np.zeros(0,np.float32);c.identity=SimpleNamespace(reset_utterance=Mock())
        c.teacher_stream=None;c.stop_feedback=SimpleNamespace(schedule=Mock())
        entered=asyncio.Event();release=asyncio.Event()
        async def slow(*args):entered.set();await release.wait()
        with patch.object(Conversation,'_on_barge',new=AsyncMock(side_effect=slow)) as parent:
            task=asyncio.create_task(c._on_barge({'onset_s':0.,'yield_prob':1.},16000))
            await entered.wait();await c._on_barge({'onset_s':0.,'yield_prob':1.},16000)
            release.set();await task
        parent.assert_awaited_once();c.ws.send_json.assert_awaited_once();c.stop_feedback.schedule.assert_called_once()
    def test_probe_echo_residual_preserves_double_talk_and_native_input(self):
        from audio_sync import suppress_aligned_leakage
        rng=np.random.default_rng(37)
        reference=rng.normal(0,.08,48000).astype(np.float32)
        user=np.zeros(48000,np.float32);user[16000:24000]=rng.normal(0,.06,8000)
        mic=user+.05*reference;original=mic.copy()
        residual=suppress_aligned_leakage(mic,reference)
        np.testing.assert_allclose(residual,user,atol=1e-7)
        np.testing.assert_array_equal(mic,original)
        np.testing.assert_array_equal(suppress_aligned_leakage(mic,np.zeros_like(mic)),mic)
        unrelated=rng.normal(0,.03,48000).astype(np.float32)
        np.testing.assert_array_equal(suppress_aligned_leakage(unrelated,reference),unrelated)
    def test_unknown_browser_acoustic_delay_preserves_the_foreground_word(self):
        from audio_sync import suppress_aligned_leakage
        rng=np.random.default_rng(1997)
        reference=rng.normal(0,.12,48000).astype(np.float32)
        for delay in (-640,1920,4000):
            shifted=np.zeros_like(reference)
            if delay>0:shifted[delay:]=reference[:-delay]
            else:shifted[:delay]=reference[-delay:]
            user=np.zeros_like(reference);user[32000:38400]=rng.normal(0,.09,6400)
            mic=shifted*.05+user
            residual=suppress_aligned_leakage(mic,reference)
            self.assertLess(float(np.sqrt(np.mean((residual-user)**2))),1e-5)
            np.testing.assert_array_equal(mic,shifted*.05+user)
    def test_phase_coloured_echo_is_removed_without_fitting_the_user(self):
        from audio_sync import suppress_aligned_leakage
        rng=np.random.default_rng(38)
        reference=rng.normal(0,.08,48000).astype(np.float32)
        echo=.04*reference+.008*np.r_[0,reference[:-1]]
        user=np.zeros(48000,np.float32);user[18000:22000]=rng.normal(0,.1,4000)
        mic=(echo+user).astype(np.float32);saved=mic.copy()
        residual=suppress_aligned_leakage(mic,reference)
        self.assertLess(np.sqrt(np.mean((residual-user)**2)),2e-5)
        np.testing.assert_array_equal(mic,saved)
    def conversation(self):
        c=RoboConversation.__new__(RoboConversation)
        c.ws=SimpleNamespace(send_json=AsyncMock())
        c.input_samples=0;c.input_blocks=0;c.last_input_notice=0.
        c.pending_barge_proposal=None
        c._send_turn=AsyncMock()
        c._on_barge=AsyncMock();c.playback=SimpleNamespace(rid=1);c.decisions=SimpleNamespace(proactive_active=False,tick=Mock(),offer=Mock())
        return c
    async def test_receipt_ack_has_level_and_duration_but_no_audio(self):
        c=self.conversation();pcm=np.full(320,3277,'<i2')
        message=SimpleNamespace(type=WSMsgType.BINARY,data=struct.pack('<ii',0,-1)+pcm.tobytes())
        with patch.object(Conversation,'_on_browser',new_callable=AsyncMock) as parent:
            await c._on_browser(message);parent.assert_awaited_once_with(message)
        packet=c.ws.send_json.call_args.args[0]
        self.assertEqual(packet['type'],'input_status');self.assertEqual(packet['seconds'],.02)
        self.assertAlmostEqual(packet['rms_dbfs'],-20.,places=1)
        self.assertEqual(set(packet),{'type','blocks','seconds','rms_dbfs'})
    async def test_shared_service_status_is_not_a_spoken_reply_or_identity_update(self):
        c=self.conversation();c.speaker_ext=SimpleNamespace(metadata=AsyncMock())
        await c._on_text(json.dumps({'status_only':True,'stage':'waiting','text':'Waiting for shared voice model.'}))
        message=c.ws.send_json.call_args.args[0]
        self.assertEqual(message['type'],'service_status');self.assertEqual(message['stage'],'waiting')
        c.speaker_ext.metadata.assert_not_awaited()
    async def test_speaker_filter_can_show_asr_without_submitting_it(self):
        c=self.conversation();c.identity_enabled=True;c.identity_mode='v4-long';c.speaking=False
        c.ears=SimpleNamespace(sr=16000)
        from unittest.mock import Mock
        c.speaker_ext=SimpleNamespace(observe=AsyncMock(),last_activity=None,note_asr=Mock(),schedule_stop_probe=Mock())
        events=[{'kind':'PARTIAL','text':'Can you hear me?'},{'kind':'ENDPOINT','transcript':'Can you hear me?'}]
        with patch.object(Conversation,'_on_events',new_callable=AsyncMock) as parent:
            await c._on_events(events,5120);parent.assert_awaited_once_with([],5120)
        preview=c.ws.send_json.call_args.args[0]
        self.assertEqual(preview['type'],'asr_preview');self.assertEqual(preview['status'],'not_submitted')
        c._send_turn.assert_not_awaited()
    async def test_filter_off_restores_native_asr_events(self):
        c=self.conversation();c.identity_enabled=False;c.identity_mode='v4-long';c.speaker_ext=None
        events=[{'kind':'PARTIAL','text':'Can you hear me?'},{'kind':'ENDPOINT','transcript':'Can you hear me?'}]
        with patch.object(Conversation,'_on_events',new_callable=AsyncMock) as parent:
            await c._on_events(events,5120);parent.assert_awaited_once_with(events,5120)
    async def test_native_interruption_is_not_authorized_by_channel_alone(self):
        c=self.conversation();c.identity_enabled=True;c.identity_mode='v4-long';c.speaking=True
        c.ears=SimpleNamespace(sr=16000)
        c.speaker_ext=SimpleNamespace(observe=AsyncMock(),last_activity={'id':'someone'},confirm_interrupt=AsyncMock(return_value=None),schedule_stop_probe=Mock(),epoch=0,spawn=asyncio.create_task)
        proposal={'kind':'BARGE','yield_prob':.7,'onset_s':1.}
        with patch.object(Conversation,'_on_events',new_callable=AsyncMock) as parent:
            await c._on_events([proposal],5120);parent.assert_awaited_once_with([],5120)
        await c.native_barge_task
        c.speaker_ext.confirm_interrupt.assert_awaited_once_with(proposal,5120)
        c.speaker_ext.confirm_interrupt.return_value='verified-user'
        with patch.object(Conversation,'_on_events',new_callable=AsyncMock) as parent:
            await c._on_events([],10240);parent.assert_awaited_once_with([],10240)
        await c.native_barge_task;c._on_barge.assert_awaited_once_with(proposal,10240)
    def test_native_settings_restored_when_speaker_filter_is_disabled(self):
        c=self.conversation();c.session=object();c.native_barge_settings=(.5,2);c.ears=SimpleNamespace(set_threshold=Mock())
        c._speaker_barge_sensitivity(True)
        self.assertEqual([call.args[2] for call in c.ears.set_threshold.call_args_list],[.2,1])
        c.ears.set_threshold.reset_mock();c._speaker_barge_sensitivity(False)
        self.assertEqual([call.args[2] for call in c.ears.set_threshold.call_args_list],[.5,2])
    async def test_hop_keeps_strongest_native_proposal_not_last_weak_frame(self):
        c=self.conversation();c.identity_enabled=True;c.identity_mode='v4-long';c.speaking=True;c.ears=SimpleNamespace(sr=16000)
        c.speaker_ext=SimpleNamespace(observe=AsyncMock(),last_activity=None,confirm_interrupt=AsyncMock(return_value=None),schedule_stop_probe=Mock(),epoch=0,spawn=asyncio.create_task)
        strong={'kind':'BARGE','yield_prob':.9};weak={'kind':'BARGE','yield_prob':.25}
        with patch.object(Conversation,'_on_events',new_callable=AsyncMock):await c._on_events([strong,weak],5120)
        await c.native_barge_task
        c.speaker_ext.confirm_interrupt.assert_awaited_once_with(strong,5120)

if __name__=='__main__':unittest.main()
