import asyncio,unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock,patch
import httpx,numpy as np
from asr_choice import ASRChoice
from sentence_session import SentenceSession
from speech_language import reply_language

class LanguageTests(unittest.TestCase):
    def test_german_default_and_explicit_language(self):
        self.assertEqual(reply_language('Hello Robby, what can we do?'),'German')
        self.assertEqual(reply_language('Antworte auf Englisch.'),'English')
        self.assertEqual(reply_language('Please write a poem in English.'),'English')
        self.assertEqual(reply_language('Weiter.',[{'role':'user','content':'Bitte auf Englisch.'}]),'English')
        self.assertEqual(reply_language('Jetzt wieder auf Deutsch.',[{'role':'user','content':'Bitte auf Englisch.'}]),'German')

class ChoiceTests(unittest.IsolatedAsyncioTestCase):
    def session(self):
        c=SimpleNamespace(ws=SimpleNamespace(send_json=AsyncMock()),speaking=False,halting=False,
            session=SimpleNamespace(reset=Mock()),ears=SimpleNamespace(sr=16000),_on_barge=AsyncMock())
        s=SentenceSession(c);s.begin=0.;s.last_speech=1.;s.mic_now=s.now=1.5
        s.records.append((0.,2.,np.zeros(32000,np.float32)))
        return s
    async def test_failed_load_retains_previous_model_and_audio(self):
        s=self.session();s.asr.http=SimpleNamespace(post=AsyncMock(side_effect=httpx.ConnectError('offline')),aclose=AsyncMock())
        await s.asr.select('parakeet_v3');await s.asr.task
        self.assertEqual(s.asr.selected,'nemotron');self.assertEqual(s.begin,0.);self.assertTrue(s.records)
        await s.close()
    async def test_original_is_available_even_if_optional_worker_is_offline(self):
        s=self.session();s.asr.selected='whisper_turbo'
        s.asr.http=SimpleNamespace(post=AsyncMock(side_effect=httpx.ConnectError('offline')),aclose=AsyncMock())
        await s.asr.select('nemotron');await s.asr.task
        self.assertEqual(s.asr.selected,'nemotron');self.assertEqual(s.epoch,1);self.assertTrue(s.records)
        await s.close()
    async def test_paused_alternative_replaces_preview_without_second_native_decode(self):
        s=self.session();s.asr.selected='whisper_turbo'
        s.asr.transcribe=AsyncMock(return_value={'text':'Das ist die korrigierte Erkennung.','model':'whisper_turbo','transcribe_ms':12})
        # No native ASR service is supplied: finalization must not depend on it.
        s._maybe_probe();await s.probe
        item=s.queue.get_nowait();self.assertEqual(item[0],'Das ist die korrigierte Erkennung.')
        self.assertEqual(item[5]['model'],'whisper_turbo');await s.close()
    async def test_resume_or_model_switch_while_alternative_decodes_prevents_stale_commit(self):
        for change in ('speech','model'):
            s=self.session();s.asr.selected='moonshine_de';entered=asyncio.Event();release=asyncio.Event()
            async def decode(*args):
                entered.set();await release.wait()
                return {'text':'Ein alter Zwischenstand.','model':'moonshine_de'}
            s.asr.transcribe=decode;s._maybe_probe();await entered.wait()
            if change=='speech':s.last_speech=1.5;s.mic_now=1.6
            else:s.epoch+=1
            release.set();await s.probe;self.assertTrue(s.queue.empty());self.assertIsNotNone(s.begin)
            await s.close()
    async def test_long_buffer_is_chunked_and_overlap_is_not_repeated(self):
        s=self.session();outputs=iter(['Das ist ein Satz. Noch ein Wort.','Noch ein Wort. Und weiter.'])
        async def post(url,content,headers):
            self.assertLessEqual(len(content),30*16000*4)
            return httpx.Response(200,json={'text':next(outputs),'model':'whisper_turbo','transcribe_ms':10},request=httpx.Request('POST','http://localhost'))
        s.asr.http=SimpleNamespace(post=post,aclose=AsyncMock())
        result=await s.asr.transcribe('whisper_turbo',np.zeros(35*16000,np.float32))
        self.assertEqual(result['text'],'Das ist ein Satz. Noch ein Wort. Und weiter.')
        self.assertEqual(result['chunks'],2);await s.close()
    async def test_unknown_model_never_contacts_worker(self):
        s=self.session();s.asr.http=SimpleNamespace(post=AsyncMock(),aclose=AsyncMock())
        await s.asr.select('../../arbitrary');s.asr.http.post.assert_not_called()
        self.assertEqual(s.asr.selected,'nemotron');await s.close()

if __name__=='__main__':unittest.main()
