import asyncio,unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock,patch
import numpy as np
from bolt_text import SentenceFramer,ASRSentenceCommitter,select_template,CATALOGUE
from sentence_session import OpenVoiceMemory,name_claim,SentenceSession

class BoltTextTests(unittest.TestCase):
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
    async def test_current_native_words_arrive_before_quiet_flush(self):
        import paths
        from server import RoboConversation
        from app.ears.conversation import Conversation
        order=[];c=RoboConversation.__new__(RoboConversation);c.ears=SimpleNamespace(sr=16000)
        c.avatar_personality=SimpleNamespace(last_asr=-100.)
        async def preview(text,at):order.append(('words',text))
        async def observe(event,end):order.append(('waveform',end))
        async def advance():order.append(('flush',None))
        c.sentence_stream=SimpleNamespace(preview_text=preview,observe=observe,advance=advance);c.ws=SimpleNamespace(send_json=AsyncMock())
        with patch.object(Conversation,'_on_events',new_callable=AsyncMock):
            await c._on_events([{'kind':'SPEAKER_FEATURES'},{'kind':'PARTIAL','text':'Complete sentence with final words.'}],5120)
        self.assertEqual(order,[('waveform',5120),('words','Complete sentence with final words.'),('flush',None)])
        self.assertGreater(c.avatar_personality.last_asr,0.,'Sentence mode must also suppress idle antics while listening')
    async def test_ready_embedding_is_reused_instead_of_recomputed_on_submit(self):
        c=SimpleNamespace(ws=SimpleNamespace(send_json=AsyncMock()),speaking=False,halting=False,
            _send_turn=AsyncMock(),current_speaker=None)
        s=SentenceSession(c);ready=asyncio.create_task(asyncio.sleep(0));await ready
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
