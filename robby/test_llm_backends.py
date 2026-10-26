import asyncio,json,os,stat,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock,patch
import httpx
from credential_store import CredentialStore
from llm_backends import config,request_body,catalog,metrics,selection

def test_key():return 'sk-or-v1-'+'unit-test-private-'*4

class VaultTests(unittest.TestCase):
    def test_authenticated_encryption_permissions_and_tamper_detection(self):
        from cryptography.fernet import InvalidToken
        with tempfile.TemporaryDirectory() as tmp:
            vault=CredentialStore(Path(tmp)/'credentials');key=test_key();vault.store_owner(key)
            self.assertEqual(vault.owner(),key)
            for name in ('openrouter.enc','master.key'):
                p=vault.root/name;self.assertNotIn(key.encode(),p.read_bytes());self.assertEqual(stat.S_IMODE(p.stat().st_mode),0o600)
            sealed=vault.seal(key);tampered=sealed[:-8]+'abcdefgh'
            with self.assertRaises(InvalidToken):vault.open(tampered)
            os.chmod(vault.root/'master.key',0o644)
            with self.assertRaises(RuntimeError):vault.owner()
    def test_provider_routing_is_allowlisted_and_key_never_enters_public_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault=CredentialStore(tmp);vault.store_owner(test_key())
            with patch('llm_backends.store',return_value=vault):
                cfg=config({'model':'qwen','provider':'cerebras/fp16'})
                body=request_body(cfg,{'chat_template_kwargs':{'test':True},'max_completion_tokens':100,'response_format':{'type':'json_object'}})
                self.assertEqual(cfg['url'],'https://openrouter.ai/api/v1/chat/completions')
                self.assertEqual(body['provider'],{'only':['cerebras/fp16'],'allow_fallbacks':False})
                self.assertEqual(body['max_tokens'],100);self.assertNotIn('chat_template_kwargs',body)
                self.assertNotIn(test_key(),json.dumps({'catalog':catalog(),'metrics':metrics(cfg),'body':body}))
                for bad in ({'model':'unknown','provider':'local'},{'model':'qwen','provider':'https://example.invalid'},{'model':'local','provider':'cerebras/fp16'}):
                    with self.assertRaises(ValueError):selection(bad)
    def test_local_transport_never_reads_the_openrouter_key(self):
        with patch('llm_backends.store',side_effect=AssertionError('No cloud key in local mode')):
            cfg=config({'model':'local','provider':'local'})
        self.assertEqual(cfg['headers'],{});self.assertFalse(cfg['remote'])

class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_public_catalog_refresh_installs_alternative_endpoint_tags(self):
        import llm_backends
        data={'data':{'endpoints':[
            {'provider_name':'Cerebras','tag':'cerebras/fp16','status':0,'quantization':'fp16','supported_parameters':['tools']},
            {'provider_name':'DeepInfra','tag':'deepinfra/bf16','status':0,'quantization':'bf16','supported_parameters':['tools','response_format']},
            {'provider_name':'Disabled','tag':'disabled','status':-2,'supported_parameters':[]}]}}
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,json=data))) as c:
            with patch('llm_backends._providers',dict(llm_backends._providers)),patch('llm_backends._updated',-1000.):
                await llm_backends.refresh_providers(c)
                self.assertEqual(selection({'model':'qwen','provider':'deepinfra/bf16'})['provider'],'deepinfra/bf16')
                self.assertNotIn('disabled',[r['id'] for r in catalog()['providers']])
    async def test_one_rate_retry_before_speech_and_no_secret_in_stream_metadata(self):
        from bolt_director import BoltDirector
        with tempfile.TemporaryDirectory() as tmp:
            vault=CredentialStore(tmp);vault.store_owner(test_key());requests=[]
            def handle(request):
                requests.append(json.loads(request.content))
                if len(requests)==1:return httpx.Response(429,headers={'retry-after':'0'})
                packet={'provider':'Cerebras','choices':[{'delta':{'content':'(interest) Hallo!'},'finish_reason':'stop'}]}
                return httpx.Response(200,content=('data: '+json.dumps(packet)+'\n\ndata: [DONE]\n\n').encode())
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as c:
                observed={}
                with patch('llm_backends.store',return_value=vault):
                    parts=[p async for p in BoltDirector().stream(c,'Hallo',{},[],backend={'model':'qwen','provider':'cerebras/fp16'},metrics=observed)]
            self.assertEqual(len(requests),2);self.assertEqual([p.text for p in parts],['Hallo!'])
            self.assertEqual(observed['llm_rate_limit_retries'],1);self.assertEqual(observed['llm_actual_provider'],'Cerebras')
            self.assertNotIn(test_key(),json.dumps(observed));self.assertNotIn(test_key(),json.dumps(requests))
    async def test_personal_key_is_session_only_and_not_in_settings(self):
        from llm_session import LLMSession
        with tempfile.TemporaryDirectory() as tmp:
            vault=CredentialStore(tmp);vault.store_owner(test_key());ws=SimpleNamespace(send_json=AsyncMock())
            with patch('llm_session.store',return_value=vault):
                personal=test_key()+'p'
                s=LLMSession(SimpleNamespace(ws=ws));await s.command('llm:key:'+personal)
                self.assertEqual(s.settings()['key_mode'],'personal_session')
                self.assertNotIn(test_key(),json.dumps(s.settings()));self.assertNotIn(s.credential,json.dumps(s.settings()))
                self.assertEqual(vault.open(s.snapshot()['llm_credential']),personal)
                self.assertEqual(vault.owner(),test_key())
                await s.command('llm:key:clear');self.assertIsNone(s.credential)
                self.assertEqual(s.settings()['key_mode'],'server')

if __name__=='__main__':unittest.main()
