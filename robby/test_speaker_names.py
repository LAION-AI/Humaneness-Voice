import json,unittest
from types import SimpleNamespace
from unittest.mock import patch
from speaker_names import NameResolver

class Client:
    def __init__(self,data,unsupported=False):self.data=data;self.unsupported=unsupported;self.calls=[]
    async def post(self,*args,**kwargs):
        self.calls.append(kwargs['json'].copy());status=400 if self.unsupported and len(self.calls)==1 else 200
        data=self.data
        class Response:
            status_code=status
            def raise_for_status(self):pass
            def json(self):return {'choices':[{'message':{'content':json.dumps(data)}}]}
        return Response()

class NameTests(unittest.IsolatedAsyncioTestCase):
    async def resolve(self,data,text='Hey Robbie, ich bin Christoph.',unsupported=False):
        client=Client(data,unsupported)
        with patch('speaker_names.shared_settings',return_value=SimpleNamespace(luna_key=lambda:'dummy',LUNA_BASE='https://invalid.example',LUNA_MODEL='configured-luna')):
            result=await NameResolver().resolve(client,text,'speaker-test')
        return result,client
    async def test_explicit_json_metadata(self):
        result,client=await self.resolve({'speaker_id':'speaker-test','display_name':'Christoph','is_self_introduction':True})
        self.assertEqual(result['display_name'],'Christoph');self.assertEqual(result['source'],'validated_luna_json')
        self.assertTrue(client.calls[0]['response_format']['json_schema']['strict'])
    async def test_cannot_reassign_other_speaker_id(self):
        result,_=await self.resolve({'speaker_id':'someone-else','display_name':'Christoph','is_self_introduction':True})
        self.assertIsNone(result)
    async def test_hallucinated_name_rejected(self):
        result,_=await self.resolve({'speaker_id':'speaker-test','display_name':'Robert','is_self_introduction':True})
        self.assertIsNone(result)
    async def test_json_mode_fallback_still_validates(self):
        result,client=await self.resolve({'speaker_id':'speaker-test','display_name':'Christoph','is_self_introduction':True},unsupported=True)
        self.assertIsNotNone(result);self.assertEqual(client.calls[1]['response_format'],{'type':'json_object'})
    async def test_third_person_not_sent_for_enrollment(self):
        result,client=await self.resolve({},text='Mein Freund heißt Robert.')
        self.assertIsNone(result);self.assertFalse(client.calls)

if __name__=='__main__':unittest.main()
