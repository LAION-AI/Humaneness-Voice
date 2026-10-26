import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from conversation_director import CompactDirector,validate,REFERENCE,ensure_addressee,provider_post


class DirectorTests(unittest.IsolatedAsyncioTestCase):
    def test_internal_playback_annotation_is_removed_from_spoken_script(self):
        script='[My preceding reply was interrupted. Only a prefix may have been heard.]\n(amusement) Robby-Tornado, los geht\'s!'
        result=validate({'general':'Playful.','script':script,'language':'de'})
        self.assertEqual(result['text'],"Robby-Tornado, los geht's!")
        self.assertNotIn('interrupted',result['manual_prompt'])
    async def test_only_transient_provider_400_has_one_retry(self):
        from unittest.mock import AsyncMock
        transient=SimpleNamespace(status_code=400,json=lambda:{'error':{'type':'client_side_error','code':'client_side_error'}})
        success=SimpleNamespace(status_code=200)
        client=SimpleNamespace(post=AsyncMock(side_effect=[transient,success]))
        response,retries=await provider_post(client,'https://example.invalid')
        self.assertIs(response,success);self.assertEqual(retries,1);self.assertEqual(client.post.await_count,2)
        client.post=AsyncMock(return_value=transient)
        self.assertEqual((await provider_post(client,'https://example.invalid'))[1],1);self.assertEqual(client.post.await_count,2)
        for status in (400,401,403,429,500):
            response=SimpleNamespace(status_code=status,json=lambda:{'error':{'code':'invalid_schema','type':'invalid_request_error'}})
            client.post=AsyncMock(return_value=response)
            self.assertEqual((await provider_post(client,'https://example.invalid'))[1],0);self.assertEqual(client.post.await_count,1)
    def test_local_addressee_preserves_acting_exact_words_and_numeric_formats(self):
        identity={'enrolled':True,'named':True,'display_name':'Selma'}
        data={'general':'Natural.','script':'(amusement) Ein Roboter fand einen Schlüssel.','language':'de'}
        result=validate(ensure_addressee(data,'Erzähl eine kleine Geschichte.',identity))
        self.assertEqual(result['text'],'Ein Roboter fand einen Schlüssel.')
        self.assertTrue(result['script'].startswith('(amusement) Ein Roboter'))
        for request in ('Sage genau: Ein Roboter fand einen Schlüssel.','Sag nur die Zahl.','Repeat exactly those words.','Antworte ohne Anrede.'):
            self.assertEqual(ensure_addressee(data,request,identity),data)
    def test_validation_preserves_acting_and_identity(self):
        result=validate({'general':'Natural and interested.','script':'(interest) Hallo [0.25 seconds pause] zusammen.','language':'de'})
        self.assertEqual(result['text'],'Hallo zusammen.')
        self.assertIn(REFERENCE,result['manual_prompt'])
        self.assertIn('(interest)',result['manual_prompt'])
    def test_complete_long_answer_is_allowed_instead_of_old_120_word_limit(self):
        script='\n'.join('(interest) '+' '.join(['Wort']*35)+'.' for _ in range(6))
        self.assertEqual(len(validate({'general':'Warm and natural.','script':script,'language':'de'})['text'].split()),210)

    def test_rejects_invalid_output(self):
        good={'general':'Natural.','script':'Hello.','language':'en'}
        for bad in ({**good,'extra':1},{**good,'script':'(interest)'},{**good,'language':'xx'},
                    {**good,'script':'word '*601},{**good,'general':''}):
            with self.assertRaises(ValueError):validate(bad)

    async def test_one_compact_call_with_history(self):
        captured=[]
        class Client:
            async def post(self,url,**kwargs):
                captured.append((url,kwargs))
                class Response:
                    def raise_for_status(self):pass
                    def json(self):return {'choices':[{'message':{'content':json.dumps({'general':'Natural.','script':'(interest) Achtundsechzig.','language':'de'})}}],'usage':{'completion_tokens':30}}
                return Response()
        settings=SimpleNamespace(LUNA_MODEL='configured-test',LUNA_BASE='https://example.invalid',luna_key=lambda:'test')
        with patch('conversation_director.shared_settings',return_value=settings):
            result=await CompactDirector().generate(Client(),'17 mal 4?',{'display_name':'Christoph'},[{'role':'assistant','content':'Previous.','playback_interrupted':True}])
        self.assertEqual(len(captured),1)
        body=captured[0][1]['json']
        self.assertEqual(body['model'],'configured-test')
        self.assertEqual(body['max_completion_tokens'],2048)
        self.assertEqual(body['messages'][1]['content'],'Previous.')
        self.assertEqual(set(body['messages'][1]),{'role','content'})
        self.assertIn('INTERNAL PLAYBACK STATE',body['messages'][0]['content'])
        self.assertEqual(result['completion_tokens'],30)
        self.assertEqual(result['text'],'Achtundsechzig.')
