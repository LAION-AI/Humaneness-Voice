import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from avatar_actions import EMOTIONS, validate
from conversation_director import CompactDirector,complete_visual_acknowledgment

class AvatarTests(unittest.IsolatedAsyncioTestCase):
    def test_empty_visual_reply_gets_only_a_bounded_local_ack(self):
        raw={'general':'','script':'(amusement) [0.2 seconds pause]','language':'de'}
        self.assertIn('Schau mal!',complete_visual_acknowledgment(raw,[{'kind':'nod'}])['script'])
        self.assertEqual(complete_visual_acknowledgment(raw,[]),raw)
        for bad in ({**raw,'extra':1},{**raw,'language':'xx'},{**raw,'script':'x'*6001},{**raw,'script':'A real answer.'}):
            self.assertEqual(complete_visual_acknowledgment(bad,[{'kind':'nod'}]),bad)
    def test_all_40_and_bounded_sequences(self):
        self.assertEqual(len(EMOTIONS),40)
        for emotion in EMOTIONS:self.assertEqual(validate({'steps':[{'kind':'emotion','emotion':emotion}]})[0]['emotion'],emotion)
        for raw in ({'steps':[]},{'steps':[{'kind':'shell'}]},{'steps':[{'kind':'emotion','emotion':'unknown'}]},
                    {'steps':[{'kind':'nod','duration_s':float('nan')}]},{'steps':[{'kind':'nod','duration_s':True}]},
                    {'steps':[{'kind':'nod','sentence':1},{'kind':'shake','sentence':0}]},
                    {'steps':[{'kind':'nod','duration_s':6}]*5},{'steps':[{'kind':'spin','direction':'whatever'}]}):
            with self.assertRaises(ValueError):validate(raw)
        self.assertEqual(validate({'steps':[{'kind':'spin','duration_s':1}]})[0]['duration_s'],3.)

    async def test_actual_tool_loop_keeps_visuals_out_of_spoken_script(self):
        seen=[]
        class Client:
            async def post(self,url,**kw):
                seen.append(json.loads(json.dumps(kw['json'])))
                msg={'tool_calls':[{'id':'avatar-1','type':'function','function':{'name':'perform_avatar','arguments':json.dumps({'steps':[{'kind':'emotion','emotion':'anger'},{'kind':'grimace','variant':'crossed'}]})}}]} if len(seen)==1 else {'content':json.dumps({'general':'Gentle and playful.','script':'(amusement) Schau mal!','language':'de'})}
                return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'choices':[{'message':msg}]})
        settings=SimpleNamespace(LUNA_MODEL='test',LUNA_BASE='https://example.invalid',luna_key=lambda:'test')
        with patch('conversation_director.shared_settings',return_value=settings):
            result=await CompactDirector().generate(Client(),'Erst wütend, dann schielen.',{},[])
        self.assertEqual([s['kind'] for s in result['avatar_actions']],['emotion','grimace'])
        self.assertEqual(result['text'],'Schau mal!');self.assertNotIn('avatar',result['manual_prompt'])
        self.assertEqual(seen[-1]['tool_choice'],'none')
        self.assertTrue(any(t['function']['name']=='perform_avatar' for t in seen[0]['tools']))

if __name__=='__main__':unittest.main()
