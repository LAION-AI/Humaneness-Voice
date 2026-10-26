import asyncio
import json
import struct
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from tts_bridge import MouthBridge
from app.wire import async_recv_msg


class FakeClient:
    def __init__(self,*args,**kwargs):pass
    async def __aenter__(self):return self
    async def __aexit__(self,*args):pass
    async def post(self,*args,**kwargs):await asyncio.sleep(60)
    async def get(self,*args,**kwargs):
        class Response:
            status_code=200
            def json(self):return {'busy':False}
        return Response()
    def stream(self,*args,**kwargs):
        class Waiting:
            async def __aenter__(self):await asyncio.sleep(60)
            async def __aexit__(self,*args):pass
        return Waiting()


def frame(kind,payload):
    raw=json.dumps(payload).encode();return kind+struct.pack('>I',len(raw))+raw


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_interrupted_reply_keeps_task_context_for_next_real_bridge_turn(self):
        from copy import deepcopy
        snapshots=[]
        async def pipeline(client,text,identity,history,**kwargs):
            snapshots.append(deepcopy(history))
            yield 'sentence',{'text':'Ein Roboter stand am Bahnhof. Er fragte nach seinem Zug.' if len(snapshots)==1 else 'Ein anderer vollständiger Witz.', 'sentence_index':0}
            yield 'done',{}
        with patch('tts_bridge.httpx.AsyncClient',FakeClient),patch.dict('os.environ',{'ROBO_DIRECTOR_MODE':'bolt'}),patch('bolt_pipeline.pipeline',pipeline):
            bridge=MouthBridge();server=await asyncio.start_server(bridge.accept,'127.0.0.1',0)
            reader,writer=await asyncio.open_connection('127.0.0.1',server.sockets[0].getsockname()[1])
            try:
                for number,text in enumerate(('Erzähl mir einen Witz.','Noch einen.')):
                    if number:writer.write(frame(b'H',{}))
                    writer.write(frame(b'T',{'transcript':text}));await writer.drain()
                    while True:
                        kind,raw=await asyncio.wait_for(async_recv_msg(reader),2)
                        if kind==b'D':break
                self.assertEqual([r['role'] for r in snapshots[1]],['user','assistant'])
                self.assertIn('Bahnhof',snapshots[1][1]['content'])
                self.assertTrue(snapshots[1][1]['playback_interrupted'])
                self.assertNotIn('only a prefix may have been heard',snapshots[1][1]['content'])
                self.assertEqual(snapshots[1][0]['content'],'Erzähl mir einen Witz.')
            finally:writer.close();await writer.wait_closed();server.close();await server.wait_closed();await asyncio.sleep(.01)
    async def test_agreed_literal_is_prepared_once_and_replayed_without_new_generation(self):
        import numpy as np
        calls=[]
        class Client(FakeClient):
            async def get(self,url,**kwargs):
                return SimpleNamespace(status_code=200,content=b'synthetic',json=lambda:{'busy':False},raise_for_status=lambda:None)
            def stream(self,*args,**kwargs):
                calls.append(kwargs['json'])
                class Response:
                    status_code=200
                    async def __aenter__(self):return self
                    async def __aexit__(self,*args):pass
                    def raise_for_status(self):pass
                    async def aiter_lines(self):yield 'data: '+json.dumps({'event':'result','candidates':[{'url':'/synthetic.mp3'}]})
                return Response()
        from unittest.mock import AsyncMock
        with patch('tts_bridge.httpx.AsyncClient',Client),patch('tts_bridge.decode_mp3',AsyncMock(return_value=np.ones(48000,np.float32)*.03)):
            bridge=MouthBridge();server=await asyncio.start_server(bridge.accept,'127.0.0.1',0)
            reader,writer=await asyncio.open_connection('127.0.0.1',server.sockets[0].getsockname()[1])
            try:
                writer.write(frame(b'W',{'replies':['Stopp, stopp!'],'language':'de'}));await writer.drain()
                while True:
                    kind,raw=await asyncio.wait_for(async_recv_msg(reader),2)
                    if kind==b'X' and json.loads(raw).get('stage')=='interjection_ready':break
                writer.write(frame(b'T',{'transcript':'Bananen','interaction_reply':'Stopp, stopp!','interaction_language':'de'}));await writer.drain();audio=0;metadata=None
                while True:
                    kind,raw=await asyncio.wait_for(async_recv_msg(reader),2)
                    if kind==b'D':break
                    if kind==b'A':audio+=len(raw)
                    if kind==b'X' and not json.loads(raw).get('status_only'):metadata=json.loads(raw)
                self.assertGreater(audio,0);self.assertEqual(len(calls),1);self.assertTrue(metadata['timing']['interjection_cache_hit'])
            finally:writer.close();await writer.wait_closed();server.close();await server.wait_closed();await asyncio.sleep(.01)
    async def test_compact_reply_uses_manual_tts_without_second_director(self):
        calls=[]
        class Client(FakeClient):
            async def post(self,*args,**kwargs):
                calls.append(('director',kwargs['json']))
                class Response:
                    def raise_for_status(self):pass
                    def json(self):return {'choices':[{'message':{'content':json.dumps({'general':'Natural.','script':'(interest) Test answer.','language':'en'})}}]}
                return Response()
            def stream(self,*args,**kwargs):
                calls.append(('tts',kwargs['json']))
                class Response:
                    async def __aenter__(self):return self
                    async def __aexit__(self,*args):pass
                    status_code=200
                    def raise_for_status(self):pass
                    async def aiter_lines(self):
                        yield 'data: '+json.dumps({'event':'direction','text':'Test answer.'})
                return Response()
        settings=SimpleNamespace(LUNA_MODEL='configured-test',LUNA_BASE='https://example.invalid',luna_key=lambda:'test')
        with patch('tts_bridge.httpx.AsyncClient',Client),patch('conversation_director.shared_settings',return_value=settings):
            bridge=MouthBridge();server=await asyncio.start_server(bridge.accept,'127.0.0.1',0)
            reader,writer=await asyncio.open_connection('127.0.0.1',server.sockets[0].getsockname()[1])
            try:
                writer.write(frame(b'T',{'transcript':'Test.'}));await writer.drain();replies=[]
                while True:
                    kind,raw=await asyncio.wait_for(async_recv_msg(reader),2)
                    if kind==b'D':break
                    if kind==b'X' and not json.loads(raw).get('status_only'):replies.append(json.loads(raw)['text'])
                self.assertEqual(replies,['Test answer.'])
                self.assertEqual([c[0] for c in calls],['director','tts'])
                body=calls[1][1];self.assertEqual(body['input_mode'],'manual')
                self.assertTrue(body['streaming']);self.assertEqual(body['chunk_frames'],20)
                self.assertIn("Preserve that speaker's voice identity.",body['message'])
                self.assertIn('SCRIPT:\n(interest) Test answer.',body['message'])
            finally:
                writer.close();await writer.wait_closed();server.close();await server.wait_closed();await asyncio.sleep(.01)

    async def check_cancel(self,delay):
        settings=SimpleNamespace(LUNA_MODEL='configured-test',LUNA_BASE='https://example.invalid',luna_key=lambda:'test')
        with patch('tts_bridge.httpx.AsyncClient',FakeClient), patch('conversation_director.shared_settings',return_value=settings):
            bridge=MouthBridge()
            server=await asyncio.start_server(bridge.accept,'127.0.0.1',0)
            reader,writer=await asyncio.open_connection('127.0.0.1',server.sockets[0].getsockname()[1])
            try:
                if delay is None:
                    writer.write(frame(b'T',{'transcript':'Synthetic test.'})+frame(b'B',{'played_s':0}))
                else:
                    writer.write(frame(b'T',{'transcript':'Synthetic test.'}));await writer.drain();await asyncio.sleep(delay)
                    writer.write(frame(b'B',{'played_s':0}))
                await writer.drain()
                kind,raw=await asyncio.wait_for(async_recv_msg(reader),2)
                if kind==b'X':
                    self.assertTrue(json.loads(raw)['status_only'])
                    kind,raw=await asyncio.wait_for(async_recv_msg(reader),2)
                self.assertEqual(kind,b'D');self.assertTrue(json.loads(raw)['barged'])
                with self.assertRaises(asyncio.TimeoutError):await asyncio.wait_for(async_recv_msg(reader),.05)
                self.assertEqual(bridge.active,0)
            finally:
                writer.close();await writer.wait_closed();server.close();await server.wait_closed();await asyncio.sleep(.01)

    async def test_immediate_cancel_acknowledges_once(self):await self.check_cancel(None)
    async def test_inflight_cancel_acknowledges_once(self):await self.check_cancel(.02)


if __name__=='__main__':unittest.main()
