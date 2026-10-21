"""Synthetic-only ASR -> full-utterance enrollment -> reply integration check.

Pass a locally generated public synthetic voice saying Hey Robo in a sentence.
It uses the actual running EARS/TTS/Luna services, not a physical microphone.
No transcript, audio or embeddings are written by this test.
"""
import argparse
import asyncio
import json
from pathlib import Path
import struct
import subprocess
import time
import aiohttp
import numpy as np


async def run(args):
    result=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',args.audio,'-ar','16000','-ac','1','-f','f32le','pipe:1'],capture_output=True,check=True)
    audio=np.frombuffer(result.stdout,'<f4')
    audio=np.r_[np.zeros(10240,np.float32),audio,np.zeros(64000,np.float32)]
    decisions=[];events={};enabled=asyncio.Event();enrolled=asyncio.Event();chunks=0;backchannels=0;counts={}
    async with aiohttp.ClientSession() as client:
        async with client.ws_connect(args.url.rstrip('/')+'/ws') as ws:
            async def receive():
                nonlocal chunks,backchannels
                async for message in ws:
                    if message.type==aiohttp.WSMsgType.BINARY:
                        rid=struct.unpack_from('<i',message.data)[0]
                        if rid:chunks+=1
                        else:backchannels+=1;continue
                        counts[rid]=counts.get(rid,0)+(len(message.data)-4)//2
                        # Virtual page acknowledgment, NOT physical DAC measurement.
                        await ws.send_str(f'played:{rid}:{counts[rid]}')
                    elif message.type==aiohttp.WSMsgType.TEXT:
                        item=json.loads(message.data);kind=item.get('type');events[kind]=events.get(kind,0)+1
                        if kind=='identity':
                            if item.get('enabled'):enabled.set()
                            if 'decision' in item:
                                decisions.append(item['decision'])
                                if item['decision'].get('new_user'):enrolled.set()
            task=asyncio.create_task(receive())
            try:
                await ws.send_str('identity:on');await asyncio.wait_for(enabled.wait(),10)
                for i in range(0,len(audio),320):
                    x=audio[i:i+320]
                    pcm=np.clip(np.round(x*32767),-32768,32767).astype('<i2')
                    await ws.send_bytes(struct.pack('<ii',0,-1)+pcm.tobytes())
                    await asyncio.sleep(.02)
                await asyncio.wait_for(enrolled.wait(),12)
                for _ in range(100):
                    if chunks:break
                    await asyncio.sleep(.1)
                await ws.send_str('interrupt')
                report={'teacher_filter_enabled':enabled.is_set(),'enrolled_from_asr_full_sentence':enrolled.is_set(),
                    'identity_decisions':decisions,'events':events,'reply_audio_chunks':chunks,
                    'backchannel_chunks':backchannels,
                    'fixture':'public synthetic speech, not a private microphone recording',
                    'playback':'virtual immediate acknowledgments, not physical sync evidence'}
                print(json.dumps(report,indent=2));assert enrolled.is_set() and chunks>0
                if args.report:Path(args.report).write_text(json.dumps(report,indent=2))
            finally:
                await ws.close();task.cancel();await asyncio.gather(task,return_exceptions=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audio',required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8793');parser.add_argument('--report')
    asyncio.run(run(parser.parse_args()))
