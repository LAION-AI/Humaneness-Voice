"""Local-only, single-copy scorer for concurrent offline Mira workers.

Not linked to the public demo. Only accepts WAVs inside this study's audio
directory. Serial scoring avoids duplicate scorer residency on the two 3090s.
"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import soundfile as sf
import torch

import settings
from mira_activation_study import ROOT, Quality

STATE={'ready':False,'error':None,'quality':None}
LOCK=threading.Lock()


def initialize():
    try:
        settings.SCORE_DEVICE='cuda:1'
        torch.set_num_threads(4);torch.cuda.set_device('cuda:1')
        STATE['quality']=Quality('cuda:1')
        STATE['ready']=True
    except Exception as exc:
        STATE['error']=str(exc)
        raise


@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(initialize)
    yield


app=FastAPI(lifespan=lifespan)


class Input(BaseModel):
    rows:list[dict]=Field(min_length=1,max_length=4)
    paths:list[str]=Field(min_length=1,max_length=4)


@app.get('/state')
def state():
    return {'ready':STATE['ready'],'error':STATE['error'],'device':'cuda:1','shared':True}


@app.post('/score')
def score(body:Input):
    if len(body.rows)!=len(body.paths):raise HTTPException(400,'Mismatched rows/paths')
    waves=[]
    for name in body.paths:
        path=Path(name).resolve()
        if not path.is_relative_to((ROOT/'audio').resolve()) or path.suffix!='.wav' or not path.is_file():
            raise HTTPException(400,'Only study WAVs are allowed')
        wave,rate=sf.read(path,dtype='float32')
        if rate!=48000 or wave.ndim!=1 or not 24000<=len(wave)<=256*3840+48000:
            raise HTTPException(400,'Unsupported study waveform')
        waves.append(wave)
    with LOCK:
        return {'scores':STATE['quality'].score(waves,body.rows)}
