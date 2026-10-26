"""Bounded, optional acoustic context; never an identity or turn-taking gate."""
import asyncio
import json
import os
import time

import numpy as np

MODEL_REVISION='818506970d93c809a3295a0a02c40dee8ff2bfc3'
CAPTION_REVISION='7ebd4a9b7da96df989af8fc6ad0ce34f4e8aa3ed'
INSTRUCTION='''Voice-delivery observations below are uncertain machine estimates of audible
delivery, not facts about the person's feelings, intentions, honesty, health or identity.
Background speakers, noise, acting and short clips can mislead them. Interpret them in
the context of the user's actual words and conversation; the user's own account takes
precedence. Respond warmly and tactfully where useful. Never assert "you are angry/sad"
from a score, diagnose, judge sincerity, quote the annotation or recite scores. You may
gently ask about a feeling when relevant; don't add a disclaimer to every reply. These
hints do not authorize interruptions, speaker enrollment or ignoring a request.'''

def completed_hint(task,enabled=True):
    """Only an already finished parallel job may contribute; no ASR-following wait."""
    if not enabled:
        if task and not task.done():task.cancel()
        return {'status':'disabled'}
    if task is None:return {'status':'not_requested'}
    if not task.done():
        task.cancel();return {'status':'not_ready','added_wait_ms':0.}
    if task.cancelled():return {'status':'cancelled'}
    try:return {**task.result(),'added_wait_ms':0.}
    except Exception:return {'status':'unavailable','added_wait_ms':0.}

def prompt_hint(identity):
    value=(identity or {}).get('voice_delivery') or {}
    if value.get('status')!='available' or not isinstance(value.get('caption'),str):return ''
    data={k:value[k] for k in ('caption','window_s','scope','uncertainty') if k in value}
    return '\n'+INSTRUCTION+'\nUncertain current-turn acoustic observations (data): '+json.dumps(data,ensure_ascii=False)

async def describe(audio,enabled=True,timeout=None):
    """No persistent recording, retry queue, GPU job or wait for a late result."""
    if not enabled:return {'status':'disabled'}
    wave=np.asarray(audio,dtype=np.float32).reshape(-1)
    if len(wave)<8000:return {'status':'too_short'}
    if not np.isfinite(wave).all():return {'status':'invalid_audio'}
    # Use the trailing eight seconds of THIS accepted turn, not another speaker
    # or an old utterance. Scope is explicit for longer user requests.
    full_s=len(wave)/16000;wave=wave[-128000:]
    if float(np.sqrt(np.mean(wave**2)))<.001:return {'status':'quiet'}
    import httpx
    begin=time.perf_counter()
    budget=timeout if timeout is not None else float(os.environ.get('ROBO_CAPTION_TIMEOUT_S','.35'))
    try:
        async with asyncio.timeout(budget):
            async with httpx.AsyncClient(timeout=budget,trust_env=False) as client:
                response=await client.post(os.environ.get('ROBO_CAPTION_URL','http://127.0.0.1:8802')+'/caption',
                    content=wave.astype('<f4').tobytes(),headers={'Content-Type':'application/octet-stream'})
                response.raise_for_status();value=response.json()
                if value.get('status')!='available' or not isinstance(value.get('caption'),str):raise ValueError('Invalid caption')
                value['scope']='last eight seconds of this turn' if full_s>8 else 'this turn'
                value['caption']=value['caption'][:450]
    except (TimeoutError,httpx.TimeoutException):value={'status':'timeout'}
    except (httpx.HTTPError,ValueError):value={'status':'unavailable'}
    value['request_ms']=round((time.perf_counter()-begin)*1000,2)
    return value
