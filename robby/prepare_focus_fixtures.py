"""Cache native speaker features for existing public synthetic fixture audio.

Requires one free research slot on the running loopback ASR service. Never
reads a microphone or uploads audio to a third party. Cache stays under logs/.
"""
import asyncio,io
from pathlib import Path
import httpx,numpy as np
from speaker_live.live_test import audio
ROOT=Path(__file__).resolve().parent
NAMES=('christoph-intro','christoph-followup','christoph-stop','christoph-wait',
       'christoph-halt','christoph-pause-clear','robert-stop','background-stop','selma-followup')
async def main():
    payload={}
    async with httpx.AsyncClient(timeout=180) as client:
        for name in NAMES:
            response=await client.post('http://127.0.0.1:8793/internal/speaker-predictions',
                content=audio(name).astype('<f4').tobytes())
            response.raise_for_status()
            with np.load(io.BytesIO(response.content)) as archive:
                for key in ('embedding','long_embedding','probabilities'):payload[name+'__'+key]=archive[key]
    path=ROOT/'logs/focus-student-fixtures.npz';path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,**payload)
    print('Cached native-hop features for',len(NAMES),'public synthetic fixtures; no private recordings.')
if __name__=='__main__':asyncio.run(main())
