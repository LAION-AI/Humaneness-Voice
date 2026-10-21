"""Actual loaded TYGER checks, never another model load; synthetic audio only."""
import asyncio
from pathlib import Path
import subprocess
import time

import numpy as np


def read16(path,seconds=6):
    p=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(path),'-t',str(seconds),
        '-ar','16000','-ac','1','-f','f32le','pipe:1'],capture_output=True,check=True)
    x=np.frombuffer(p.stdout,'<f4').copy()
    if not len(x):raise ValueError('Empty synthetic fixture')
    return x


async def model_checks(ears):
    root=Path(__file__).resolve().parents[1]
    # A sustained reply, not a five-second clip followed by three seconds of
    # "speaking" silence (which tests a different turn transition).
    agent=read16(root/'demo_de.wav',8)
    user=read16(root/'humaneness-small/full-ft-speaker-study/audio/diagnostic-idris-natural_duration-0.mp3',4)
    n=16000*8;agent=np.pad(agent[:n],(0,max(0,n-len(agent))))
    rows=[]
    for name in ['silence','normal_conversation','bot_echo','user_interrupts']:
      for echo_db in ([-26,-40,-60,-90] if name in ('bot_echo','user_interrupts') else [None]):
        for pairing in (['arrival_fixed_150ms','timestamps'] if name in ('bot_echo','user_interrupts') else ['timestamps']):
            await asyncio.sleep(.1)
            session=ears.open_session(ears.defaults)
            if session is None:raise RuntimeError('Close the browser microphone session before synchronization checks')
            mic=np.zeros(n,np.float32);reference=np.zeros(n,np.float32);mode='listen'
            if name=='normal_conversation':mic[:len(user)]=user
            if name in ('bot_echo','user_interrupts'):
                mode='speaking';mic=agent*10**(echo_db/20);reference=agent.copy()
                if name=='user_interrupts':
                    at=24000;length=min(len(user),n-at);mic[at:at+length]+=user[:length]
                if pairing=='arrival_fixed_150ms':
                    # Recreate a 550 ms actual arrival/buffer path minus a 150 ms fixed guess.
                    ahead=6400;reference=np.r_[reference[ahead:],np.zeros(ahead,np.float32)]
            events=[];tick=time.perf_counter();peaks=[]
            try:
                for i in range(0,n,5120):
                    a=mic[i:i+5120];b=reference[i:i+5120]
                    if len(a)<5120:a=np.pad(a,(0,5120-len(a)));b=np.pad(b,(0,5120-len(b)))
                    out=await ears.feed(session,a,b,mode)
                    peaks.extend(e.get('yield_prob',0.) for e in out if e['kind']=='FRAME')
                    events.extend((e['kind'],(i+5120)/16000) for e in out if e['kind']!='FRAME')
            finally:ears.close(session)
            rows.append({'scenario':name,'pairing':pairing,'echo_db':echo_db,'seconds':time.perf_counter()-tick,
                'endpoint_count':sum(k=='ENDPOINT' for k,t in events),'barge_count':sum(k=='BARGE' for k,t in events),
                'barge_times_s':[t for k,t in events if k=='BARGE'],
                'max_yield':max(peaks,default=0.),
                'contains_transcripts':False})
    return {'conditions':rows,'note':'Running Nemotron/TYGER; synthetic fixtures, reconstructed arrival-order control, not a measured before-state on the lost machine. Hardware microphone accuracy requires physical testing.'}
