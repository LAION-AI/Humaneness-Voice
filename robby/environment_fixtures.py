"""Authored procedural audio only (CC0); never samples from real microphones."""
from pathlib import Path
import numpy as np
import soundfile as sf

def fixtures():
    sr=16000;rng=np.random.default_rng(27);n=sr*2;t=np.arange(n)/sr
    impact=np.zeros(n,np.float32)
    for at in (.25,.53,.71):
        i=round(at*sr);length=2000;u=np.arange(length)/sr
        impact[i:i+length]+=(.22*rng.normal(size=length)*np.exp(-u*75)+.15*np.sin(2*np.pi*150*u)*np.exp(-u*22)).astype(np.float32)
    beeps=(.12*np.sin(2*np.pi*900*t)*(np.sin(2*np.pi*2*t)>0)).astype(np.float32)
    music=sum(.03*np.sin(2*np.pi*f*t) for f in (220,277.18,329.63)).astype(np.float32)
    return {'impacts':impact,'beeps':beeps,'quiet':rng.normal(0,.001,n).astype(np.float32),'tone_chord':music}

def microphone_file(path,at=18.,duration=50.):
    wave=np.random.default_rng(8).normal(0,.0003,round(duration*16000)).astype(np.float32)
    sound=fixtures()['impacts'];start=round(at*16000);wave[start:start+len(sound)]+=sound
    sf.write(path,wave,16000,subtype='PCM_16');return Path(path)

def music_fixture(seconds=8.,sr=16000):
    """CC0 authored melody, bass and drums, not a copyrighted song recording."""
    n=round(seconds*sr);out=np.zeros(n,np.float32);rng=np.random.default_rng(927)
    notes=(60,64,67,72,69,67,64,62,60,65,69,72,67,64,62,60)
    for step,at in enumerate(np.arange(0,seconds,.25)):
        i=round(at*sr);count=min(round(.42*sr),n-i);t=np.arange(count)/sr
        freq=440*2**((notes[step%len(notes)]-69)/12)
        env=(1-np.exp(-t*180))*np.exp(-t*6)
        tone=sum(np.sin(2*np.pi*freq*h*t)/h**1.8 for h in (1,2,3,4))
        out[i:i+count]+=(tone*env*.14).astype(np.float32)
        if step%2==0:
            bass=65.406 if step%32<16 else 87.307
            out[i:i+count]+=(.09*np.sin(2*np.pi*bass*t)*np.exp(-t*5)).astype(np.float32)
            out[i:i+count]+=(.16*np.sin(2*np.pi*(48*t+.9*(1-np.exp(-t*24))))*np.exp(-t*22)).astype(np.float32)
        if step%4==2:out[i:i+count]+=(rng.normal(0,.07,count)*np.exp(-t*35)).astype(np.float32)
        out[i:i+count]+=(rng.normal(0,.025,count)*np.exp(-t*110)).astype(np.float32)
    return out

if __name__=='__main__':
    root=Path(__file__).resolve().parent/'logs';root.mkdir(exist_ok=True)
    microphone_file(root/'environment-microphone.wav')
