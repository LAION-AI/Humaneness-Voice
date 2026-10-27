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

if __name__=='__main__':
    root=Path(__file__).resolve().parent/'logs';root.mkdir(exist_ok=True)
    microphone_file(root/'environment-microphone.wav')
