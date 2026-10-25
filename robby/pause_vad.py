"""Small CPU neural speech detector; per-call state never leaves session RAM.

Uses the upstream Silero 16-kHz ONNX interface (512 new + 64 context samples).
The runtime downloads only a pinned, hash-checked model at service startup.
"""
import hashlib
import os
from pathlib import Path
import threading
import numpy as np

REVISION='066409f3d9db45a30d21dfcbf9147826ffdd7052'
SHA256='7ed98ddbad84ccac4cd0aeb3099049280713df825c610a8ed34543318f1b2c49'
URL=f'https://raw.githubusercontent.com/snakers4/silero-vad/{REVISION}/src/silero_vad/data/silero_vad_16k_op15.onnx'
_model=None
_lock=threading.Lock()

def model_path():
    return Path(os.environ.get('ROBO_VAD_MODEL',str(Path.home()/'.cache/robo-vad/silero_vad_16k_op15.onnx')))

def get_model():
    global _model
    with _lock:
        if _model is not None:return _model
        path=model_path()
        if not path.exists():
            import httpx
            data=httpx.get(URL,timeout=30,follow_redirects=True).raise_for_status().content
            if hashlib.sha256(data).hexdigest()!=SHA256:raise ValueError('VAD download hash mismatch')
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_suffix('.download');temporary.write_bytes(data);temporary.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=SHA256:raise ValueError('VAD model hash mismatch')
        import onnxruntime as ort
        options=ort.SessionOptions();options.inter_op_num_threads=1;options.intra_op_num_threads=1
        _model=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        return _model

class SpeechClock:
    """32-ms neural decisions, hysteresis and a 64-ms minimum speech onset.

    No RMS/energy threshold determines whether the pause timer is reset.
    Background speech can still be speech; this is not source separation.
    """
    def __init__(self,model=None):
        self.model=model if model is not None else get_model()
        self.state=np.zeros((2,1,128),np.float32);self.context=np.zeros((1,64),np.float32)
        self.pending=np.zeros(0,np.float32);self.end=0.;self.last_speech=-100.
        self.active=False;self.positive=0;self.probability=0.;self.calls=0;self.compute_ms=0.
    def feed(self,audio,end):
        import time
        started=time.perf_counter()
        self.pending=np.r_[self.pending,np.asarray(audio,np.float32)]
        # Pending samples precede the new block. Absolute capture timestamps
        # therefore remain correct even for 20-ms packets and 32-ms VAD frames.
        start=end-len(self.pending)/16000
        count=len(self.pending)//512
        for index in range(count):
            chunk=self.pending[index*512:(index+1)*512][None,:]
            inputs=np.concatenate((self.context,chunk),axis=1)
            probability,self.state=self.model.run(None,{'input':inputs,'state':self.state,'sr':np.array(16000,np.int64)})
            self.context=inputs[:,-64:];self.probability=float(probability.reshape(-1)[0]);self.calls+=1
            if self.active:
                self.active=self.probability>=.35
            else:
                self.positive=self.positive+1 if self.probability>=.5 else 0
                self.active=self.positive>=2
            if self.active:self.last_speech=start+(index+1)*.032
            else:
                if self.probability<.35:self.positive=0
        self.pending=self.pending[count*512:].copy();self.end=end
        self.compute_ms+=(time.perf_counter()-started)*1000
        return self.last_speech
