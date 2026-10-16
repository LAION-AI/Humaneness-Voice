"""Real PCM/FFmpeg ingestion and CPU codec spies; no GPU inference claims."""
import io
import math
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

import settings
from engine import SmallEngine


@pytest.mark.parametrize('cap,duration,expected',[(15.,7.5,7.5),(15.,15.,15.),(15.,20.,15.),(20.,20.,20.)])
def test_full_reference_waveform_reaches_codec_not_just_first_37_frames(tmp_path,monkeypatch,cap,duration,expected):
    monkeypatch.setattr(settings,'REFERENCE_SECONDS',cap)
    sr=16000
    wave=np.full(int(duration*sr),.02,np.float32)
    wave[int(min(expected-1.,6.)*sr):int(expected*sr)]=.2
    path=tmp_path/'reference.wav';sf.write(path,wave,sr,subtype='FLOAT')
    calls=[]
    def encode(wav,rate,n_vq):
        calls.append(wav.clone())
        assert rate==sr and n_vq==12
        return [torch.ones(math.ceil(wav.shape[-1]/sr*12.5),12,dtype=torch.long)]
    engine=SmallEngine.__new__(SmallEngine)
    engine.refs={};engine.reference_info={}
    engine.proc=SimpleNamespace(encode_audios_from_wav=encode)
    codes=engine.reference(path)
    assert len(calls[0][0])==int(expected*sr)
    assert torch.equal(calls[0][0],torch.from_numpy(wave[:int(expected*sr)]))
    assert len(codes)==math.ceil(expected*12.5)>37
    assert engine.reference_info[str(path)]['used_seconds']==expected
    assert engine.reference_info[str(path)]['truncated']==(duration>cap)
    assert engine.reference(path) is codes and len(calls)==1
    engine.forget_reference(path)
    assert not engine.refs and not engine.reference_info


@pytest.mark.parametrize('duration,expected',[(7.5,7.5),(15.,15.),(20.,15.)])
def test_real_upload_normalization_preserves_long_speech_context(tmp_path,monkeypatch,duration,expected):
    import small_app
    from storage import AudioStore
    monkeypatch.setattr(small_app,'store',AudioStore(tmp_path))
    monkeypatch.setattr(settings,'REFERENCE_SECONDS',15.)
    sr=16000
    wave=np.full(int(sr*duration),.15,np.float32)
    stream=io.BytesIO();sf.write(stream,wave,sr,format='WAV',subtype='PCM_16')
    path=tmp_path/'normalized.wav'
    seconds=small_app.normalize(stream.getvalue(),path)
    assert seconds==pytest.approx(expected,abs=.02)
    assert seconds>2.96
    assert sf.info(path).samplerate==48000 and sf.info(path).channels==1
    data,_=sf.read(path)
    assert abs(data[-48000:].mean())>.1


def test_api_reports_new_limit_and_study_defaults():
    from fastapi.testclient import TestClient
    from small_app import app,Turn
    state=TestClient(app).get('/api/state').json()
    assert state['reference_seconds_max']==15
    assert state['adapters']['default'] is None
    assert Turn(message='Hello').adapters.id == state['adapters']['default']
    assert state['generation_default']=='single' and state['streaming_default']
    assert state['n_default']==1 and Turn(message='Hello').n == 1 and Turn(message='Hello').streaming
    assert state['model_default']=='small_rl'
    assert state['speaker_default']=={'method':'ecapa','weight':.6}
    assert state['search_chunk_frames_default']==10
