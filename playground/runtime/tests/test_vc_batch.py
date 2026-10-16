from types import SimpleNamespace
import threading
import numpy as np
import torch

from vc_batch import SEEDS, install_seeded_decoder
from voice_conversion import ConversionSession
from pydantic import ValidationError
import pytest


def test_per_candidate_noise_matches_serial_seed_and_context_is_restored():
    class Decoder:
        def forward(self,mu,mask,n_timesteps,temperature=1.,spks=None,cond=None,noised_mels=None,meanflow=False):
            return torch.randn_like(mu) if noised_mels is None else noised_mels
    decoder=Decoder();model=SimpleNamespace(s3gen=SimpleNamespace(flow=SimpleNamespace(decoder=decoder)))
    install_seeded_decoder(model);install_seeded_decoder(model)
    seeds=[17,31,102];mu=torch.zeros(3,80,20)
    expected=[]
    for seed in seeds:
        torch.manual_seed(seed);expected.append(torch.randn_like(mu[:1]))
    token=SEEDS.set(seeds)
    try:actual=decoder.forward(mu,None,10)
    finally:SEEDS.reset(token)
    assert torch.equal(actual,torch.cat(expected)) and SEEDS.get() is None
    torch.manual_seed(14);original=decoder.forward(mu,None,10)
    torch.manual_seed(14);assert torch.equal(original,torch.randn_like(mu))


def test_batched_streams_match_serial_seams_lengths_and_candidate_seeds():
    calls=[]
    def infer(w,conditioning,seed):return w.copy()+np.float32(seed%100)*.001
    def infer_many(waves,conditioning,seeds):
        calls.append((list(map(len,waves)),seeds))
        return [infer(w,conditioning,s) for w,s in zip(waves,seeds)]
    serial=ConversionSession(SimpleNamespace(lock=threading.RLock(),cb=None,infer=infer),777)
    batch=ConversionSession(SimpleNamespace(lock=threading.RLock(),cb=None,infer_many=infer_many),777)
    waves=[np.sin(np.arange(n,dtype=np.float32)*.01)*.1 for n in (160000,210000,33000)]
    old=[]
    for i,wave in enumerate(waves):
        state=serial.new_stream(i);parts=[]
        for start in range(0,len(wave),76800):parts.append(serial.chunk(wave[start:start+76800],state))
        parts.append(serial.chunk(np.zeros(0,np.float32),state,True));old.append(np.concatenate(parts))
    new=batch.full(waves,streaming=True)
    assert len(calls)<serial.report['conversion_calls']
    for a,b,wave in zip(old,new,waves):
        assert len(a)==len(b)==len(wave) and np.array_equal(a,b)


def test_offline_conversion_calls_one_batch_and_preserves_order():
    seen=[]
    def infer_many(waves,conditioning,seeds):
        seen.append(seeds);return [w+s*.00001 for w,s in zip(waves,seeds)]
    session=ConversionSession(SimpleNamespace(infer_many=infer_many),100)
    waves=[np.ones(48000,np.float32)*i for i in range(4)]
    result=session.full(waves)
    assert len(seen)==1 and seen[0]==[100+104729*i for i in range(4)]
    assert len(result)==4 and session.report['conversion_calls']==4


def test_precision_is_explicit_optional_and_has_fp32_comparison():
    from small_app import Turn
    assert not Turn(message='Hello').voice_conversion
    assert Turn(message='Hello').vc_precision=='bf16'
    assert Turn(message='Hello',voice_conversion=True,vc_precision='fp32').vc_precision=='fp32'
    with pytest.raises(ValidationError):Turn(message='Hello',vc_precision='int8')
