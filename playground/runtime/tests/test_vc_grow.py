from types import SimpleNamespace
import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from vc_grow import TARGET_STATIC,StyledSpeakerProjection,LoRALinear,mix_scores,verified_download
from ears_reward import EarsModels


def names():
    import json
    from pathlib import Path
    root=next((Path(__file__).resolve().parents[1]/'model-cache/chatterbox/hub').glob(
        'models--laion--chatterbox-s3gen-vc-grow-ce-speaker/snapshots/*/score_order_and_preparation.json'),None)
    if root:return json.loads(root.read_text())['score_order']
    from adaptive_reward import DIMENSIONS
    emo=sorted(k[4:] for k in DIMENSIONS if k.startswith('eiv:') and k[4:] in ('Affection','Anger'))
    # Test namespace need not reflect predictor selection; only 99 native axes.
    return ['emotion'+str(i) for i in range(40)]+sorted(TARGET_STATIC)+[
        'dynamic'+str(i) for i in range(43)]+['genuineness_0_6','blend_0_10']


def test_only_static_axes_are_replaced_with_target_traits():
    order=names();assert len(order)==99
    source=np.arange(3*99,dtype=np.float32).reshape(3,99)
    target=np.full(99,-17,dtype=np.float32)
    mixed=mix_scores(source,target,order)
    for i,key in enumerate(order):
        assert np.array_equal(mixed[:,i],np.full(3,-17) if key in TARGET_STATIC else source[:,i])
    assert not np.shares_memory(source,mixed)
    with pytest.raises(ValueError):mix_scores(source[:,:98],target,order)
    source[0,0]=np.nan
    with pytest.raises(ValueError):mix_scores(source,target,order)


def test_projector_requires_exact_request_local_conditioning():
    projection=StyledSpeakerProjection(nn.Linear(192,80))
    with pytest.raises(RuntimeError):projection(torch.zeros(2,192))
    projection.condition=torch.ones(2,99);projection.present=torch.ones(2,99)
    assert projection(torch.zeros(2,192)).shape==(2,80)
    with pytest.raises(ValueError):projection(torch.zeros(1,192))


def test_lora_shapes_rank_and_zero_scale_parity():
    base=nn.Linear(10,20);layer=LoRALinear(base);layer.scale=0
    x=torch.randn(3,10)
    assert torch.equal(layer(x),base(x))
    assert layer.lora_down.weight.shape==(128,10)
    assert layer.lora_up.weight.shape==(20,128)


def test_verified_download_rejects_corrupt_checkpoint(tmp_path,monkeypatch):
    import huggingface_hub
    p=tmp_path/'trusted';p.write_bytes(b'abc')
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',lambda *a,**k:str(p))
    assert verified_download('repo','file','revision',hashlib.sha256(b'abc').hexdigest())==p
    with pytest.raises(RuntimeError,match='integrity'):verified_download('repo','file','revision','wrong')


def test_99_score_native_denormalization_and_training_scales():
    order=names()
    mapped=['emo_'+n if i<40 else 'vn_'+n+'_reg' if i<97 else n for i,n in enumerate(order)]
    mean=[6.]*98+[10.]
    norm={'score_names':mapped,'score_mean':mean,'score_std':[2.]*99}
    def extractor(batch,**kwargs):
        return {'input_features':np.zeros((len(batch),80,10),np.float32),
                'attention_mask':np.ones((len(batch),10),np.int64)}
    def model(mel,mask,**kwargs):return {'scores':torch.zeros(len(mel),99)}
    ears=EarsModels('cpu');ears.models['medium']=(model,extractor,norm)
    ears.load=lambda size:None
    got=ears.predict_conditioning([np.zeros(16000,np.float32),np.zeros(496000,np.float32)],order)
    assert got.shape==(2,99) and np.allclose(got,1.)
    with pytest.raises(ValueError):ears.predict_conditioning([np.zeros(100)],order[:98])
