"""CPU materialization and concurrent prefetch contracts, no GPU required."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
import torch

import acting_models
from checkpoint_ram import CheckpointRAM


def test_cpu_copy_is_retained_across_live_parameter_dtype_move(tmp_path,monkeypatch):
    calls=[];original={'weight':torch.ones(2,3),'bias':torch.ones(2)}
    monkeypatch.setattr(acting_models,'download',lambda id:tmp_path)
    from safetensors import torch as safe_torch
    monkeypatch.setattr(safe_torch,'load_file',lambda *a,**k:calls.append(1) or original)
    ram=CheckpointRAM();root,state=ram.get('small_alpha')
    assert root==tmp_path and ram.get('small_alpha')[1] is state and len(calls)==1
    assert all(v.device.type=='cpu' and v.dtype==torch.bfloat16 for v in state.values())
    assert state['weight'].data_ptr()!=original['weight'].data_ptr()
    module=torch.nn.Linear(3,2);module.load_state_dict(state,assign=True)
    module.to(dtype=torch.float32)
    assert state['weight'].dtype==torch.bfloat16
    assert ram.describe()['models']['small_alpha']['status']=='ready'
    assert ram.describe()['bytes']==16


def test_concurrent_get_has_one_owner_and_metadata_does_not_block(tmp_path,monkeypatch):
    entered=threading.Event();release=threading.Event();calls=[]
    def download(id):entered.set();release.wait(3);calls.append(id);return tmp_path
    monkeypatch.setattr(acting_models,'download',download)
    from safetensors import torch as safe_torch
    monkeypatch.setattr(safe_torch,'load_file',lambda *a,**k:{'weight':torch.ones(1)})
    ram=CheckpointRAM()
    with ThreadPoolExecutor(2) as pool:
        a=pool.submit(ram.get,'small_alpha');assert entered.wait(1)
        b=pool.submit(ram.get,'small_alpha')
        assert ram.describe()['models']['small_alpha']['status']=='loading'
        release.set();assert a.result()[1] is b.result()[1]
    assert calls==['small_alpha']


def test_failed_prefetch_is_visible_and_retryable(monkeypatch):
    def fail(id):raise RuntimeError('fixture failure')
    monkeypatch.setattr(acting_models,'download',fail)
    ram=CheckpointRAM()
    with pytest.raises(RuntimeError,match='fixture failure'):ram.get('small_alpha')
    assert ram.describe()['models']['small_alpha']['status']=='error'
    assert 'small_alpha' not in ram.loading
