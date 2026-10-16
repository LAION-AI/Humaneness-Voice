import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest
from pydantic import ValidationError
from safetensors.torch import save
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rank1_adapters as adapters


@pytest.fixture
def bank(tmp_path, monkeypatch):
    base = tmp_path / 'base.pt'
    base.write_bytes(b'fixture base checkpoint')
    digest = adapters.sha256(base)
    monkeypatch.setattr(adapters, 'BASE_SHA256', digest)
    meta = {}
    archive_path = tmp_path / 'adapters.tar'
    with tarfile.open(archive_path, 'w:') as archive:
        for id_, value in [('profile/test', .125), ('emotion/Test', .25), ('vn/AROU_high', .5)]:
            tensors = {'proj.a.weight': torch.ones(1, 3), 'proj.b.weight': torch.full((2, 1), value)}
            data = save(tensors)
            member = id_ + '.safetensors'
            item = tarfile.TarInfo(member)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
            meta[id_] = {'archive': 'adapters.tar', 'member': member,
                'member_sha256': hashlib.sha256(data).hexdigest()}
    index = {'adapters': meta}
    manifest = {'archives': [{'file': 'adapters.tar', 'bytes': archive_path.stat().st_size,
        'sha256': adapters.sha256(archive_path)}]}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(adapters, 'repository', lambda: tmp_path)
    monkeypatch.setattr(adapters, 'index_data', lambda: index)
    monkeypatch.setattr(adapters, 'hf_hub_download', lambda *a, **kw: str(archive_path))
    monkeypatch.setattr(adapters.settings, 'STAGE', 'S3')
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.proj.weight.fill_(1.)
    result = adapters.Rank1Bank(model, base, module_names=['proj'])
    return result


def selection(**slots):
    return adapters.AdapterSelection.model_validate(slots)


def test_reversible_fusion_preserves_parameter_storage_and_does_not_accumulate(bank):
    weight = bank.model.proj.weight
    original = weight.clone()
    pointer = weight.data_ptr()
    pick = selection(voice={'id': 'profile/test'}, emotion={'id': 'emotion/Test', 'weight': .5})
    report = bank.apply(pick)
    assert report['experimental_mix'] and len(report['active']) == 2
    assert torch.equal(weight, original + .25)
    bank.apply(pick)
    assert torch.equal(weight, original + .25)
    bank.apply(selection(style={'id': 'vn/AROU_high', 'weight': 2}))
    assert torch.equal(weight, original + 1)
    bank.apply(selection())
    assert torch.equal(weight, original)
    assert weight.data_ptr() == pointer


def test_zero_weight_is_disabled_and_base_reset_is_exact(bank):
    original = bank.model.proj.weight.clone()
    bank.apply(selection(voice={'id': 'profile/test'}))
    report = bank.apply(selection(voice={'id': 'profile/test', 'weight': 0}))
    assert report['active'] == []
    assert torch.equal(bank.model.proj.weight, original)


@pytest.mark.parametrize('body', [
    {'voice': {'id': 'emotion/Test'}}, {'emotion': {'id': 'profile/test'}},
    {'style': {'id': 'profile/test'}}, {'voice': {'id': '../private'}},
    {'voice': {'id': 'profile/test', 'weight': -1}},
    {'voice': {'id': 'profile/test', 'weight': 2.1}},
    {'voice': {'id': 'profile/test', 'weight': float('nan')}},
    {'voice': {'id': 'profile/test', 'weight': float('inf')}},
    {'unknown': {'id': 'profile/test'}},
])
def test_invalid_selections_rejected(body):
    with pytest.raises(ValidationError):
        selection(**body)


def test_unknown_member_is_rejected_without_modifying_current_weights(bank):
    bank.apply(selection(voice={'id': 'profile/test'}))
    old = bank.model.proj.weight.clone()
    with pytest.raises(ValueError, match='Unknown LoRA'):
        bank.apply(selection(emotion={'id': 'emotion/Unknown'}))
    assert torch.equal(old, bank.model.proj.weight)


def test_member_hash_checked_before_any_live_parameter_update(bank):
    original = bank.model.proj.weight.clone()
    bank.index['adapters']['emotion/Test']['member_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='member hash'):
        bank.apply(selection(voice={'id': 'profile/test'}, emotion={'id': 'emotion/Test'}))
    assert torch.equal(original, bank.model.proj.weight)


def test_archive_integrity_checked(bank):
    bank.manifest['archives'][0]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='archive hash'):
        bank.apply(selection(voice={'id': 'profile/test'}))


def test_all_real_catalog_entries_and_family_counts(monkeypatch):
    # Legacy rank-1 adapters deliberately remain restricted to their S3 base.
    monkeypatch.setattr(adapters.settings,'STAGE','S3')
    data = adapters.catalog()
    assert data['counts'] == {'voice': 500, 'emotion': 40, 'style': 17}
    assert data['revision'] == adapters.settings.LORA_REVISION
    all_ids = set()
    for slot, rows in data['groups'].items():
        for row in rows:
            adapters.validate_selection(selection(**{slot: {'id': row['id']}}))
            all_ids.add(row['id'])
    assert len(all_ids) == 557
    styles = {row['id']: row['label'] for row in data['groups']['style']}
    assert 'Arousal shift' in styles['vn/ARSH_high']
    assert 'Valence shift' in styles['vn/VALS_high']
    assert 'volatility' in styles['vn/VOLT_high']


def test_catalog_api_and_unknown_selection_does_not_reserve_server(monkeypatch):
    from fastapi.testclient import TestClient
    import small_app
    client = TestClient(small_app.app)
    response = client.get('/api/adapters/catalog')
    assert response.status_code == 200
    assert response.json()['counts']['dpo'] == 6
    assert client.get('/lora-ui.js').status_code == 200
    monkeypatch.setitem(small_app.STATE, 'phase', 'ready')
    monkeypatch.setitem(small_app.STATE, 'busy', False)
    response = client.post('/api/turn', json={'message': 'Test', 'adapters': {'voice': {'id': 'profile/NOT_REAL'}}})
    assert response.status_code == 422 and not small_app.STATE['busy']


def test_api_applies_selection_to_each_turn_and_resets_on_next_turn(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import numpy as np
    from fastapi.testclient import TestClient
    import small_app
    from storage import AudioStore
    calls = []
    class Engine:
        last_performance = None
        def configure_adapters(self, value):
            calls.append(value.model_dump())
            return {'selection': value.model_dump(), 'active': [x.model_dump() for x in value.active()], 'switch_ms': 0}
        def generate(self, *args, **kwargs):
            return [np.zeros(48000, dtype=np.float32)]
    class Scorers:
        def score(self, *args):
            return [{'rank': 0, 'wer': 0., 'sec': 1., 'hyp': 'Ready.', 'reward': .5,
                     'clap': .2, 'genuineness': 3., 'blend': 5.}]
    monkeypatch.setattr(small_app, 'store', AudioStore(tmp_path))
    monkeypatch.setattr(small_app, 'STATE', {'phase': 'ready', 'busy': False, 'engine': Engine(),
        'scorers': Scorers(), 'finisher': SimpleNamespace(finish=lambda waves, *a: (waves, [{}]))})
    client = TestClient(small_app.app)
    from dpo_adapters import AdapterSelection
    owner='b'*32
    client.cookies.set('small_session',owner)
    reference=small_app.store.new(owner,'Fixture reference','reference','A')
    import soundfile as sf
    sf.write(reference.path,np.full(48000,.1,dtype=np.float32),48000)
    chosen = {'id': 'rebalanced_rank064', 'weight': .5}
    selections=[chosen,{**chosen,'ref15':True,'ref15_weight':.25},{'ref15':True},{'id':None},{}]
    for picks in selections:
        response = client.post('/api/turn', json={'message': 'CAPTION: warm.\nTRANSCRIPT: "Ready."','voice_conversion':False,'format':'caption',
            'input_mode':'manual','generation':'single','streaming':False,'adapters':picks,'speaker':{'method':'off'},'ranking':{'mode':'legacy'},
            'voice_mode':'custom' if picks.get('ref15') else 'none',
            'reference_id':reference.id if picks.get('ref15') else None})
        if AdapterSelection(**picks).active():
            assert response.status_code==422
            continue
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        assert not any(event['event'] == 'error' for event in events)
        result = next(event for event in events if event['event'] == 'result')
        assert result['adapters']['selection'] == AdapterSelection(**picks).model_dump()
        assert any(event['event'] == 'adapters' for event in events)
    assert calls == [AdapterSelection(**pick).model_dump() for pick in selections if not AdapterSelection(**pick).active()]
    assert not small_app.STATE['busy']


def test_gpu_waiter_rejects_foreign_jobs_and_requires_both_gpus():
    from wait_for_gpus import available
    devices = {0: {'uuid': 'a', 'memory_mib': 25, 'utilization_percent': 0},
               1: {'uuid': 'b', 'memory_mib': 12, 'utilization_percent': 0}}
    assert available(devices, [], [0, 1])
    assert not available(devices, [{'uuid': 'a', 'pid': 123, 'memory_mib': 1}], [0, 1])
    assert not available(devices, [], [0, 1, 2])
    devices[1]['memory_mib'] = 20000
    assert not available(devices, [], [0, 1])
    assert available(devices, [{'uuid': 'b', 'pid': 111, 'memory_mib': 19900}], [0, 1], own_pid=111)
    devices[0]['utilization_percent'] = 10
    assert not available(devices, [], [0, 1])
