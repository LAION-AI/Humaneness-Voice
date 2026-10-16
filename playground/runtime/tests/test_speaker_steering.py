from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speaker_steering import SpeakerSteeringState, catalog_speaker_steering, resolve_speaker_steering


class Block(nn.Module):
    def forward(self, hidden): return hidden + 1, "cache"


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(Block() for _ in range(17))
    def forward(self, inputs_embeds, attention_mask=None):
        hidden = inputs_embeds
        for layer in self.layers: hidden, _ = layer(hidden)
        return hidden


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_dormant_and_zero_factor_exact_parity(dtype):
    model = Transformer()
    values = torch.randn(4, 3, 4).to(dtype)
    values[0, -1, 0] = -17.
    mask = torch.ones(4, 3, dtype=torch.bool)
    baseline = model(values, attention_mask=mask)
    state = SpeakerSteeringState(4).install(model)
    assert torch.equal(model(values, attention_mask=mask), baseline)
    state.configure(torch.ones(4), alpha=0.)
    assert torch.equal(model(values, attention_mask=mask), baseline)
    assert not model.state_dict()  # dormant nonpersistent state adds no keys


def test_only_last_position_both_cfg_branches_and_finished_masks():
    model = Transformer()
    values = torch.ones(4, 3, 4)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    baseline = model(values, attention_mask=mask)
    state = SpeakerSteeringState(4).install(model)
    state.configure(torch.arange(4).float(), alpha=.75)
    out = model(values, attention_mask=mask)
    assert torch.equal(out[:, :-1], baseline[:, :-1])
    delta = out[:, -1] - baseline[:, -1]
    assert delta.tolist() == [[0., .75, 1.5, 2.25], [0., 0., 0., 0.], [0., .75, 1.5, 2.25], [0., 0., 0., 0.]]
    # Separate legacy branch calls share the identical vector, not a doubled one.
    assert torch.equal(model(values[:2], attention_mask=mask[:2]) - baseline[:2], out[:2] - baseline[:2])


def test_switches_do_not_accumulate_invalid_vector_clears_previous_voice():
    model = Transformer()
    state = SpeakerSteeringState(4).install(model)
    values, mask = torch.ones(2, 1, 4), torch.ones(2, 1, dtype=torch.bool)
    state.configure(torch.ones(4), alpha=.75)
    first = model(values, attention_mask=mask)
    state.configure(torch.full((4,), 2.), alpha=.75)
    second = model(values, attention_mask=mask)
    torch.testing.assert_close(second - first, torch.full_like(first, .75))
    with pytest.raises(ValueError): state.configure(torch.ones(3))
    assert not bool(state.enabled)
    assert torch.equal(model(values, attention_mask=mask), torch.full_like(values, 18.))
    state.configure(torch.ones(4))
    assert torch.equal(model(values), torch.full_like(values, 18.))  # no mask -> bypass


def test_compiled_forward_hook_buffers_switch_without_recompile():
    model = Transformer()
    state = SpeakerSteeringState(4).install(model)
    compilations = []
    def backend(graph, inputs):
        compilations.append(graph)
        return graph.forward
    model.forward = torch.compile(model.forward, backend=backend, dynamic=True)
    values = torch.ones(2, 3, 4); mask = torch.ones(2, 3, dtype=torch.bool)
    baseline = model(values, attention_mask=mask)
    count = len(compilations)
    state.configure(torch.ones(4), .75)
    out = model(values, attention_mask=mask)
    assert len(compilations) == count
    assert torch.equal(out[:, :-1], baseline[:, :-1])
    torch.testing.assert_close(out[:, -1] - baseline[:, -1], torch.full((2, 4), .75))
    state.configure(None)
    assert torch.equal(model(values, attention_mask=mask), baseline)
    assert len(compilations) == count


def fixture_bank(tmp_path):
    ref = tmp_path / 'reference.wav'; ref.write_bytes(b'exact reference audio')
    bank = tmp_path / 'bank'; bank.mkdir()
    file = bank / 'mira.npz'; np.savez_compressed(file, layer_16__contrast=np.ones(4, dtype=np.float32))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'version':1, 'checkpoint_sha256':'exact_checkpoint', 'layer':16, 'alpha':.75,
        'voices':{'mira':{'status':'ready','name':'Mira','vector_file':'mira.npz','vector_key':'layer_16__contrast',
                         'dimension':4,'sha256':sha(file),'reference_sha256':sha(ref),
                         'validation':{'eligible':True,'n':20,'private_path':'never exposed'}}}}
    (bank / 'manifest.json').write_text(json.dumps(manifest))
    engine = SimpleNamespace(checkpoint_info={'sha256':'exact_checkpoint'},
        adapter_bank=SimpleNamespace(active_key=()), model=SimpleNamespace(config=SimpleNamespace(hidden_size=4)))
    return engine, ref, bank, manifest


def test_exact_bank_reference_checkpoint_validation_and_public_catalog(tmp_path):
    engine, ref, bank, manifest = fixture_bank(tmp_path)
    vector, report = resolve_speaker_steering(engine, 'mira', ref, bank=bank)
    assert report['active'] and torch.equal(vector, torch.ones(4))
    catalog = catalog_speaker_steering(bank)
    serialized = json.dumps(catalog)
    assert 'private_path' not in serialized and 'vector_file' not in serialized and str(tmp_path) not in serialized
    engine.checkpoint_info['sha256'] = 'different_checkpoint'
    assert resolve_speaker_steering(engine, 'mira', ref, bank=bank)[1]['reason'] == 'checkpoint_mismatch'


@pytest.mark.parametrize('mode,reason', [('custom','custom_reference_no_matching_vector'),('no_ref','no_reference'),
    ('no_voice','no_catalog_voice'),('adapter','adapters_active_vector_base_mismatch'),('disabled','disabled_by_user'),
    ('wrong_reference','reference_mismatch'),('corrupt','vector_integrity_mismatch')])
def test_bypass_never_uses_wrong_or_unavailable_vector(tmp_path, mode, reason):
    engine, ref, bank, manifest = fixture_bank(tmp_path)
    voice, enabled, custom = 'mira', True, False
    if mode == 'custom': custom = True
    if mode == 'no_ref': ref = None
    if mode == 'no_voice': voice = None
    if mode == 'adapter': engine.adapter_bank.active_key = (('DPO', 1.),)
    if mode == 'disabled': enabled = False
    if mode == 'wrong_reference': ref.write_bytes(b'another clip')
    if mode == 'corrupt': (bank / 'mira.npz').write_bytes(b'corrupt vector')
    vector, report = resolve_speaker_steering(engine, voice, ref, enabled=enabled, custom=custom, bank=bank)
    assert vector is None and not report['active'] and report['reason'] == reason


def test_real_fast_loop_wiring_offline_batch_and_live_chunk_sizes():
    from test_activation_steering import fake_engine, branches, exhaust
    from performance_core import FastGenerator
    for chunk_frames in (1, 20):
        engine = fake_engine()
        state = SpeakerSteeringState(2).install(engine.model.transformer, layer=0)
        state.configure(torch.ones(2), .75)
        p = {'max_new_tokens':8,'audio_repetition_penalty':1.}
        exhaust(FastGenerator.stream_frames(SimpleNamespace(engine=engine, frame=engine.fast.frame),
            branches(), p, chunk_frames, guidance=2.))
        # Third predictor row1 is finished in BOTH CFG branches, despite its
        # fake head intentionally trying to emit a new acoustic token again.
        torch.testing.assert_close(engine.fast.inputs[0][:2], torch.tensor([[11.75,6.75],[15.75,10.75]]))
        last = engine.fast.inputs[2]
        torch.testing.assert_close(last[0], torch.tensor([7.75,4.75]))
        torch.testing.assert_close(last[1], torch.tensor([2.,1.]))
        assert torch.equal(last[:2], last[2:])


def test_beam_global_forward_supplies_same_attention_mask_contract():
    # Beam directly invokes the SAME model.transformer module as fast/legacy,
    # so permanent global hooks cover it without per-search hook registration.
    import ast
    root = Path(__file__).resolve().parents[1]
    for file in ('beam_search.py','performance_core.py'):
        tree = ast.parse((root / file).read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'transformer']
        assert calls and all(any(keyword.arg == 'attention_mask' for keyword in call.keywords) for call in calls)


def test_engine_configure_contract_and_request_finally_cleanup():
    import threading
    from engine import SmallEngine
    state = SpeakerSteeringState(4)
    engine = SimpleNamespace(lock=threading.Lock(), speaker_steering=state, speaker_steering_report={})
    report = {'active':True,'enabled':True,'voice':'Mira','alpha':.75}
    try:
        returned = SmallEngine.configure_speaker_steering(engine, torch.ones(4), report)
        assert returned == report and bool(state.enabled)
        assert returned is not report and returned is not engine.speaker_steering_report
        raise RuntimeError('Aborted generation')
    except RuntimeError:
        pass
    finally:
        SmallEngine.configure_speaker_steering(engine, None, {'active':False,'reason':'request_complete'})
    assert not bool(state.enabled) and torch.equal(state.direction, torch.zeros(4))
    assert engine.speaker_steering_report['reason'] == 'request_complete'
