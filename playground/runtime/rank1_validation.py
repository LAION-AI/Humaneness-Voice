"""Real packed-adapter CPU integrity/fusion checks, no GPU or audio efficacy claim."""
import json
from pathlib import Path
import time

from huggingface_hub import hf_hub_download
import torch

import settings
from rank1_adapters import AdapterSelection, Rank1Bank, catalog


def main():
    torch.set_num_threads(2)
    path = hf_hub_download(settings.REPO, 'checkpoints/S3/model_bf16.pt', revision=settings.REVISION,
        cache_dir=str(Path(settings.HF_CACHE) / 'hub'), token=False)
    started = time.perf_counter()
    state = torch.load(path, map_location='cpu', weights_only=True)
    # Real base attention weights only; no GPU allocation, codec or TTS run.
    model = torch.nn.Module()
    model.transformer = torch.nn.Module()
    model.transformer.layers = torch.nn.ModuleList()
    for layer in range(28):
        block = torch.nn.Module()
        block.self_attn = torch.nn.Module()
        for projection in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            weight = state[f'transformer.layers.{layer}.self_attn.{projection}.weight']
            linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype)
            with torch.no_grad():
                linear.weight.copy_(weight)
            setattr(block.self_attn, projection, linear)
        model.transformer.layers.append(block)
    del state
    bank = Rank1Bank(model, path)
    checked = []
    for id_ in bank.index['adapters']:
        bank.tensors(id_)
        checked.append(id_)
    originals = {name: module.weight.detach().clone() for name, module in bank.modules.items()}
    pointers = {name: module.weight.data_ptr() for name, module in bank.modules.items()}
    profiles = [
        {'voice': {'id': 'profile/anime_000'}},
        {'emotion': {'id': 'emotion/Affection'}},
        {'style': {'id': 'vn/S_RANT_high'}},
        {'voice': {'id': 'profile/anime_000'}, 'emotion': {'id': 'emotion/Affection', 'weight': .5},
         'style': {'id': 'vn/S_RANT_high', 'weight': .5}},
        {},
    ]
    switches = []
    for fields in profiles:
        selection = AdapterSelection.model_validate(fields)
        result = bank.apply(selection)
        for name, module in bank.modules.items():
            expected = originals[name].float().clone()
            for choice in selection.active():
                tensors = bank.tensors(choice.id)
                expected.addmm_(tensors[name + '.b.weight'], tensors[name + '.a.weight'], alpha=choice.weight)
            assert torch.equal(module.weight, expected.to(module.weight.dtype)), name
            assert pointers[name] == module.weight.data_ptr(), name
        switches.append(result)
    assert all(torch.equal(module.weight, originals[name]) for name, module in bank.modules.items())
    result = {'verification': 'Real pinned archives and actual S3 attention weights, CPU only; no TTS efficacy or throughput claim',
        'repo': settings.LORA_REPO, 'revision': settings.LORA_REVISION, 'verified_adapter_count': len(checked),
        'module_count': len(bank.modules), 'tensor_shapes_hashes_finiteness': 'passed',
        'fused_weight_formula': 'passed', 'stable_parameter_storage': 'passed',
        'base_restoration_bit_exact': 'passed', 'switches': switches,
        'cpu_seconds': time.perf_counter() - started}
    (settings.HERE / 'test-results/rank1-validation.json').write_text(json.dumps(result, indent=2))
    (settings.HERE / 'test-results/rank1-catalog.json').write_text(json.dumps(catalog(), indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != 'switches'}), flush=True)


if __name__ == '__main__':
    main()
