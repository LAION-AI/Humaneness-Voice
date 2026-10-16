"""Published RL r128 residual arithmetic, without merging or BF16 adapters."""
import json

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file


class RLLinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.empty(base.out_features, rank, device=base.weight.device, dtype=torch.float32))

    # The demo's static local attention cache reads these properties. They
    # expose the original projection; they do not fold the adapter into it.
    @property
    def weight(self): return self.base.weight

    @property
    def bias(self): return self.base.bias

    def forward(self, x):
        result = self.base(x)
        with torch.autocast(x.device.type, enabled=False):
            residual = F.linear(F.linear(x.float(), self.a), self.b)
        return result + residual.to(result.dtype)


def attach_rl(model, root, release):
    """Accept only the exact pinned full adapter, including every local layer."""
    config = json.loads((root.parent/'adapter_config.json').read_text())
    if (config['base_weight_sha256'] != release['sha256'] or
        config['adapter_sha256'] != release['adapter_sha256'] or
        config['rank'] != 128 or config['alpha'] != 128 or config['residual_scale'] != 1):
        raise ValueError('RL adapter/base/scaling contract mismatch')
    model.requires_grad_(False)
    groups = {'semantic': 0, 'local': 0}
    for name, module in list(model.named_modules()):
        leaf = name.rsplit('.', 1)[-1]
        semantic = name.startswith('transformer.') and leaf in ('q_proj','k_proj','v_proj','o_proj')
        local = name.startswith('local_transformer.') and '.attn.' in name and leaf in ('c_attn','c_proj')
        if (semantic or local) and isinstance(module, nn.Linear):
            parent, _, leaf = name.rpartition('.')
            setattr(model.get_submodule(parent), leaf, RLLinear(module, config['rank']))
            groups['semantic' if semantic else 'local'] += 1
    if not all(groups.values()): raise ValueError('RL adapter must cover semantic and local attention')
    targets = {k:p for k,p in model.named_parameters() if p.requires_grad}
    state = load_file(str(root.parent/'adapter.safetensors'), device='cpu')
    if set(state) != set(targets): raise ValueError('RL adapter tensor names do not match the architecture')
    if len(state) != config['adapter_tensors'] or sum(p.numel() for p in targets.values()) != config['adapter_parameters']:
        raise ValueError('RL adapter tensor/parameter count mismatch')
    with torch.no_grad():
        for key, value in state.items():
            if targets[key].shape != value.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError('Invalid RL adapter tensor: '+key)
            targets[key].copy_(value)
    model.eval().requires_grad_(False)
    return {'enabled':True,'rank':128,'dtype':'fp32','merged':False,'sha256':release['adapter_sha256'],**groups}
