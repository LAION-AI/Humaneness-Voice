"""Opt-in RL inference arithmetic; official FP32 remains the default.

Apply once at checkpoint load, before compile/CUDA-graph capture. Precision and
adapter merging are numerical inference variants, not changed source weights.
"""
import torch
from rl_adapter import RLLinear

MODES=('fp32_strict','fp32_tf32','bf16_base_fp32_lora','bf16_all','fp32_merged','bf16_merged')

def configure(engine,mode='fp32_strict'):
    if mode not in MODES:raise ValueError('Unsupported RL compute mode: '+str(mode))
    if getattr(engine,'precision_configured',False):raise RuntimeError('RL compute precision is configured once at model load')
    model=engine.model;wrappers={name:module for name,module in model.named_modules() if isinstance(module,RLLinear)}
    if not wrappers:raise ValueError('Expected the released unmerged RL policy')
    merged=mode.endswith('_merged');bf16=mode.startswith('bf16')
    # Build W+B@A in the original strict FP32 arithmetic, independently of
    # whatever backend precision the user selected for subsequent inference.
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    saved={name:(module.a.detach().clone(),module.b.detach().clone()) for name,module in wrappers.items()} if mode=='bf16_base_fp32_lora' else {}
    frequencies={name:value.detach().clone() for name,value in model.named_buffers() if 'inv_freq' in name}
    if merged:
        with torch.no_grad():
            for name,module in wrappers.items():
                layer=torch.nn.Linear(module.base.in_features,module.base.out_features,
                    bias=module.base.bias is not None,device=module.base.weight.device,dtype=torch.float32)
                layer.weight.copy_(module.base.weight.float()+module.b.float()@module.a.float())
                if layer.bias is not None:layer.bias.copy_(module.base.bias.float())
                layer.requires_grad_(False)
                parent,_,leaf=name.rpartition('.');setattr(model.get_submodule(parent),leaf,layer)
    if bf16:
        model.to(dtype=torch.bfloat16)
        for name,value in frequencies.items():
            parent,_,leaf=name.rpartition('.');setattr(model.get_submodule(parent),leaf,value)
        if saved:
            for name,(a,b) in saved.items():wrappers[name].a.data=a;wrappers[name].b.data=b
    tf32=mode=='fp32_tf32'
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')
    torch.backends.cuda.matmul.allow_tf32=tf32;torch.backends.cudnn.allow_tf32=tf32
    engine.policy_fp32=not bf16;engine.precision_configured=True
    result={'mode':mode,'model_dtype':'bf16' if bf16 else 'fp32','tf32':tf32,'merged':merged,
        'adapter_projection_count':len(wrappers),'extra_adapter_gemms_removed_per_projection_set':2*len(wrappers) if merged else 0,
        'official_reference_arithmetic':mode=='fp32_strict',
        'note':'Precision/merging may alter sampled tokens. Compare quality with the strict FP32 reference.'}
    engine.precision_report=result
    return result
