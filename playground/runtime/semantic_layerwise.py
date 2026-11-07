"""Optional bounded compile unit for the Large 1.7 semantic continuation.

Compile one tensor-only transformer block, reuse it across all 36 layers, then
capture the complete decode in a CUDA graph. This avoids compiling a gigantic
36-layer Python graph for every candidate batch shape. No weight quantization.
"""
import torch
import torch.nn.functional as F
from semantic_graph import SemanticBank


def rms(x,w,eps):
    f=x.float();f=f*torch.rsqrt(f.square().mean(-1,keepdim=True)+eps)
    return w*f.to(x.dtype)


def layer_step(hidden,key_cache,value_cache,position,allowed,valid,cos,sin,
               iw,qw,kw,vw,qn,kn,ow,pw,gw,uw,dw,eps,heads,kv_heads,dim):
    batch=hidden.shape[0];residual=hidden;x=rms(hidden,iw,eps)
    query=rms(F.linear(x,qw).view(batch,1,heads,dim),qn,eps)
    key=rms(F.linear(x,kw).view(batch,1,kv_heads,dim),kn,eps)
    value=F.linear(x,vw).view(batch,1,kv_heads,dim)
    def rotate(a):
        first,second=a.chunk(2,dim=-1);return torch.cat((-second,first),dim=-1)
    query=query*cos+rotate(query)*sin;key=key*cos+rotate(key)*sin
    key_cache.index_copy_(1,position,key);value_cache.index_copy_(1,position,value)
    def repeat(a):
        b,t,h,d=a.shape
        return a[:,:,:,None,:].expand(b,t,h,heads//kv_heads,d).reshape(b,t,heads,d).transpose(1,2)
    out=F.scaled_dot_product_attention(query.transpose(1,2),repeat(key_cache),repeat(value_cache),
        attn_mask=allowed[:,None,None,:],dropout_p=0.,is_causal=False,scale=dim**-.5)
    hidden=residual+F.linear(out.transpose(1,2).contiguous().reshape(batch,1,-1),ow)
    x=rms(hidden,pw,eps)
    return (hidden+F.linear(F.silu(F.linear(x,gw))*F.linear(x,uw),dw))*valid


compiled_layer=torch.compile(layer_step,dynamic=False,fullgraph=True,mode='max-autotune-no-cudagraphs')


class LayerwiseSemanticBank(SemanticBank):
    def __init__(self,model,nb,capacity):
        super().__init__(model,nb,capacity,compile_decode=False)
        self.weights=[]
        for layer in model.transformer.layers:
            a=layer.self_attn;m=layer.mlp
            norms=[layer.input_layernorm,a.q_norm,a.k_norm,layer.post_attention_layernorm]
            eps=norms[0].variance_epsilon
            if a.attn_implementation!='sdpa' or any(n.variance_epsilon!=eps for n in norms):
                raise ValueError('Layerwise continuation requires SDPA and matching RMSNorm epsilons')
            linear=[a.q_proj,a.k_proj,a.v_proj,a.o_proj,m.gate_proj,m.up_proj,m.down_proj]
            if any(x.bias is not None for x in linear) or type(m.act_fn).__name__ not in ('SiLU','SiLUActivation'):
                raise ValueError('Unsupported biased or non-SiLU semantic block')
            self.weights.append((layer.input_layernorm.weight,a.q_proj.weight,a.k_proj.weight,a.v_proj.weight,
                a.q_norm.weight,a.k_norm.weight,a.o_proj.weight,layer.post_attention_layernorm.weight,
                m.gate_proj.weight,m.up_proj.weight,m.down_proj.weight,eps,a.num_heads,a.num_key_value_heads,a.head_dim))

    def _decode(self):
        transformer=self.model.transformer
        hidden=self.model._build_inputs_embeds(self.ids)*self.query_valid
        positions=(self.allowed.long().sum(-1)-1).clamp_min(0).view(self.nb,1)
        cos,sin=transformer.rotary_emb(hidden,positions);cos=cos.unsqueeze(-2);sin=sin.unsqueeze(-2)
        for (key,value),weights in zip(self.cache,self.weights):
            hidden=compiled_layer(hidden,key,value,self.position,self.allowed,self.query_valid,cos,sin,*weights)
        hidden=transformer.norm(hidden)*self.query_valid;self.position.add_(1)
        return self.model._global_hidden_to_local(hidden[:,-1]).to(self.model.local_transformer.ln_f.weight.dtype)
