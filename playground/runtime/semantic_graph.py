"""Opt-in static semantic KV cache + CUDA-graph decode (single-frame queries).

Prefill remains eager. Each graph reads a GPU cache-position counter, a static
input row and a full-capacity boolean key mask. No model weights are modified.
The default dynamic/compiled path stays available for correctness comparison.
"""
from collections import OrderedDict
import os
import torch

BUCKET=int(os.environ.get('HVS_SEMANTIC_BUCKET','128'))

def raw_forward(transformer):
    fn=transformer.forward
    if getattr(fn,'__closure__',None):
        return dict(zip(fn.__code__.co_freevars,[v.cell_contents for v in fn.__closure__])).get('forward',fn)
    return fn

class SemanticBank:
    def __init__(self,model,nb,capacity):
        self.model=model;self.forward=raw_forward(model.transformer)
        if type(model.transformer).__name__!='MossQwen3Model':
            raise ValueError('Semantic graph currently supports the released MossQwen3Model only')
        self.cache=[]
        self.nb=nb;self.capacity=capacity;self.graph=None;self.output=None
        self.device=next(model.parameters()).device
        self.ids=torch.zeros((nb,1,model.config.n_vq+1),dtype=torch.long,device=self.device)
        self.allowed=torch.zeros((nb,capacity),dtype=torch.bool,device=self.device)
        self.position=torch.zeros(1,dtype=torch.long,device=self.device)
        self.query_valid=torch.zeros((nb,1,1),dtype=torch.bool,device=self.device)
        # Optional: fuse the many small decode kernels (norms, RoPE, residuals)
        # with Inductor before the CUDA graph is captured; weights are unchanged.
        if os.environ.get('HVS_SEMANTIC_COMPILE','0')=='1':self._decode=torch.compile(self._decode,dynamic=False,mode='max-autotune-no-cudagraphs')
    def prefill(self,ids,mask):
        self.allowed.zero_();self.allowed[:,:mask.shape[1]].copy_(mask)
        length=ids.shape[1]
        out=self.forward(inputs_embeds=self.model._build_inputs_embeds(ids),input_ids=None,
            attention_mask=mask,past_key_values=None,use_cache=True,return_dict=True)
        self.position.fill_(length)
        if not self.cache:
            for key,value in out.past_key_values:
                shape=(self.nb,self.capacity,*key.shape[2:])
                self.cache.append((key.new_zeros(shape),value.new_zeros(shape)))
        for (key,value),(stored_key,stored_value) in zip(out.past_key_values,self.cache):
            stored_key[:,:length].copy_(key);stored_value[:,:length].copy_(value)
        return self.model._global_hidden_to_local(out.last_hidden_state[:,-1]).to(self.model.local_transformer.ln_f.weight.dtype)
    def _decode(self):
        # The release has a custom tuple KV cache, not Transformers StaticCache.
        # Reuse its trained modules/attention kernels with fixed-address storage.
        transformer=self.model.transformer
        hidden=self.model._build_inputs_embeds(self.ids)*self.query_valid
        positions=(self.allowed.long().sum(-1)-1).clamp_min(0).view(self.nb,1)
        cos,sin=transformer.rotary_emb(hidden,positions)
        cos=cos.unsqueeze(-2);sin=sin.unsqueeze(-2)
        def rotate(x):
            a,b=x.chunk(2,dim=-1);return torch.cat((-b,a),dim=-1)
        for layer,(stored_key,stored_value) in zip(transformer.layers,self.cache):
            attn=layer.self_attn;residual=hidden;x=layer.input_layernorm(hidden)
            query=attn.q_norm(attn.q_proj(x).view(self.nb,1,attn.num_heads,attn.head_dim))
            key=attn.k_norm(attn.k_proj(x).view(self.nb,1,attn.num_key_value_heads,attn.head_dim))
            value=attn.v_proj(x).view(self.nb,1,attn.num_key_value_heads,attn.head_dim)
            query=query*cos+rotate(query)*sin;key=key*cos+rotate(key)*sin
            stored_key.index_copy_(1,self.position,key);stored_value.index_copy_(1,self.position,value)
            # Single query attends every valid key; padded future slots are
            # excluded by allowed. The release's kernel repeats KV identically.
            attend=attn._sdpa_attention if attn.attn_implementation=='sdpa' else attn._eager_attention
            out=attend(query,stored_key,stored_value,self.allowed)
            hidden=residual+attn.o_proj(out.reshape(self.nb,1,-1).contiguous())
            hidden=(hidden+layer.mlp(layer.post_attention_layernorm(hidden)))*self.query_valid
        hidden=transformer.norm(hidden)*self.query_valid
        self.position.add_(1)
        return self.model._global_hidden_to_local(hidden[:,-1]).to(self.model.local_transformer.ln_f.weight.dtype)
    def decode(self,ids,mask):
        self.ids.copy_(ids);self.allowed[:,:mask.shape[1]].copy_(mask)
        self.query_valid.copy_(mask[:,-1:].unsqueeze(-1))
        if self.graph is None:
            position=self.position.clone()
            stream=torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                self._decode();self._decode()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            self.position.copy_(position)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.output=self._decode()
            self.position.copy_(position)
        self.graph.replay()
        return self.output

class SemanticGraphs:
    def __init__(self,model,limit=2):self.model=model;self.limit=limit;self.banks=OrderedDict()
    def prepare(self,ids,budget):
        capacity=((ids.shape[1]+budget+BUCKET-1)//BUCKET)*BUCKET;key=(ids.shape[0],capacity)
        if key not in self.banks:
            self.banks[key]=SemanticBank(self.model,ids.shape[0],capacity)
            while len(self.banks)>self.limit:self.banks.popitem(last=False)
        self.banks.move_to_end(key);return self.banks[key]
