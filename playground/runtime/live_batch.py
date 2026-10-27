"""Continuous single-weight TTS batching for the pinned Small/CFG/PCM stack.

One owner thread performs all CUDA work. New utterances have independent global
KV and codec state; decode frames from active utterances share a real batch.
No text/audio is persisted here. The ordinary Playground engine is unchanged.
"""
import contextlib,queue,time
from dataclasses import dataclass,field
import numpy as np
import torch
import torch.nn.functional as F
from performance_core import combine_branches,StreamingCodec

def warm_up(engine):
    """Capture the default T=0.8, one/two-stream shapes before service readiness.

    Only six 80-ms frames per synthetic utterance, no files or audible output.
    Warmup and production use the same scheduler, weights and reference path.
    """
    import threading
    from prompting import build
    from dynamic_references import fixed_clip
    reference,_=fixed_clip('emolia_c1699')
    engine.fast.graph_limit=max(engine.fast.graph_limit,8)
    commands=queue.Queue();cancelled=threading.Event()
    for i,text in enumerate(('Hallo, schön dich zu sehen.','Schauen wir uns das gemeinsam an.')):
        spec=build({'text':text,'script':text,'general':'Gentle, warm, natural speech.'},'structured','de',2.,reference_identity=True)
        commands.put({'type':'submit','id':str(i),'spec':spec,'order':i,'max_frames':6,'align':False})
    commands.put({'type':'finish'})
    batch=ContinuousBatch(engine,reference,commands,lambda event:None,cancelled,temperature=.8,max_active=2)
    batch.run();return batch.stats


def merge_caches(states):
    """Pinned Qwen KV layout is [branch, time, heads, head_dim].

    Left padding is masked. Qwen derives RoPE positions from the cumulative
    attention mask, so cache lengths need not match and old keys are unchanged.
    """
    width=max(s.past[0][0].shape[1] for s in states)
    past=tuple(tuple(torch.cat([F.pad(s.past[layer][kind],
        (0,0,0,0,width-s.past[layer][kind].shape[1],0)) for s in states],0)
        for kind in range(2)) for layer in range(len(states[0].past)))
    masks=torch.cat([F.pad(s.mask,(width+1-s.mask.shape[1],0),value=False) for s in states],0)
    inputs=torch.cat([s.input for s in states],0)
    return past,masks,inputs


class CodecState:
    """Swap only per-stream state; codec weights are shared under engine.lock."""
    def __init__(self,codec,audio_pad):
        self.codec=codec;self.decoder=StreamingCodec(codec,1,audio_pad)
        with self.decoder:
            self.states=[(m,m._streaming_state) for m in codec.modules()
                         if getattr(m,'_streaming_state',None) is not None]
        # The pinned StreamingState.__exit__ is a no-op; the objects and their
        # KV tensors remain owned by this utterance after module pointers clear.
    def decode(self,frames):
        if any(getattr(m,'_streaming_state',None) is not None for m,_ in self.states):
            raise RuntimeError('Codec state already attached')
        try:
            for m,s in self.states:m._streaming_state=s
            pcm,lengths=self.decoder.decode(torch.stack(frames,dim=1))
            return pcm[0,:int(lengths[0])].cpu().numpy().copy()
        finally:
            for m,_ in self.states:m._streaming_state=None


@dataclass
class Utterance:
    id:str
    spec:dict
    order:int
    first:int=2
    chunk:int=5
    max_frames:int|None=None
    past:object=None
    mask:object=None
    input:object=None
    codec:object=None
    local:object=None
    steps:int=0
    pending:list=field(default_factory=list)
    samples:int=0
    started:float=0
    first_pcm:float|None=None
    cancelled:bool=False
    align:bool=False
    waves:list=field(default_factory=list)


class ContinuousBatch:
    def __init__(self,engine,reference_path,commands,emit,cancelled,*,temperature=.8,guidance=2.,
                 seed=777,max_active=3,batch_decode=True,finisher=None):
        self.e=engine;self.reference_path=reference_path;self.commands=commands;self.emit=emit
        self.cancelled=cancelled;self.temperature=temperature;self.guidance=guidance
        self.seed=seed;self.max_active=max_active;self.batch_decode=batch_decode
        self.finisher=finisher
        self.active=[];self.pending=[];self.known=set();self.sealed=False
        self.stats={'scheduler':'continuous CFG batch','weight_instances':1,'batch_histogram':{},
                    'prefill_ms':[],'frame_ms':[],'max_active':0,'jobs':0,'cancelled_jobs':0}
    def command(self,msg):
        kind=msg['type']
        if kind=='submit':
            if msg['id'] in self.known:raise ValueError('Duplicate utterance id')
            if len(self.pending)+len(self.active)>=8:raise ValueError('Too many pending utterances')
            self.known.add(msg['id']);self.pending.append(Utterance(msg['id'],msg['spec'],msg['order'],
                msg.get('first',2),msg.get('chunk',5),msg.get('max_frames'),align=msg.get('align',False)))
            self.pending.sort(key=lambda x:x.order)
        elif kind=='cancel':
            for s in self.active+self.pending:
                if s.id==msg['id']:s.cancelled=True
        elif kind=='finish':self.sealed=True
        else:raise ValueError('Unknown scheduler command')
    def drain(self):
        while True:
            try:self.command(self.commands.get_nowait())
            except queue.Empty:break
    def prefill(self,s,reference):
        e=self.e;mdl=e.model;cfg=mdl.config;s.started=time.perf_counter()
        branches=[e.pack(s.spec['prompt'],s.spec,reference,1),e.pack(s.spec['unconditional'],s.spec,reference,1)]
        s.input,s.mask=combine_branches(branches,cfg.pad_token_id,cfg.audio_pad_token_id)
        out=mdl.transformer(inputs_embeds=mdl._build_inputs_embeds(s.input),input_ids=None,
            attention_mask=s.mask,past_key_values=None,use_cache=True,position_ids=None,
            output_attentions=False,output_hidden_states=False,return_dict=True,cu_seqlens=None,num_sequences=None)
        s.past=out.past_key_values
        s.local=mdl._global_hidden_to_local(out.last_hidden_state[:,-1]).to(mdl.local_transformer.ln_f.weight.dtype)
        s.codec=CodecState(e.proc.audio_tokenizer,cfg.audio_pad_token_id)
        self.stats['prefill_ms'].append(round((time.perf_counter()-s.started)*1000,2))
        self.stats['jobs']+=1;self.emit({'event':'job_started','id':s.id,'order':s.order})
    def flush(self,s):
        if not s.pending:return
        pcm=s.codec.decode(s.pending);s.pending=[]
        if not np.isfinite(pcm).all():raise RuntimeError('Non-finite streaming PCM')
        s.samples+=len(pcm)
        if s.align:s.waves.append(pcm)
        if s.first_pcm is None:s.first_pcm=time.perf_counter()-s.started
        if len(pcm):self.emit({'event':'audio_chunk','id':s.id,'_wave':pcm,'sample_rate':48000,'channels':1})
    def complete(self,s):
        if not s.cancelled:self.flush(s)
        else:self.stats['cancelled_jobs']+=1
        aligned=None;alignment=None;tick=time.perf_counter()
        if not s.cancelled and s.align and s.waves and self.finisher:
            waves,reports=self.finisher.finish([np.concatenate(s.waves)],s.spec,True)
            aligned=waves[0];alignment=reports[0]
        self.emit({'event':'job_done','id':s.id,'cancelled':s.cancelled,'samples':s.samples,
            'first_audio_ms':None if s.first_pcm is None else round(s.first_pcm*1000,2),
            'generation_ms':round((time.perf_counter()-s.started)*1000,2) if s.started else 0,'frames':s.steps,
            '_aligned':aligned,'alignment':alignment,'alignment_ms':round((time.perf_counter()-tick)*1000,2)})
    def step(self,states):
        e=self.e;mdl=e.model;cfg=mdl.config;tick=time.perf_counter()
        continued=[s for s in states if s.local is None]
        if continued:
            past,mask,inputs=merge_caches(continued)
            out=mdl.transformer(inputs_embeds=mdl._build_inputs_embeds(inputs),input_ids=None,
                attention_mask=mask,past_key_values=past,use_cache=True,position_ids=None,
                output_attentions=False,output_hidden_states=False,return_dict=True,cu_seqlens=None,num_sequences=None)
            local=mdl._global_hidden_to_local(out.last_hidden_state[:,-1]).to(mdl.local_transformer.ln_f.weight.dtype)
            for i,s in enumerate(continued):
                length=s.mask.shape[1]
                s.past=tuple(tuple(v[i*2:i*2+2,-length:].contiguous() for v in layer) for layer in out.past_key_values)
                s.local=local[i*2:i*2+2]
        n=len(states)
        local=torch.stack([s.local for s in states]).transpose(0,1).reshape(2*n,-1).contiguous()
        p={'audio_temperature':self.temperature,'text_temperature':0. if self.temperature==0 else 1.,
           'audio_top_k':50,'audio_top_p':.95,'text_top_k':50,'text_top_p':1.}
        stop,codes=e.fast.frame(local,n,2,self.guidance,p,graph=True)
        # Clone graph outputs before the next replay; each utterance owns tokens.
        codes=codes.clone();continuing=stop.eq(cfg.audio_assistant_slot_token_id).cpu().tolist()
        done=[]
        for i,s in enumerate(states):
            s.local=None
            budget=min(810,s.spec['frames']+60,s.max_frames or 810)
            if not continuing[i] or s.steps>=budget:
                self.complete(s);done.append(s);continue
            frame=codes[i:i+1];s.pending.append(frame);s.steps+=1
            s.input=mdl._build_generation_row(1,frame.device,frame).repeat(2,1,1)
            s.mask=F.pad(s.mask,(0,1),value=True)
            if len(s.pending)>=(s.first if s.samples==0 else s.chunk):self.flush(s)
        key=str(n);self.stats['batch_histogram'][key]=self.stats['batch_histogram'].get(key,0)+1
        self.stats['frame_ms'].append(round((time.perf_counter()-tick)*1000,2))
        return done
    @torch.inference_mode()
    def run(self):
        e=self.e;started=time.perf_counter()
        with e.lock,torch.cuda.device(e.device),torch.autocast('cuda',dtype=torch.bfloat16,enabled=not getattr(e,'policy_fp32',False)):
            # Save/restore the shared RNG; batching changes the sampling draw
            # order, so stochastic samples need not equal serial samples.
            rng=torch.cuda.get_rng_state(e.device)
            try:
                torch.cuda.manual_seed(self.seed)
                reference=e.reference(self.reference_path) if self.reference_path else None
                while not self.cancelled.is_set():
                    self.drain()
                    for s in list(self.active):
                        if s.cancelled:self.complete(s);self.active.remove(s)
                    for s in list(self.pending):
                        if s.cancelled:self.complete(s);self.pending.remove(s)
                    # First audible chunk has priority over admission of new
                    # prefixes. Afterwards arrivals join at every frame boundary.
                    can_admit=not self.active or all(s.samples>0 for s in self.active)
                    if can_admit:
                        admission_limit=1 if self.stats['jobs']==0 else self.max_active
                        while self.pending and len(self.active)<admission_limit:
                            s=self.pending.pop(0);self.prefill(s,reference);self.active.append(s)
                    self.stats['max_active']=max(self.stats['max_active'],len(self.active))
                    if not self.active:
                        if self.sealed:break
                        try:self.command(self.commands.get(timeout=.05))
                        except queue.Empty:pass
                        continue
                    groups=[self.active[:]] if self.batch_decode else [[s] for s in self.active]
                    for group in groups:
                        for s in self.step(group):self.active.remove(s)
                if self.cancelled.is_set():
                    for s in self.active:s.cancelled=True;self.complete(s)
                self.stats['total_ms']=round((time.perf_counter()-started)*1000,2)
                self.stats['gpu_allocated_mb']=round(torch.cuda.memory_allocated(e.device)/2**20,1)
                self.emit({'event':'session_done','performance':self.stats})
            finally:
                self.active.clear();self.pending.clear();torch.cuda.set_rng_state(rng,e.device)
