"""Same-length Chatterbox batching with per-candidate diffusion noise.

No padded acoustic contexts, precision change or reduced flow-step count.
The original decoder stays available outside this request-local seed context.
"""
from contextvars import ContextVar
from types import MethodType

SEEDS=ContextVar('chatterbox_candidate_seeds',default=None)


def install_seeded_decoder(model):
    decoder=model.s3gen.flow.decoder
    if getattr(decoder,'_candidate_seeds_installed',False):return
    original=decoder.forward
    def forward(self,mu,mask,n_timesteps,temperature=1.,spks=None,cond=None,noised_mels=None,meanflow=False):
        seeds=SEEDS.get()
        if seeds is not None:
            import torch
            if len(seeds)!=mu.shape[0] or noised_mels is not None or meanflow:
                raise ValueError('Unsupported seeded Chatterbox batch')
            noises=[]
            for seed in seeds:
                generator=torch.Generator(device=mu.device).manual_seed(seed%(2**31-1))
                noises.append(torch.randn((1,*mu.shape[1:]),device=mu.device,dtype=mu.dtype,generator=generator))
            # Supplying the full mel extent also fixes the random prompt part.
            noised_mels=torch.cat(noises,dim=0)*temperature
        return original(mu,mask,n_timesteps,temperature=temperature,spks=spks,cond=cond,
            noised_mels=noised_mels,meanflow=meanflow)
    decoder.forward=MethodType(forward,decoder)
    decoder._candidate_seeds_installed=True


def batch_native(model,waves,conditioning,seeds,device):
    import numpy as np
    import torch
    from chatterbox.models.s3gen.s3gen import get_resampler
    if not waves or len(waves)!=len(seeds) or len({len(w) for w in waves})!=1:
        raise ValueError('VC microbatch requires equal nonzero source lengths')
    audio=torch.as_tensor(np.stack(waves),dtype=torch.float32,device=device)
    source=get_resampler(48000,16000,torch.device(device))(audio)
    codes,lengths=tokenize_equal(model.s3gen.tokenizer,source)
    if not bool((lengths==codes.shape[1]).all()):
        raise RuntimeError('Unexpected tokenizer padding in equal-length VC microbatch')
    token=SEEDS.set(tuple(seeds))
    try:
        output,_=model.s3gen.inference(speech_tokens=codes,speech_token_lens=lengths,
            ref_dict=dict(conditioning))
    finally:SEEDS.reset(token)
    return output.detach().float().cpu().numpy()


def tokenize_equal(tokenizer,source):
    """Vectorized STFT; retain the upstream PER-WAVE log-mel floor."""
    import torch
    stft=torch.stft(source,tokenizer.n_fft,160,window=tokenizer.window,return_complex=True)
    mel=tokenizer._mel_filters @ (stft[...,:-1].abs()**2)
    log=mel.clamp(min=1e-10).log10()
    log=torch.maximum(log,log.amax(dim=(-2,-1),keepdim=True)-8.)
    features=(log+4.)/4.
    lengths=torch.full((len(source),),features.shape[-1],dtype=torch.int32,device=source.device)
    codes,code_lengths=tokenizer.quantize(features,lengths)
    return codes.long().detach(),code_lengths.long().detach()
