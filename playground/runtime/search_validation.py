"""Validate continuation, emitted PCM and shared scoring; no subjective claims."""
import json
import numpy as np
import torch
import torchaudio

import settings
from engine import SmallEngine, Scorers
from beam_search import ChunkSearch
from prompting import build


class FixedScorer:
    device='none (constant validation scorer)'
    last_timing={'asr_seconds':0.,'quality_seconds':0.,'total_seconds':0.}
    def score(self,waves,text,final):
        return [{'reward':1.,'prefix_wer':0.,'hyp':'','sec':len(w)/48000} for w in waves]


def main():
    engine=SmallEngine()
    spec=build({'text':'You knew what this meant to me. Tell me the truth.',
                'caption':'Intense contained anger, clear words.'},'caption','en',seconds=6.6)
    reference=settings.HERE/'performance/benchmark-reference.wav'
    baseline=engine.generate_many([spec]*4,1.5,seed=991,reference_path=reference,streaming=True,
                                  batch_size=4,chunk_frames=10,max_frames=20)
    codes=engine.last_codes[0].clone()
    emitted=[]
    wave,p=ChunkSearch(engine,FixedScorer()).generate(spec,n=4,beam_width=1,chunk_frames=10,
        max_seconds=1.6,seed=991,reference_path=reference,on_commit=lambda w,e:emitted.append(w))
    new=engine.last_codes[0]
    continuation={'same_tokens':bool(torch.equal(codes,new)),
        'token_agreement':float((codes[:min(len(codes),len(new))]==new[:min(len(codes),len(new))]).float().mean()),
        'same_lengths':len(baseline[0])==len(wave),
        'pcm_rms_difference':float(np.sqrt(np.mean((baseline[0][:len(wave)]-wave)**2))) if len(baseline[0])>=len(wave) else None,
        'commits_equal_final':bool(np.array_equal(np.concatenate(emitted),wave)),
        'chunk_count':len(emitted)}
    assert continuation['same_tokens'] and continuation['commits_equal_final'],continuation
    scorers=Scorers()
    # Load the packaged copies independently even after production switched to
    # sharing one tower; comparing aliases would be a vacuous validation.
    from transformers import AutoModel
    gen_tower=AutoModel.from_pretrained('/mnt/nvme/moss-15-v2-assets/vc_genuineness/voiceclap_commercial',
        trust_remote_code=True).to(scorers.retriever.device).eval()
    blend_tower=AutoModel.from_pretrained('/mnt/nvme/moss-15-v2-assets/vc_blend/voiceclap_commercial',
        trust_remote_code=True).to(scorers.retriever.device).eval()
    audio=torchaudio.functional.resample(torch.tensor(wave,device=scorers.retriever.device),48000,16000)[None]
    with torch.inference_mode():
        reference_embedding=scorers.retriever.model.encode_waveform(audio).float()
        gen_embedding=gen_tower.encode_waveform(audio).float()
        blend_embedding=blend_tower.encode_waveform(audio).float()
        shared={'genuineness_embedding_max_abs':float((reference_embedding-gen_embedding).abs().max()),
            'blend_embedding_max_abs':float((reference_embedding-blend_embedding).abs().max())}
    assert shared['genuineness_embedding_max_abs']<1e-6 and shared['blend_embedding_max_abs']<1e-6,shared
    result={'continuation':continuation,'shared_audio_tower':shared,
            'notes':'Numerical/causal validation on one prompt, four candidates, twenty frames; not a human perceptual-equivalence study.'}
    (settings.HERE/'search/validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
