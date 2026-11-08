"""Conservative boundaries: Qwen supported languages, Parakeet TDT otherwise.

Parakeet fallback is ASR-based boundary estimation, not forced alignment. It must
recognize the script with <=15% WER and match its opening/closing two words before
we trim. Uncertain cases retain original audio. Neither path changes selection.
"""
import os
from recipe import errors,tokens
from languages import language_name,QWEN_ALIGNMENT_LANGUAGES

def transcript_supports_edges(reference,hypothesis):
    ref,hyp=tokens(reference),tokens(hypothesis);k=min(2,len(ref))
    return bool(ref and hyp and errors(reference,hypothesis)['wer']<=.15 and ref[:k]==hyp[:k] and ref[-k:]==hyp[-k:])

class BoundaryAligner:
    def __init__(self,device):
        from align_engine import QwenAligner
        self.device=device;self.qwen=QwenAligner(device=device);self.ok=self.qwen.ok;self.model=None
    def align(self,wave,words,language):
        if language in QWEN_ALIGNMENT_LANGUAGES:
            result=self.qwen.align(wave,words,language=language_name(language))
            if result and all(x[3]>.5 for x in result):return result,dict(method='qwen_forced_alignment',timing_basis='forced_alignment_estimate')
        return self.parakeet(wave,' '.join(words))
    def parakeet(self,wave,reference):
        import torch
        from transformers import ParakeetProcessor,ParakeetForTDT
        if self.model is None:
            repo=os.environ.get('PODCAST_ASR','nvidia/parakeet-tdt-0.6b-v3')
            self.proc=ParakeetProcessor.from_pretrained(repo,local_files_only=True)
            self.model=ParakeetForTDT.from_pretrained(repo,dtype=torch.bfloat16,local_files_only=True).to(self.device).eval()
        with torch.inference_mode():
            x=self.proc([wave],sampling_rate=16000,return_tensors='pt',padding=True)
            x={k:v.to(self.device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in x.items()}
            out=self.model.generate(**x,max_new_tokens=2048)
            text=self.proc.batch_decode(out.sequences,skip_special_tokens=True)[0]
            durations=out.durations[0].cpu();ids=out.sequences[0].cpu();starts=durations.cumsum(0)-durations
        meta=dict(method='parakeet_tdt_boundaries',timing_basis='asr_token_boundary_estimate',recognized=text,script_wer=errors(reference,text)['wer'],supported_edges=transcript_supports_edges(reference,text))
        if not meta['supported_edges']:return None,meta
        frame_s=self.proc.feature_extractor.hop_length/self.proc.feature_extractor.sampling_rate*8
        skip={self.proc.tokenizer.pad_token_id,self.proc.blank_token_id};result=[];limit=len(wave)/16000
        for token,start,duration in zip(ids.tolist(),starts.tolist(),durations.tolist()):
            if token in skip:continue
            piece=self.proc.tokenizer.decode([token],skip_special_tokens=True,group_tokens=False)
            if not tokens(piece):continue
            a=start*frame_s;b=min(limit,(start+max(1,duration))*frame_s)
            if 0<=a<b<=limit:result.append((piece,a,b,1.))
        if not result or result[0][1]>=result[-1][2]:return None,meta
        return result,meta
