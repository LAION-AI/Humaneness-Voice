"""Batched frozen Ears + Parakeet, on the SAME selected CUDA device as TTS.

HTTP Parakeet is an explicit memory-saving alternative for this shared host;
the normal local backend supports actual batched ASR without a second service.
"""
import os
import re
import time
import numpy as np
import torch
from scipy.signal import resample_poly
from recipe import errors, reward

KEYS = ['genuineness_0_6', 'audiobox:CE', 'audiobox:PQ', 'dnsmos:P808_MOS']

def audio16(wave):
    return resample_poly(np.asarray(wave, dtype=np.float32), 1, 3).astype('<f4')

class Scorer:
    def __init__(self, device, asr_url=None, load_asr=True):
        from ears_reward import EarsModels
        self.device = device
        self.ears = EarsModels(device)
        self.ears.load('medium')
        self.url = asr_url
        self.asr_microbatch = 16
        if not load_asr:
            return
        if not self.url:
            from transformers import ParakeetProcessor, ParakeetForTDT
            self.proc = ParakeetProcessor.from_pretrained(os.environ.get('PODCAST_ASR','nvidia/parakeet-tdt-0.6b-v3'), local_files_only=True)
            self.asr = ParakeetForTDT.from_pretrained(os.environ.get('PODCAST_ASR','nvidia/parakeet-tdt-0.6b-v3'), dtype=torch.bfloat16,
                                                    local_files_only=True).to(device).eval()
        else:
            import httpx
            self.client = httpx.Client(timeout=120)

    @torch.inference_mode()
    def transcribe(self, waves16):
        tick = time.perf_counter()
        if self.url:
            results = []
            for wave in waves16:
                r = self.client.post(self.url, content=wave.tobytes())
                r.raise_for_status()
                report = r.json()
                assert report['model'] == 'parakeet_v3'
                results.append(report['text'])
        else:
            def decode(group):
                inp = self.proc(group, sampling_rate=16000, return_tensors='pt', padding=True)
                inp = {k: v.to(self.device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                       for k, v in inp.items()}
                out = self.asr.generate(**inp, max_new_tokens=2048)
                return self.proc.batch_decode(out.sequences)
            results=[];position=0
            while position<len(waves16):
                batch=min(self.asr_microbatch,len(waves16)-position)
                try:
                    results.extend(decode(waves16[position:position+batch]))
                    position+=batch
                except torch.cuda.OutOfMemoryError:
                    if batch==1:raise
                    self.asr_microbatch=max(1,batch//2)
                    torch.cuda.empty_cache()
                    print('ASR memory fallback: microbatch',self.asr_microbatch,'(candidate N unchanged)',flush=True)
        return [re.sub(r'<[^>]+>', '', s).strip() for s in results], time.perf_counter()-tick

    @torch.inference_mode()
    def quality(self, waves16):
        tick = time.perf_counter()
        model, extractor, norm = self.ears.models['medium']
        indices = [norm['score_names'].index(k) for k in KEYS]
        f = extractor(waves16, sampling_rate=16000, padding='longest', truncation=True,
                      max_length=480000, return_attention_mask=True, return_tensors='np')
        mel = f['input_features'].astype(np.float32)
        mask = f['attention_mask'].astype(np.int64)
        if mel.shape[-1] % 2:
            mel = np.pad(mel, ((0,0), (0,0), (0,1)))
            mask = np.pad(mask, ((0,0), (0,1)))
        with torch.cuda.device(self.device), torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(torch.from_numpy(mel).to(self.device),
                        torch.from_numpy(mask).to(self.device), predict_events=False)
        raw = out['scores'][:,indices].float().cpu().numpy()*np.asarray(norm['score_std'])[indices] + np.asarray(norm['score_mean'])[indices]
        assert np.isfinite(raw).all()
        return [dict(zip(KEYS, map(float, row))) for row in raw], time.perf_counter()-tick

    def score(self, waves, text, finals=None):
        tick = time.perf_counter()
        # Prefixes are at most the per-turn cap (<30s); do not silently truncate ASR.
        waves16 = [audio16(w) for w in waves]
        waves16 = [np.pad(w, (0, max(0,4800-len(w)))) for w in waves16]
        quality, quality_s = self.quality(waves16)
        hypotheses, asr_s = self.transcribe(waves16)
        finals = finals if finals is not None else [True]*len(waves)
        scores = []
        for q, hyp, final in zip(quality, hypotheses, finals):
            e = errors(text, hyp, final)
            scores.append({'ears': q, 'asr': hyp, 'final': bool(final), **e,
                           'reward': reward(q[KEYS[0]], e['wer']) if e['hypothesis_words'] else 0.})
        return scores, {'ears_seconds': quality_s, 'asr_seconds': asr_s,
                        'asr_microbatch': 1 if self.url else min(self.asr_microbatch,len(waves)),
                        'scoring_seconds': time.perf_counter()-tick}
