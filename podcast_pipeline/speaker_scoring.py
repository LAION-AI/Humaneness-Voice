"""Exactly 50% native-normalized genuineness + 50% reference ECAPA cosine.

WER is a final diagnostic, never a multiplier or candidate eligibility gate.
Reference embeddings are computed once and reused. No per-batch normalization.
"""
import math
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from recipe import references
from scoring import Scorer, audio16

ECAPA_REPO = 'speechbrain/spkrec-ecapa-voxceleb'
ECAPA_REVISION = '0f99f2d0ebe89ac095bcc5903c4dd8f72b367286'
FORMULA = '0.5*clip(genuineness/6,0,1) + 0.5*clip(ecapa_cosine,0,1)'

def speaker_reward(genuineness, cosine):
    if not all(math.isfinite(x) for x in (genuineness, cosine)):
        raise ValueError('Non-finite speaker reward')
    return .5*max(0., min(1., genuineness/6.)) + .5*max(0., min(1., cosine))

class SpeakerScorer(Scorer):
    def __init__(self, device, reference_config, load_asr=False,asr_url=None):
        super().__init__(device, asr_url=asr_url, load_asr=load_asr)
        from speechbrain.inference.speaker import EncoderClassifier
        from huggingface_hub import snapshot_download
        import settings
        root = snapshot_download(ECAPA_REPO, revision=ECAPA_REVISION,
            cache_dir=str(Path(settings.HF_CACHE)/'hub'), local_files_only=True,
            allow_patterns=['hyperparams.yaml','*.ckpt','label_encoder.txt'])
        self.ecapa = EncoderClassifier.from_hparams(source=root,
            overrides={'pretrained_path':root},
            savedir=str(Path(settings.HF_CACHE)/'small-ecapa-pinned'),
            run_opts={'device':device})
        self.ecapa.eval()
        self.targets={}
        for speaker,path in references(reference_config).items():
            x,sr=sf.read(path,dtype='float32',always_2d=True)
            divisor=math.gcd(sr,16000)
            x=resample_poly(x.mean(1),16000//divisor,sr//divisor).astype(np.float32)
            self.targets[speaker]=self.embed([x])[0]
        self.reference_seconds={s:sf.info(p).duration for s,p in references(reference_config).items()}

    @torch.inference_mode()
    def embed(self,waves16):
        vectors=[]
        # ECAPA's activation footprint grows with duration. Bound just this stage.
        for offset in range(0,len(waves16),4):
            group=waves16[offset:offset+4]
            length=max(len(x) for x in group)
            x=np.zeros((len(group),length),np.float32)
            for i,w in enumerate(group):x[i,:len(w)]=w
            lens=torch.tensor([len(w)/length for w in group],device=self.device)
            with torch.autocast('cuda',enabled=False):
                v=self.ecapa.encode_batch(torch.from_numpy(x).to(self.device),wav_lens=lens).reshape(len(group),-1).float()
            assert torch.isfinite(v).all() and bool((v.norm(dim=-1)>1e-8).all())
            vectors.append(torch.nn.functional.normalize(v,dim=-1))
        return torch.cat(vectors)

    @torch.inference_mode()
    def score_speakers(self,waves,speakers):
        assert len(waves)==len(speakers)
        tick=time.perf_counter()
        x=[audio16(w) for w in waves]
        x=[np.pad(w,(0,max(0,4800-len(w)))) for w in x]
        quality,ears_s=self.quality(x)
        start=time.perf_counter()
        vectors=self.embed(x)
        target=torch.stack([self.targets[s] for s in speakers])
        cosine=(vectors*target).sum(-1).clamp(-1,1).cpu().tolist()
        speaker_s=time.perf_counter()-start
        rows=[dict(ears=q,ecapa_cosine=float(c),genuineness_normalized=float(np.clip(q['genuineness_0_6']/6,0,1)),
            speaker_similarity_normalized=float(np.clip(c,0,1)),reward=speaker_reward(q['genuineness_0_6'],c),reference_speaker=s)
            for q,c,s in zip(quality,cosine,speakers)]
        return rows,dict(ears_seconds=ears_s,speaker_seconds=speaker_s,scoring_seconds=time.perf_counter()-tick)
