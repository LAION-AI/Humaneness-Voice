"""Real GPU checks of official native scoring paths; no live uploads used."""
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModel

import settings
from adaptive_reward import CATALOG, RewardPlan
from dimension_predictors import DimensionPredictors


def main():
    device='cuda:1'
    root=snapshot_download(CATALOG['voicenet_repo'],revision=CATALOG['voicenet_revision'],
        allow_patterns=['voiceclap_commercial/*','predict.py'],cache_dir=settings.HF_CACHE+'/hub')
    tower=AutoModel.from_pretrained(str(Path(root)/'voiceclap_commercial'),trust_remote_code=True,
                                  dtype=torch.float32).to(device).eval()
    shared=hf_hub_download('laion/voiceclap-commercial','model.safetensors',
        revision='4e0170bf9a7e29704dd3a26ca92855b3cdd8706c',cache_dir=settings.HF_CACHE+'/hub')
    with open(shared,'rb') as stream:shared_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    with open(Path(root)/'voiceclap_commercial/model.safetensors','rb') as stream:
        bundled_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    assert shared_hash==bundled_hash,'Cannot reuse a different VoiceCLAP tower'
    plan=RewardPlan(terms=[{'dimension':'vn:AROU','objective':'target','target':5},
        {'dimension':'vn:S_RANT'},{'dimension':'eiv:Anger','objective':'target','target':3},
        {'dimension':'eiv:score_speech_quality'}])
    models=DimensionPredictors(device)
    models.prepare(plan)
    sample=sorted((settings.HERE/'search/raw').glob('quality-*anger_en-*.wav'))[0]
    wave,sr=sf.read(sample,dtype='float32')
    audio=torchaudio.functional.resample(torch.tensor(wave,device=device),sr,16000)
    import_spec=importlib.util.spec_from_file_location('upstream_voicenet_predict',Path(root)/'predict.py')
    upstream=importlib.util.module_from_spec(import_spec); import_spec.loader.exec_module(upstream)
    with torch.inference_mode():
        emb=tower.encode_waveform(audio[None],sample_rate=16000).float()
        started=time.perf_counter()
        outputs=models.predict([audio,audio],emb.repeat(2,1),plan)
        wall=time.perf_counter()-started
        diffs={}
        for key in ['vn:AROU','vn:S_RANT']:
            path=hf_hub_download(CATALOG['voicenet_repo'],f'regression/{key[3:]}.pt',
                revision=CATALOG['voicenet_revision'],cache_dir=settings.HF_CACHE+'/hub')
            reference=upstream._load_head(path,device)
            value=float(reference['net']((emb-reference['mu'])/reference['sd']).squeeze())
            diffs[key]=abs(value-outputs[0][key])
        features=models.extractor(audio.cpu().numpy(),sampling_rate=16000,return_tensors='pt').input_features
        hidden=models.encoder(features.to(device)).last_hidden_state.float()
        for key in ['eiv:Anger','eiv:score_speech_quality']:
            meta=next(d for d in CATALOG['dimensions'] if d['id']==key)
            path=hf_hub_download(CATALOG['eiv_repo'],meta['filename'],revision=CATALOG['eiv_revision'],
                                 cache_dir=settings.HF_CACHE+'/hub')
            state={k.removeprefix('_orig_mod.'):v.to(device) for k,v in torch.load(path,map_location='cpu',weights_only=True).items()}
            value=hidden.flatten(1) if state['proj.weight'].shape[1]>3072 else torch.cat(
                [hidden.mean(1),hidden.min(1).values,hidden.max(1).values,hidden.std(1)],-1)
            value=torch.nn.functional.linear(value,state['proj.weight'],state['proj.bias'])
            for i in [2,5,8,11]:
                value=torch.nn.functional.linear(value.relu(),state[f'mlp.{i}.weight'],state[f'mlp.{i}.bias'])
            diffs[key]=abs(float(value.squeeze())-outputs[0][key])
        assert max(diffs.values())<.0002,diffs
        assert outputs[0]==outputs[1]
    result={'source_audio':sample.name,'device':device,'batch':2,'voicenet_activation':'GELU',
        'eiv_activation':'ReLU','eiv_encoder':CATALOG['encoder_repo'],'encoder_revision':CATALOG['encoder_revision'],
        'shared_tower_sha256':shared_hash,'independent_native_score_abs_differences':diffs,
        'dimension_inference_wall_seconds':wall,'scores':outputs,
        'note':'Numerical smoke, not a quality study or isolated complete search throughput measurement.'}
    (settings.HERE/'test-results/adaptive-validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
