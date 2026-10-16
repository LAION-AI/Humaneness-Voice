"""Pinned real-GPU numerical and timing checks; not a perceptual quality study."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModel, WhisperConfig, WhisperModel

import settings
from adaptive_reward import CATALOG
from burst_models import BurstModels, extract_events, math_frames
from burst_reward import (BurstOptions, DETECTOR_REPO, DETECTOR_REVISION,
    LOCATOR_REPO, LOCATOR_REVISION, WHISPER_REVISION, evaluate_bursts)


@torch.inference_mode()
def main():
    device='cuda:1'
    root=snapshot_download(CATALOG['voicenet_repo'],revision=CATALOG['voicenet_revision'],
        allow_patterns=['voiceclap_commercial/*'],cache_dir=settings.HF_CACHE+'/hub')
    tower=AutoModel.from_pretrained(str(Path(root)/'voiceclap_commercial'),trust_remote_code=True,
                                  dtype=torch.float32).to(device).eval()
    models=BurstModels(SimpleNamespace(device=device,model=tower));models.prepare(True)
    sources=sorted((settings.HERE/'search/raw').glob('quality-*burst_en-*.wav'))[:6]
    audio=[]
    for source in sources:
        wave,sr=sf.read(source,dtype='float32')
        audio.append(torchaudio.functional.resample(torch.tensor(wave,device=device),sr,16000))
    assert audio,'Missing local synthetic research samples'
    module_path=hf_hub_download(LOCATOR_REPO,'inference.py',revision=LOCATOR_REVISION,
                               cache_dir=settings.HF_CACHE+'/hub')
    spec=importlib.util.spec_from_file_location('upstream_burst_locator',module_path)
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    config_path=hf_hub_download('openai/whisper-small','config.json',revision=WHISPER_REVISION,
                               cache_dir=settings.HF_CACHE+'/hub')
    config=WhisperConfig.from_json_file(config_path);config._attn_implementation='sdpa'
    with patch.object(WhisperModel,'from_pretrained',side_effect=lambda *a,**k:WhisperModel(config)):
        reference=upstream.WhisperSegmenter()
    checkpoint=hf_hub_download(LOCATOR_REPO,'model_v2.pt',revision=LOCATOR_REVISION,
                              cache_dir=settings.HF_CACHE+'/hub')
    reference.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    reference.whisper.decoder=None;reference=reference.to(device).eval()
    padded=np.pad(audio[0].cpu().numpy(),(0,480000-len(audio[0])))
    features=models.extractor(padded,sampling_rate=16000,return_tensors='pt').input_features.to(device)
    difference=float((reference(features)-models.locator(features)).abs().max())
    assert difference<.0001,difference
    probabilities=models.probabilities(audio)
    for p in probabilities:
        native=upstream.extract_events(p,threshold=.5,merge_gap=.1,min_dur=.1)
        ours=extract_events(p)
        assert len(native)==len(ours),(native,ours)
        for expected,actual in zip(native,ours):
            assert np.max(np.abs(np.array(expected)-np.array([
                actual['start'],actual['end'],actual['confidence']])))<=.000501
    events=models.detect(audio,True)
    # Independent functional BatchNorm/GELU/Linear inference for all five
    # shipped heads, compared with our mean-logits-then-softmax ensemble.
    embedding=torch.nn.functional.normalize(tower.encode_waveform(audio[0][None],sample_rate=16000).float(),dim=-1)
    logits=[]
    for seed in range(5):
        path=hf_hub_download(DETECTOR_REPO,f'vocal_burst_mlp_prod_s{seed}.pt',
            revision=DETECTOR_REVISION,cache_dir=settings.HF_CACHE+'/hub')
        s={k:v.to(device) for k,v in torch.load(path,map_location='cpu',weights_only=False)['state_dict'].items()}
        value=torch.nn.functional.linear(embedding,s['net.0.weight'],s['net.0.bias'])
        value=torch.nn.functional.batch_norm(value,s['net.1.running_mean'],s['net.1.running_var'],
                                             s['net.1.weight'],s['net.1.bias'],training=False,eps=1e-5)
        value=torch.nn.functional.gelu(value)
        logits.append(torch.nn.functional.linear(value,s['net.4.weight'],s['net.4.bias']))
    reference_probs=torch.stack(logits).mean(0).softmax(-1)
    our_probs=torch.stack([head(embedding) for head in models.heads]).mean(0).softmax(-1)
    head_difference=float((reference_probs-our_probs).abs().max())
    assert head_difference<.000001
    found=[(i,e) for i,row in enumerate(events) for e in row]
    assert found,'Locator produced no events on six synthetic burst-recipe samples'
    i,observed=max(found,key=lambda pair:pair[1]['confidence'])
    target={'label':observed['top_label'],'start':observed['start'],
            'duration':observed['end']-observed['start']}
    options=BurstOptions(mode='location',source='manual',targets=[target],penalize_extra=False)
    aligned=evaluate_bursts([target],[observed],options,'',[],len(audio[i])/16000,True)['score']
    shifted=dict(observed,start=observed['start']+1.,end=observed['end']+1.)
    misaligned=evaluate_bursts([target],[shifted],options,'',[],len(audio[i])/16000+1,True)['score']
    assert aligned>0 and misaligned==0
    # Real 31.2-second PCM checks the second locator window and crop extent.
    long_audio=torch.cat([torch.zeros(480000,device=device),audio[i][:19200]])
    long_probs=models.probabilities([long_audio])[0]
    assert len(long_probs)==math_frames(len(long_audio)) and np.isfinite(long_probs).all()
    result={'device':device,'locator_revision':LOCATOR_REVISION,'detector_revision':DETECTOR_REVISION,
        'locator_logit_max_abs_difference':difference,'classifier_probability_max_abs_difference':head_difference,
        'v2_postprocessing':{'threshold':.5,'merge_gap':.1,'min_duration':.1},
        'source_samples':[p.name for p in sources],'detected_events':events,
        'position_control':{'aligned':aligned,'shifted_one_second':misaligned},
        'long_prefix_seconds':len(long_audio)/16000,'long_prefix_frames':len(long_probs),
        'note':'Numerical/smoke and self-derived position control only. Not independent detector accuracy, human listening, or quality-improvement evidence.'}
    (settings.HERE/'test-results/burst-validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='detected_events'}),flush=True)


if __name__=='__main__':main()
