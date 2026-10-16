"""Quality scales, pure ranking semantics and API contracts; no model download."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest
import torch

from ears_reward import RankingOptions, EarsModels, METRICS, combine_ears, normalized


@pytest.mark.parametrize('metric',['ce','pq','cu','pc','mos','ovrl'])
def test_fixed_scale(metric):
    _,_,lo,hi=METRICS[metric]
    assert normalized(lo,metric)==0
    assert normalized(hi,metric)==1
    assert normalized((hi+lo)/2,metric)==.5
    assert normalized(-100,metric)==0 and normalized(100,metric)==1
    with pytest.raises(ValueError):normalized(float('nan'),metric)


@pytest.mark.parametrize('weight',[0,.25,.5,1])
def test_mixture_has_no_hidden_identity_genuine_blend_weights(weight):
    item={'quality':.99,'n_clap':.2,'gate':.75,'genuineness':6,'blend':10,'speaker_score':1}
    options=RankingOptions(weight=weight)
    result=combine_ears(item,{'raw':8.2,'normalized':.8,'available':True},options)
    assert result['quality']==pytest.approx((1-weight)*.2+weight*.8)
    assert result['reward']==pytest.approx(result['quality']*.75)
    assert result['ears_weight']==weight and result['clap_weight']==1-weight


def test_pure_sort_preserves_raw_order_and_ignores_wer_even_outside_scale():
    rewards=[]
    for raw in [10.2,10.9]:
        item={'quality':1,'n_clap':0,'gate':0}
        combine_ears(item,{'raw':raw,'normalized':1,'available':True},RankingOptions(mode='ears_only'))
        assert item['ranking_gate']==1 and item['clap_weight']==0
        rewards.append(item['reward'])
    assert rewards==[10.2,10.9]


def test_empty_unavailable_does_not_become_positive_quality():
    item={'quality':1,'n_clap':1,'gate':1}
    combine_ears(item,{'raw':None,'normalized':None,'available':False},RankingOptions())
    assert item['reward']==0 and not item['ranking_available']


def test_default_and_legacy_validation():
    from small_app import Turn
    assert Turn(message='Hello').ranking.model_dump()=={'mode':'ears_mix','model':'medium','metric':'ce','weight':.5}
    assert Turn(message='Hello',ranking={'mode':'legacy'},reward_mode='adaptive_auto').ranking.mode=='legacy'
    with pytest.raises(ValueError):Turn(message='Hello',reward_mode='adaptive_auto')
    for kwargs in [{'weight':float('nan')},{'weight':1.1},{'model':'large'},{'metric':'made_up'}]:
        with pytest.raises(ValueError):RankingOptions(**kwargs)


def test_all_audio_is_covered_and_windows_are_duration_weighted(monkeypatch):
    names=[x[0] for x in METRICS.values()]
    calls=[]
    def extractor(waves,**kwargs):
        calls.extend(len(w) for w in waves)
        # Encode a segment's constant level in the synthetic mel features.
        return {'input_features':np.asarray([[[w[0],w[0]]] for w in waves],np.float32),
                'attention_mask':np.ones((len(waves),2),np.int64)}
    def model(mel,mask,**kwargs):return {'scores':mel[:,0,:1].repeat(1,len(names))}
    m=EarsModels('cpu');m.models['medium']=(model,extractor,
        {'score_names':names,'score_mean':[1]*len(names),'score_std':[2]*len(names)})
    monkeypatch.setattr(m,'load',lambda size:None)
    wave=np.concatenate([np.ones(480000,np.float32),np.ones(16000,np.float32)*3])
    result=m.predict([wave,np.zeros(0,np.float32),np.ones(100,np.float32)])
    assert result[0]['window_count']==2 and result[0]['duration_seconds']==31
    assert result[0]['raw']==pytest.approx((3*30+7)/31)
    assert not result[1]['available'] and result[1]['raw'] is None
    assert result[2]['raw']==3 and calls==[480000,16000,1600]
    assert m.last_timing['ears_windows']==3


def test_search_scores_are_overwritten_by_ears_not_speaker(monkeypatch):
    import search_scoring
    from search_scoring import SearchScorer
    class Tower:
        def encode_waveform(self,audio,**kwargs):return torch.ones((len(audio),4))
    retriever=SimpleNamespace(device='cpu',model=Tower(),qmu=np.zeros(4))
    judge=SimpleNamespace(want_vector=lambda *args:np.ones(4),
        gen=SimpleNamespace(_predict=lambda embeddings:torch.ones(len(embeddings))*6),
        bl=SimpleNamespace(mu=torch.zeros(4),sd=torch.ones(4),mlp=lambda e:torch.ones(len(e),1)*10))
    ears=SimpleNamespace(load=lambda size:None,last_timing={'ears_seconds':.01},
        predict=lambda waves,size,metric:[{'available':True,'raw':5.5,'normalized':.5} for w in waves])
    scorers=SimpleNamespace(asr=None,retriever=retriever,judge=judge,ears=ears)
    scorer=SearchScorer(scorers,{'general':'Warm','tagged':'','text':'Hello'},ranking=RankingOptions(mode='ears_only'))
    monkeypatch.setattr(scorer,'transcribe_batch',lambda audio:(['Wrong word'],[[]]))
    item=scorer.score([np.ones(48000,np.float32)],'Hello',[True])[0]
    assert item['reward']==5.5 and item['ranking_gate']==1 and item['gate']==0
    assert item['ears']['raw']==5.5 and scorer.last_timing['ears_seconds']==.01
    with pytest.raises(ValueError):SearchScorer(scorers,{'general':'Warm','tagged':''},
        ranking=RankingOptions(),speaker_reward=object())
