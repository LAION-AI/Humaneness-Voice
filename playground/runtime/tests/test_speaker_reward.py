from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alignment import LiveCutoff
from speaker_reward import SpeakerOptions, IDENTITY_KEYS, identity_vector, combine_identity, SpeakerReward


def test_defaults_and_closed_speaker_options():
    from small_app import Turn
    turn=Turn(message='Hello')
    assert turn.voice_mode=='profile' and turn.voice_profile=='emolia_c1699'
    assert turn.speaker.method=='ecapa' and turn.speaker.weight==.6
    assert turn.search_chunk_frames==10 and turn.chunk_frames==10
    assert turn.align
    assert Turn(message='x',reference_id='legacy').voice_mode=='custom'
    for kwargs in ({'method':'arbitrary'}, {'weight':float('nan')},{'weight':1.1},{'extra':'ignored'}):
        with pytest.raises(ValueError):SpeakerOptions(**kwargs)


def test_identity_vector_uses_fixed_centered_native_scales_and_no_style_axes():
    assert len(IDENTITY_KEYS)==15
    assert {'vn:AGEV','vn:GEND','vn:REGS','vn:R_CHST','vn:R_THRT'}<=set(IDENTITY_KEYS)
    assert not {'vn:AROU','vn:VALN','vn:TEMP','vn:RANG','vn:S_RANT'} & set(IDENTITY_KEYS)
    raw=torch.tensor([[0.]*15,[3.]*15,[6.]*15])
    centered=identity_vector(raw)
    assert torch.equal(centered[0],torch.full((15,),-1.))
    assert torch.equal(centered[1],torch.zeros(15))
    assert torch.equal(centered[2],torch.ones(15))
    with pytest.raises(ValueError):identity_vector(torch.full((1,15),float('nan')))


@pytest.mark.parametrize('weight',[0.,.34,.7,1.])
def test_exact_style_identity_weights_and_wer_gate(weight):
    item={'n_clap':.8,'quality':.1,'gate':.75,'style_score':.6}
    report={'available':True,'cosine':.2,'normalized':.6}
    result=combine_identity(item.copy(),report,weight)
    assert result['quality']==pytest.approx((1-weight)*.8+weight*.6)
    assert result['reward']==pytest.approx(result['quality']*.75)
    result=combine_identity(item.copy(),report,weight,adaptive=True)
    assert result['quality']==pytest.approx(.6)
    result=combine_identity(item.copy(),{'available':False,'normalized':None,'cosine':None},weight)
    assert result['quality']==.8 and result['speaker_weight']==0
    assert result['prompt_weight']==1


def test_chunk_scoring_truncates_candidates_not_fixed_anchor_and_masks_short_speech():
    calls=[]
    class Models:
        lock=__import__('threading').RLock()
        def target(self,path,method):return torch.tensor([1.,0.]),None
        def embed(self,waves,method,embeddings):
            calls.append(([len(w) for w in waves],embeddings))
            return torch.tensor([[1.,0.]]*len(waves)),None
    reward=SpeakerReward(Models(),SpeakerOptions(method='voicenet'),'fixed.wav',.8,'fixed profile anchor')
    waves=[torch.ones(32000),torch.ones(6000),torch.zeros(32000),torch.ones(32000)]
    scores=reward.score(waves,torch.ones(4,2),[True,True,True,False])
    assert calls[0][0]==[12800] and calls[0][1] is None
    assert scores[0]['cosine']==1 and scores[0]['normalized']==1
    assert all(not s['available'] for s in scores[1:])
    reward.window_seconds=None
    reward.score([waves[0]],torch.ones(1,2),[True])
    assert calls[-1][0]==[32000] and calls[-1][1] is not None


def test_live_fades_preserve_speech_and_stop_only_after_complete_text():
    raw=np.ones(38400,np.float32)
    guard=LiveCutoff({'text':'Hello world'})
    prefix={'stable_hyp':'Hello','prefix_wer':0.,'timestamps':[{'word':'Hello','start':.5,'end':1.}]}
    assert len(guard.feed(raw,prefix))==0 and not guard.done
    full={'stable_hyp':'Hello world','prefix_wer':0.,'timestamps':[
        {'word':'Hello','start':.5,'end':1.},{'word':'world','start':1.,'end':1.3}]}
    output=guard.feed(raw,full)
    assert guard.done and guard.report['cut_s']==pytest.approx(1.42)
    assert len(output)==int(1.42*48000)+int(.15*48000)
    assert np.all(output[:19200]==0) and output[-1]==0
    assert np.all(output[24000:62400]==1) and np.all(raw==1)
    assert len(guard.feed(raw,full))==0 and len(guard.finish(raw))==0


def test_live_cutoff_protects_closing_burst_and_cannot_retract_committed_audio():
    guard=LiveCutoff({'text':'Hello','edge_script':'Hello (sigh, 0.5 seconds)'})
    score={'stable_hyp':'Hello','prefix_wer':0.,'timestamps':[{'word':'Hello','start':0.,'end':1.3}]}
    out=guard.feed(np.ones(76800,np.float32),score)
    assert not guard.done
    out2=guard.feed(np.ones(76800,np.float32),score)
    assert guard.done and guard.report['cut_s']==pytest.approx(2.54)
    assert (len(out)+len(out2))/48000==pytest.approx(2.69,abs=1/48000)
    guard=LiveCutoff({'text':'Hello world'})
    prefix={'stable_hyp':'Hello','timestamps':[]}
    out=guard.feed(np.ones(96000,np.float32),prefix)
    assert not guard.done
    assert len(guard.finish(np.ones(120000,np.float32)))+len(out)==120000


def test_live_late_alignment_fades_held_tail_instead_of_abrupt_stop():
    guard=LiveCutoff({'text':'Hello world'})
    out=guard.feed(np.ones(144000,np.float32),{'stable_hyp':'Hello','timestamps':[]})
    assert len(out)==96000
    tail=guard.finish(np.ones(48000,np.float32))
    assert guard.report['fallback_tail_fade']
    assert len(tail)==7200 and tail[0]==1 and tail[-1]==0
    assert np.all(np.diff(tail)<=0)


def test_real_search_score_uses_identity_mix_before_optional_burst(monkeypatch):
    from search_scoring import SearchScorer
    retriever=SimpleNamespace(device='cpu',qmu=np.zeros(3),model=SimpleNamespace(
        encode_waveform=lambda batch,**kw:torch.tensor([[1.,0.,0.]]*len(batch))))
    judge=SimpleNamespace(want_vector=lambda *a:np.array([1.,0.,0.]),
        gen=SimpleNamespace(_predict=lambda e:torch.full((len(e),),3.)),
        bl=SimpleNamespace(mu=torch.zeros(3),sd=torch.ones(3),mlp=lambda x:torch.full((len(x),1),5.)))
    identity=SimpleNamespace(options=SpeakerOptions(method='voicenet',weight=.34),window_seconds=None,last_timing={},score=lambda waves,*a:
        [{'available':True,'method':'voicenet','cosine':.5,'normalized':.75} for _ in waves])
    scorer=SearchScorer(SimpleNamespace(asr=None,retriever=retriever,judge=judge),
        {'general':'warm delivery','text':'Hello'},speaker_reward=identity)
    monkeypatch.setattr(scorer,'transcribe_batch',lambda audio:(['Hello']*len(audio),
        [[{'word':'Hello','start':0.,'end':.4}]]*len(audio)))
    score=scorer.score([np.ones(48000,np.float32)],'Hello',[True])[0]
    assert score['quality']==pytest.approx(.66+.34*.75)
    assert score['reward']==score['quality'] and score['speaker_weight']==.34
