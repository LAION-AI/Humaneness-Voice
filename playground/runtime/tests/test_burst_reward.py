import numpy as np
import pytest

from burst_models import extract_events
from burst_reward import BurstOptions, BurstReward, evaluate_bursts, script_targets, word_anchors


def event(start=.3,end=.6,label='Chuckle',prob=.9):
    return {'start':start,'end':end,'confidence':1.,'classes':['Chuckle','Breathy Giggle','no_burst'],
            'families':['laugh','laugh','no_burst'],'probabilities':[prob,.05,1-prob-.05],
            'top_label':label}


def target(start=.3,label='chuckle',duration=.3):
    return {'start':start,'label':label,'duration':duration}


def score(requests,events,final=True,seconds=2,**kwargs):
    options=BurstOptions(mode='typed',source='manual',targets=[target()],**kwargs)
    return evaluate_bursts(requests,events,options,'hello',[],seconds,final)


def test_position_type_and_one_to_one_matching():
    right=score([target()],[event()],match='exact')
    wrong_time=score([target()],[event(1.3,1.6)],match='exact')
    wrong_type=score([target()],[event(prob=.01)],match='exact')
    assert right['score']==pytest.approx(.9)
    assert right['score']>wrong_type['score']>wrong_time['score']
    duplicate=score([target(),target()],[event()],match='exact')
    assert duplicate['score']==pytest.approx(.45)
    assert sum(t['status']=='matched' for t in duplicate['targets'])==1


def test_future_targets_not_penalized_until_final():
    report=score([target(10)],[],final=False)
    assert report['score'] is None and report['pending_targets']==1
    assert score([target(10)],[])['score']==0


def test_family_match_and_unknown_type_are_transparent():
    exact=score([target()],[event()],match='exact')
    family=score([target()],[event()],match='family')
    assert family['score']>exact['score']
    unknown=score([target(label='coughing')],[event()],match='exact')
    assert unknown['targets'][0]['match_kind']=='unsupported_type_location_only'
    assert not unknown['targets'][0]['type_supported']


def test_no_targets_and_extra_events():
    assert score([],[])['score'] is None
    penalized=score([target()],[event(),event(1.1,1.4)],match='exact')
    assert penalized['extra_penalty']==.15
    assert penalized['score']==pytest.approx(.75)


def test_native_v2_short_event_and_no_padding_false_events():
    probabilities=np.zeros(50);probabilities[10:15]=.9
    events=extract_events(probabilities)
    assert len(events)==1 and events[0]['start']==.2 and events[0]['end']==.3
    assert extract_events(np.zeros(40))==[]


def test_overlapping_windows_preserve_events_after_thirty_seconds():
    import torch
    from types import SimpleNamespace
    from burst_models import BurstModels, math_frames
    models=BurstModels(SimpleNamespace(device='cpu'))
    models.extractor=lambda waves,**kw:SimpleNamespace(input_features=torch.tensor(np.stack(waves)))
    models.locator=lambda features:features.reshape(-1,1500,320).mean(-1)*40-20
    wave=np.zeros(499200,dtype=np.float32)  # 31.2 seconds, beyond first window
    wave[481600:488000]=1.  # 30.1–30.5 seconds
    probabilities=models.probabilities([wave])[0]
    assert len(probabilities)==math_frames(len(wave))
    events=extract_events(probabilities)
    assert len(events)==1
    assert events[0]['start']==pytest.approx(30.1)
    assert events[0]['end']==pytest.approx(30.5)


def test_script_clock_and_word_gap_anchors():
    spec={'tagged':'[0.3 seconds pause] [2 seconds duration] hello there (chuckle, 0.3 seconds) [1 seconds duration] my friend','text':'hello there my friend'}
    requests,first=script_targets(spec)
    assert requests[0]['word_index']==2 and requests[0]['start']==2.3 and first==.3
    stamps=[{'word':'hello','start':0,'end':.25},{'word':'there','start':.3,'end':.5},
            {'word':'my','start':1.,'end':1.2},{'word':'friend','start':1.3,'end':1.6}]
    words=BurstOptions(mode='location',timing='words')
    clock=BurstOptions(mode='location',timing='clock')
    located=[event(.6,.9)]
    assert evaluate_bursts(requests,located,words,spec['text'],stamps,2,True)['score']>0
    assert evaluate_bursts(requests,located,clock,spec['text'],stamps,2,True)['score']==0


def test_repeated_words_use_prefix_not_later_suffix():
    mapping=word_anchors('yes no yes',[{'word':'yes','start':0,'end':.2}])
    assert list(mapping)==[0]


def test_manual_script_durations_are_not_renderer_clamped():
    requests,_=script_targets({'tagged':'(chuckle, 2.0 seconds) hello (short sigh) (scream, 0.05 seconds)'})
    assert [t['duration'] for t in requests]==[2.,.14,.05]
    assert requests[1]['start']==pytest.approx(2.4)


def test_leading_burst_can_precede_asr_but_final_missing_words_fail():
    spec={'tagged':'[0.3 seconds pause] (chuckle, 0.4 seconds) [2 seconds duration] hello there','text':'hello there'}
    requests,first=script_targets(spec)
    options=BurstOptions(mode='location',timing='clock')
    prefix=evaluate_bursts(requests,[event(.3,.7)],options,spec['text'],[],.8,False,first)
    assert prefix['speech_not_due']
    assert not evaluate_bursts(requests,[event(.3,.7)],options,spec['text'],[],1.6,False,first)['speech_not_due']
    assert not evaluate_bursts(requests,[event(.3,.7)],options,spec['text'],[],.8,True,first)['speech_not_due']


def test_configuration_bounds_and_manual_time_basis():
    from small_app import Turn
    assert Turn(message='hello').burst.mode=='off'
    with pytest.raises(ValueError):BurstOptions(mode='typed',source='manual')
    with pytest.raises(ValueError):BurstOptions(weight=1.5)
    with pytest.raises(ValueError):BurstOptions(tolerance=0)
    with pytest.raises(ValueError):BurstOptions(targets=[target()]*9)
    class Models:
        def prepare(self,typed):pass
    reward=BurstReward(Models(),{'text':'hello'},BurstOptions(mode='location',source='manual',targets=[target()]))
    assert reward.options.timing=='clock'
