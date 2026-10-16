import numpy as np
import pytest
import torch

from search_scoring import prefix_errors, search_reward, token_spans_to_words, style_query
from beam_search import expansion_indices, common_prefix, reorder_codec


def test_unspoken_suffix_is_not_a_deletion():
    assert prefix_errors('one two three four','one two')['prefix_wer']==0
    assert prefix_errors('one two three four','one three')['prefix_wer']>0
    assert prefix_errors('one two three four','one two',final=True)['prefix_wer']==.5
    assert prefix_errors('one two','')['prefix_wer']==1
    assert prefix_errors('one two','nonsense')['prefix_wer']==1


def test_reward_finite_zero_wer_and_requested_weights():
    score={'clap':1.,'genuineness':6.,'blend':10.,'prefix_wer':0.}
    assert search_reward(score)['reward']==1.
    score['prefix_wer']=2
    assert search_reward(score)['reward']==0
    score={'clap':-1.,'genuineness':6.,'blend':10.,'prefix_wer':0.}
    assert search_reward(score)['reward']==.5


def test_branching_budget_and_irreversible_prefix():
    for n in range(1,17):
        winners=[3] if n==1 else [3,5]
        children=expansion_indices(winners,n)
        assert len(children)==n
        assert children.count(3)==(n if n==1 else (n+1)//2)
        assert children.count(5)==(0 if n==1 else n//2)
    assert expansion_indices([3,5],8)==[3]*4+[5]*4
    assert expansion_indices([3],4)==[3]*4
    assert expansion_indices([1,2],3)==[1,1,2]
    assert expansion_indices([1,2],7)==[1]*4+[2]*3
    with pytest.raises(ValueError): expansion_indices([1,2],1)
    codes=torch.ones(4,3,12,dtype=torch.long)
    codes[1,1,0]=2
    assert common_prefix(codes,[0,1])==1
    assert common_prefix(codes,[0])==3


def test_codec_cloning_does_not_alias_rows():
    from types import SimpleNamespace
    state=SimpleNamespace(cached_keys=torch.arange(12).reshape(4,3),offset=torch.arange(4))
    class Codec:
        def modules(self): return [SimpleNamespace(_streaming_state=state)]
    reorder_codec(Codec(),torch.tensor([2,2,1,1]),4)
    assert state.offset.tolist()==[2,2,1,1]
    state.cached_keys[0,0]=99
    assert state.cached_keys[1,0]!=99


def test_word_timestamps_merge_pieces():
    spans=[{'token':' You','start':0.,'end':.1},{'token':' came','start':.12,'end':.3},
           {'token':' back','start':.31,'end':.4},{'token':'.','start':.4,'end':.4}]
    words=token_spans_to_words(spans)
    assert [x['word'] for x in words]==['You','came','back.']
    assert words[-1]['end']==.4


def test_search_api_bounds_and_modes():
    from small_app import Turn
    for mode in ['search_winner','search_beam']:
        for model,limit in [('large_acting',4),('small_acting',8)]:
            for n in range(1,limit+1):
                assert Turn(message='hello',model=model,generation=mode,search_n=n).search_n==n
            with pytest.raises(ValueError):Turn(message='hello',model=model,generation=mode,search_n=limit+1)
    with pytest.raises(ValueError):Turn(message='hello',search_n=0)
    with pytest.raises(ValueError):Turn(message='hello',search_n=17)
    with pytest.raises(ValueError):Turn(message='hello',search_chunk_frames=5)
    with pytest.raises(ValueError):Turn(message='hello',search_seconds=0)


def test_style_query_does_not_score_the_spoken_words():
    query=style_query({'general':'Angry delivery. Say "You betrayed me."',
        'text':'You betrayed me.','tagged':'(tight jaw) You betrayed me.'})
    assert 'betrayed' not in query and 'Angry' in query and 'tight jaw' in query


def test_search_api_streams_private_mp3_before_final(tmp_path,monkeypatch):
    import json
    import small_app
    import beam_search
    import search_scoring
    from storage import AudioStore
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    class Scorers:
        def score(self,waves,spec):
            return [{'rank':0,'reward':1.,'sec':len(waves[0])/48000,'wer':0.,'hyp':spec['text'],
                     'genuineness':1.,'blend':1.,'clap':.2}]
    class Finisher:
        def finish(self,waves,spec,align):return waves,[{'enabled':False,'available':False,'applied':False,'raw_seconds':.8,'seconds':.8}]
    wave=.1*np.sin(2*np.pi*220*np.arange(38400)/48000).astype('float32')
    def generate(self,spec,n,width,chunk,seconds,g,t,seed,ref,commit):
        assert n==4 and width==2 and chunk==20
        commit(wave,{'step':1,'committed_until_seconds':.8,'retained_rows':[0]})
        return wave,{'search_rtf':.5,'n':n}
    class SearchScores:
        def score(self,waves,text,final):
            return [{'prefix_wer':0.,'reward':1.,'sec':len(waves[0])/48000,'hyp':text,
                     'genuineness':1.,'blend':1.,'clap':.2}]
    monkeypatch.setattr(search_scoring,'SearchScorer',lambda *args:SearchScores())
    monkeypatch.setattr(beam_search.ChunkSearch,'generate',generate)
    monkeypatch.setattr(small_app,'store',AudioStore(tmp_path))
    monkeypatch.setitem(small_app.STATE,'engine',object())
    monkeypatch.setitem(small_app.STATE,'scorers',Scorers())
    monkeypatch.setitem(small_app.STATE,'finisher',Finisher())
    monkeypatch.setitem(small_app.STATE,'phase','ready')
    monkeypatch.setitem(small_app.STATE,'busy',False)
    a,b=TestClient(small_app.app),TestClient(small_app.app)
    response=a.post('/api/turn',json={'message':'CAPTION: warm\nTRANSCRIPT: "hello"',
        'input_mode':'manual','format':'caption','generation':'search_beam','search_n':4,'search_chunk_frames':20,
        'adapters':{'id':None},'voice_mode':'none','speaker':{'method':'off'},'ranking':{'mode':'legacy'}})
    events=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert [e['event'] for e in events].index('search_chunk') < [e['event'] for e in events].index('result')
    chunk=next(e for e in events if e['event']=='search_chunk')
    assert '_wave' not in chunk
    assert a.get(chunk['url']).headers['content-type']=='audio/mpeg'
    assert b.get(chunk['url']).status_code==404
    assert events[-1]['search_performance']['search_rtf']==.5
    assert not small_app.STATE['busy']
