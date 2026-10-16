"""Conversion ownership, seam timing and post-VC rewards; synthetic models."""
import json
from types import SimpleNamespace
import threading

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from pydantic import ValidationError

from voice_conversion import ConversionSession,common_pcm_prefix


def session(transform=lambda w:w+.3):
    manager=SimpleNamespace(lock=threading.RLock(),cb=None,
        infer=lambda w,conditioning,seed:transform(w).copy())
    return ConversionSession(manager,777)


def test_crossfade_never_drops_time_or_repeats_the_new_words():
    vc=session(lambda w:w)
    wave=np.linspace(-.2,.2,6*76800+1200,dtype=np.float32)
    out=vc.full([wave],streaming=True)[0]
    assert len(out)==len(wave)
    assert np.allclose(out,wave,atol=1e-7)


def test_small_initial_chunks_are_buffered_and_short_eos_is_converted():
    vc=session();state=vc.new_stream()
    parts=[vc.chunk(np.full(7680,.1,np.float32),state) for _ in range(2)]
    assert not any(len(p) for p in parts)
    parts.append(vc.chunk(np.zeros(0,np.float32),state,final=True))
    out=np.concatenate(parts)
    assert len(out)==15360 and np.allclose(out,.4)
    assert state.done
    with pytest.raises(RuntimeError):vc.chunk(np.ones(10),state)


def test_branch_forks_do_not_share_context_tail_or_pending():
    vc=session();parent=vc.new_stream();vc.chunk(np.ones(76800,np.float32),parent)
    child=parent.fork(123)
    for name in ('context','tail','pending'):
        assert not np.shares_memory(getattr(parent,name),getattr(child,name))
    child.context[:]=.2
    assert np.all(parent.context==1)
    a=np.array([1,2,3,4],np.float32);b=np.array([1,2,8,4],np.float32)
    assert common_pcm_prefix([a,b],[0,1],4)==2
    assert common_pcm_prefix([a,b],[0],3)==3


@pytest.mark.parametrize('transform',[lambda w:np.zeros(0),lambda w:w*np.nan,lambda w:np.zeros(len(w)*2)])
def test_invalid_conversion_never_falls_back_to_raw_source(transform):
    with pytest.raises(RuntimeError):session(transform).full([np.ones(48000,np.float32)])


def test_vc_requires_reference_and_is_optional():
    from small_app import Turn
    assert not Turn(message='Hello').voice_conversion
    assert not Turn(message='Hello',voice_conversion=False).voice_conversion
    assert not Turn(message='Hello',voice_mode='none').voice_conversion
    assert Turn(message='Hello',voice_conversion=True).voice_conversion
    with pytest.raises(ValidationError):Turn(message='Hello',voice_mode='none',voice_conversion=True)


@pytest.mark.parametrize('mode,streaming',[('single',True),('single',False),('best',False),('all',True),
    ('search_winner',False),('search_beam',False)])
def test_every_mode_aligns_scores_and_serves_converted_audio(tmp_path,monkeypatch,mode,streaming):
    import small_app
    from storage import AudioStore
    import beam_search
    vc=session();received=[];scored=[]
    def prepare(path,seed):
        assert sf.info(path).duration==1
        received.append(str(path));return vc
    def generate(spec,n,*args,**kwargs):
        waves=[np.ones(76800*2,np.float32)*(.1+i*.05) for i in range(n)]
        if kwargs.get('on_audio'):
            for chunk in [waves[0][:76800],waves[0][76800:]]:kwargs['on_audio'](chunk)
        return waves
    def finish(waves,*args):
        assert all(float(w.min())>=.39 for w in waves)
        return waves,[{} for w in waves]
    def score(waves,*args):
        assert all(float(w.min())>=.39 for w in waves)
        scored.extend(waves)
        return [{'rank':i,'sec':len(w)/48000,'reward':1.,'wer':0.,'prefix_wer':0.,
                 'hyp':'Hello world','clap':.2,'genuineness':3.,'blend':5.} for i,w in enumerate(waves)]
    def search(self,spec,n,width,chunk,seconds,g,t,seed,ref,commit,conversion=None):
        assert conversion is vc
        # Exercise the API's converted-search contract; actual GPU search is
        # checked separately, not claimed by this synthetic integration test.
        converted=conversion.full([np.full(76800*2,.1,np.float32)],streaming=True)[0]
        self.scorer.score([converted],spec['text'],[True])
        commit(converted,{'step':1,'committed_until_seconds':3.2,'retained_rows':[0],
            'scores':[{'prefix_wer':0.,'hyp':'Hello world'}]})
        return converted,{'n':n,'search_rtf':1.}
    monkeypatch.setattr(small_app,'vc_manager',SimpleNamespace(prepare=prepare))
    monkeypatch.setattr(small_app,'store',AudioStore(tmp_path))
    monkeypatch.setattr(small_app,'STATE',{'phase':'ready','busy':False,
        'engine':SimpleNamespace(generate=generate,last_performance=None),
        'scorers':SimpleNamespace(score=score),'finisher':SimpleNamespace(finish=finish)})
    import search_scoring
    monkeypatch.setattr(search_scoring,'SearchScorer',lambda *args:SimpleNamespace(score=score))
    monkeypatch.setattr(beam_search.ChunkSearch,'generate',search)
    client=TestClient(small_app.app);owner='c'*32;client.cookies.set('small_session',owner)
    ref=small_app.store.new(owner,'Private fixture','reference','A')
    sf.write(ref.path,np.ones(48000)*.1,48000)
    data={'message':'CAPTION: Quiet.\nTRANSCRIPT: "Hello world"','input_mode':'manual','format':'caption',
        'generation':mode,'streaming':streaming,'voice_conversion':True,'voice_mode':'custom',
        'reference_id':ref.id,'align':False,'speaker':{'method':'off'},'adapters':{'id':None},'n':2,'ranking':{'mode':'legacy'}}
    response=client.post('/api/turn',json=data)
    events=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert not [e for e in events if e['event']=='error']
    result=next(e for e in events if e['event']=='result')
    assert result['voice_conversion']['enabled'] and result['voice_conversion']['conversion_calls']>0
    assert received==[str(ref.path)] and scored
    if mode=='single' and streaming:
        chunks=[e for e in events if e['event']=='audio_chunk']
        assert chunks and events.index(chunks[0])<events.index(result)
    for candidate in result['candidates']:
        audio=client.get(candidate['url'])
        assert audio.status_code==200 and audio.headers['content-type']=='audio/mpeg'
    assert vc.conditioning is None and not small_app.STATE['busy']
