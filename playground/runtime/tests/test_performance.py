import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from performance_core import combine_branches
from metrics import word_errors, tokens


def test_left_padding_keeps_audio_pad_and_exact_mask():
    first = torch.tensor([[[11, 4], [12, 5]]])
    second = torch.tensor([[[20, 6]]])
    ids, mask = combine_branches([(first, torch.ones(1, 2, dtype=torch.bool)),
                                  (second, torch.ones(1, 1, dtype=torch.bool))], 0, -1)
    assert ids.tolist() == [[[11, 4], [12, 5]], [[0, -1], [20, 6]]]
    assert mask.tolist() == [[True, True], [False, True]]


@pytest.mark.parametrize("ref,hyp,expected", [("one two three", "one four three", 1/3),
    ("one two", "one", .5), ("one two", "one two three", .5),
    ("Das wäre schön!", "das wäre schön", 0), ("We did it", "", 1), ("one", "two three", 2)])
def test_conventional_wer(ref, hyp, expected):
    assert word_errors(ref, hyp)["wer"] == pytest.approx(expected)


def test_german_letters_and_apostrophes():
    assert tokens("Äußere Grüße, don’t.") == ["äussere", "grüsse", "don't"]


def test_report_routes_never_expose_references_or_runtime():
    from fastapi.testclient import TestClient
    import small_app
    client = TestClient(small_app.app)
    assert client.get("/performance/").status_code == 200
    assert client.get("/performance/sweep.json").status_code == 200
    assert client.get("/performance/benchmark-reference.wav").status_code == 404
    assert client.get("/study/reference-b.wav").status_code == 404
    assert client.get("/study/design.json").status_code == 200
    assert client.get("/study/audio/private.wav").status_code == 404


def test_realization_report_and_all_public_audio_links():
    import re
    from fastapi.testclient import TestClient
    import small_app
    client = TestClient(small_app.app)
    response = client.get("/realization-check.html")
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/html')
    assert response.headers['cache-control'] == 'no-store'
    assert 'Fresh CPU model inference' in response.text
    audio_links = re.findall(r'<audio[^>]*src="\.\./(search/audio/[^"]+)"', response.text)
    assert len(audio_links) == 20
    for link in audio_links:
        audio = client.get('/' + link)
        assert audio.status_code == 200
        assert audio.headers['content-type'].startswith('audio/mpeg')
    assert client.get('/test-results/inprocess-tests.json').status_code == 404
    assert client.get('/test-results/realization-check.json').status_code == 404


def test_study_recommendation_enforces_both_predeclared_constraints():
    from reports import study_summary
    def row(g,t,wer,genuine,emotion):
        return {'stage':'main','format':'caption','guidance':g,'temperature':t,'seed':777,
            'emotion':'Anger','raw_word_errors':{'wer':wer},'processed_word_errors':{'wer':wer},
            'genuineness':genuine,'emotion_percentile':emotion,'emotion_rank':1,
            'target_emotion_score':emotion,'clap':emotion,'hit_frame_limit':False,'alignment':{'applied':True}}
    rows=[row(1.5,1,.02,2,.5),row(2,1.2,.07,1.9,.8),row(3,1,.11,3,1),row(1,1,.01,1.4,.9)]
    summary=study_summary({'main_samples':4,'rows':rows,'selection_rule':'test'},rows)
    chosen=summary['recommendations']['caption']['chosen']
    assert chosen['cfg']==2 and chosen['temperature']==1.2
