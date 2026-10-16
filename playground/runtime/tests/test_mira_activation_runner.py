"""Research vector provenance and strict public-artifact routes, without GPUs."""
import hashlib

import numpy as np
import pytest


def test_alternate_vector_bank_preserves_primary_and_records_hash(tmp_path, monkeypatch):
    import mira_activation_study as study
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    (tmp_path / 'vectors').mkdir()
    primary = tmp_path / 'vectors/selected.npz'
    np.savez(primary, layer_0__raw_mean=np.ones(1024, dtype=np.float32))
    original = primary.read_bytes()
    alternative = tmp_path / 'balanced.npz'
    np.savez(alternative, semantic_local__high_minus_all=np.zeros(2560, dtype=np.float32))
    bank, provenance = study.load_vectors('balanced.npz')
    assert bank['semantic_local__high_minus_all'].shape == (2560,)
    assert provenance['vector_file'] == 'balanced.npz'
    assert provenance['vector_sha256'] == hashlib.sha256(alternative.read_bytes()).hexdigest()
    assert primary.read_bytes() == original
    with pytest.raises(ValueError, match='inside this study'):
        study.load_vectors(tmp_path.parent / 'not-allowed.npz')


def test_mira_report_routes_expose_only_explicit_artifacts(tmp_path, monkeypatch):
    import small_app
    from fastapi.testclient import TestClient
    monkeypatch.setattr(small_app.settings, 'HERE', tmp_path)
    folder = tmp_path / 'mira-activation-study'
    (folder / 'vectors').mkdir(parents=True)
    (folder / 'index.html').write_text('<html>Research audio</html>')
    (folder / 'summary.json').write_text('{"takes":100}')
    (folder / 'vectors/selected.npz').write_bytes(b'research-vectors')
    client = TestClient(small_app.app)
    assert client.get('/mira-activation-study/').status_code == 200
    assert client.get('/mira-activation-study/summary.json').json() == {'takes': 100}
    assert client.get('/mira-activation-study/vectors.npz').content == b'research-vectors'
    for private in ('reference.wav', 'activations/foo.npz', 'audio/foo.wav',
                    'vector-diagnostics/report.json', 'runtime/private.mp3', 'unknown.json'):
        assert client.get('/mira-activation-study/' + private).status_code == 404
