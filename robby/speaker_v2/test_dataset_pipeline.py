"""Small offline safety tests; synthetic byte fixtures, no network or model load."""
import io
import json
import tarfile
import pytest

from speaker_v2 import prepare_emotion_pilot as emotion
from speaker_v2 import prepare_pilot as core


def fixture_tar():
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode='w:') as archive:
        for index in range(2):
            uid = f'voice_001__E__Anger__C__en.c{index:03d}'
            for ext, data in [('mp3', bytes([index+1])*2048),
                              ('json', json.dumps({'uid':uid, 'dur_s':8}).encode())]:
                header = tarfile.TarInfo(uid+'.'+ext)
                header.size = len(data)
                archive.addfile(header, io.BytesIO(data))
    return out.getvalue()


def test_tar_resynchronizes_after_partial_member():
    # Starts halfway through first audio payload, at a global512-byte boundary.
    pairs = emotion.tar_pairs(fixture_tar()[1024:], 'voice_001')
    assert len(pairs) == 1
    assert pairs[0][0].endswith('c001')
    assert pairs[0][1] == bytes([2])*2048


def test_tar_wrong_voice_never_labeled_as_requested():
    assert emotion.tar_pairs(fixture_tar(), 'other_voice') == []


def test_full_archive_response_rejected_before_body(monkeypatch):
    class Response:
        status_code = 200
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *args): return False
        @property
        def raw(self): raise AssertionError('Must never download a full archive')
    monkeypatch.setattr(emotion.requests, 'get', lambda *a,**kw: Response())
    monkeypatch.setattr(emotion, 'charged_bytes', 0)
    with pytest.raises(RuntimeError, match='Range not honored'):
        emotion.byte_range('fake.tar', 0, 1024)


def test_hard_budget_checked_before_network(monkeypatch):
    monkeypatch.setattr(emotion, 'charged_bytes', emotion.LIMIT-10)
    monkeypatch.setattr(emotion.requests, 'get', lambda *a,**kw: pytest.fail('Network after budget exceeded'))
    with pytest.raises(RuntimeError, match='hard limit'):
        emotion.byte_range('fake.tar', 0, 11)


def test_resume_manifest_preserves_preextracted_heldout(tmp_path):
    dest = tmp_path/'conversations'
    dest.mkdir()
    row = {'path':'conversations/test-028-0.npz', 'split':'test'}
    (dest/'test-028-0.json').write_text(json.dumps(row))
    # No new anchors: purely reconcile already completed records, without
    # silently dropping a held-out-first batch from the streaming manifest.
    core.extract([], output_root=tmp_path)
    saved = json.loads((tmp_path/'manifest.json').read_text())
    assert saved['conversations'] == [row]
