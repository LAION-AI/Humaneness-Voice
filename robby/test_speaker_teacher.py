"""Offline, lightweight teacher contracts. Real-model benchmark is separate."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from speaker_teacher import (EcapaTeacher, StreamingTeacher, REQUIRED_FILES,
                             get_cached_teacher, resolve_model_root,preserve_encoder_precision,enrollment_reference)


class FakeTeacher:
    def __init__(self):
        self.calls = []

    def embed(self, audio):
        self.calls.append(audio.copy())
        return np.eye(192, dtype=np.float32)[0]


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.speech = np.full(5120, .03, np.float32)
    def test_reference_uses_available_speech_without_padding(self):
        x=np.full(24000,.05,np.float32)
        np.testing.assert_array_equal(enrollment_reference(x,3.04),x)
    def test_reference_window_excludes_quiet_room_preroll(self):
        x=np.r_[np.full(32000,.008,np.float32),np.full(64000,.06,np.float32)]
        y=enrollment_reference(x,3.04)
        self.assertEqual(len(y),48640);self.assertGreater(float(y.mean()),.059)
        with self.assertRaises(ValueError):enrollment_reference(x,.32)

    def test_speechbrain_tf32_quirk_is_disabled_without_flag_changes(self):
        import os,torch
        from unittest.mock import patch
        before=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        with patch.dict(os.environ,{'SB_DISABLE_QUIRKS':'other_quirk'}):
            preserve_encoder_precision()
            self.assertEqual(set(os.environ['SB_DISABLE_QUIRKS'].split(',')),{'other_quirk','allow_tf32'})
        self.assertEqual(before,(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32))

    def test_missing_local_snapshot_never_downloads(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                resolve_model_root(d)
            for name in REQUIRED_FILES:
                (Path(d)/name).touch()
            self.assertEqual(resolve_model_root(d), Path(d))
            a = get_cached_teacher(d)
            self.assertIs(a, get_cached_teacher(d))
            self.assertFalse(a.loaded)

    def test_embed_rejects_invalid_before_loading(self):
        teacher = EcapaTeacher('/missing/offline/snapshot')
        for audio in [np.zeros(20000), self.speech, np.full(11000, np.nan), np.ones((2, 16000))]:
            with self.assertRaises(ValueError):
                teacher.embed(audio)
        self.assertFalse(teacher.loaded)

    def test_embedding_normalized_and_cpu_only(self):
        import torch
        class FakeModel:
            def encode_batch(self, audio):
                self.assert_cpu = audio.device.type == 'cpu'
                return torch.ones(1, 1, 192)
        teacher = EcapaTeacher('/not/used')
        teacher._model = FakeModel()
        threads = torch.get_num_threads()
        value = teacher.embed(np.tile(self.speech, 3))
        self.assertAlmostEqual(float(np.linalg.norm(value)), 1., places=6)
        self.assertEqual(value.shape, (192,))
        self.assertTrue(teacher._model.assert_cpu)
        self.assertEqual(torch.get_num_threads(), threads)

    def test_context_cadence_and_no_repeated_evidence(self):
        fake = FakeTeacher()
        stream = StreamingTeacher(fake)
        for _ in range(4):
            self.assertEqual(stream.update(self.speech, speech=True)['kind'], 'warming')
        first = stream.update(self.speech, speech=True)
        self.assertTrue(first['fresh'])
        self.assertEqual(first['context_ms'], 1600)
        self.assertEqual(stream.update(self.speech, speech=True)['kind'], 'waiting')
        self.assertTrue(stream.update(self.speech, speech=True)['fresh'])
        self.assertEqual(len(fake.calls), 2)

    def test_silence_overlap_unknown_reset_and_memory_bound(self):
        fake = FakeTeacher()
        stream = StreamingTeacher(fake, max_context_seconds=2.)
        for _ in range(50):
            stream.update(self.speech, speech=True)
        self.assertEqual(stream._samples, 32000)
        self.assertTrue(all(len(x) <= 32000 for x in fake.calls))
        for kwargs, expected in [({'speech':False}, 'silence'),
                                 ({'speech':True, 'overlap':True}, 'overlap'),
                                 ({}, 'uncertain')]:
            result = stream.update(self.speech, **kwargs)
            self.assertEqual(result['kind'], expected)
            self.assertEqual(result['context_ms'], 0)
            self.assertIsNone(result['embedding'])

    def test_one_second_update_cadence(self):
        stream = StreamingTeacher(FakeTeacher(), update_seconds=1.)
        chunk = np.full(1600, .03, np.float32)
        fresh = [i for i in range(40) if stream.update(chunk, speech=True)['fresh']]
        self.assertEqual(fresh, [15, 25, 35])

    def test_numpy_activity_and_oversized_chunk_keep_bounded_storage(self):
        stream = StreamingTeacher(FakeTeacher(), max_context_seconds=2.)
        result = stream.update(np.full(160000, .03, np.float32), speech=np.bool_(True))
        self.assertTrue(result['fresh'])
        self.assertEqual(sum(x.nbytes for x in stream._chunks), 32000 * 4)
        self.assertTrue(all(x.base is None for x in stream._chunks))


if __name__ == '__main__':
    unittest.main()
