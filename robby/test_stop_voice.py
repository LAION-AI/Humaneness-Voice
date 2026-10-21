import unittest
from types import SimpleNamespace
import numpy as np
import torch
from stop_voice import isolated_stop_embedding


class IsolatedStopTests(unittest.TestCase):
    def make(self,fail=False):
        layers=[torch.nn.Identity() for _ in range(15)]
        def encode(waves):
            self.assertEqual([len(w) for w in waves],[128000]*3)
            h=torch.zeros(3,6,1024)
            for k in (3,7,11):layers[k](h)
            if fail:raise RuntimeError('synthetic failure')
            return list(h)
        def fast(x,energy):
            self.assertEqual(tuple(x.shape),(3,4,4,1024))
            self.assertEqual(tuple(energy.shape),(3,4))
            return torch.eye(3,192),torch.zeros(3,3)
        return SimpleNamespace(enc=SimpleNamespace(layers=layers),encode=encode),SimpleNamespace(fast=fast)
    def test_fixed_phases_are_averaged_and_hooks_removed(self):
        asr,predictor=self.make();wave=np.full(9600,.05,np.float32);saved=wave.copy()
        result=isolated_stop_embedding(asr,predictor,wave,8000)
        np.testing.assert_allclose(result[:3],np.full(3,1/np.sqrt(3)),atol=1e-6)
        np.testing.assert_array_equal(wave,saved)
        self.assertTrue(all(not layer._forward_hooks for layer in asr.enc.layers))
    def test_encoder_failure_also_removes_hooks(self):
        asr,predictor=self.make(True)
        with self.assertRaises(RuntimeError):isolated_stop_embedding(asr,predictor,np.ones(9600,np.float32),8000)
        self.assertTrue(all(not layer._forward_hooks for layer in asr.enc.layers))
