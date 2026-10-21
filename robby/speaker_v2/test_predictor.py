"""Fast architecture, streaming-state and verification-protocol checks."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .inference import Predictor
from .models import SpeakerPredictor, parameter_count,load_state
from .train import roc_metrics


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_architecture_dimensions_and_budget(self):
        for arch in ("mlp", "gru", "tcn"):
            for width in (96, 192):
                model = SpeakerPredictor(arch,width)
                self.assertGreaterEqual(parameter_count(model), 100000)
                self.assertLessEqual(parameter_count(model), 1000000)
                for context in (4,8,12,25):
                    embedding, activity = model(torch.randn(2,context,1024))
                    self.assertEqual(tuple(embedding.shape),(2,192))
                    self.assertEqual(tuple(activity.shape),(2,3))
                    self.assertTrue(torch.allclose(embedding.norm(dim=-1),torch.ones(2),atol=1e-5))

    def test_incremental_silence_and_reset(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"test.pt"
            model = SpeakerPredictor("gru",96)
            with torch.no_grad():
                model.activity.weight.zero_()
                model.activity.bias.copy_(torch.tensor([10.,-10.,-10.]))
            torch.save({"model":model.state_dict(),"config":{"architecture":"gru","width":96,"context":12}},path)
            predictor = Predictor(path)
            result = predictor.update(np.random.default_rng(1).normal(size=(4,1024)),energy=.1)
            self.assertEqual(result["kind"],"speech")
            self.assertEqual(result["embedding"].shape,(192,))
            self.assertLess(result["confidence"],.34)
            result = predictor.update(np.zeros((4,1024)),energy=0.)
            self.assertEqual(result["kind"],"silence")
            self.assertIsNone(result["embedding"])
            result = predictor.update(np.random.default_rng(3).normal(size=(4,1024)),energy=.1)
            self.assertLess(result["confidence"],.34)  # silence must reset confidence accumulation
            predictor.reset()
            self.assertEqual(len(predictor.frames),0)
            with self.assertRaises(ValueError): predictor.update(np.zeros((4,192)))

    def test_validation_threshold_fixed_for_test(self):
        val = roc_metrics([.8,.9],[.1,.2,.3])
        threshold = val["far05_threshold"]
        test = roc_metrics([.25,.95],[.15,.35],threshold)
        self.assertEqual(test["threshold"],threshold)
        self.assertEqual(test["false_accept_rate"],.5)
        self.assertEqual(test["false_reject_rate"],.5)

    def test_centered_output_and_original_checkpoint_compatibility(self):
        model = SpeakerPredictor("mlp",96,centered=True)
        with torch.no_grad():
            model.teacher_mean.copy_(torch.randn(192)*.03)
            model.teacher_radius.fill_(.8)
        embedding,_ = model(torch.randn(2,8,1024))
        self.assertTrue(torch.allclose(embedding.norm(dim=-1),torch.ones(2),atol=1e-5))
        original = {k:v for k,v in model.state_dict().items() if k not in ("teacher_mean","teacher_radius")}
        load_state(SpeakerPredictor("mlp",96),original)
        original.pop("embedding.weight")
        with self.assertRaises(ValueError):load_state(SpeakerPredictor("mlp",96),original)


if __name__ == "__main__": unittest.main()
