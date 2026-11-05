"""Keep greedy generation available when the optional fast sampler is enabled."""
import unittest,torch
from performance_core import fast_sample
class FastSampleTests(unittest.TestCase):
 def test_zero_temperature_is_argmax_even_with_top_p(self):
  logits=torch.tensor([[1.,5.,3.],[7.,2.,1.]])
  self.assertEqual(fast_sample(None,logits,0.,2,.95).tolist(),[1,0])
 def test_positive_temperature_respects_top_k(self):
  logits=torch.tensor([[1.,5.,3.]])
  self.assertEqual(fast_sample(None,logits,.8,1,.95).tolist(),[1])
