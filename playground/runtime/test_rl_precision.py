import unittest
from types import SimpleNamespace
import torch
from rl_adapter import RLLinear
from rl_precision import configure

class PrecisionTests(unittest.TestCase):
    def engine(self):
        torch.manual_seed(777);base=torch.nn.Linear(8,6)
        module=RLLinear(base,2)
        with torch.no_grad():module.a.normal_(std=.1);module.b.normal_(std=.1)
        return SimpleNamespace(model=torch.nn.Sequential(module),policy_fp32=True)
    def test_default_retains_exact_fp32_residual(self):
        e=self.engine();x=torch.randn(3,8);expected=e.model(x)
        report=configure(e);self.assertTrue(report['official_reference_arithmetic'])
        self.assertTrue(torch.equal(expected,e.model(x)))
    def test_fp32_merge_removes_extra_linear_calls_and_preserves_algebra(self):
        e=self.engine();x=torch.randn(3,8);expected=e.model(x)
        report=configure(e,'fp32_merged')
        self.assertTrue(report['merged']);self.assertIsInstance(e.model[0],torch.nn.Linear)
        torch.testing.assert_close(expected,e.model(x),atol=1e-6,rtol=1e-6)
    def test_mixed_bf16_keeps_original_float32_factors(self):
        e=self.engine();a=e.model[0].a.detach().clone();b=e.model[0].b.detach().clone()
        configure(e,'bf16_base_fp32_lora')
        self.assertEqual(e.model[0].base.weight.dtype,torch.bfloat16)
        self.assertEqual(e.model[0].a.dtype,torch.float32)
        self.assertTrue(torch.equal(a,e.model[0].a));self.assertTrue(torch.equal(b,e.model[0].b))
        self.assertTrue(torch.isfinite(e.model(torch.randn(3,8,dtype=torch.bfloat16))).all())
    def test_full_bf16_and_invalid_or_repeated_configuration(self):
        e=self.engine();configure(e,'bf16_all')
        self.assertTrue(torch.isfinite(e.model(torch.randn(3,8,dtype=torch.bfloat16))).all())
        with self.assertRaises(RuntimeError):configure(e)
        with self.assertRaises(ValueError):configure(self.engine(),'unknown')

if __name__=='__main__':unittest.main()
