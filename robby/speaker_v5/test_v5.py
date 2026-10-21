import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from aiohttp import web
from feature_tap import extract_context
from speaker_v4.train import MixedStudent
from .data import Resident
from .train import paired

class ContextTests(unittest.TestCase):
    def test_gru_all_spans_finite_gradients(self):
        torch.set_num_threads(2)
        for frames in (4,25,38):
            model=MixedStudent(False);x=torch.randn(2,1,frames,1024);energy=torch.full((2,frames),-20.)
            emb,states=model(x,energy)
            self.assertEqual(tuple(emb.shape),(2,192));self.assertEqual(tuple(states.shape),(2,3))
            (emb.square().sum()+states.square().sum()).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
    def test_prefix_suffix_and_energy_alignment(self):
        n=6;arrays={'q':np.full((n,25,1024),2,np.int8),'scale':np.ones((n,25,1),np.float16),
            'energy':np.zeros((n,25),np.float16),'local_target':np.zeros((n,192),np.float16),
            'speaker':np.arange(n),'activity':np.zeros(n,int),'emotion':np.zeros(n,int)}
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory)
            for key,value in {'q':np.ones((n,13,1024),np.int8),'scale':np.full((n,13,1),3,np.float16),
                'energy':np.tile(np.arange(38),(n,1)).astype(np.float16),'filled':np.ones(n,np.uint8)}.items():np.save(cache/(key+'.npy'),value)
            with patch('speaker_v5.data.cache_path',return_value=cache),patch('speaker_v5.data.onset_ages',return_value=np.ones(n)),\
                 patch('speaker_v5.data.v4.subset',return_value=np.arange(n)),patch('speaker_v5.data.v4.arrays',return_value=arrays),\
                 patch('speaker_v5.data.v4.splits',return_value={'train':np.arange(4),'validation':np.array([4]),'test':np.array([5])}):
                r=Resident(38,'cpu');x,e,*_=r.batch([1,3])
                self.assertEqual(tuple(x.shape),(2,1,38,1024));self.assertTrue((x[:,:,:13]==3).all());self.assertTrue((x[:,:,-25:]==2).all())
                self.assertTrue(torch.equal(e[0],torch.arange(38).float()))
                self.assertEqual(r.split['test'].tolist(),[5])
    def test_paired_bootstrap_averages_seeds_first(self):
        def rows(offset):return [{'test':{'per_speaker':{str(i):{'accuracy':.3+i*.1+offset} for i in range(10)}}} for _ in range(3)]
        result=paired(rows(.02),rows(0),'test');self.assertAlmostEqual(result['delta_macro_accuracy_pp'],2)
        np.testing.assert_allclose(result['paired_identity_bootstrap_95ci_pp'],[2,2])

class EndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_context_rejected_before_reserving_live_asr(self):
        class Request:
            remote='127.0.0.1';host='127.0.0.1:8793';headers={};query={'frames':'39'}
        with self.assertRaises(web.HTTPBadRequest):await extract_context(Request(),None)
    async def test_public_proxy_forbidden(self):
        class Request:
            remote='127.0.0.1';host='127.0.0.1:8793';headers={'CF-Ray':'proxy'};query={'frames':'38'}
        with self.assertRaises(web.HTTPForbidden):await extract_context(Request(),None)

if __name__=='__main__':unittest.main()
