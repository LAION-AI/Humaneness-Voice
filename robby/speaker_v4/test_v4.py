import unittest
from unittest.mock import patch
import numpy as np
import torch
from .train import MixedStudent,regression,objective
from .train import Pairs
from feature_tap import reserve_research,ResearchLocked

class SlotTests(unittest.IsolatedAsyncioTestCase):
    async def test_423_is_a_real_http_exception(self):
        self.assertEqual(ResearchLocked().status,423)
        class Ears:
            defaults={};browser_waiting=1
            def open_session(self,_):raise AssertionError('Must not reserve ahead of browser')
        with self.assertRaises(ResearchLocked):await reserve_research(Ears())
    async def test_async_close_can_finish_before_next_batch(self):
        class Ears:
            defaults={};browser_waiting=0;n=0
            def open_session(self,_):
                self.n+=1
                return 'slot' if self.n==3 else None
        self.assertEqual(await reserve_research(Ears()),'slot')

class V4Tests(unittest.TestCase):
    def test_layer_shapes_gradients_and_normalization(self):
        torch.set_num_threads(2)
        for mix in (False,True):
            model=MixedStudent(mix);x=torch.randn(3,4,4,1024);e=torch.full((3,4),-20.)
            emb,logits=model(x,e)
            self.assertEqual(tuple(emb.shape),(3,192));self.assertEqual(tuple(logits.shape),(3,3))
            self.assertTrue(torch.allclose(emb.norm(dim=-1),torch.ones(3),atol=1e-5))
            loss=regression(emb,torch.randn(3,192),model.teacher_mean);loss.backward()
            if mix:self.assertTrue(torch.isfinite(model.layer_logits.grad).all())
    def test_l15_only_ignores_other_layers(self):
        m=MixedStudent(False).eval();x=torch.randn(2,4,4,1024);y=x.clone();y[:,:3]*=100
        e=torch.full((2,4),-30.)
        with torch.no_grad():self.assertTrue(torch.equal(m(x,e)[0],m(y,e)[0]))
    def test_objective_predeclared(self):
        self.assertAlmostEqual(objective({'macro_speaker_accuracy':.8,'eer':.1,'source_utterance_teacher_cosine':.6}),.81)
    def test_pairs_use_training_identities_and_other_emotions(self):
        class Resident:
            indices=np.arange(8);ids_cpu=np.array([0,0,1,1,2,2,480,490]);classes_cpu=np.zeros(8,int)
            emotions_cpu=np.array([1,2,1,2,1,2,1,2]);split={'train':np.arange(6),'validation':np.array([6]),'test':np.array([7])}
        stable=np.random.default_rng(1).normal(size=(500,192)).astype(np.float32);stable/=np.linalg.norm(stable,axis=-1,keepdims=True)
        with patch('speaker_v4.data.metadata',return_value=(np.arange(8),[{'text_hash':'same'}]*8)):
            sampler=Pairs(Resident(),stable);anchor=np.array([0,2,4]);p,n=sampler.sample(anchor,np.random.default_rng(3))
        self.assertTrue(np.all(Resident.ids_cpu[p]==Resident.ids_cpu[anchor]))
        self.assertTrue(np.all(Resident.emotions_cpu[p]!=Resident.emotions_cpu[anchor]))
        self.assertTrue(np.all(Resident.ids_cpu[n]!=Resident.ids_cpu[anchor]));self.assertTrue(np.all(n<6))

if __name__=='__main__':unittest.main()
