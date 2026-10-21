"""Tiny artificial-data integration test, NOT evidence of speaker quality.

Tests the complete four-model train/checkpoint/validation/test/report workflow.
All temporary weights/data are removed automatically. No real source identities,
held-out utterances, live microphone data, or published quality metrics are used.
"""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from .train import run_four,Student,CONFIGS


class TrainingTests(unittest.TestCase):
    def test_shapes_and_parameter_budget(self):
        torch.set_num_threads(2)
        for cfg in CONFIGS:
            m=Student(cfg['architecture']);self.assertTrue(100000<=sum(p.numel() for p in m.parameters())<=1000000)
            for context in (4,8,13,25):
                e,l=m(torch.randn(2,context,1024),torch.randn(2,2,1024),torch.full((2,context),-30.))
                self.assertEqual(tuple(e.shape),(2,192));self.assertTrue(torch.isfinite(e).all());self.assertEqual(tuple(l.shape),(2,3))

    def test_full_workflow_on_artificial_data(self):
        rng=np.random.default_rng(13);count=400
        with tempfile.TemporaryDirectory(prefix='tyger-training-selfcheck-') as name:
            root=Path(name);folder=root/'arrays';folder.mkdir()
            prototypes=rng.normal(size=(500,192)).astype(np.float32);prototypes/=np.linalg.norm(prototypes,axis=-1,keepdims=True)
            ids=np.r_[np.arange(200)%2,480+np.arange(100)%2,490+np.arange(100)%2].astype(np.int16)
            activity=np.where(np.arange(count)%7==0,1,0).astype(np.uint8)
            targets=prototypes[ids].copy();targets[activity!=0]=0;ids[activity!=0]=-1
            data={'q':rng.integers(-127,128,(count,25,1024),dtype=np.int8),'scale':np.full((count,25,1),.01,np.float16),
                  'pool':rng.normal(size=(count,2,1024)).astype(np.float16),'energy':np.full((count,25),-25,np.float16),
                  'local_target':targets.astype(np.float16),'speaker':ids,'activity':activity,
                  'scenario':(np.arange(count)%5).astype(np.uint8),'emotion':(np.arange(count)%9).astype(np.uint8),'filled':np.ones(count,np.uint8)}
            for key,value in data.items():np.save(folder/(key+'.npy'),value)
            np.save(root/'prototypes.npy',prototypes)
            (root/'sources_manifest.json').write_text(json.dumps([{'speaker_idx':i,'split':'train' if i<480 else 'validation' if i<490 else 'test'} for i in range(500)]))
            (root/'splits.json').write_text(json.dumps({'train':list(range(200)),'validation':list(range(200,300)),'test':list(range(300,400))}))
            events=[];run_four(root,lambda **fields:events.append(fields))
            result=json.loads((root/'models/results.json').read_text());self.assertTrue(result['finished']);self.assertEqual(len(result['results']),4)
            selected=[r for r in result['results'] if r.get('selected')];self.assertEqual(len(selected),1)
            self.assertEqual(len(selected[0]['test_contexts']),5);self.assertIn('test_controls',selected[0]);self.assertTrue(events)
            self.assertTrue((root/'models/report.html').exists())

if __name__=='__main__':unittest.main()
