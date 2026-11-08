"""Meaningful failure cases for prefix selection and immutable CFG cache forking."""
import unittest
from recipe import errors, reward, GENERAL, chunk_choice

class RewardTests(unittest.TestCase):
    def test_unspoken_suffix_is_free_but_internal_omission_is_not(self):
        self.assertEqual(errors('one two three four','one two',False)['wer'],0)
        self.assertGreater(errors('one two three four','one three',False)['wer'],0)
        self.assertEqual(errors('one two three four','one two',True)['wer'],.5)
    def test_silence_cannot_be_a_perfect_prefix(self):
        self.assertEqual(errors('hello world','',False)['wer'],1)
        self.assertEqual(errors('hello world','',False)['matched_words'],0)
    def test_genuineness_no_ce_and_finite_at_zero_wer(self):
        self.assertAlmostEqual(reward(3,0),.5)
        self.assertGreater(reward(4,.1),reward(3,.1))
        with self.assertRaises(ValueError): reward(float('nan'),0)
    def test_prompt(self):
        self.assertIn('Very authentic, very genuine',GENERAL)
        self.assertNotIn('pleasant',GENERAL.lower())
        self.assertNotIn('nice to listen',GENERAL.lower())
    def test_wrong_ending_cannot_displace_better_unfinished_candidate(self):
        prefix='Vielleicht bleibt das große Rätsel offen Wir können über die Welt verschieden denken und trotzdem spüren dass jetzt jemand Zuwendung braucht '
        text=prefix+'Wir müssen nicht alles wissen bevor wir etwas lieben dürfen'
        hypotheses=[prefix+'Wir müssen nicht alles wissen bevor wir nicht mehr wissen',
                    prefix+'Wir müssen nicht alles wissen bevor']
        scores=[dict(**errors(text,h,False),asr=h,reward=r) for h,r in zip(hypotheses,[.65,.70])]
        self.assertLessEqual(scores[0]['wer'],.12)
        self.assertEqual(scores[0]['matched_words'],len(text.split()))
        winner,complete,_=chunk_choice(scores,text,[False,False])
        self.assertEqual(winner,1);self.assertFalse(complete)
        scores[1]=dict(**errors(text,text,False),asr=text,reward=.70)
        winner,complete,_=chunk_choice(scores,text,[False,False])
        self.assertEqual(winner,1);self.assertTrue(complete)

class StateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from run import setup
        setup('cpu')
    def test_cfg_rows_and_graph_addresses_survive_fork(self):
        import torch
        from types import SimpleNamespace
        from generation import fork_semantic
        cache=torch.arange(4*5*2).reshape(4,5,2).float()
        bank=SimpleNamespace(device='cpu',cache=[(cache,cache.clone())],
            allowed=torch.ones(4,5,dtype=torch.bool),ids=torch.arange(4).reshape(4,1,1),
            query_valid=torch.ones(4,1,1,dtype=torch.bool))
        before=cache.clone();address=cache.data_ptr()
        indexes=fork_semantic(bank,1,2,2,3)
        self.assertEqual(indexes.tolist(),[1,1,3,3])
        self.assertEqual(cache.data_ptr(),address)
        self.assertTrue(torch.equal(cache[:,:3],before[[1,1,3,3],:3]))
        cache[0,0,0]=-999
        self.assertNotEqual(cache[1,0,0].item(),-999)
    def test_codec_fork_preserves_addresses_without_aliasing(self):
        import torch
        from types import SimpleNamespace
        from generation import fork_codec
        state=SimpleNamespace(offset=torch.tensor([4,9]),cache=torch.arange(12).reshape(2,3,2).float())
        module=SimpleNamespace(_streaming_state=state)
        codec=SimpleNamespace(decoder=SimpleNamespace(modules=lambda:[module]))
        addr=state.cache.data_ptr();before=state.cache.clone()
        fork_codec(codec,torch.tensor([1,1]),2)
        self.assertEqual(state.cache.data_ptr(),addr)
        self.assertTrue(torch.equal(state.cache,before[[1,1]]))
        state.cache[0,0,0]=-999
        self.assertNotEqual(state.cache[1,0,0].item(),-999)

if __name__=='__main__': unittest.main()
