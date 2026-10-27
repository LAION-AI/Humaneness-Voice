"""CPU contracts for the pinned KV layout and bounded scheduler admission."""
import queue,threading,unittest
import torch
from live_batch import Utterance,merge_caches,ContinuousBatch

class BatchState(unittest.TestCase):
    def state(self,n,offset):
        s=Utterance(str(n),{},n)
        s.past=tuple(tuple(torch.arange(2*n*3*4).reshape(2,n,3,4)+offset+k for k in (0,1)) for _ in range(2))
        s.mask=torch.ones(2,n+1,dtype=torch.bool);s.mask[:,0]=False
        s.input=torch.ones(2,1,13,dtype=torch.long)*offset
        return s
    def test_padding_preserves_cache_and_cumulative_rope_positions(self):
        a,b=self.state(3,10),self.state(5,100)
        past,mask,inputs=merge_caches([a,b])
        self.assertEqual(tuple(past[0][0].shape),(4,5,3,4))
        self.assertTrue(torch.equal(past[0][0][:2,-3:],a.past[0][0]))
        self.assertTrue(torch.equal(past[0][0][2:],b.past[0][0]))
        self.assertFalse(mask[:2,:2].any())
        self.assertTrue(torch.equal(mask[:2].sum(-1),a.mask.sum(-1)))
        self.assertTrue(torch.equal(mask[2:].sum(-1),b.mask.sum(-1)))
        self.assertTrue(torch.equal(inputs[:2],a.input))
        self.assertTrue(torch.equal(inputs[2:],b.input))
    def test_pending_cancellation_completes_without_allocating_cuda(self):
        events=[];scheduler=ContinuousBatch(None,None,queue.Queue(),events.append,threading.Event())
        scheduler.command({'type':'submit','id':'job','order':0,'spec':{}})
        scheduler.command({'type':'cancel','id':'job'})
        scheduler.complete(scheduler.pending[0])
        self.assertTrue(events[0]['cancelled']);self.assertEqual(events[0]['samples'],0)
        self.assertEqual(events[0]['generation_ms'],0)
    def test_admission_is_bounded_and_ids_unique(self):
        scheduler=ContinuousBatch(None,None,queue.Queue(),lambda x:None,threading.Event())
        for i in range(8):scheduler.command({'type':'submit','id':str(i),'order':i,'spec':{}})
        with self.assertRaises(ValueError):scheduler.command({'type':'submit','id':'8','order':8,'spec':{}})
        with self.assertRaises(ValueError):scheduler.command({'type':'submit','id':'0','order':0,'spec':{}})

if __name__=='__main__':unittest.main()
