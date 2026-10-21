import unittest
import numpy as np
from enrollment_cues import aligned_reference,foreground_context_allowed

class EnrollmentCuesTests(unittest.TestCase):
    def test_reference_uses_native_time_interval_not_old_or_quiet_room(self):
        a,b=np.eye(192,dtype=np.float32)[:2];records=[]
        for i in range(12):
            target=4<=i<10
            records.append((i*.32,(i+1)*.32,{'raw_embedding':a if target else b},
                np.full(5120,.05 if target else .008,np.float32)))
        fast,long=aligned_reference(records,1.,3.52)
        np.testing.assert_allclose(fast,a);self.assertIsNone(long)
        self.assertEqual(aligned_reference(records,0.,.64),(None,None))
    def test_sparse_loud_foreground_can_be_checked_but_quiet_room_cannot(self):
        records=[(i*.32,(i+1)*.32,{},np.full(5120,.01,np.float32)) for i in range(20)]
        self.assertTrue(foreground_context_allowed(records,[.06,.05,.07]))
        self.assertFalse(foreground_context_allowed(records,[.008,.01,.014]))
        self.assertFalse(foreground_context_allowed([],[]))

if __name__=='__main__':unittest.main()
