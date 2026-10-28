import unittest
from lfm3_timeline import normalize_timeline

class TimelineTests(unittest.TestCase):
 def test_relative_and_absolute_times(self):
  x=normalize_timeline({'events':[{'start_s':.5,'end_s':1.5,'kind':'gesture','person_id':'person_1','description':'waves'}]},10,3)
  self.assertEqual(x['events'][0]['absolute_start_s'],10.5);self.assertEqual(x['events'][0]['absolute_end_s'],11.5);self.assertEqual(x['events'][0]['timing_status'],'model_estimate')
 def test_invalid_times_do_not_get_invented(self):
  for a,b in [(None,1),(-1,1),(2,1),(0,9),(float('nan'),1),(True,1)]:
   x=normalize_timeline({'events':[{'start_s':a,'end_s':b,'description':'turn'}]},10,3)['events'][0]
   self.assertEqual(x['timing_status'],'unavailable_or_out_of_range');self.assertNotIn('absolute_start_s',x);self.assertEqual(x['description'],'turn')
 def test_missing_and_unstructured_events_are_not_fabricated(self):
  x=normalize_timeline({'actions':'someone moves'},5,3);self.assertNotIn('events',x);self.assertIn('did not return',x['timeline_status'])
  x=normalize_timeline({'events':['waves']},5,3)['events'][0];self.assertEqual(x['timing_status'],'unavailable');self.assertNotIn('absolute_start_s',x)
 def test_time_strings_and_tiny_end_rounding(self):
  x=normalize_timeline({'events':[{'start_s':'00:01.2','end_s':'3.02s'}]},5,3)['events'][0]
  self.assertEqual(x['absolute_start_s'],6.2);self.assertEqual(x['absolute_end_s'],8.)
 def test_unconfirmed_person_and_explicit_non_event(self):
  x=normalize_timeline({'people':[],'events':[{'person_id':'person_1','description':'No visible person or gesture.'},{'person_id':'person_2','description':'waves','start_s':0,'end_s':1}]},2,3)
  self.assertEqual(len(x['events']),1);self.assertEqual(x['events'][0]['person_status'],'unconfirmed_label')
if __name__=='__main__':unittest.main()
