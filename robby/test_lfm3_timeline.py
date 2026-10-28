import unittest,tempfile,secrets
from lfm3_timeline import normalize_timeline,normalize_relevance,normalize_scores,recover_repaired_video,note_unreported_signal
from lfm3_context import build_context,flag_context_echo,MAX_BOUND
from knowledge_memory import MemoryStore,cost

class TimelineTests(unittest.TestCase):
 def test_relative_and_absolute_times(self):
  x=normalize_timeline({'events':[{'start_s':.5,'end_s':1.5,'kind':'gesture','person_id':'person_1','description':'waves'}]},10,3)
  self.assertEqual(x['events'][0]['absolute_start_s'],10.5);self.assertEqual(x['events'][0]['absolute_end_s'],11.5);self.assertEqual(x['events'][0]['timing_status'],'model_estimate')
  self.assertIn('3 fps',normalize_timeline({'events':[]},0,3,fps=3)['timeline_status'])
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
 def test_relevance_only_accepts_three_scores(self):
  self.assertEqual([normalize_relevance(x) for x in [0,1,2,'2',True,3,'high',None]],[0,1,2,2,None,None,None,None])
  x=normalize_scores({'observations':[{'description':'speech','relevance':2},{'description':'noise','relevance':5}]})
  self.assertIsNone(x['observations'][1]['relevance'])
  x=normalize_scores({'observations':[{'category':'relevant_memory','description':'Bob likes jazz','relevance':2},{'category':'speech','description':'hello','relevance':1}]})
  self.assertEqual(x['context_only_items_filtered'],1);self.assertEqual(len(x['observations']),1)
  self.assertIsNone(note_unreported_signal({'observations':[]},False)['observations'][0]['relevance'])
  self.assertEqual(note_unreported_signal({'observations':[]},True)['observations'],[])
 def test_private_context_is_bounded_and_scoped(self):
  with tempfile.TemporaryDirectory() as root:
   token=secrets.token_hex(32);store=MemoryStore(token,root);quote='Bob likes jazz.'
   store.update({'entries':[{'title':'Bob','kind':'person','triggers':['Bob'],'facts':[{'key':'interests','value':'jazz','evidence':quote,'related':None}]}]},quote,0)
   recent=[{'role':'user','content':'Bob and jazz are playing.'}]
   ctx,meta=build_context(token,recent,root);self.assertTrue(meta['memory_used']);self.assertLessEqual(cost(ctx),MAX_BOUND)
   self.assertNotIn(token,str(ctx));self.assertIn('Bob',str(ctx))
   other,_=build_context(secrets.token_hex(32),recent,root);self.assertFalse(other['relevant_memory'])
   disabled,_=build_context(None,recent,root);self.assertFalse(disabled['relevant_memory'])
 def test_context_echo_is_flagged_not_erased(self):
  context={'recent_conversation':[{'role':'user','content':'Bob likes jazz music'}],'relevant_memory':[]}
  out=flag_context_echo({'observations':[{'description':'Bob likes jazz music'},{'description':'Hello Robby'}]},context)
  self.assertTrue(out['observations'][0]['possible_context_echo']);self.assertNotIn('possible_context_echo',out['observations'][1]);self.assertEqual(out['observations'][0]['description'],'Bob likes jazz music')
 def test_repaired_nested_events_are_only_lifted_if_unambiguous(self):
  one={'people':[{'id':'person_1','description':'person'},{'events':[{'description':'moves'}]}]}
  x=recover_repaired_video(one);self.assertEqual(len(x['people']),1);self.assertEqual(x['events'][0]['description'],'moves')
  two={'people':[{'events':[]},{'events':[{'description':'moves'}]}]}
  self.assertNotIn('events',recover_repaired_video(two));self.assertIsNone(recover_repaired_video({'irrelevant':1}))
  x=recover_repaired_video({'foreground':'blue','observations':[{'category':'foreground','description':'blue'},{'people':[],'events':[],'changes':'none'}]})
  self.assertEqual(x['people'],[]);self.assertEqual(x['events'],[]);self.assertEqual(len(x['observations']),1)
if __name__=='__main__':unittest.main()
