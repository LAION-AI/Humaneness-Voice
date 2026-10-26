"""Isolated private-store, provenance, budgeting and lifecycle regressions."""
import asyncio,json,secrets,tempfile,unittest
from unittest.mock import patch,AsyncMock
from pathlib import Path
from knowledge_memory import MemoryStore,entry_id,retrieve,settings,cost,bounded_history,validate_updates
from memory_session import MemorySession

def changes(title='Bob',key='partner',value='Alice',evidence='Ich bin Bob, meine Freundin ist Alice.',related='Alice'):
 return {'entries':[{'title':title,'kind':'person','triggers':[title],'facts':[{'key':key,'value':value,'evidence':evidence,'related':related}]}]}

class Store(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.token=secrets.token_hex(32);self.store=MemoryStore(self.token,self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def test_private_namespaces_and_reopen(self):
  quote='Ich bin Bob, meine Freundin ist Alice.';self.store.update(changes(),quote,0)
  other=MemoryStore(secrets.token_hex(32),self.tmp.name)
  self.assertFalse(other.snapshot()[1]);self.assertEqual(MemoryStore(self.token,self.tmp.name).snapshot()[1][entry_id('Bob')]['facts']['partner']['value'],'Alice')
  self.assertNotIn(self.token,str(self.store.path));self.assertEqual(self.store.path.stat().st_mode&0o777,0o700)
 def test_provenance_rejects_assistant_invention_and_path_inputs(self):
  with self.assertRaises(ValueError):self.store.update(changes(), 'Hallo.',0)
  for token in ['../secret','x'*64,'a'*63]:
   with self.assertRaises(ValueError):MemoryStore(token,self.tmp.name)
  with self.assertRaises(ValueError):self.store.delete('../state')
 def test_update_preserves_other_fields_and_keeps_five_revisions(self):
  self.store.update(changes(), 'Ich bin Bob, meine Freundin ist Alice.',0)
  for i in range(7):
   quote=f'Bob wohnt jetzt in Ort {i}.';self.store.update(changes(key='home',value='Ort '+str(i),evidence=quote,related=None),quote,0)
  bob=self.store.snapshot()[1][entry_id('Bob')]
  self.assertEqual(bob['facts']['partner']['value'],'Alice');self.assertEqual(bob['facts']['home']['value'],'Ort 6');self.assertEqual(len(bob['versions']),5)
 def test_reset_invalidates_late_jobs_and_erases_versions_and_history(self):
  self.store.update(changes(),'Ich bin Bob, meine Freundin ist Alice.',0)
  self.store.save_history([{'role':'user','content':'Hallo Bob.'}],5000,0)
  epoch=self.store.reset();self.assertEqual(epoch,1)
  self.assertFalse(self.store.update(changes(),'Ich bin Bob, meine Freundin ist Alice.',0))
  self.assertFalse(self.store.save_history([{'role':'user','content':'late'}],5000,0))
  state,entries=self.store.snapshot();self.assertEqual(state['history'],[]);self.assertFalse(entries)
 def test_zero_budget_unicode_phrase_boundaries_and_cycle_safe_hops(self):
  for title,target in [('Bob','Alice'),('Alice','Jazz'),('Jazz','Bob')]:
   quote=f'{title} kennt {target}.';self.store.update(changes(title,key='knows',value=target,evidence=quote,related=target),quote,0)
  entries=self.store.snapshot()[1]
  rows,m=retrieve(entries,'Bob',10000,2);self.assertEqual(len(rows),3);self.assertEqual(len({x['id'] for x in rows}),3)
  self.assertFalse(retrieve(entries,'Bobby',10000,0)[0]);self.assertFalse(retrieve(entries,'Bob',0,2)[0])
  for budget in (30,200,500,1000):
   rows,meta=retrieve(entries,'Bob',budget,2);self.assertLessEqual(meta['knowledge_used_bound'],budget)
   if rows:self.assertLessEqual(cost(rows),budget)
 def test_history_budget_keeps_interruption_metadata(self):
  h=[{'role':'user','content':'Frage?\nÄÖ😊'},{'role':'assistant','content':'Antwort.','playback_interrupted':True}]
  out,used=bounded_history(h,5000);self.assertTrue(out[-1]['playback_interrupted']);self.assertLessEqual(used,5000)
  self.assertFalse(bounded_history(h,0)[0])
 def test_possessive_name_retrieval_and_multiple_interests(self):
  quote='Alice liebt Jazz und ihren Garten.'
  value=changes('Alice','interests','Jazz',quote,None)
  value['entries'][0]['facts'].append({'key':'interest','value':'Garten','evidence':quote,'related':None})
  self.store.update(value,quote,0)
  entries=self.store.snapshot()[1];found,_=retrieve(entries,'Was sind Alices Hobbys?',10000,0)
  self.assertEqual(found[0]['title'],'Alice');self.assertIn('Jazz',found[0]['detail']);self.assertIn('Garten',found[0]['detail'])
 def test_settings_strict_limits(self):
  self.assertFalse(settings({})['enabled'])
  for data in ({'knowledge_budget':10001},{'history_budget':-1},{'hops':3},{'enabled':'false'},{'history_budget':True}):
   with self.assertRaises(ValueError):settings(data)

class Session(unittest.IsolatedAsyncioTestCase):
 async def asyncSetUp(self):self.tmp=tempfile.TemporaryDirectory();self.sent=[];self.token=secrets.token_hex(32)
 async def asyncTearDown(self):self.tmp.cleanup()
 async def send(self,data):self.sent.append(data)
 async def test_disabled_is_noop_and_does_not_create_storage(self):
  m=MemorySession(self.send,self.tmp.name);history=[{'role':'user','content':'Bob'}]
  recent,context=await m.prepare('Bob',{},history,{'model':'qwen'})
  self.assertIs(recent,history);self.assertIsNone(context);m.enqueue({'text':'Bob'},history)
  self.assertFalse(list(Path(self.tmp.name).iterdir()));self.assertIsNone(m.worker)
 async def test_enable_restore_off_and_reset(self):
  s=MemoryStore(self.token,self.tmp.name);s.update(changes(),'Ich bin Bob, meine Freundin ist Alice.',0)
  s.save_history([{'role':'user','content':'Ich bin Bob.'},{'role':'assistant','content':'Hallo.'}],5000,0)
  m=MemorySession(self.send,self.tmp.name);h=[]
  await m.command({'enabled':True,'capability':self.token},h);self.assertEqual(len(h),2)
  recent,context=await m.prepare('Wer ist Alice?',{},h,{'model':'qwen'})
  self.assertTrue(context['entries']);self.assertNotIn(self.token,json.dumps(context))
  await m.command({'enabled':False},h);self.assertIsNone((await m.prepare('Alice',{},h,{}))[1])
  await m.command({'action':'reset'},h);self.assertEqual(h,[]);self.assertFalse(s.snapshot()[1])
 async def test_local_context_is_smaller_and_no_pre_optin_history_is_queued(self):
  m=MemorySession(self.send,self.tmp.name);h=[{'role':'user','content':'Before consent'}]
  await m.command({'enabled':True,'capability':self.token},h)
  _,ctx=await m.prepare('Hallo',{},h,{'model':'local'})
  self.assertLessEqual(sum(ctx['settings'][k] for k in ('knowledge_budget','history_budget')),2500)
  async def hold():await asyncio.sleep(30)
  m.worker=asyncio.create_task(hold());h.append({'role':'user','content':'After consent'})
  m.enqueue({'text':'After consent'},h);self.assertEqual([r['content'] for r in m.pending['history']],['After consent'])
  await m.cancel()

if __name__=='__main__':unittest.main()
