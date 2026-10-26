"""Real Qwen extraction/retrieval across fresh sessions; authored test data only."""
import asyncio,datetime,json,secrets,tempfile,time
from pathlib import Path
import httpx
from memory_session import MemorySession
from bolt_director import BoltDirector
from bolt_text import join_sentences
from knowledge_memory import entry_id
ROOT=Path(__file__).resolve().parent
async def main():
 report={'passed':False,'scope':__doc__,'updates':[],'answers':[]};events=[]
 async def send(data):events.append(data)
 token=secrets.token_hex(32)
 with tempfile.TemporaryDirectory(prefix='robby-wiki-test-') as work:
  m=MemorySession(send,work);h=[];await m.command({'enabled':True,'capability':token},h)
  try:
   turns=[('Hallo, ich bin Bob. Meine Freundin ist Alice. Ich wohne in Hamburg und spiele gern Saxofon.', 'Hallo Bob!'),
          ('Alice liebt Jazz und kümmert sich gern um ihren Garten.', 'Das klingt schön.'),
          ('Kleine Korrektur: Ich bin Bob und wohne jetzt in Bremen, nicht mehr in Hamburg.', 'Danke für die Korrektur.'),
          ('Mein Fahrrad heißt Blitz. Ich fahre damit jeden Sonntag zum See.', 'Dein Fahrrad ist bestimmt grün.')] # robot invention must NOT be stored
   for user,assistant in turns:
    h.extend([{'role':'user','content':user},{'role':'assistant','content':assistant}]);t=time.perf_counter()
    m.enqueue({'text':user,'reply':assistant,'llm_selection':{'model':'qwen','provider':'cerebras/fp16'}},h)
    await m.worker
    state,entries=m.store.snapshot();report['updates'].append({'user':user,'ms':round((time.perf_counter()-t)*1000,2),'state':m.last,'titles':[r['title'] for r in entries.values()]})
    assert m.last['state']=='ready',m.last
   _,entries=m.store.snapshot();report['entries']=list(entries.values())
   bob=entries[entry_id('Bob')];assert 'bremen' in bob['detail'].casefold() and 'hamburg' not in bob['detail'].casefold(),bob
   assert 'alice' in bob['detail'].casefold()
   assert not any('grün' in r['detail'].casefold() for r in entries.values()),'Assistant invention persisted'
   # Fresh identity, ZERO recent-chat tokens: all recall must come from the wiki.
   again=MemorySession(send,work);fresh=[];await again.command({'enabled':True,'capability':token,'history_budget':0},fresh)
   assert fresh==[]
   async with httpx.AsyncClient(timeout=40) as c:
    for question,expected in [('Wie heißt Bobs Freundin?',['alice']),('Wo wohnt Bob jetzt?',['bremen']),('Was macht Alice gern?',['jazz','garten'])]:
     history,context=await again.prepare(question,{},fresh,{'model':'qwen'});metrics={};parts=[]
     async for s in BoltDirector().stream(c,question,{},history,{'knowledge_memory':context},metrics,backend={'model':'qwen','provider':'cerebras/fp16'}):parts.append(s)
     text=join_sentences(parts);passed=all(x in text.casefold() for x in expected)
     report['answers'].append({'question':question,'text':text,'passed':passed,'context':context['metrics'],'metrics':metrics});assert passed,text
   stranger=MemorySession(send,work);await stranger.command({'enabled':True,'capability':secrets.token_hex(32),'history_budget':0},[])
   assert not (await stranger.prepare('Bob Alice',{},[],{'model':'qwen'}))[1]['entries'];report['isolated_browser']=True
   await again.command({'action':'reset'},fresh)
   assert not (await again.prepare('Bob Alice',{},fresh,{'model':'qwen'}))[1]['entries'];report['reset_verified']=True
   report['passed']=True
  except Exception as e:report['error']=type(e).__name__+': '+str(e)[:1000]
  finally:await m.close()
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');raw=json.dumps(report,indent=2,ensure_ascii=False)
 (ROOT/'logs'/('knowledge-memory-model-'+stamp+'.json')).write_text(raw);(ROOT/'static/knowledge-memory-model.json').write_text(raw)
 print(json.dumps({'passed':report['passed'],'error':report.get('error'),'updates':len(report['updates']),'answers':len(report['answers'])}),flush=True)
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':asyncio.run(main())
