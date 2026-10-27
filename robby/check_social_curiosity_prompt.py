"""Five real-provider proactive voice questions after authored introductions."""
import asyncio,datetime,json,re,time
from pathlib import Path
import httpx
from bolt_director import BoltDirector
from bolt_text import join_sentences
from llm_backends import default_selection
from run_voice_assistant_checks import deployed_source_hash
async def main():
 rows=[]
 async with httpx.AsyncClient(timeout=45) as client:
  for i in range(5):
   history=[{'role':'user','content':'Hey Robby, ich bin Christoph.'},{'role':'assistant','content':'Hallo Christoph! Schön, dass du da bist!'}]
   if i%2:history += [{'role':'user','content':'Meine Freundin heißt Alice.'},{'role':'assistant','content':'Schön, von Alice zu hören.'}]
   context={'environment_proactive':True,'environment':{'kind':'unfamiliar_voice','caption':'A salient voice did not confidently match the current partner. It might be a TV, recording or the same person speaking differently.'}}
   identity={'named':True,'enrolled':True,'display_name':'Christoph','needs_name':False,'self_introduction':False}
   metrics={};start=time.perf_counter()
   parts=[s async for s in BoltDirector().stream(client,'',identity,history,context,metrics,backend=default_selection())]
   reply=join_sentences(parts);passed=bool('?' in reply and re.search(r'stimme|fernseh|jemand|\bwer\b|sprech|\btv\b',reply,re.I))
   row={'case':i,'reply':reply,'passed':passed,'wall_ms':round((time.perf_counter()-start)*1000,2),'metrics':metrics};rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='metrics'},ensure_ascii=False),flush=True)
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 report={'scope':__doc__,'source_sha256':deployed_source_hash(),'passed':all(r['passed'] for r in rows),'rows':rows,'limits':'Lexical question check plus manual reading; five development probes are not a general semantic accuracy estimate.'}
 (Path(__file__).parent/'static'/('social-proactive-prompt-'+stamp+'.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False))
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':asyncio.run(main())
