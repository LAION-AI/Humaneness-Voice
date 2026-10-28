"""Authored conversational probes; no actual user transcripts or voiceprints."""
import asyncio,json,re,time,datetime
from pathlib import Path
import httpx
from bolt_director import BoltDirector
from bolt_text import join_sentences
from knowledge_memory import MEMORY_NOTE
from run_voice_assistant_checks import deployed_source_hash

async def main():
 rows=[];history=[{'role':'user','content':'Hallo, ich bin Christoph und meine Freundin heißt Alice.'},{'role':'assistant','content':'Hallo Christoph, schön dich kennenzulernen!'}]
 memory={'note':MEMORY_NOTE,'entries':[{'title':'Christoph','kind':'person','summary':'partner: Alice','detail':'partner: Alice'}]}
 cases=[('greeting_named',"Hey Robby, wie geht's?",{'named':True,'enrolled':True,'display_name':'Christoph','needs_name':False}),
        ('greeting_uncertain',"Hey Robby, wie geht's?",{'named':False,'enrolled':True,'display_name':'Speaker 1','needs_name':True}),
        ('memory_partner','Wie heißt meine Freundin?',{'named':True,'enrolled':True,'display_name':'Christoph','needs_name':False}),
        ('story','Erzähl mir einen kurzen Witz über Äpfel.',{'named':True,'enrolled':True,'display_name':'Christoph','needs_name':False})]
 async with httpx.AsyncClient(timeout=50) as client:
  for backend in ({'model':'local','provider':'local'},{'model':'qwen','provider':'cerebras/fp16'}):
   for label,utterance,identity in cases:
    start=time.perf_counter();metrics={};row={'model':backend['model'],'case':label}
    try:
     parts=[p async for p in BoltDirector().stream(client,utterance,identity,history,{'knowledge_memory':memory},metrics,backend=backend)]
     reply=join_sentences(parts);row.update(reply=reply,ms=round((time.perf_counter()-start)*1000),passed=bool(reply) and not re.search(r'wie (?:heißt|heisst) du|wie darf ich dich nennen|wie ist dein name|what(?: is|’s|\x27s) your name',reply,re.I))
     if label=='memory_partner':row['passed'] &= 'alice' in reply.casefold()
    except Exception as exc:row.update(passed=False,error=type(exc).__name__+': '+str(exc)[:150])
    rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 report={'scope':__doc__,'source_sha256':deployed_source_hash(),'rows':rows,'passed':all(x['passed'] for x in rows),'limits':['Authored text and memory, not real microphone or proof of acoustic identity.','Few cases do not quantify model reliability.']}
 (Path(__file__).parent/'static'/('name-continuity-'+stamp+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print('overall',report['passed'],flush=True)
if __name__=='__main__':asyncio.run(main())
