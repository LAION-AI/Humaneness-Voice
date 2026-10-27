"""Authored live-API identity/role probes. No private dialogue is retained."""
import asyncio,json,time,re,datetime
from pathlib import Path
from unittest.mock import patch
import httpx
from bolt_director import BoltDirector
from bolt_text import join_sentences
from llm_backends import default_selection
from run_voice_assistant_checks import deployed_source_hash
ROOT=Path(__file__).resolve().parent
async def main():
 rows=[]
 cases=[('identity','Und wer bin ich?', [{'role':'user','content':'Ich bin Christoph.'},{'role':'assistant','content':'Hallo Christoph!'}], 'Christoph'),
 ('correct_role','Nein, ich bin Christoph. Du bist Robby.', [{'role':'user','content':'Ich bin Christoph.'},{'role':'assistant','content':'Ich bin also Christoph.'}], 'Christoph'),
 ('two_people','Und wer bin ich?', [{'role':'user','content':'Ich bin Christoph.'},{'role':'assistant','content':'Hallo Christoph!'},{'role':'user','content':'Meine Freundin heißt Alice.'},{'role':'assistant','content':'Schön, von Alice zu hören.'}], 'Christoph'),
 ('other_user','Wie heiße ich?', [{'role':'user','content':'Ich bin Christoph.'},{'role':'assistant','content':'Hallo Christoph!'},{'role':'user','content':'Ich bin Alice.'},{'role':'assistant','content':'Hallo Alice!'}], 'Alice')]
 cases.append(('freeform_roles','Sag mir kurz, wer von uns Robby ist und wer Christoph ist.', [], 'Christoph'))
 async with httpx.AsyncClient(timeout=45) as c:
  for mode in ('explicit_role_grounding',):
   for repeat in range(2):
    for name,text,history,person in cases:
     identity={'named':True,'enrolled':True,'display_name':person,'speaker_id':'speaker-1','needs_name':False,'self_introduction':name=='correct_role'}
     metrics={};start=time.perf_counter()
     async def run():return [s async for s in BoltDirector().stream(c,text,identity,history,metrics=metrics,backend=default_selection())]
     if mode=='prior_role_prompt':
      with patch('social_profiles.context_note',lambda identity:''):parts=await run()
     else:parts=await run()
     reply=join_sentences(parts)
     wrong_self=bool(re.search(r'\bich (?:bin|heiße|heisse)(?: also| doch| auch)? '+person+r'\b',reply,re.I))
     reask=bool(re.search(r'wie (?:heißt|heisst) du|wie darf ich dich nennen|wie ist dein name',reply,re.I))
     grammar=bool(re.search('wie du dir selbst genannt hast',reply,re.I))
     passed=person.casefold() in reply.casefold() and not (wrong_self or reask or grammar)
     row={'mode':mode,'repeat':repeat,'case':name,'input':text,'reply':reply,'passed':passed,'role_error':wrong_self,'repeated_name_question':reask,'observed_bad_phrase':grammar,'metrics':metrics,'wall_ms':round((time.perf_counter()-start)*1000,2)}
     rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='metrics'},ensure_ascii=False),flush=True)
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 report={'scope':__doc__,'created':stamp,'source_sha256':deployed_source_hash(),'rows':rows,'current_passed':all(r['passed'] for r in rows if r['mode']=='explicit_role_grounding'),
 'limits':['Two repetitions are diagnostic, not a reliable error-rate estimate.','Prior prompt can pass these probes despite the user-reported failure.','Simple lexical checks catch specified regressions, not arbitrary German grammar errors.','No exact provider attribution for the original user quote: private transcripts were not retained.']}
 (ROOT/'static'/('social-dialogue-'+stamp+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print({'current_passed':report['current_passed']})
if __name__=='__main__':asyncio.run(main())
