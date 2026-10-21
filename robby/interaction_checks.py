"""Actual configured Luna tool calls, public text fixtures only; no live mic."""
import asyncio,json,time
from pathlib import Path
import httpx
from conversation_director import CompactDirector
from interaction_policy import RuleBook
from test_live_decisions import REQUEST

async def main():
    report={'started':time.time(),'passed':False,'actual_configured_luna_api':True,'private_microphone_used':False,'cases':[]}
    target=Path(__file__).parent/'logs/interaction-tools-checks.json'
    target.write_text(json.dumps(report))
    identity={'speaker_id':'synthetic-christoph','display_name':'Christoph','enrolled':True,'named':True}
    state={'enabled':True,'rules':[]};book=RuleBook();history=[]
    async with httpx.AsyncClient() as client:
        for label,text in [('install',REQUEST),('ordinary','Erzähl mir einen kurzen Satz über Äpfel.'),('cancel','Beende das Spiel und unterbrich mich nicht mehr wegen Bananen.')]:
            result=await CompactDirector().generate(client,text,identity,history,interaction_state=state)
            actions=result.get('interaction_actions',[]);book.apply(actions,identity['speaker_id'],text)
            if label=='install':assert book.rules and any(r.matches('Bananen sind gesund.') for r in book.rules.values())
            if label=='ordinary':assert not actions and book.rules
            if label=='cancel':assert not book.rules
            state={'enabled':True,'rules':book.public()};history.extend([{'role':'user','content':text},{'role':'assistant','content':result['text']}])
            row={'case':label,'reply':result['text'],'actions':actions,'director_ms':result['director_ms'],'passed':True}
            report['cases'].append(row);target.write_text(json.dumps(report,indent=2));print(json.dumps(row,ensure_ascii=False),flush=True)
    report.update(passed=True,finished=time.time());target.write_text(json.dumps(report,indent=2))

if __name__=='__main__':asyncio.run(main())
