"""Actual local/API multi-turn handoff; keeps full generated answers in history."""
import asyncio,datetime,json,time
from pathlib import Path
import httpx
from bolt_director import BoltDirector
from quick_reaction import generate,eligible,repeated_opening
ROOT=Path(__file__).resolve().parent
TURNS=['Erzähl mir einen Katzenwitz.','Noch einen bitte.','Und noch einen!',
       'Der war lahm. Noch einen, aber lustiger.','Erzähl stattdessen einen Pinguinwitz.','Noch einen bitte.']
async def main():
    path=ROOT/'static/quick-quality-dialogue.json'
    if path.exists():raise SystemExit('Refusing to overwrite a trial')
    report={'created':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Authored real-model text conversation; no TTS playback.','rows':[]};history=[]
    async with httpx.AsyncClient(timeout=40) as client:
        for text in TURNS:
            start=time.perf_counter();parts=[];first=None
            async def answer():
                nonlocal first
                async for part in BoltDirector().stream(client,text,{},history,avatar_context={'parallel_quick_reaction':True},backend={'model':'qwen','provider':'cerebras/fp16'}):
                    if first is None:first=(time.perf_counter()-start)*1000
                    parts.append(part.text)
            reaction,_=await asyncio.gather(generate(client,text,{},history),answer())
            picked=reaction.get('text') if first is not None and reaction['ms']<first else None
            reply=' '.join(([picked] if picked else [])+parts)
            row={'input':text,'reaction':reaction,'played_reaction':picked,'main_first_ms':first,'main_answer':parts,
                 'repeated_played_opening':bool(picked and repeated_opening(picked,history))}
            report['rows'].append(row);path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
            history.extend([{'role':'user','content':text},{'role':'assistant','content':reply}])
            print(json.dumps(row,ensure_ascii=False),flush=True)
if __name__=='__main__':asyncio.run(main())
