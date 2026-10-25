"""Compare real local decode engines on short German/English voice replies."""
import asyncio,json,os,time
from pathlib import Path
import httpx
from bolt_director import BoltDirector

CASES=['Was können wir heute nachmittags gemeinsam machen?',
       'Bitte erzähle mir eine kurze lustige Geschichte.',
       'Was ist siebzehn mal vier? Antworte kurz.',
       'Please name three fruits in one sentence.']

async def main():
    rows=[]
    async with httpx.AsyncClient(timeout=40) as client:
        for repeat in range(3):
            for text in CASES:
                for label,port in ([('baseline',8799),('DFlash',8800)] if repeat%2==0 else [('DFlash',8800),('baseline',8799)]):
                    os.environ['ROBO_BOLT_BASE']=f'http://127.0.0.1:{port}';metrics={};reply=[]
                    try:
                        async for sentence in BoltDirector().stream(client,text,{},[],metrics=metrics,reply_words=7):reply.append(sentence.text)
                        row={'engine':label,'repeat':repeat,'input':text,'reply':' '.join(reply),**metrics}
                    except Exception as exc:row={'engine':label,'repeat':repeat,'input':text,'error':str(exc)}
                    rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    report={'rows':rows,'finished':time.time(),'scope':'Real requests in rotated serial order, same model weights, 7-word response target; cache details are retained.'}
    (Path(__file__).parent/'static/bolt-engine-comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':asyncio.run(main())
