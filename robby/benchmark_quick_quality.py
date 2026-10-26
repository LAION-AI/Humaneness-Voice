"""Reproducible local-model quality/latency probe with synthetic conversation.

Run baseline BEFORE edits; held-out evaluation AFTER selecting a candidate.
Seeds are controlled; calls are serial to avoid measuring self-induced queues.
This logs text for semantic review; format validity alone is not quality.
"""
import argparse,asyncio,datetime,hashlib,json,time
from pathlib import Path
import httpx
import quick_reaction as qr
from quick_reaction_cases import CASES
from speech_language import reply_language

ROOT=Path(__file__).resolve().parent

async def main(args):
    rows=[]
    path=ROOT/'static'/('quick-quality-'+args.label+'.json')
    if path.exists():raise SystemExit('Refusing to overwrite an existing experiment: '+str(path))
    report={'created':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':args.label,
        'scope':'Authored synthetic text; local E4B calls only. Quality needs semantic review. No TTS or microphone.',
        'source_sha256':hashlib.sha256((ROOT/'quick_reaction.py').read_bytes()).hexdigest(),
        'system':qr.SYSTEM,'temperature':getattr(qr,'TEMPERATURE',.35),'top_p':getattr(qr,'TOP_P',.9),
        'split':args.split,'rows':rows}
    async with httpx.AsyncClient(timeout=10) as client:
        class SeedClient:
            seed=0
            last_system=None
            async def post(self,url,*,json,**kwargs):
                self.last_system=json['messages'][0]['content']
                return await client.post(url,json={**json,'seed':self.seed},**kwargs)
        seeded=SeedClient()
        # Prime caches, record rather than silently discarding the cold call.
        report['warmup']=await qr.generate(seeded,'Hallo Robby!',{},[],timeout=2.)
        for c in CASES:
            if args.split!='all' and c['split']!=args.split:continue
            enabled=qr.eligible(c['text'],c['history'])
            for repeat in range(args.repeats if enabled else 1):
                seeded.seed=1103+repeat*101
                out=await qr.generate(seeded,c['text'],{},c['history'],reply_language(c['text'],c['history'])) if enabled else {'text':None,'status':'ineligible','ms':0}
                row={**c,'repeat':repeat,'seed':seeded.seed,'eligible':enabled,'reaction':out,
                     'request_system':seeded.last_system if enabled else None};rows.append(row)
                path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
                print(json.dumps({'case':c['name'],'repeat':repeat,'text':out.get('text'),'raw':out.get('raw'),'ms':out['ms'],'status':out['status']},ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--split',choices=['development','held_out','all'],default='development');p.add_argument('--repeats',type=int,default=3)
    asyncio.run(main(p.parse_args()))
