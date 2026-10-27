"""Authored two-stage trials: real Bolt, Qwen and shared-GPU TTS at T=0.8.

Run from this folder with the existing services idle. No microphone is used.
Receipts are immutable by default; pass another --output for a new run.
"""
import argparse,asyncio,base64,datetime,json,time
from pathlib import Path
import httpx,numpy as np
from bolt_pipeline import pipeline
ROOT=Path(__file__).resolve().parent
CASES=[
 ('sky','Warum ist der Himmel blau?',[]),
 ('joke','Erzähl mir einen kurzen lustigen Katzenwitz.',[]),
 ('tired','Heute war ein anstrengender Tag. Wie kann ich etwas abschalten?',[]),
 ('english','Please speak English. Why do cats purr?',[]),
]
async def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='two-stage-audio.json');p.add_argument('--cases',default='all');p.add_argument('--legacy',action='store_true');args=p.parse_args()
    if args.legacy:
        import os
        os.environ['ROBO_INFORMED_BRIDGE']='0'
    path=ROOT/'static'/args.output
    if path.exists():raise SystemExit('Refusing to overwrite an existing trial')
    report={'created':datetime.datetime.now(datetime.timezone.utc).isoformat(),'temperature':.8,'legacy':args.legacy,
        'scope':'Authored text -> Bolt + Qwen -> PCM; excludes ASR/network-to-browser/physical playback. Virtual playback gaps use a fixed 220 ms initial reserve, then uninterrupted consumption.','rows':[]}
    async with httpx.AsyncClient(base_url='http://127.0.0.1:8792',timeout=120) as client:
        for name,text,history in CASES:
            if args.cases!='all' and name not in args.cases.split(','):continue
            state=(await client.get('/api/state')).json()
            if state.get('busy') or state.get('phase')!='ready':raise RuntimeError('Shared TTS not idle and ready')
            events=[];waves=[];role=None;cursor=None;gaps=[];first=None;start=time.perf_counter()
            try:
                async for kind,value in pipeline(client,text,{},history,llm_selection={'model':'qwen','provider':'cerebras/fp16'}):
                    now=time.perf_counter()-start
                    if kind=='sentence':role=value['sentence_role']
                    if kind=='audio':
                        if first is None:first=now;cursor=now+.22
                        gap=max(0.,now-cursor)
                        if gap>.005:gaps.append({'at_ms':round(now*1000,2),'gap_ms':round(gap*1000,2),'role':role})
                        cursor=max(cursor,now)+len(value)/24000;waves.append(value)
                        events.append({'event':kind,'at_ms':round(now*1000,2),'samples':len(value),'role':role})
                    else:events.append({'event':kind,'at_ms':round(now*1000,2),'value':value})
            except Exception as exc:
                events.append({'event':'error','value':type(exc).__name__+': '+str(exc)})
            pcm=np.clip(np.concatenate(waves) if waves else np.zeros(0),-1,1)
            proc=await asyncio.create_subprocess_exec('ffmpeg','-v','error','-f','s16le','-ar','24000','-ac','1','-i','pipe:0','-b:a','96k','-f','mp3','pipe:1',stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE)
            mp3,_=await proc.communicate((pcm*32767).astype('<i2').tobytes());assert proc.returncode==0
            row={'case':name,'input':text,'history':history,'events':events,'first_pcm_ms':round(first*1000,2) if first else None,
                'audio_s':len(pcm)/24000,'virtual_gaps':gaps,'listen_mp3_b64':base64.b64encode(mp3).decode()}
            report['rows'].append(row);path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
            print(json.dumps({k:v for k,v in row.items() if k not in ('events','listen_mp3_b64','history')},ensure_ascii=False),flush=True)
            for _ in range(40):
                if not (await client.get('/api/state')).json().get('busy'):break
                await asyncio.sleep(.05)
if __name__=='__main__':asyncio.run(main())
