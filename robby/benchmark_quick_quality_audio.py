"""Real final bridge -> Qwen -> streaming TTS, authored listening examples.

No private recordings. PCM timings start at text submission, exclude ASR and
physical playback. Store mono 96-kbps MP3 for reproducible listening.
"""
import asyncio,base64,datetime,json,time
from pathlib import Path
import httpx,numpy as np
from quick_reaction_cases import CASES
ROOT=Path(__file__).resolve().parent

async def main():
    path=ROOT/'static/quick-quality-audio.json'
    if path.exists():raise SystemExit('Refusing to overwrite a trial')
    report={'created':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Four authored live API/Qwen/TTS trials; not microphone-to-ear or a human listening rating.','rows':[]}
    async with httpx.AsyncClient(base_url='http://127.0.0.1:8798',timeout=120) as client:
        for name in ['cat_joke','another_joke','bereavement','english_loss']:
            c=next(x for x in CASES if x['name']==name)
            if (await client.get('http://127.0.0.1:8792/api/state')).json().get('busy'):raise RuntimeError('TTS occupied; leave the user alone')
            events=[];audio=[];role=None;first=None;answer=None;start=time.perf_counter()
            body={'text':c['text'],'history':c['history'],'llm_model':'qwen','llm_provider':'cerebras/fp16','quick_replies':True,'chunk_frames':20}
            async with client.stream('POST','/api/bolt/chat',json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith('data: '):continue
                    value=json.loads(line[6:]);ms=round((time.perf_counter()-start)*1000,2)
                    if value.get('error'):raise RuntimeError(value['error'])
                    if 'sentence_role' in value:role=value['sentence_role']
                    if value.get('audio'):
                        if first is None:first=ms
                        if role=='answer' and answer is None:answer=ms
                        audio.append(np.frombuffer(base64.b64decode(value.pop('audio')),'<i2'))
                    events.append({'at_ms':ms,**value})
            pcm=np.concatenate(audio) if audio else np.zeros(0,'<i2')
            proc=await asyncio.create_subprocess_exec('ffmpeg','-v','error','-f','s16le','-ar','24000','-ac','1','-i','pipe:0','-b:a','96k','-f','mp3','pipe:1',stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE)
            mp3,_=await proc.communicate(pcm.tobytes());assert proc.returncode==0
            row={'case':name,'input':c['text'],'history':c['history'],'events':events,'first_pcm_ms':first,
                 'first_main_audio_ms':answer,'audio_s':len(pcm)/24000,'listen_mp3_b64':base64.b64encode(mp3).decode()}
            report['rows'].append(row);path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
            print(json.dumps({k:v for k,v in row.items() if k not in ('events','listen_mp3_b64','history')},ensure_ascii=False),flush=True)
if __name__=='__main__':asyncio.run(main())
