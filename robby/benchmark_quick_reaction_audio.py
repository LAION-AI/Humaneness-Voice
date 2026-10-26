"""Paired real API/LLM/TTS trials; saves only authored synthetic responses."""
import asyncio,base64,datetime,json,time
from pathlib import Path
import httpx,numpy as np
from benchmark_bolt import measure

ROOT=Path(__file__).resolve().parent
INPUTS=['Heute war ein wirklich anstrengender Tag. Erzähl mir etwas Aufmunterndes.',
        'Was ist der Sinn des Lebens? Gib mir eine kurze persönliche Einschätzung.',
        'Warum ist der Himmel blau? Erklär es einfach.']

async def main():
    rows=[];stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    async with httpx.AsyncClient(base_url='http://127.0.0.1:8798',timeout=90) as c:
        for case,text in enumerate(INPUTS):
            for enabled in ([False,True] if case%2==0 else [True,False]):
                state=(await c.get('http://127.0.0.1:8792/api/state')).json()
                if state.get('busy'):raise RuntimeError('Existing TTS request in progress; do not displace it')
                start=time.perf_counter();events=[];audio=[];role=None;first=None;first_answer=None
                body={'text':text,'llm_model':'qwen','llm_provider':'cerebras/fp16','quick_replies':enabled,'chunk_frames':20}
                async with c.stream('POST','/api/bolt/chat',json=body) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith('data: '):continue
                        value=json.loads(line[6:]);at=round((time.perf_counter()-start)*1000,2)
                        if value.get('error'):raise RuntimeError(value['error'])
                        if 'sentence_role' in value:role=value['sentence_role']
                        if value.get('audio'):
                            if first is None:first=at
                            if role=='answer' and first_answer is None:first_answer=at
                            wave=np.frombuffer(base64.b64decode(value.pop('audio')),'<i2');audio.append(wave)
                        events.append({'at_ms':at,**value})
                pcm=np.concatenate(audio) if audio else np.zeros(0,'<i2')
                # MP3 export for blind listening, not an input to any scoring claim.
                process=await asyncio.create_subprocess_exec('ffmpeg','-v','error','-f','s16le','-ar','24000','-ac','1','-i','pipe:0','-b:a','96k','-f','mp3','pipe:1',stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE)
                mp3,_=await process.communicate(pcm.tobytes());assert process.returncode==0
                row={'case':case,'input':text,'quick_replies':enabled,'first_pcm_ms':first,'first_answer_pcm_ms':first_answer,
                    'audio_s':len(pcm)/24000,'events':events,'listen_mp3_b64':base64.b64encode(mp3).decode()}
                rows.append(row)
                raw=json.dumps({'scope':'Six real paired generation trials; no physical microphone or playback latency. Model outputs are stochastic, not identical between conditions.','rows':rows},indent=2,ensure_ascii=False)
                (ROOT/'logs'/('quick-reaction-audio-'+stamp+'.json')).write_text(raw);(ROOT/'static/quick-reaction-audio.json').write_text(raw)
                print(json.dumps({k:row[k] for k in ('case','quick_replies','first_pcm_ms','first_answer_pcm_ms','audio_s')}),flush=True)
if __name__=='__main__':asyncio.run(main())
