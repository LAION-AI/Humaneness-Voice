"""Real-model local latency measurements; authored synthetic audio only.

Every row records actual input/output words and all failures. The socket test
adds real-time pacing/endpointing separately; offline ASR is not called live E2E.
"""
import argparse,asyncio,base64,html,json,time,statistics
from pathlib import Path
import httpx,numpy as np
from scipy.signal import resample_poly

ROOT=Path(__file__).resolve().parent
INPUT='Was können wir heute nachmittags gemeinsam machen?'
OUTPUT='Natürlich, ich erzähle dir eine kleine Geschichte.'
INPUTS=[INPUT,'Welche kleine Geschichte kannst du mir erzählen?','Welches lustige Spiel könnten wir jetzt spielen?']

async def measure(client,route,body):
    start=time.perf_counter();first_sentence=None;first_audio=None;pcm=[];metadata={};texts=[]
    async with client.stream('POST',route,json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith('data: '):continue
            item=json.loads(line[6:]);now=(time.perf_counter()-start)*1000
            if item.get('error'):raise RuntimeError(item['error'])
            if item.get('audio'):
                if first_audio is None:first_audio=now
                pcm.append(np.frombuffer(base64.b64decode(item['audio']),'<i2').astype(np.float32)/32768)
            elif item.get('sentence_text') or item.get('manual_prompt'):
                if first_sentence is None:first_sentence=now
                texts.append(item.get('sentence_text') or item['text'])
            else:metadata=item
    audio=np.concatenate(pcm) if pcm else np.zeros(0,np.float32)
    return {'first_sentence_ms':first_sentence,'first_audio_ms':first_audio,
            'total_ms':(time.perf_counter()-start)*1000,'metrics':metadata,
            'output':' '.join(texts),'output_words':len(' '.join(texts).split()),'audio_s':len(audio)/24000},audio

def render(report):
    rows=report['rows'];summaries=[]
    for group in sorted({r['group'] for r in rows}):
        selected=[r for r in rows if r['group']==group and not r.get('error')]
        cells=[group,str(len(selected))+'/'+str(sum(r['group']==group for r in rows))]
        for key in ('llm_ttft_ms','first_sentence_ms','asr_ms','first_audio_ms','total_ms'):
            vals=[r[key] for r in selected if r.get(key) is not None]
            cells.append(f'{statistics.median(vals):.0f}' if vals else '—')
        summaries.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Robby: local Nemotron latency</title>
<style>body{font:16px/1.6 system-ui;max-width:1150px;margin:30px auto;padding:20px;background:#faf8f4;color:#20333a}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bfcbd0;padding:10px;text-align:left}.scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#176175}</style>
<h1>Robby: local Nemotron sentence streaming</h1><p>Measured on 25 October 2026. Values below are median milliseconds, measured at the local FastAPI client. Model weights: NVIDIA Nemotron 3.5 Bolt E4B Q4_K_M, served with llama.cpp CUDA; shared Humaneness Small RL TTS and frozen Nemotron ASR. Two RTX 3090 GPUs. Only newly authored synthetic speech is used.</p>
<div class="scroll"><table><tr><th>Condition</th><th>Successful runs</th><th>LLM first token</th><th>First sentence</th><th>ASR</th><th>First audio</th><th>Full generation</th></tr>'''+''.join(summaries)+'''</table></div>
<p>TTFT means first text token. First audio means the first playable PCM packet received, not the end of the spoken reply. Full generation is the time to finish generating all audio, not its playback duration. Warm steady-state samples use an already loaded model. Raw rows include actual input and output word counts, failures and per-sentence metrics.</p>
<p>“ASR + LLM + TTS (offline audio)” sends a complete prerecorded synthetic request to ASR, then its recognized text to the real LLM/TTS path. It excludes speaking duration, streaming endpoint detection, microphone hardware and browser/device/network buffering. A separately documented real-time browser/socket test measures that additional path; these numbers must not be presented as microphone-to-speaker latency.</p>
<h2>Pipeline and choices</h2><p>Stable ASR sentences are labelled using nearest full-utterance ECAPA voice prototypes. Unknown speakers are answered, numbered and optionally asked their name. Explicit self-identification can attach several acoustic prototypes to the same session person. The LLM outputs short acting cues and spoken words; 100 fixed GENERAL templates are matched in memory. A completed output sentence immediately enters streaming TTS while the remaining LLM output continues. New input sentences queue in order and enter the next LLM request with conversation history. Already decoding LLM requests cannot be retroactively edited.</p>
<p>The report compares 160, 400, 800 and 1600 ms requested TTS frame chunks (2, 5, 10, 20 frames). Actual MP3/PCM packet lengths can differ because of codec and final-tail boundaries. Smaller chunks may reduce first audio at the cost of more decoding/transport work. See the measured rows before choosing a default.</p>
<h2>Raw reproducible measurements</h2><pre>'''+html.escape(json.dumps(report,indent=2,ensure_ascii=False))+'''</pre><p><a href="/docs">FastAPI reference</a> · <a href="/desk-robot/">Robby demo</a></p></html>'''

async def main(args):
    report={'hardware':'2 x NVIDIA RTX 3090 24GB; LLM + ASR GPU0, TTS GPU1','input':INPUT,'input_words':len(INPUT.split()),
            'tts_only_text':OUTPUT,'tts_only_words':len(OUTPUT.split()),'started':time.time(),'runs':args.runs,'rows':[]}
    async with httpx.AsyncClient(base_url=args.base,timeout=240) as client:
        for repeat in range(args.runs):
            try:
                utterance=INPUTS[repeat%len(INPUTS)]
                row,_=await measure(client,'/api/bolt/llm',{'text':utterance,'reply_words':7})
                row.update(group='LLM only',repeat=repeat,input=utterance,input_words=len(utterance.split()),llm_ttft_ms=row['metrics'].get('llm_ttft_ms'))
            except Exception as exc:row={'group':'LLM only','repeat':repeat,'error':str(exc)}
            report['rows'].append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
        fixture=None
        for frames in args.chunks:
            for repeat in range(args.runs):
                try:
                    row,audio=await measure(client,'/api/bolt/tts',{'text':OUTPUT,'cue':'contentment, soft','chunk_frames':frames})
                    row.update(group=f'TTS only · {frames*80} ms chunks',repeat=repeat,chunk_frames=frames)
                    if fixture is None:
                        # A separate request sentence is synthesized below.
                        fixture=audio
                except Exception as exc:row={'group':f'TTS only · {frames*80} ms chunks','repeat':repeat,'error':str(exc)}
                report['rows'].append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
        successful=[r for r in report['rows'] if r['group'].startswith('TTS only') and not r.get('error')]
        if not successful:raise RuntimeError('No TTS measurements succeeded')
        best=min(args.chunks,key=lambda f:statistics.median(r['first_audio_ms'] for r in successful if r.get('chunk_frames')==f))
        report['lowest_first_audio_chunk_frames']=best
        _,input_audio=await measure(client,'/api/bolt/tts',{'text':INPUT,'cue':'interest, natural','chunk_frames':best,'voice':'anime_088'})
        audio16=resample_poly(input_audio,2,3).astype('<f4')
        work=ROOT/'logs/bolt-benchmark';work.mkdir(exist_ok=True);np.save(work/'input16.npy',audio16)
        for repeat in range(args.runs):
            for group in ('LLM + TTS','ASR + LLM + TTS (offline audio)'):
                start=time.perf_counter()
                try:
                    recognized=INPUTS[repeat%len(INPUTS)];asr_ms=None
                    if group.startswith('ASR'):
                        response=await client.post('/api/bolt/asr',content=audio16.tobytes());response.raise_for_status()
                        data=response.json();recognized=data['text'];asr_ms=data['asr_ms']
                    row,_=await measure(client,'/api/bolt/chat',{'text':recognized,'reply_words':7,'chunk_frames':best})
                    if asr_ms is not None:
                        row['first_audio_ms']+=asr_ms;row['total_ms']=(time.perf_counter()-start)*1000
                    row.update(group=group,repeat=repeat,asr_ms=asr_ms,input=recognized,input_words=len(recognized.split()),
                        llm_ttft_ms=row['metrics'].get('llm_ttft_ms'),chunk_frames=best)
                except Exception as exc:row={'group':group,'repeat':repeat,'error':str(exc)}
                report['rows'].append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
        report.update(finished=time.time(),scope='Local server generation; no physical microphone/device latency')
        (ROOT/'static/bolt-speed.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
        (ROOT/'static/bolt-speed.html').write_text(render(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8798');p.add_argument('--runs',type=int,default=3)
    p.add_argument('--chunks',type=lambda s:[int(x) for x in s.split(',')],default=[2,5,10,20]);asyncio.run(main(p.parse_args()))
