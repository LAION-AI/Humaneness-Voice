"""Compare actual Robby director streams; synthetic prompts, no TTS or private memory."""
import asyncio, datetime, json, statistics, time
from pathlib import Path
import httpx
from bolt_director import BoltDirector

CASES = [
    ('greeting', 'Hallo Robby, wie geht es dir?'),
    ('joke', 'Erzähl mir einen kurzen Witz über Katzen.'),
    ('explanation', 'Warum ist der Himmel blau? Erklär es mir einfach.'),
    ('poem', 'Schreib mir ein kurzes Gedicht über einen freundlichen Roboter mit vier Zeilen.'),
]
MODELS = [('qwen', 'cerebras/fp16'), ('gemma12', 'local')]

async def main():
    rows=[]
    async with httpx.AsyncClient(timeout=90) as client:
        for label, prompt in CASES:
            for model, provider in MODELS:
                metrics={}; start=time.perf_counter(); parts=[]; error=None
                try:
                    async for part in BoltDirector().stream(client,prompt,{},[],backend={'model':model,'provider':provider},metrics=metrics):
                        parts.append(part.text)
                except Exception as exc:
                    error=type(exc).__name__+': '+str(exc)[:160]
                row={'case':label,'input':prompt,'model':model,'elapsed_ms':round((time.perf_counter()-start)*1000,1),
                     'metrics':metrics,'sentences':parts,'error':error}
                rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    report={'created':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope':'Actual Robby text director; synthetic independent prompts, no ASR, TTS or browser. First call may be cold.',
            'rows':rows,'medians':{model:{key:round(statistics.median(r['metrics'][key] for r in rows if r['model']==model and key in r['metrics']),1)
                                      for key in ('llm_ttft_ms','llm_first_sentence_ms','llm_total_ms')
                                      if any(r['model']==model and key in r['metrics'] for r in rows)} for model,_ in MODELS}}
    Path('static/gemma12-latency.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    if any(r['error'] or not r['sentences'] for r in rows):raise SystemExit('A model did not return a usable answer')

if __name__=='__main__':asyncio.run(main())
