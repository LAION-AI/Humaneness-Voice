"""Read-only local-LLM benchmark. DOES NOT install or activate gpu-unlock.

Run only when other users have finished latency-sensitive tests. Requests
are serial, bounded and use only synthetic public prompts. The JSON includes
actual server timings and shared-GPU load, not an inferred unlock speedup.
"""
import argparse, ast, hashlib, json, statistics, subprocess, threading, time
from pathlib import Path
import httpx

ROOT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def gpu():
    fields=['index','name','driver_version','memory.used','utilization.gpu','pstate','power.draw','power.limit','clocks.sm','clocks.mem','temperature.gpu']
    result=subprocess.run(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return [dict(zip(fields,[p.strip() for p in line.split(',')])) for line in result.stdout.splitlines()]

def processes():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return [dict(zip(('pid','process_name','used_memory_mib'),[p.strip() for p in line.split(',')])) for line in result.stdout.splitlines()]

def system_prompt():
    file=ROOT.parent/'robo-integration/bolt_director.py'
    tree=ast.parse(file.read_text())
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SYSTEM' for t in n.targets)))

def request(client,url,messages):
    payload={'model':'nemotron-bolt','messages':messages,'stream':True,
        'stream_options':{'include_usage':True},'max_tokens':96,'temperature':0,
        'seed':20261026,'top_p':1.,'cache_prompt':False,
        'chat_template_kwargs':{'enable_thinking':False,'conversational_mode':False}}
    start=time.perf_counter();first=None;text=[];timings=None;usage=None;finish=None;samples=[];done=threading.Event()
    def monitor():
        while not done.is_set():
            try:samples.append({'elapsed_ms':round((time.perf_counter()-start)*1000,2),'gpus':gpu()})
            except (OSError,subprocess.CalledProcessError):pass
            done.wait(.5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        with client.stream('POST',url+'/v1/chat/completions',json=payload,timeout=90) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith('data: ') or line[6:]=='[DONE]':continue
                event=json.loads(line[6:]);choices=event.get('choices') or []
                if event.get('error'):raise RuntimeError('Local model stream reported an error')
                if event.get('timings'):timings=event['timings']
                if event.get('usage'):usage=event['usage']
                if not choices:continue
                delta=choices[0].get('delta',{}).get('content') or ''
                if delta:
                    if first is None:first=(time.perf_counter()-start)*1000
                    text.append(delta)
                if choices[0].get('finish_reason'):finish=choices[0]['finish_reason']
    finally:done.set();thread.join(timeout=2)
    spoken=''.join(text)
    return {'client_ttft_ms':None if first is None else round(first,3),
        'client_total_ms':round((time.perf_counter()-start)*1000,3),
        'server_timings':timings,'usage':usage,'finish_reason':finish,
        'output_sha256':hashlib.sha256(spoken.encode()).hexdigest(),'output_text':spoken,
        'request_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        'gpu_samples':samples}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8799');p.add_argument('--label',default='unchanged-shared-gpu')
    p.add_argument('--repetitions',type=int,default=3);p.add_argument('--out',default=str(ROOT/'baseline.json'))
    a=p.parse_args()
    if not 1<=a.repetitions<=10:p.error('Use 1–10 repetitions')
    url=a.url.rstrip('/');system=system_prompt();synthetic_notes='\n'.join(f'Notiz {i}: Robby hat {i%7+1} kleine bunte Murmeln und besucht Station {i}.' for i in range(96))
    cases=[('short_prompt',[{'role':'system','content':'Antworte auf Deutsch. Schreibe nur den gewünschten Inhalt.'},{'role':'user','content':'Nenne zwanzig freundliche Aktivitäten für einen kleinen Roboter.'}]),
        ('production_system',[{'role':'system','content':system},{'role':'user','content':'Erzähle eine freundliche Geschichte über einen kleinen Roboter in mindestens zehn Sätzen.'}]),
        ('long_prefill',[{'role':'system','content':'Nutze die folgenden synthetischen Notizen als Kontext. Antworte auf Deutsch.\n'+synthetic_notes},{'role':'user','content':'Erzähle daraus eine lange freundliche Geschichte mit mindestens zehn Sätzen.'}])]
    with httpx.Client(timeout=90) as client:
        props=client.get(url+'/props');props.raise_for_status();props=props.json()
        model_path=Path(props['model_path']);upstream=ROOT/'upstream'
        report={'schema_version':1,'label':a.label,'started_unix':time.time(),
            'scope':'Local llama.cpp serial synthetic generation; existing server unchanged; shared GPU, not an unlocked-vs-locked experiment.',
            'gpu_unlock_applied_by_this_script':False,'gpu_unlock_module_present':Path('/sys/module/gpu_unlock').exists(),
            'gpu_unlock_source_commit':subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip(),
            'llama_source_commit':subprocess.check_output(['git','-C','/home/c4r33u19/moss15v2/llama.cpp','rev-parse','HEAD'],text=True).strip(),
            'model':{'file':model_path.name,'bytes':model_path.stat().st_size,'sha256':sha(model_path),'ftype':props.get('model_ftype'),'slots':props.get('total_slots'),'context_per_slot':props['default_generation_settings'].get('n_ctx')},
            'system_sha256':hashlib.sha256(system.encode()).hexdigest(),'gpu_before':gpu(),'processes_before':processes(),
            'sampling':{'temperature':0,'seed':20261026,'max_tokens':96,'cache_prompt':False,'concurrency':1},'cases':[]}
        out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
        for name,messages in cases:
            print('Warmup',name,flush=True);warmup=request(client,url,messages);rows=[]
            for repeat in range(a.repetitions):
                print('Measured',name,repeat,flush=True);rows.append({'repeat':repeat,**request(client,url,messages)})
                out.write_text(json.dumps(report|{'in_progress':True},indent=2,ensure_ascii=False))
            row={'case':name,'messages':messages,'warmup':warmup,'runs':rows,
                'summary':{'repetitions':len(rows),'ttft_median_ms':round(statistics.median(r['client_ttft_ms'] for r in rows),3),
                    'total_median_ms':round(statistics.median(r['client_total_ms'] for r in rows),3),
                    'decode_median_tokens_s':round(statistics.median(r['server_timings']['predicted_per_second'] for r in rows),3) if all(r['server_timings'] and 'predicted_per_second' in r['server_timings'] for r in rows) else None,
                    'prefill_median_tokens_s':round(statistics.median(r['server_timings']['prompt_per_second'] for r in rows),3) if all(r['server_timings'] and 'prompt_per_second' in r['server_timings'] for r in rows) else None,
                    'identical_output_across_repeats':len({r['output_sha256'] for r in rows})==1}}
            report['cases'].append(row);out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
            print(json.dumps({'case':name,**row['summary']}),flush=True)
        report.update(finished_unix=time.time(),gpu_after=gpu(),processes_after=processes(),passed=all(r['server_timings'] for c in report['cases'] for r in c['runs']))
        out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('Saved',out,flush=True)

if __name__=='__main__':main()
