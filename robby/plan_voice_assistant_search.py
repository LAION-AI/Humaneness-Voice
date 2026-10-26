"""Create a deterministic configuration plan; never changes the running server."""
import argparse, hashlib, itertools, json, shlex, sys
from pathlib import Path

def configurations(matrix):
    axes=matrix['implemented_search_axes']; keys=sorted(axes)
    for values in itertools.product(*(axes[k] for k in keys)):
        settings=dict(zip(keys,values))
        identity=hashlib.sha256(json.dumps(settings,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:12]
        yield {'id':identity,'settings':settings}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='logs/voice-assistant-search-plan.json')
    p.add_argument('--url',default='http://127.0.0.1:8793')
    a=p.parse_args();root=Path(__file__).parent
    matrix=json.loads((root/'voice_assistant_test_matrix.json').read_text());rows=[]
    for row in configurations(matrix):
        s=row['settings']
        command=[sys.executable,'run_voice_assistant_checks.py','--suites','unit,stress',
            '--threshold',str(s['focus_threshold']),'--pause-ms',str(s['speech_pause_ms']),
            '--url',a.url,'--speakers',','.join(matrix['development_speakers']),
            '--seeds',','.join(map(str,matrix['development_augmentation_seeds']))]
        rows.append({**row,'component_screen_command':shlex.join(command),
            'finalist_browser_command':shlex.join([sys.executable,'run_voice_assistant_checks.py',
                '--suites','decision,browser,stops','--threshold',str(s['focus_threshold']),
                '--pause-ms',str(s['speech_pause_ms']),'--url',a.url])})
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'executed':False,'configurations':rows,
        'selection':'Filter on task/false-trigger gates first; inspect Pareto frontier of recall, errors and p95 latency. Never select on held-out results.',
        'pause_warning':'Component cropping alone does not establish endpoint latency; replay the real browser for every finalist.'},indent=2))
    print(f'Wrote {len(rows)} candidate configurations to {out}; no experiments started.')

if __name__=='__main__':main()
