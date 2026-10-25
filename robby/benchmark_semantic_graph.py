"""Paired full TTS comparison: compiled dynamic KV vs static KV CUDA graph.

Uses a temporary GPU0 copy. Generation hashes and cross-request cache resets
are checked; production GPU1 is untouched. Run without competing ASR tests.
"""
import argparse,copy,hashlib,json,os,sys,time,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'humaneness-small'))
os.environ['HVS_RL_PRECISION']='fp32_merged'
os.environ.pop('HVS_SEMANTIC_CUDA_GRAPH',None)
import settings
settings.ACTING_DEVICE=settings.DEVICE='cuda:0'
from acting_engine import ActingEngine
from semantic_graph import SemanticGraphs
from benchmark_tts_precision import CASES
from bolt_text import Sentence
from prompting import build
from dynamic_references import fixed_clip

def main(target_kind='semantic'):
    torch.set_num_threads(4)
    report={'started':time.time(),'precision':'fp32_merged','rows':[],
        'scope':'Temporary RTX3090 GPU0 copy, shared production unchanged. Both arms use local acoustic CUDA graphs.'}
    target=ROOT/'static'/('bolt-local-compile.json' if target_kind=='local' else 'bolt-semantic-graph.json')
    def save():target.write_text(json.dumps(report,indent=2))
    engine=ActingEngine('small_rl');report['checkpoint']=engine.checkpoint_info
    reference,_=fixed_clip('emolia_c1699')
    specs={name:build({'text':text,'script':f'({direction}) '+text,'general':Sentence(text,direction).spec()['general']},
        'structured',language,literal=Sentence(text,direction).spec()['manual_prompt'],reference_identity=True)
        for name,text,direction,language in CASES}
    modes=('local_eager_graph','local_compiled_graph') if target_kind=='local' else ('compiled_dynamic','static_graph')
    original_local=engine.fast.local_frame
    for mode in modes:
        if mode=='local_compiled_graph':
            engine.fast.clear_graphs()
            engine.fast.local_frame=torch.compile(original_local,dynamic=False,fullgraph=True)
        engine.fast.semantic=SemanticGraphs(engine.model) if mode=='static_graph' else None
        started=time.perf_counter()
        try:
            engine.generate_many([specs[CASES[0][0]]],guidance=2.,seed=777,reference_path=reference,
                temperature=.7,streaming=True,batch_size=1,chunk_frames=20)
            report.setdefault('warmup_ms',{})[mode]=(time.perf_counter()-started)*1000
            for repeat in range(1 if target_kind=='local' else 2):
                for name,text,direction,language in CASES:
                    for seed in (777,2026,4242):
                        wave=engine.generate_many([specs[name]],guidance=2.,seed=seed,reference_path=reference,
                            temperature=.7,streaming=True,batch_size=1,chunk_frames=20)[0]
                        perf=copy.deepcopy(engine.last_performance)
                        row={'mode':mode,'case':name,'seed':seed,'repeat':repeat,
                            'device_ms':perf['device_seconds']*1000,
                            'first_audio_device_ms':perf['batches'][0]['first_audio_seconds']*1000,
                            'audio_s':len(wave)/engine.sr,'rtf':perf['aggregate_rtf'],
                            'codes_sha256':hashlib.sha256(engine.last_codes[0].numpy().tobytes()).hexdigest(),
                            'wave_sha256':hashlib.sha256(wave.tobytes()).hexdigest()}
                        report['rows'].append(row);save();print(json.dumps(row),flush=True)
        except Exception:
            report.setdefault('failures',[]).append({'mode':mode,'traceback':traceback.format_exc()});save()
            traceback.print_exc()
    report['finished']=time.time();save()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--target',choices=['semantic','local'],default='semantic')
    main(parser.parse_args().target)
