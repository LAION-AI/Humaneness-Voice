"""Actual search-vs-single/BoN measurements, atomic resume, no selected-only study.

Run with the demo stopped (one instance owns the GPUs). References stay local.
The benchmark reports search+scoring critical-path RTF per *delivered* audio,
not aggregate candidate RTF. Codecs/ASR preserve raw audio for fair comparison.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

import settings
from engine import SmallEngine, Scorers
from beam_search import ChunkSearch
from search_scoring import SearchScorer
from prompting import build
from small_app import encode_mp3

ROOT = settings.HERE / 'search'
CASES = [
    ('anger_en','en','You knew what this meant to me. Tell me the truth.',
     'Intense contained anger at a betrayal. Tight jaw, clipped attacks and hard consonants, believable confrontation, clear words.'),
    ('fear_en','en','Wait a second. I can hear someone outside the door.',
     'Strong intimate fear, breath held and trembling, urgent restrained panic, clear words rather than screaming.'),
    ('affection_de','de','Du bist wieder da. Ich habe dich wirklich sehr vermisst.',
     'Deep warm affection in a private reunion, gentle smiling breath and tender unhurried phrasing, sincere and natural.'),
    ('burst_en','en','Stop. You knew what this meant to me.',
     'Furious shouted confrontation, a frustrated groan before speech, a forceful angry voice with clear words.'),
]


def save(path, value):
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    tmp.replace(path)


def move_scorers(scorers, device):
    scorers.asr.device=device
    scorers.asr.model.to(device)
    scorers.retriever.device=device
    scorers.retriever.model.to(device)
    for head in [scorers.judge.gen,scorers.judge.bl]:
        head.device=torch.device(device)
        head.mlp.to(device)
        head.mu=head.mu.to(device)
        head.sd=head.sd.to(device)
    # Only one audio tower is necessary for the new batch-scoring path.
    # Final proxy scoring below also uses SearchScorer, not legacy Judge.score.
    scorers.judge.gen.embedder=scorers.retriever.model
    scorers.judge.bl.embedder=scorers.retriever.model
    torch.cuda.empty_cache()


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['smoke','measure','quality','audit'],default='smoke')
    args=parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    (ROOT/'audio').mkdir(exist_ok=True)
    (ROOT/'raw').mkdir(exist_ok=True)
    engine=SmallEngine()
    scorers=Scorers()
    reference=settings.HERE/'performance'/'benchmark-reference.wav'
    tasks=[]
    if args.phase=='smoke':
        tasks=[('two_gpu',0,777,4,1,10,'search')]
    elif args.phase=='measure':
        for layout in ['two_gpu','one_gpu']:
            for n in [4,8,12,16]:
                for chunk in [10,20]:
                    for width in [1,2]:
                        for seed in [777,2027]:
                            tasks.append((layout,0,seed,n,width,chunk,'search'))
    elif args.phase=='quality':
        for case in range(len(CASES)):
            for seed in [777,2027]:
                tasks.extend([('two_gpu',case,seed,1,1,10,'single'),('two_gpu',case,seed,8,1,10,'best_of_n')])
                for width in [1,2]:
                    for chunk in [10,20]:
                        tasks.append(('two_gpu',case,seed,8,width,chunk,'search'))
    else:
        # Fresh, withheld text for a systems/robustness check. All performance
        # configurations above used one anger text; do not call it a language-
        # universal real-time guarantee.
        for case in [1,2]:
            for n in [4,8,12,16]:
                for chunk in [10,20]:
                    tasks.append(('one_gpu',case,12345,n,1,chunk,'search'))
    environment={'model':settings.REPO,'revision':settings.REVISION,'stage':settings.STAGE,
        'tts_parameters':engine.parameters,'gpu':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'torch':torch.__version__,'cuda':torch.version.cuda,'cfg':1.5,'temperature':1.,'max_seconds':6.6,
        'cases':CASES,'seeds':[777,2027],'reference_provenance':json.loads((settings.HERE/'study/design.json').read_text()).get('reference_provenance'),
        'selection_weights':{'clap':.5,'genuineness':.25,'blend':.25},
        'reference_used':'A, same locally prepared clip as throughput benchmark',
        'reference_wav_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
        'extra_burst_case':'Exploratory structured groan/shout case added after the first three speech-only cases, to test an actual requested non-lexical event. No settings re-tuned.',
        'normalization':'fixed CLAP (cos+1)/2, genuine /6, blend /10; no within-set minmax',
        'intelligibility_gate':'max(0,1-prefixWER); final EOS/cap uses full-target WER',
        'device_arrangements':'two_gpu=TTS cuda:0/scorers cuda:1; one_gpu=all active compute/models on cuda:0 (second GPU unused)',
        'quality_endpoints':'untrimmed output, final full-target Parakeet WER and shared VoiceCLAP proxies; same reward models used for search and evaluation, not independent human validation'}
    save(ROOT/'design.json',environment)
    current_device=None
    for layout,case,seed,n,width,chunk,kind in tasks:
        name=f'{args.phase}-{layout}-{CASES[case][0]}-{kind}-n{n}-b{width}-c{chunk}-s{seed}'
        path=ROOT/(name+'.json')
        if path.is_file():
            continue
        device='cuda:0' if layout=='one_gpu' else 'cuda:1'
        if device!=current_device:
            move_scorers(scorers,device)
            current_device=device
        label,lang,text,direction=CASES[case]
        if label=='burst_en':
            spec=build({'text':text,'general':direction,'script':
                '(frustrated groan, 0.3 seconds)\n(angry shout, forceful and clear) [3.2 seconds duration] '+text},
                'structured',lang,seconds=6.6)
        else:
            spec=build({'text':text,'caption':direction},'caption',lang,seconds=6.6)
        scorer=SearchScorer(scorers,spec)
        # The first seed is a warm-up/systems check; keep it as an actual
        # observation with capture costs flagged, never silently discard it.
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
        started=time.perf_counter()
        try:
            commits=[]
            if kind=='search':
                wave,performance=ChunkSearch(engine,scorer).generate(spec,n,width,chunk,6.6,1.5,1.,seed,reference,
                    on_commit=lambda w,e:commits.append({'seconds':len(w)/48000,'at_wall_seconds':time.perf_counter()-started}))
            else:
                waves=engine.generate_many([spec]*n,1.5,seed,reference,1.,streaming=False,batch_size=n,max_frames=83)
                scores=scorer.score(waves,text,[True]*n)
                winner=max(range(n),key=lambda i:scores[i]['reward'])
                wave=waves[winner]
                elapsed=time.perf_counter()-started
                performance={'n':n,'beam_width':1,'chunk_ms':None,'search_wall_seconds':elapsed,
                    'actual_seconds':len(wave)/48000,'search_rtf':elapsed/(len(wave)/48000),
                    'first_commit_seconds':elapsed,'graph_capture_seconds':engine.last_performance['graph_capture_seconds'],
                    'tts_codec_seconds':engine.last_performance['device_seconds'],'scoring_seconds':scorer.last_timing['total_seconds'],
                    'candidate_scores':scores,'winner':winner,'candidate_device_performance':engine.last_performance}
            # Independent *endpoint* pass, but not an independent reward model.
            final=scorer.score([wave],text,[True])[0]
            filename=name+'.mp3'
            encode_mp3(wave,ROOT/'audio'/filename)
            sf.write(ROOT/'raw'/(name+'.wav'),wave,48000,subtype='FLOAT')
            result={'id':name,'layout':layout,'case':label,'seed':seed,'n':n,'beam_width':width,'chunk_frames':chunk,
                'kind':kind,'phase':args.phase,'final':final,'performance':performance,'commits':commits,
                'peak_vram_gib':[torch.cuda.max_memory_allocated(i)/2**30 for i in range(torch.cuda.device_count())],
                'audio':'audio/'+filename,'pcm_sha256':hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest()}
            save(path,result)
            print(json.dumps({'id':name,'rtf':performance['search_rtf'],'first':performance['first_commit_seconds'],
                'wer':final['prefix_wer'],'reward':final['reward'],'capture':performance['graph_capture_seconds'],
                'vram':result['peak_vram_gib']}),flush=True)
        except torch.cuda.OutOfMemoryError:
            engine.fast.clear_graphs()
            torch.cuda.empty_cache()
            save(path,{'id':name,'layout':layout,'case':label,'seed':seed,'n':n,'beam_width':width,
                       'chunk_frames':chunk,'kind':kind,'phase':args.phase,'error':'CUDA out of memory'})
            print(json.dumps({'id':name,'error':'CUDA out of memory'}),flush=True)
        if args.phase=='smoke':
            print('SMOKE complete',flush=True)


if __name__=='__main__':
    run()
