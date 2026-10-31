"""Offline TTS speed + correctness check (loads its own engine copy on cuda:0).

Reports milliseconds of GPU work per 80-ms frame and time to the first audio
chunk, compares codec CUDA graphs against eager decoding for bit-identical
waveforms, and checks that the reference-voice cache hits for fresh file copies.
Run with the same environment as the server (see INFERENCE_OPTIMIZATIONS.md).
"""
import os,shutil,tempfile,time
import numpy as np
import torch
import settings
settings.ACTING_DEVICE=settings.DEVICE=os.environ.get('ACTING_DEVICE','cuda:0')
from acting_engine import ActingEngine
from dynamic_references import fixed_clip
from prompting import build
import performance_core

TEXT='Natürlich, ich erzähle dir gerne eine kleine Geschichte über einen mutigen kleinen Roboter, der das Meer sehen wollte.'
DIRECTION='contentment, soft'

def main():
    engine=ActingEngine('small_rl');reference,_=fixed_clip('emolia_c1699')
    general=("Use the same speaker as the reference audio. Preserve that speaker's voice identity. "
             "Use warm, subtle delivery, clear articulation and natural spontaneous phrasing.")
    literal=f'GENERAL: {general}\nSCRIPT:\n({DIRECTION}) {TEXT}'
    spec=build({'text':TEXT,'script':f'({DIRECTION}) {TEXT}','general':general},'structured','de',
               literal=literal,reference_identity=True)
    folder=tempfile.mkdtemp();timings=[]
    for i in range(3):
        copy=os.path.join(folder,f'reference-{i}.wav');shutil.copyfile(reference,copy)
        torch.cuda.synchronize();start=time.perf_counter();engine.reference(copy);torch.cuda.synchronize()
        timings.append(round((time.perf_counter()-start)*1000,1));engine.forget_reference(copy)
    print('reference preparation ms (first encodes, later copies should be ~1 ms):',timings)
    def generate(seed):
        torch.cuda.synchronize();start=time.perf_counter()
        wave=engine.generate_many([spec],guidance=2.,seed=seed,reference_path=reference,temperature=.8,
                                  streaming=True,batch_size=1,chunk_frames=20)[0]
        torch.cuda.synchronize()
        return wave,(time.perf_counter()-start)/len(engine.last_codes[0]),engine.last_performance['batches'][0]['first_audio_seconds']
    results={}
    for graph in (False,True):
        performance_core.CODEC_GRAPH=graph
        for seed in (1,2):generate(seed)  # warm compile/graph shapes
        runs=[generate(777+i) for i in range(3)];results[graph]=runs
        print(f'codec_graph={graph}: ms/frame',[round(r[1]*1000,1) for r in runs],
              'first_audio_ms',[round(r[2]*1000) for r in runs])
    identical=all(np.array_equal(a[0],b[0]) for a,b in zip(results[False],results[True]))
    print('waveforms bit-identical with and without codec graphs:',identical)

if __name__=='__main__':main()
