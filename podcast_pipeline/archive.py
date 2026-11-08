"""Bounded CPU archival: publish a turn receipt only after every artifact exists."""
import time
from pathlib import Path
import soundfile as sf
from safetensors.torch import save_file
from recipe import dump,specification

def archive_batch(out,batch,waves,codes,scores,perf,refs,n,total_s):
    tick=time.perf_counter()
    for group,(lang,turn) in enumerate(batch):
        local=scores[group*n:(group+1)*n]
        winner=max(range(n),key=lambda i:(local[i]['reward'],-i));idx=group*n+winner
        folder=Path(out)/f'bestof{n}-speaker5050'/f"{lang}-{turn['index']:02d}";folder.mkdir(parents=True,exist_ok=True)
        for i in range(n):sf.write(folder/f'candidate-{i}.flac',waves[group*n+i],48000,subtype='PCM_24')
        sf.write(folder/'raw.flac',waves[idx],48000,subtype='PCM_24')
        assert not codes[idx].is_cuda,'Archival must not touch the GPU'
        save_file({'codes':codes[idx]},str(folder/'selected-codes.safetensors'))
        per=dict(**perf,mode='turn',first_commit_seconds=total_s,raw_seconds=len(waves[idx])/48000,
            final_scores=local,winner=winner,group_index=group,group_count=len(batch),
            timing_allocation='batch wall time divided equally among independent segments; first_commit is full batch latency')
        dump(folder/'result.json',dict(language=lang,turn=turn,variant=f'bestof{n}-speaker5050',
            reference=refs[turn['speaker']],performance=per,prompt=specification(turn,lang)['prompt'],generated_unix=time.time()))
        print('TURN',lang,turn['index'],'winner',winner,'G',round(local[winner]['ears']['genuineness_0_6'],3),
            'ECAPA',round(local[winner]['ecapa_cosine'],3),'batch_s',round(total_s,2),flush=True)
    return dict(seconds=time.perf_counter()-tick,turns=[t['index'] for _,t in batch],language=batch[0][0])
