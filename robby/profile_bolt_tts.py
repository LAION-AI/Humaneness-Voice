"""Isolated same-checkpoint GPU stage profiling, without changing the live TTS.

Allocates a temporary model on GPU 0, reusing cached released weights. Run only
when no ASR/browser latency benchmark is active. CUDA events are recorded without
synchronizing each layer, so the measured AR path keeps its ordinary overlap.
"""
import json,sys,time,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SMALL=ROOT.parent/'humaneness-small'
sys.path.insert(0,str(SMALL))
import torch
import settings
settings.ACTING_DEVICE='cuda:0'
settings.DEVICE='cuda:0'
from acting_engine import ActingEngine
from performance_core import StreamingCodec
from prompting import build,parse_literal
from dynamic_references import fixed_clip

def main():
    torch.set_num_threads(4)
    engine=ActingEngine('small_rl')
    prompt='GENERAL: Use the same speaker as the reference audio. Gentle, natural and spontaneous delivery.\nSCRIPT:\n(contentment, soft) Natürlich, ich erzähle dir eine kleine Geschichte.'
    literal=parse_literal(prompt,'structured')
    spec=build(literal,'structured','de',literal=prompt,reference_identity=True)
    reference,_=fixed_clip('emolia_c1699')
    print('Warming same-checkpoint path',flush=True)
    engine.generate_many([spec],guidance=2.,seed=777,temperature=.7,reference_path=reference,
                         streaming=True,batch_size=1,chunk_frames=20)
    original_global=engine.model.transformer.forward
    original_frame=engine.fast.frame
    original_codec=StreamingCodec.decode
    samples=[]
    def measure(label,fn,*args,**kwargs):
        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record();result=fn(*args,**kwargs);end.record()
        samples.append((label,start,end));return result
    def global_forward(*args,**kwargs):
        return measure('semantic_prefill' if kwargs.get('past_key_values') is None else 'semantic_step',original_global,*args,**kwargs)
    def local_frame(*args,**kwargs):return measure('acoustic_token_prediction',original_frame,*args,**kwargs)
    def codec_decode(self,*args,**kwargs):return measure('waveform_codec',original_codec,self,*args,**kwargs)
    engine.model.transformer.forward=global_forward
    engine.fast.frame=local_frame
    StreamingCodec.decode=codec_decode
    rows=[]
    try:
        for repeat in range(3):
            samples.clear();start=time.perf_counter()
            waves=engine.generate_many([spec],guidance=2.,seed=777,temperature=.7,reference_path=reference,
                streaming=True,batch_size=1,chunk_frames=20)
            torch.cuda.synchronize(engine.device)
            stages={}
            for label,a,b in samples:stages.setdefault(label,[]).append(a.elapsed_time(b))
            row={'repeat':repeat,'wall_ms':round((time.perf_counter()-start)*1000,2),
                'stages_ms':{k:round(sum(v),3) for k,v in stages.items()},
                'calls':{k:len(v) for k,v in stages.items()},'audio_s':len(waves[0])/engine.sr,
                'performance':engine.last_performance}
            rows.append(row);print(json.dumps(row),flush=True)
    finally:
        engine.model.transformer.forward=original_global;engine.fast.frame=original_frame;StreamingCodec.decode=original_codec
    report={'model':'small_rl','checkpoint':engine.checkpoint_info,'policy_dtype':str(next(engine.model.parameters()).dtype),
        'parameters':{'semantic_backbone':sum(p.numel() for p in engine.model.transformer.parameters()),
                      'local_transformer':sum(p.numel() for p in engine.model.local_transformer.parameters()),
                      'waveform_codec':sum(p.numel() for p in engine.proc.audio_tokenizer.parameters())},
        'device':'RTX 3090 GPU 0; temporary duplicate while live GPU1 server remains untouched',
        'scope':'CUDA event spans around semantic backbone, whole local 12-code token step, and stateful waveform codec; excludes prompt/reference setup and server postprocessing',
        'caveat':'Instrumentation adds event launch overhead. No per-layer synchronizations. Same weights/precision/reference/CFG/chunk length as production, but an isolated process on the other RTX 3090.',
        'rows':rows,'finished':time.time()}
    (ROOT/'static/bolt-tts-profile.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('MEDIANS',json.dumps({key:round(statistics.median(r['stages_ms'].get(key,0) for r in rows),3) for key in rows[0]['stages_ms']}),flush=True)

if __name__=='__main__':main()
