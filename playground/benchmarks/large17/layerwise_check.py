"""Compare a smaller compile unit after the unchanged Small-optimization sweep."""
import gc,importlib,json,time
from pathlib import Path
import torch
from prompting import build
import settings,performance_core,semantic_graph

def main(e):
    e.fast.clear_graphs();gc.collect();torch.cuda.empty_cache()
    importlib.reload(semantic_graph)
    from semantic_layerwise import LayerwiseSemanticBank
    e.fast.semantic=semantic_graph.SemanticGraphs(e.model)
    e.fast.semantic.bank_class=LayerwiseSemanticBank
    settings.FAST_INFERENCE=True;performance_core.CODEC_GRAPH=True
    text='Hallo, schön, dass du da bist. Was möchtest du heute machen?'
    spec=build({'text':text,'script':'(warm, natural, spontaneous) '+text,
        'general':'Use the same speaker identity as the reference. Warm, natural spontaneous conversation.'},
        'structured','de',seconds=5.,reference_identity=True)
    reference=settings.HERE/'fixed-references/audio/emolia_c1699.wav'
    out={'variant':'layerwise semantic compile + full continuation CUDA graph','conditions':[]}
    path=Path(__file__).with_name('layerwise.json')
    def save():path.write_text(json.dumps(out,indent=2))
    cases=[(2.,b,s) for b in (1,2,4,8,16) for s in (True,False)]
    cases += [(1.,b,s) for b in (1,4,8,16) for s in (True,False)]
    cases += [(2.5,1,s) for s in (True,False)]
    for guidance,batch,streaming in cases:
     condition={'batch':batch,'streaming':streaming,'cfg':guidance,'runs':[]};out['conditions'].append(condition);save()
     try:
      for repeat in range(4):
       start=time.perf_counter();chunks=[];torch.cuda.reset_peak_memory_stats(e.device)
       def receive(offset,waves):
           chunks.append({'seconds':time.perf_counter()-start,'samples':[len(x) for x in waves]});return False
       waves=e.generate_many([spec]*batch,guidance=guidance,seed=1700+repeat,reference_path=reference,temperature=.8,
           streaming=streaming,batch_size=batch,chunk_frames=20,max_frames=125,on_audio_batch=receive if streaming else None)
       condition['runs'].append({'warmup':repeat==0,'wall_seconds':time.perf_counter()-start,
           'chunks':chunks,'peak_allocated_gb':torch.cuda.max_memory_allocated(e.device)/1e9,
           'performance':e.last_performance,'audio_seconds':[len(w)/e.sr for w in waves]});save()
       print('LAYERWISE',guidance,batch,streaming,repeat,round(condition['runs'][-1]['wall_seconds'],3),flush=True)
     except torch.cuda.OutOfMemoryError as exc:
      condition['error']=str(exc);save();e.fast.clear_graphs();gc.collect();torch.cuda.empty_cache()
    import soundfile as sf
    longtext='Ich kann dir gerne eine kleine Geschichte erzählen. Heute hat ein neugieriger Roboter zum ersten Mal das Meer gesehen, und er konnte vor Freude kaum stillstehen.'
    longspec=build({'text':longtext,'caption':'Warm, natural, spontaneous conversation.'},'caption','de',seconds=10.,reference_identity=True)
    out['long_streams']=[]
    for seed in (1801,1802,1803):
        start=time.perf_counter();chunks=[]
        waves=e.generate_many([longspec],guidance=2.,seed=seed,reference_path=reference,temperature=.8,
            streaming=True,chunk_frames=20,max_frames=200,on_audio=lambda w:chunks.append({'seconds':time.perf_counter()-start,'samples':len(w)}))
        name=f'long-{seed}.wav';sf.write(Path(__file__).with_name(name),waves[0],e.sr)
        out['long_streams'].append({'seed':seed,'chunks':chunks,'performance':e.last_performance,'audio_file':name});save()
    from checks import run
    out['checks']=run(e,spec,reference);out['finished']=True;save()
