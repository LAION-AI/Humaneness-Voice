"""Large 1.7 controlled resident-GPU benchmark; no scoring/network in RTF.

Run with the server stopped and rtx3090-speed.env exported. The live assistant's
other workers stay resident. Cold capture rows are retained but never pooled
with the three measured seeds. Batch RTF divides by SUM of candidate durations.
"""
import gc,json,time,os,sys,hashlib
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
sys.path.insert(0,os.environ.get('HVS_RUNTIME',str(Path(__file__).resolve().parents[1]/'humaneness-small')))
import settings,performance_core
from prompting import build
ROOT=Path(__file__).resolve().parent

def main(engine=None):
    # Retain the unchanged monolithic control, even when serving uses the
    # layerwise flag. The separate bounded compiler experiment follows below.
    os.environ['HVS_LARGE17_LAYERWISE']='0'
    from large17_engine import Large17Engine
    torch.set_num_threads(6)
    torch._dynamo.config.cache_size_limit=32
    e=engine or Large17Engine();e.max_batch_size=16
    reference=settings.HERE/'fixed-references/audio/emolia_c1699.wav'
    texts=[('Hallo, schön, dass du da bist. Was möchtest du heute machen?',5.),
           ('Ich kann dir gerne eine kleine Geschichte erzählen. Heute hat ein neugieriger Roboter zum ersten Mal das Meer gesehen, und er konnte vor Freude kaum stillstehen.',10.)]
    specs=[build({'text':text,'script':'(warm, natural, spontaneous) '+text,
        'general':'Use the same speaker identity as the reference. Warm, natural spontaneous conversation.'},
        'structured','de',seconds=seconds,reference_identity=True) for text,seconds in texts]
    out={'hardware':torch.cuda.get_device_name(e.device),'torch':torch.__version__,
         'revision':e.release['revision'],'temperature':.8,'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
         'parameters':e.parameters,'precision':e.precision_report,'conditions':[],
         'scope':'Resident TTS, prefill, frame generation and causal PCM codec. Excludes LLM, ASR, alignment, ranking, MP3 and network.'}
    path=ROOT/'benchmark.json'
    def save():path.write_text(json.dumps(out,indent=2))
    t=time.perf_counter();e.reference(reference);torch.cuda.synchronize(e.device)
    out['reference_encode_seconds']=time.perf_counter()-t
    for optimized in (False,True):
      settings.FAST_INFERENCE=optimized;performance_core.CODEC_GRAPH=optimized
      import codec_fused
      if not optimized:codec_fused.uninstall(e.proc.audio_tokenizer)
      else:
       cached=performance_core._CODEC_GRAPHS.get(id(e.proc.audio_tokenizer))
       codec_fused.install(e.proc.audio_tokenizer,cached.states if cached else None)
      for guidance in (1.,2.,2.5):
       for streaming in (False,True):
        for batch in (1,):
         # Full monolithic compilation takes minutes per new shape. Batch
         # 1 is the matched control; layerwise_check runs the full batch sweep.
         # Keep the common deployed CFG=2 full sweep; add card settings at N=1/4/8.
         if guidance!=2 and batch not in (1,4,8):continue
         condition={'optimized':optimized,'cfg':guidance,'streaming':streaming,'batch':batch,'runs':[]}
         out['conditions'].append(condition);save()
         print('CONDITION',optimized,guidance,streaming,batch,flush=True)
         try:
          for repeat in range(4):
           chunks=[];torch.cuda.reset_peak_memory_stats(e.device);started=time.perf_counter()
           def receive(offset,waves):
            chunks.append({'seconds':time.perf_counter()-started,'samples':[len(x) for x in waves],'rms':[float(np.sqrt(np.mean(x*x))) if len(x) else 0. for x in waves]})
            return False
           waves=e.generate_many([specs[0]]*batch,guidance=guidance,seed=1700+repeat,reference_path=reference,
                temperature=.8,streaming=streaming,batch_size=batch,chunk_frames=20,max_frames=125,
                on_audio_batch=receive if streaming else None)
           wall=time.perf_counter()-started
           row={'warmup':repeat==0,'seed':1700+repeat,'wall_seconds':wall,'chunks':chunks,
                'peak_allocated_gb':torch.cuda.max_memory_allocated(e.device)/1e9,
                'peak_reserved_gb':torch.cuda.max_memory_reserved(e.device)/1e9,
                'actual_audio_seconds':[len(w)/e.sr for w in waves],
                'hit_frame_cap':[len(c)>=125 for c in e.last_codes],
                'rms':[float(np.sqrt(np.mean(w*w))) for w in waves],
                'performance':e.last_performance}
           condition['runs'].append(row);save()
           if repeat==1 and batch==1:
            name=f"sample-{'fast' if optimized else 'eager'}-cfg{guidance}-{'stream' if streaming else 'offline'}.wav"
            sf.write(ROOT/name,waves[0],e.sr);row['audio_file']=name;save()
           print('RUN',repeat,'first',round(chunks[0]['seconds'],3) if chunks else '-',
                 'wall',round(wall,3),'RTF',round(e.last_performance['aggregate_rtf'],4),flush=True)
         except Exception as exc:
          condition['error']=type(exc).__name__+': '+str(exc);save();print('FAIL',condition['error'],flush=True)
          if isinstance(exc,torch.cuda.OutOfMemoryError):
           e.fast.clear_graphs();gc.collect();torch.cuda.empty_cache()
          else:raise
    # A longer sentence confirms that the fast onset is not only a short fixture.
    out['long_streams']=[]
    for seed in (1801,1802,1803):
     chunks=[];started=time.perf_counter()
     waves=e.generate_many([specs[1]],guidance=2.,seed=seed,reference_path=reference,temperature=.8,
        streaming=True,chunk_frames=20,max_frames=200,on_audio=lambda w:chunks.append({'seconds':time.perf_counter()-started,'samples':len(w)}))
     name=f'long-monolithic-{seed}.wav';sf.write(ROOT/name,waves[0],e.sr)
     out['long_streams'].append({'seed':seed,'chunks':chunks,'performance':e.last_performance,'audio_file':name});save()
    from checks import run
    out['checks']=run(e,specs[0],reference)
    out['finished']=True;save()
    print('BENCHMARK COMPLETE',flush=True)

if __name__=='__main__':main()
