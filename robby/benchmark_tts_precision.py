"""Same-checkpoint precision / adapter-GEMM ablation on a temporary GPU copy.

Keeps the live TTS untouched. Baseline vs TF32, mixed/full BF16 and algebraic
LoRA merging. All warm-up/compilation costs and numerical changes are explicit.
Only newly authored synthetic speech is saved for listening and evaluation.
"""
import argparse,base64,copy,gc,hashlib,json,os,re,statistics,subprocess,sys,time,types
from pathlib import Path
import numpy as np,soundfile as sf,torch
from scipy.signal import resample_poly

ROOT=Path(__file__).resolve().parent
from bolt_text import Sentence
sys.path.insert(0,str(ROOT.parent/'humaneness-small'))
import settings
settings.ACTING_DEVICE='cuda:0';settings.DEVICE='cuda:0'
from acting_engine import ActingEngine,compile_decode_only
from rl_adapter import RLLinear
from performance_core import FastGenerator
from prompting import build,parse_literal
from dynamic_references import fixed_clip

VARIANTS=('fp32_strict','fp32_tf32','bf16_base_fp32_lora','bf16_all','fp32_merged','bf16_merged')
CASES=[('neutral_de','Natürlich, ich erzähle dir eine kleine Geschichte.','contentment, soft','de'),
       ('playful_de','Hui, ich drehe mich und habe Spaß!','amusement, playful','de'),
       ('firm_en','Stop right there, this is not acceptable!','anger, firm','en')]

def native_bf16_forward(self,x):
    base=self.base(x)
    return base+torch.nn.functional.linear(torch.nn.functional.linear(x.to(self.a.dtype),self.a),self.b).to(base.dtype)

def to16(wave,sr=48000):return resample_poly(wave,16000//np.gcd(sr,16000),sr//np.gcd(sr,16000)).astype(np.float32)
def wer(reference,hypothesis):
    a=re.findall(r'\w+',reference.casefold());b=re.findall(r'\w+',hypothesis.casefold());row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]/max(1,len(a))

def main(args):
    torch.set_num_threads(4);folder=ROOT/'logs/bolt-precision';folder.mkdir(exist_ok=True)
    report={'model':'small_rl','variants':list(args.variants),'seeds':args.seeds,'started':time.time(),'rows':[],
        'scope':'Isolated RTX3090 GPU0 copy; live GPU1 TTS unchanged. Paired prompts/seeds, but precision/merge can change sampled tokens.',
        'acceptance':'Small exploratory quality comparison, not proof of perceptual equivalence. Official reference FP32 remains available.'}
    target=ROOT/'static/bolt-precision.json'
    if args.resume and target.exists():
        previous=json.loads(target.read_text());report.update(previous)
        report['variants']=list(dict.fromkeys(previous.get('variants',[])+list(args.variants)))
    def save():target.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    engine=ActingEngine('small_rl');model=engine.model
    report['checkpoint']=engine.checkpoint_info
    # Preserve original CPU weights and nonpersistent rotary buffers so that
    # every arm starts from the same official FP32 residual policy.
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
    wrappers={name:module for name,module in model.named_modules() if isinstance(module,RLLinear)}
    raw_global=model.transformer.forward.__closure__
    # The compile-decode wrapper closes over the original forward. Recover
    # via closure name, avoiding nested compilation across precision arms.
    uncompiled=dict(zip(model.transformer.forward.__code__.co_freevars,
        [cell.cell_contents for cell in model.transformer.forward.__closure__ or ()])).get('forward',model.transformer.forward)
    reference,_=fixed_clip('emolia_c1699');reference_wave,ref_sr=sf.read(reference,dtype='float32',always_2d=True)
    specs={name:build({'text':text,'script':f'({direction}) '+text,'general':Sentence(text,direction).spec()['general']},
        'structured',language,literal=Sentence(text,direction).spec()['manual_prompt'],reference_identity=True)
        for name,text,direction,language in CASES}
    audio_paths=[]
    for variant in args.variants:
        completed={(r['case'],r['seed']) for r in report['rows'] if r['variant']==variant}
        if all((case[0],seed) in completed for case in CASES for seed in args.seeds):
            print('Reusing saved completed arm',variant,flush=True);continue
        for name,module in wrappers.items():
            parent,_,leaf=name.rpartition('.');setattr(model.get_submodule(parent),leaf,module)
            if 'forward' in module.__dict__:delattr(module,'forward')
        engine.fast.clear_graphs();model.transformer.forward=uncompiled
        model.to(dtype=torch.float32);model.load_state_dict(state,strict=True)
        with torch.no_grad():
            for name,value in buffers.items():
                parent,_,leaf=name.rpartition('.');getattr(model.get_submodule(parent),leaf).copy_(value.to(engine.device))
        torch.set_float32_matmul_precision('high' if variant=='fp32_tf32' else 'highest')
        torch.backends.cuda.matmul.allow_tf32=variant=='fp32_tf32';torch.backends.cudnn.allow_tf32=variant=='fp32_tf32'
        bf16=variant.startswith('bf16')
        if variant.endswith('_merged'):
            with torch.no_grad():
                for name,module in wrappers.items():
                    merged=torch.nn.Linear(module.base.in_features,module.base.out_features,bias=module.base.bias is not None,
                        device=engine.device,dtype=torch.float32)
                    merged.weight.copy_(module.base.weight+module.b@module.a)
                    if merged.bias is not None:merged.bias.copy_(module.base.bias)
                    parent,_,leaf=name.rpartition('.');setattr(model.get_submodule(parent),leaf,merged)
        if bf16:
            model.to(dtype=torch.bfloat16)
            # Rotary frequencies retain their original higher precision.
            for name,value in buffers.items():
                if 'inv_freq' in name:
                    parent,_,leaf=name.rpartition('.');setattr(model.get_submodule(parent),leaf,value.to(engine.device))
            if variant=='bf16_base_fp32_lora':
                for name,module in wrappers.items():
                    module.a.data=state[name+'.a'].to(engine.device,dtype=torch.float32)
                    module.b.data=state[name+'.b'].to(engine.device,dtype=torch.float32)
            elif variant=='bf16_all':
                for module in wrappers.values():module.forward=types.MethodType(native_bf16_forward,module)
        engine.policy_fp32=not bf16
        model.transformer.forward=compile_decode_only(uncompiled)
        engine.fast=FastGenerator(engine,graph_limit=2)
        warm_start=time.perf_counter()
        try:
            engine.generate_many([specs[CASES[0][0]]],guidance=2.,seed=777,reference_path=reference,
                temperature=.7,streaming=True,batch_size=1,chunk_frames=20)
            report.setdefault('warmup_ms',{})[variant]=round((time.perf_counter()-warm_start)*1000,2)
            for name,text,direction,language in CASES:
                for seed in args.seeds:
                    if (name,seed) in completed:continue
                    wave=engine.generate_many([specs[name]],guidance=2.,seed=seed,reference_path=reference,
                        temperature=.7,streaming=True,batch_size=1,chunk_frames=20)[0]
                    perf=copy.deepcopy(engine.last_performance);stem=f'{variant}-{name}-{seed}'
                    path=folder/(stem+'.npy');np.save(path,to16(wave,engine.sr));audio_paths.append(path)
                    row={'variant':variant,'case':name,'text':text,'seed':seed,'dtype':'bf16' if bf16 else 'fp32',
                        'merged':variant.endswith('_merged'),'device_ms':round(perf['device_seconds']*1000,3),
                        'first_audio_device_ms':round(perf['batches'][0]['first_audio_seconds']*1000,3),
                        'audio_s':len(wave)/engine.sr,'rtf':perf['aggregate_rtf'],
                        'codes_sha256':hashlib.sha256(engine.last_codes[0].numpy().tobytes()).hexdigest(),
                        'audio_file':str(path.relative_to(ROOT)),'performance':perf}
                    if seed==args.seeds[0]:
                        wav=folder/(stem+'.wav');mp3=folder/(stem+'.mp3');sf.write(wav,wave,engine.sr)
                        subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(wav),'-ac','1','-b:a','96k',str(mp3)],check=True)
                        row['listen_mp3_b64']=base64.b64encode(mp3.read_bytes()).decode()
                    report['rows'].append(row);save();print(json.dumps({k:row[k] for k in ('variant','case','seed','device_ms','first_audio_device_ms','audio_s')}),flush=True)
        except Exception as exc:
            report.setdefault('failures',[]).append({'variant':variant,'error':type(exc).__name__+': '+str(exc)[:500]});save();print(variant,'FAILED',type(exc).__name__,str(exc)[:180],flush=True)
    # Restore strict math before evaluating all arms with identical frozen
    # independent scoring models; release the temporary TTS allocation first.
    model.transformer.forward=uncompiled;engine.fast.clear_graphs();engine.model=None
    del model,wrappers,state,buffers,engine;gc.collect();torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(ROOT))
    from speaker_teacher import get_cached_teacher
    teacher=get_cached_teacher();reference_embedding=teacher.embed(to16(reference_wave.mean(1),ref_sr))
    from ears_reward import EarsModels
    ears=EarsModels('cuda:0');ears.load('medium')
    import httpx
    with httpx.Client(timeout=60) as client:
        for row in report['rows']:
            if 'quality' in row:continue
            audio=np.load(ROOT/row['audio_file'])
            try:
                response=client.post('http://127.0.0.1:8793/internal/transcribe',content=audio.astype('<f4').tobytes());response.raise_for_status()
                hyp=response.json()['text'];score=ears.predict([audio],size='medium')[0]
                row.update(hypothesis=hyp,wer=wer(row['text'],hyp),speaker_cosine=round(float(teacher.embed(audio)@reference_embedding),5),
                    quality=score['scores_raw'])
            except Exception as exc:row['quality_error']=type(exc).__name__+': '+str(exc)[:160]
            save()
    report['summary']=[]
    for variant in report['variants']:
        a=[r for r in report['rows'] if r['variant']==variant];q=[r for r in a if 'quality' in r]
        if not a:continue
        report['summary'].append({'variant':variant,'n':len(a),'scored':len(q),
            'median_device_ms':statistics.median(r['device_ms'] for r in a),
            'median_first_audio_device_ms':statistics.median(r['first_audio_device_ms'] for r in a),
            'median_rtf':statistics.median(r['rtf'] for r in a),
            'mean_wer':float(np.mean([r['wer'] for r in q])) if q else None,
            'mean_speaker_cosine':float(np.mean([r['speaker_cosine'] for r in q])) if q else None,
            'quality_means':{key:float(np.mean([r['quality'][key] for r in q])) for key in q[0]['quality']} if q else {}})
    report['finished']=time.time();save();print('SUMMARY',json.dumps(report['summary']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variants',type=lambda s:s.split(','),default=list(VARIANTS));p.add_argument('--seeds',type=lambda s:[int(v) for v in s.split(',')],default=[777,2026,4242]);p.add_argument('--resume',action='store_true');main(p.parse_args())
