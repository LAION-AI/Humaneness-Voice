"""Four preregistered small students; speaker-disjoint, causal evaluation.

No live microphone data is read. All teachers and Nemotron remain frozen.
Run via python -m speaker_v3.overnight --stage train (resumes checkpoints).
"""
import html,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from speaker_v2.train import roc_metrics

CONFIGS=[{'name':'attentive_stats','architecture':'stats','width':128},
         {'name':'query_gru','architecture':'gru','width':128},
         {'name':'causal_tcn','architecture':'tcn','width':128},
         {'name':'temporal_transformer','architecture':'transformer','width':128}]


def checkpoint(value,path):
    temporary=path.with_suffix('.tmp.pt');torch.save(value,temporary);temporary.replace(path)


class CausalBlock(nn.Module):
    def __init__(self,width,dilation):
        super().__init__();self.d=dilation
        self.conv=nn.Conv1d(width,width,3,dilation=dilation,groups=width)
        self.mix=nn.Conv1d(width,width,1);self.norm=nn.LayerNorm(width)
    def forward(self,x):
        z=self.conv(F.pad(x.transpose(1,2),(2*self.d,0)))
        return self.norm(x+F.gelu(self.mix(z).transpose(1,2)))


class Student(nn.Module):
    """Predict newest 320-ms identity from a causal 80-ms hidden-state sequence.

    Query-conditioned attention selects relevant past frames after a change of
    speaker. Long-window mean/std is optional, separately ablated, never required.
    """
    def __init__(self,architecture='stats',width=128):
        super().__init__();self.architecture=architecture;self.width=width
        self.input=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,width),nn.GELU())
        self.energy=nn.Linear(1,width)
        self.long=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,64),nn.GELU())
        if architecture=='gru':self.temporal=nn.GRU(width,width,batch_first=True)
        elif architecture=='tcn':self.temporal=nn.Sequential(*[CausalBlock(width,d) for d in (1,2,4)])
        elif architecture=='transformer':
            layer=nn.TransformerEncoderLayer(width,4,width*2,.1,batch_first=True,norm_first=True)
            self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            self.position=nn.Parameter(torch.randn(25,width)*.01)
        else:self.temporal=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,width))
        self.key=nn.Linear(width,width,bias=False);self.query=nn.Linear(width,width,bias=False)
        self.head=nn.Sequential(nn.Linear(width*3+64,384),nn.GELU(),nn.Dropout(.15))
        self.embedding=nn.Linear(384,192);self.activity=nn.Linear(384,3)
        self.register_buffer('teacher_mean',torch.zeros(192));self.register_buffer('teacher_radius',torch.tensor(1.))

    def forward(self,x,pool,energy,use_pool=True):
        x=self.input(x)+self.energy(((energy.float()+40)/40).clamp(-2,2).unsqueeze(-1))
        if self.architecture=='gru':x,_=self.temporal(x)
        elif self.architecture=='transformer':
            x=x+self.position[-x.shape[1]:]
            mask=torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool).triu(1)
            x=self.temporal(x,mask=mask)
        else:x=self.temporal(x)
        current=x[:,-4:].mean(1)
        weights=(self.key(x)*self.query(current).unsqueeze(1)).sum(-1)/(self.width**.5)
        # Soft energy prior, not a hard VAD: low-energy speakers remain possible.
        weights=F.softmax(weights+(energy.float()+45).clamp(-20,0)/10,dim=1)
        mean=(weights.unsqueeze(-1)*x).sum(1)
        std=((weights.unsqueeze(-1)*(x-mean.unsqueeze(1)).square()).sum(1)+1e-5).sqrt()
        long=self.long(pool.flatten(1)) if use_pool else torch.zeros(len(x),64,device=x.device,dtype=x.dtype)
        z=self.head(torch.cat([current,mean,std,long],-1))
        direction=F.normalize(self.embedding(z).float(),dim=-1)
        return F.normalize(self.teacher_mean+direction*self.teacher_radius,dim=-1),self.activity(z).float()


def batch(arrays,indices,device,context=25,pool=True):
    x=np.asarray(arrays['q'][indices,-context:],np.float32)*np.asarray(arrays['scale'][indices,-context:],np.float32)
    values=[x,np.asarray(arrays['pool'][indices],np.float32),np.asarray(arrays['energy'][indices,-context:],np.float32),
            np.asarray(arrays['local_target'][indices],np.float32),np.asarray(arrays['activity'][indices],np.int64),
            np.asarray(arrays['speaker'][indices],np.int64)]
    return [torch.from_numpy(np.ascontiguousarray(a)).to(device) for a in values]


@torch.inference_mode()
def predict(model,arrays,indices,device,context=25,use_pool=False):
    outputs=[];states=[]
    for offset in range(0,len(indices),128):
        x,p,e,_,_,_=batch(arrays,indices[offset:offset+128],device,context)
        embedding,activity=model(x,p,e,use_pool)
        outputs.append(embedding.cpu().numpy());states.append(activity.argmax(-1).cpu().numpy())
    return np.concatenate(outputs),np.concatenate(states)


def metrics(pred,arrays,indices,prototypes,threshold=None):
    emb,states=pred;ids=np.asarray(arrays['speaker'][indices]);classes=np.asarray(arrays['activity'][indices]);valid=classes==0
    gallery=np.unique(ids[valid]);speaker=ids[valid];e=emb[valid]
    if len(gallery)<2:raise ValueError('At least two evaluation voices required')
    scores=e@prototypes[gallery].T;own=speaker[:,None]==gallery[None,:]
    result=roc_metrics(scores[own],scores[~own],threshold)
    correct=gallery[scores.argmax(1)]==speaker
    result.update(speaker_accuracy=float(correct.mean()),macro_speaker_accuracy=float(np.mean([correct[speaker==s].mean() for s in gallery])),
                  activity_accuracy=float((states==classes).mean()),speakers=int(len(gallery)),windows=len(indices),speech_windows=int(valid.sum()))
    target=np.asarray(arrays['local_target'][indices[valid]],np.float32)
    target/=np.maximum(np.linalg.norm(target,axis=-1,keepdims=True),1e-8)
    result['source_utterance_teacher_cosine']=float((e*target).sum(-1).mean())
    result['per_speaker']={str(int(s)):{'windows':int((speaker==s).sum()),'accuracy':float(correct[speaker==s].mean()),
        'cosine':float(scores[own][speaker==s].mean())} for s in gallery}
    per_speaker=np.array([r['accuracy'] for r in result['per_speaker'].values()])
    bootstrap=np.random.default_rng(77).choice(per_speaker,(2000,len(per_speaker))).mean(1)
    result['macro_accuracy_speaker_bootstrap_95ci']=[float(v) for v in np.quantile(bootstrap,[.025,.975])]
    for i,name in enumerate(('speech','silence','overlap')):
        use=classes==i;result[name+'_recall']=float((states[use]==i).mean()) if use.any() else None
    result['scenarios']={}
    scenario=np.asarray(arrays['scenario'][indices])[valid]
    for i,name in enumerate(('clean','noise','reverb','telephone','echo')):
        use=scenario==i
        if use.any():result['scenarios'][name]={'windows':int(use.sum()),'accuracy':float(correct[use].mean()),'cosine':float(scores[own][use].mean())}
    if 'emotion' in arrays:
        from .overnight import EMOTIONS
        emotion=np.asarray(arrays['emotion'][indices])[valid];result['emotions']={}
        for i,name in enumerate(EMOTIONS):
            use=emotion==i
            if use.any():result['emotions'][name]={'windows':int(use.sum()),'accuracy':float(correct[use].mean()),'cosine':float(scores[own][use].mean())}
    if threshold is not None:
        enrolled=gallery[::2];known=np.isin(speaker,enrolled);score=(e@prototypes[enrolled].T).max(1)
        result['known_acceptance']=float((score[known]>=threshold).mean())
        result['unknown_rejection']=float((score[~known]<threshold).mean())
        result['unknown_protocol']='Alternating evaluation speaker IDs enrolled; remaining identities unknown; validation-calibrated threshold'
    return result


def write_report(out,results,finished=False):
    table=''.join('<tr><td>'+html.escape(r['name'])+'</td><td>'+str(r['parameters'])+'</td><td>'+str(r['best_epoch'])+
        '</td><td>'+f"{r['validation']['positive_cosine']:.3f}"+'</td><td>'+f"{100*r['validation']['macro_speaker_accuracy']:.1f}%"+
        '</td><td>'+f"{100*r['validation']['eer']:.1f}%"+'</td></tr>' for r in results)
    payload=json.dumps({'finished':finished,'results':results},indent=2,allow_nan=False)
    (out/'results.json').write_text(payload)
    (out/'report.html').write_text('''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>TYGER V3 speaker study</title>
<style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:20px;background:#faf8f3;color:#302f2b}td,th{padding:12px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap;font-size:13px}a{color:#735b38}</style>
<h1>Speaker prediction from frozen Nemotron hidden states</h1><p>'''+('Completed evaluation.' if finished else 'Training in progress; provisional validation results.')+'''</p>
<table><tr><th>Candidate</th><th>Parameters</th><th>Selected epoch</th><th>Enrollment cosine</th><th>Macro speaker accuracy</th><th>EER</th></tr>'''+table+'''</table>
<p>480 training / 10 validation / 10 test synthetic voice identities, disjoint. Each prediction is aligned to a new 320-ms step (four 80-ms L15 states). State-pooling spans: 320, 640, 1040, and 2000 ms; optional pooled eight-second representation is reported separately. These are NOT audio-history limits: each hidden state already has the past context of its causal eight-second encoder window. No future audio. These are intermediate states from the current window, not just states cached when earlier windows ended.</p>
<p>Targets: frozen ECAPA 192D embeddings of clean source utterances; separate enrollment-only utterances define identity prototypes. Training combines raw and centered cosine, normalized MSE, prototype contrastive loss and an activity head. Silence/ambiguous overlap have no regression target. This is a synthetic-voice study, not secure identity verification.</p>
<p>Four architectures and context tests are declared before test evaluation. Selection uses validation only; held-out test is evaluated once after selection and never used to retune. Windows share utterances and voices, so they are not independent statistical trials. Random source excerpts, five augmentation scenarios and small same-split cohorts do not replace real-room or real-human evaluation. The echo scenario is a synthetic delayed source track, not a full independent robot/user interaction.</p>
<h2>Reproducibility and complete ablations</h2><pre>'''+html.escape(payload)+'''</pre><p><a href="/speaker-training/">Live training status</a> · <a href="/">Robby</a></p>''')


def run_four(work,update):
    work=Path(work);out=work/'models';out.mkdir(exist_ok=True)
    arrays={p.stem:np.load(p,mmap_mode='r') for p in (work/'arrays').glob('*.npy')}
    splits={s:np.asarray(v,np.int64) for s,v in json.loads((work/'splits.json').read_text()).items()}
    if not arrays['filled'].all():raise ValueError('Incomplete feature cache; refusing incomplete training')
    rows=json.loads((work/'sources_manifest.json').read_text());prototypes=np.load(work/'prototypes.npy').astype(np.float32)
    prototypes/=np.maximum(np.linalg.norm(prototypes,axis=-1,keepdims=True),1e-8)
    trainvoices=np.array(sorted(r['speaker_idx'] for r in rows if r['split']=='train'))
    device='cuda:0' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    if device!='cpu':torch.cuda.set_per_process_memory_fraction(.10,0)
    train_indices=splits['train'];val_indices=splits['validation'];test_indices=splits['test']
    counts=np.bincount(arrays['activity'][train_indices],minlength=3).clip(1)
    majority_activity=int(counts.argmax())
    class_weights=torch.tensor(counts**-.5,dtype=torch.float32,device=device);class_weights/=class_weights.mean()
    proto=torch.from_numpy(prototypes).to(device);mean=proto[trainvoices].mean(0);radius=(proto[trainvoices]-mean).norm(dim=-1).mean()
    contrast=F.normalize(proto[trainvoices]-mean,dim=-1)
    lookup=torch.full((500,),-1,dtype=torch.long,device=device);lookup[torch.tensor(trainvoices,device=device)]=torch.arange(len(trainvoices),device=device)
    results=[];rng=np.random.default_rng(1137)
    for number,cfg in enumerate(CONFIGS):
        seed=1730+number;torch.manual_seed(seed);random.seed(seed)
        model=Student(cfg['architecture'],cfg['width']).to(device);model.teacher_mean.copy_(mean);model.teacher_radius.copy_(radius)
        parameters=sum(p.numel() for p in model.parameters())
        if not 100000<=parameters<=1000000:raise ValueError('Student parameter budget violated: '+str(parameters))
        optimizer=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=.003)
        latest=out/(cfg['name']+'-resume.pt');bestfile=out/(cfg['name']+'-best.pt')
        start_epoch=0;best=-1e9;history=[];best_epoch=0
        if latest.exists():
            saved=torch.load(latest,map_location=device,weights_only=False);model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
            start_epoch=saved['epoch'];history=saved['history'];best=saved['best'];best_epoch=saved['best_epoch']
            if 'numpy_rng' in saved:rng.bit_generator.state=saved['numpy_rng']
            if 'torch_rng' in saved:torch.set_rng_state(saved['torch_rng'].cpu())
            if device!='cpu' and 'cuda_rng' in saved:torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),0)
        start=time.perf_counter();batch_size=128
        for epoch in range(start_epoch,18):
            model.train();order=rng.permutation(train_indices);running=0.;epoch_start=time.perf_counter()
            for step,offset in enumerate(range(0,len(order),batch_size)):
                from .overnight import guard
                if step%100==0:guard()
                selected=order[offset:offset+batch_size];context=int(rng.choice([4,8,13,25]));use_pool=bool(rng.random()<.25)
                x,p,e,target,activity,speaker=batch(arrays,selected,device,context)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device!='cpu'):
                    embedding,logits=model(x,p,e,use_pool);valid=activity==0
                    regression=embedding.sum()*0
                    if valid.any():
                        a=embedding[valid];b=F.normalize(target[valid],dim=-1);ac=F.normalize(a-mean,dim=-1);bc=F.normalize(b-mean,dim=-1)
                        raw=(1-(a*b).sum(-1)).mean();center=(1-(ac*bc).sum(-1)).mean()
                        identity=F.cross_entropy(ac@contrast.T/.12,lookup[speaker[valid]])
                        regression=.4*raw+.6*center+.05*F.mse_loss(a,b)*192+.2*identity
                    loss=regression+.2*F.cross_entropy(logits,activity,weight=class_weights)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; checkpoint retained')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step();running+=float(loss.detach())*len(selected)
                if step%100==0:update(phase='Training four causal students',candidate=cfg['name'],candidate_number=number+1,epoch=epoch+1,
                    batch=step,train_seconds=time.perf_counter()-start,parameters=parameters)
            model.eval();validation=metrics(predict(model,arrays,val_indices,device),arrays,val_indices,prototypes)
            objective=validation['macro_speaker_accuracy']-.5*validation['eer']+.1*validation['source_utterance_teacher_cosine']
            history.append({'epoch':epoch+1,'loss':running/len(order),'seconds':time.perf_counter()-epoch_start,'validation':validation})
            if objective>best:
                best=objective;best_epoch=epoch+1
                checkpoint({'model':model.state_dict(),'config':cfg,'parameters':parameters,'epoch':best_epoch,'seed':seed,
                    'validation':validation,'format_version':3,'teacher_revision':'0f99f2d0ebe89ac095bcc5903c4dd8f72b367286'},bestfile)
            checkpoint({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':cfg,'epoch':epoch+1,'history':history,
                'best':best,'best_epoch':best_epoch,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                **({'cuda_rng':torch.cuda.get_rng_state(0)} if device!='cpu' else {})},latest)
            update(phase='Training four causal students',candidate=cfg['name'],epoch=epoch+1,validation=validation)
            if epoch>=8 and epoch+1-best_epoch>=5:break
        saved=torch.load(bestfile,map_location=device,weights_only=False);model.load_state_dict(saved['model']);model.eval()
        validation=saved['validation'];threshold=validation['far05_threshold'];contexts={}
        for context in (4,8,13,25):
            contexts[str(context*80)+'ms']=metrics(predict(model,arrays,val_indices,device,context),arrays,val_indices,prototypes,threshold)
        contexts['2000ms_plus_8s_pool']=metrics(predict(model,arrays,val_indices,device,25,True),arrays,val_indices,prototypes,threshold)
        # CUDA timing excludes feature extraction, transport and teacher inference.
        x,p,e,*_=batch(arrays,val_indices[:1],device,25)
        with torch.inference_mode():
            for _ in range(10):model(x,p,e,False)
            if device!='cpu':torch.cuda.synchronize()
            t=time.perf_counter()
            for _ in range(100):model(x,p,e,False)
            if device!='cpu':torch.cuda.synchronize()
        result={**cfg,'parameters':parameters,'best_epoch':saved['epoch'],'validation':validation,'validation_contexts':contexts,
            'student_batch1_ms':(time.perf_counter()-t)*10,'device':device,'train_seconds_this_run':time.perf_counter()-start,'history':history}
        results.append(result);write_report(out,results)
        del model,optimizer
        if device!='cpu':torch.cuda.empty_cache()
    winner=max(results,key=lambda r:r['validation']['macro_speaker_accuracy']-.5*r['validation']['eer']+.1*r['validation']['source_utterance_teacher_cosine'])
    # One held-out test pass for every preregistered context of the selected model.
    saved=torch.load(out/(winner['name']+'-best.pt'),map_location=device,weights_only=False)
    model=Student(saved['config']['architecture'],saved['config']['width']).to(device);model.load_state_dict(saved['model']);model.eval()
    winner['selected']=True;threshold=winner['validation']['far05_threshold'];winner['test_contexts']={}
    for split,ids in [('validation',val_indices),('test',test_indices)]:
        baseline=np.broadcast_to(F.normalize(mean,dim=-1).cpu().numpy(),(len(ids),192))
        majority=np.full(len(ids),majority_activity)
        source=np.asarray(arrays['local_target'][ids],np.float32)
        source/=np.maximum(np.linalg.norm(source,axis=-1,keepdims=True),1e-8)
        winner[split+'_controls']={
            'constant_training_teacher_mean':metrics((baseline,majority),arrays,ids,prototypes,threshold),
            'clean_full_source_utterance_teacher':metrics((source,majority),arrays,ids,prototypes,threshold),
            'note':'Teacher uses clean full source audio, not equal-context/noise; activity is a training-majority constant in both controls.'}
    for context in (4,8,13,25):
        winner['test_contexts'][str(context*80)+'ms']=metrics(predict(model,arrays,test_indices,device,context),arrays,test_indices,prototypes,threshold)
    winner['test_contexts']['2000ms_plus_8s_pool']=metrics(predict(model,arrays,test_indices,device,25,True),arrays,test_indices,prototypes,threshold)
    write_report(out,results,True);update(phase='Four students complete; held-out evaluation ready',selected=winner['name'],results_path=str(out/'report.html'))


if __name__=='__main__':
    from .overnight import WORK,update
    run_four(WORK,update)
