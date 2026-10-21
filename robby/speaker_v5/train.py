"""Matched context-specific finetuning; held-out test is delayed until selection."""
import json,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from speaker_v3.train import checkpoint,metrics
from speaker_v4.train import MixedStudent,Pairs,regression,objective,aggregates
from speaker_v4 import data as v4
from .common import OLD,WORK,ARMS,atomic,guard,update,report

@torch.inference_mode()
def predict(model,r,indices):
    model.eval();emb=[];states=[]
    for offset in range(0,len(indices),256):
        x,e,*_=r.batch(indices[offset:offset+256]);a,b=model(x,e)
        emb.append(a.cpu().numpy());states.append(b.argmax(-1).cpu().numpy())
    return np.concatenate(emb),np.concatenate(states)

def evaluate(model,r,split,prototypes,threshold=None):
    indices=r.split[split];emb,states=predict(model,r,indices)
    result=metrics((emb,states),r.arrays,r.indices[indices],prototypes,threshold)
    valid=r.classes_cpu[indices]==0;who=r.ids_cpu[indices]
    # Same split gallery as main metrics, not all 500 identities.
    gallery=np.unique(who[valid]);pred=gallery[(emb@prototypes[gallery].T).argmax(-1)]
    age=r.ages[indices];result['onset_slices']={}
    for label,lo,hi in [('0-320ms',0,.32),('320-640ms',.32,.64),('640-1000ms',.64,1.),('at-least-1s',1.,float('inf'))]:
        use=valid&(age>=lo)&(age<hi)
        if use.any():
            speakers=np.unique(who[use]);correct=pred==who
            result['onset_slices'][label]={'speech_windows':int(use.sum()),
                'macro_accuracy':float(np.mean([correct[use&(who==s)].mean() for s in speakers])),
                'speakers':int(len(speakers))}
    return result

def prototypes():
    x=np.load(OLD/'prototypes.npy').astype(np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)

def train_one(r,frames,recipe,seed,epochs=8):
    guard();alpha,pair_mode=ARMS[recipe];arm=f'{frames*80}ms/{recipe}';name=f'f{frames}-{recipe}-s{seed}'
    folder=WORK/'models';folder.mkdir(exist_ok=True);done=folder/(name+'.json');bestfile=folder/(name+'-best.pt');resume=folder/(name+'-resume.pt')
    if done.exists():return json.loads(done.read_text())
    torch.manual_seed(seed);model=MixedStudent(False).to(r.device)
    warm=torch.load(OLD/'models/query_gru-best.pt',map_location=r.device,weights_only=False)
    model.load_state_dict(warm['model']);del warm
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.003)
    train=r.split['train'];proto_np=prototypes();proto=torch.from_numpy(proto_np).to(r.device);mean=model.teacher_mean
    stable=torch.from_numpy(v4.stable_targets()).to(r.device);pairs=Pairs(r,v4.stable_targets()) if pair_mode else None
    voices=np.unique(r.ids_cpu[train]);voices=voices[voices>=0];contrast=F.normalize(proto[voices]-mean,dim=-1)
    lookup=torch.full((500,),-1,dtype=torch.long,device=r.device);lookup[torch.tensor(voices,device=r.device)]=torch.arange(len(voices),device=r.device)
    counts=np.bincount(r.classes_cpu[train],minlength=3).clip(1)
    weights=torch.tensor(counts**-.5,dtype=torch.float32,device=r.device);weights/=weights.mean()
    batch_size=256;best=-1e9;best_epoch=0;history=[];first_epoch=0;examples=0;prior_pairs=prior_matches=0
    if resume.exists():
        z=torch.load(resume,map_location=r.device,weights_only=False);model.load_state_dict(z['model']);optimizer.load_state_dict(z['optimizer'])
        first_epoch=z['epoch'];history=z['history'];best=z['best'];best_epoch=z['best_epoch'];examples=z['examples']
        prior_pairs=z.get('total_pairs',0);prior_matches=z.get('matched_pairs',0)
        torch.set_rng_state(z['torch_rng'].cpu());torch.cuda.set_rng_state(z['cuda_rng'].cpu());del z
    started=time.perf_counter()
    for epoch in range(first_epoch,epochs):
        guard();model.train();order=np.random.default_rng(seed*100+epoch).permutation(train);rng=np.random.default_rng(seed*200+epoch)
        loss_sum=0.;t=time.perf_counter()
        for step,offset in enumerate(range(0,len(order),batch_size)):
            if step%100==0:guard()
            anchor=order[offset:offset+batch_size];n=len(anchor);chosen=anchor
            if pair_mode:
                pos,neg=pairs.sample(anchor,rng,hard=pair_mode=='hard');chosen=np.r_[anchor,pos,neg]
            x,e,target,activity,speaker=r.batch(chosen);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                embedding,logits=model(x,e);valid=activity==0;reg=embedding.sum()*0
                if valid.any():
                    a=embedding[valid];b=target[valid];s=speaker[valid]
                    reg=(1-alpha)*regression(a,b,mean)+alpha*regression(a,stable[s],mean)
                    reg=reg+.2*F.cross_entropy(F.normalize(a-mean,dim=-1)@contrast.T/.12,lookup[s])
                loss=reg+.2*F.cross_entropy(logits,activity,weight=weights)
                if pair_mode:
                    active=activity[:n]==0
                    if active.any():
                        center=F.normalize(embedding-mean,dim=-1);a=center[:n][active];p=center[n:2*n][active];negative=center[2*n:][active]
                        positive=(a*p).sum(-1);negative_cos=(a*negative).sum(-1)
                        loss=loss+.1*(1-positive).mean()+.1*F.relu(negative_cos-positive+.15).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite context-training objective')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step()
            loss_sum+=float(loss.detach())*n;examples+=len(chosen)
            if step%100==0:update(phase='Training context-specific students',context_ms=frames*80,recipe=recipe,seed=seed,epoch=epoch+1,batch=step)
        val=evaluate(model,r,'validation',proto_np);score=objective(val)
        history.append({'epoch':epoch+1,'loss':loss_sum/len(train),'seconds':time.perf_counter()-t,'validation':val})
        if score>best:
            best=score;best_epoch=epoch+1
            checkpoint({'model':model.state_dict(),'frames':frames,'recipe':recipe,'seed':seed,'epoch':best_epoch,'validation':val,'format_version':5},bestfile)
        checkpoint({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'history':history,
            'best':best,'best_epoch':best_epoch,'examples':examples,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
            'total_pairs':prior_pairs+(pairs.pairs if pairs else 0),'matched_pairs':prior_matches+(pairs.matched if pairs else 0)},resume)
        update(phase='Training context-specific students',context_ms=frames*80,recipe=recipe,seed=seed,epoch=epoch+1,
            validation_accuracy=val['macro_speaker_accuracy'],validation_eer=val['eer'])
        if epoch>=4 and epoch+1-best_epoch>=3:break
    saved=torch.load(bestfile,map_location=r.device,weights_only=False);model.load_state_dict(saved['model']);model.eval()
    x,e,*_=r.batch(r.split['validation'][:1])
    with torch.inference_mode():
        for _ in range(10):model(x,e)
        torch.cuda.synchronize();t=time.perf_counter()
        for _ in range(100):model(x,e)
        torch.cuda.synchronize();latency=(time.perf_counter()-t)*10
    result={'arm':arm,'frames':frames,'context_ms':frames*80,'recipe':recipe,'seed':seed,'alpha':alpha,'pair_mode':pair_mode,
        'validation':saved['validation'],'best_epoch':saved['epoch'],'checkpoint':bestfile.name,'history':history,
        'parameters':sum(p.numel() for p in model.parameters()),'batch_size':batch_size,'effective_examples_seen':examples,
        'committed_training_epoch_seconds':sum(h['seconds'] for h in history),'training_seconds_this_run':time.perf_counter()-started,
        'student_gpu_batch1_ms':latency,'pair_counts':{'total':prior_pairs+(pairs.pairs if pairs else 0),'transcript_matched':prior_matches+(pairs.matched if pairs else 0)}}
    atomic(done,result)
    if resume.exists():resume.unlink() # redundant optimizer only, after completed best/result
    del model,optimizer,pairs,saved;torch.cuda.empty_cache();return result

def summarize_groups(runs):
    groups=aggregates(runs)
    for arm,group in groups.items():group['student_gpu_batch1_ms']=float(np.mean([r['student_gpu_batch1_ms'] for r in runs if r['arm']==arm]))
    return groups

def paired(a,b,split):
    people=sorted(a[0][split]['per_speaker']);delta=[]
    for person in people:
        delta.append(np.mean([r[split]['per_speaker'][person]['accuracy'] for r in a])-np.mean([r[split]['per_speaker'][person]['accuracy'] for r in b]))
    draws=np.random.default_rng(731).choice(delta,(5000,len(delta))).mean(1)*100
    return {'delta_macro_accuracy_pp':float(np.mean(delta)*100),'paired_identity_bootstrap_95ci_pp':np.quantile(draws,[.025,.975]).tolist(),
        'identities':len(people),'note':'Seeds averaged before speaker bootstrap; repeated synthetic benchmark, not real-human validation'}
