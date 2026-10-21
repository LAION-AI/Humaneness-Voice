"""Four-frame specialization and a matched layer/hard-example factorial."""
import json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from speaker_v3.train import Student,metrics,checkpoint
from .common import OLD,WORK,FEATURES,SEEDS,atomic,guard,update,report
from . import data

class MixedStudent(Student):
    def __init__(self,mix_layers=False):
        super().__init__('gru',128);self.mix_layers=mix_layers
        if mix_layers:self.layer_logits=nn.Parameter(torch.tensor([-1.5,-1.5,-1.5,3.]))
        else:self.register_parameter('layer_logits',None)
    def forward(self,x,energy):
        if self.mix_layers:
            x=(F.layer_norm(x.float(),(1024,))*self.layer_logits.softmax(0)[None,:,None,None]).sum(1)
        else:x=x[:,-1].float()
        pool=torch.zeros(len(x),2,1024,device=x.device,dtype=x.dtype)
        return super().forward(x,pool,energy,False)

class Resident:
    """Bounded GPU int8 cache: no repeated random disk I/O per epoch."""
    def __init__(self,indices,with_layers=False,device='cuda:0'):
        self.indices=np.asarray(indices,np.int64);self.device=device;self.arrays=data.arrays();self.layers=with_layers
        n=len(indices);channels=4 if with_layers else 1
        self.q=torch.empty(n,channels,4,1024,dtype=torch.int8,device=device)
        self.scales=torch.empty(n,channels,4,1,dtype=torch.float16,device=device)
        extra_q=np.load(FEATURES/'q.npy',mmap_mode='r') if with_layers else None
        extra_s=np.load(FEATURES/'scale.npy',mmap_mode='r') if with_layers else None
        for start in range(0,n,8192):
            end=min(n,start+8192);g=self.indices[start:end]
            self.q[start:end,-1]=torch.from_numpy(self.arrays['q'][g,-4:].copy()).to(device)
            self.scales[start:end,-1]=torch.from_numpy(self.arrays['scale'][g,-4:].copy()).to(device)
            if with_layers:
                self.q[start:end,:3]=torch.from_numpy(extra_q[start:end].copy()).to(device)
                self.scales[start:end,:3]=torch.from_numpy(extra_s[start:end].copy()).to(device)
        self.energy=torch.from_numpy(self.arrays['energy'][self.indices,-4:].copy()).to(device)
        self.target=torch.from_numpy(self.arrays['local_target'][self.indices].copy()).to(device)
        self.ids_cpu=np.asarray(self.arrays['speaker'][self.indices],np.int64)
        self.classes_cpu=np.asarray(self.arrays['activity'][self.indices],np.int64)
        self.emotions_cpu=np.asarray(self.arrays['emotion'][self.indices],np.int64)
        self.ids=torch.tensor(self.ids_cpu,device=device);self.classes=torch.tensor(self.classes_cpu,device=device)
        masks=data.splits();self.split={s:np.flatnonzero(np.isin(self.indices,v)) for s,v in masks.items()}
    def batch(self,indices):
        index=torch.as_tensor(indices,dtype=torch.long,device=self.device)
        x=self.q[index].float()*self.scales[index].float()
        return x,self.energy[index].float(),self.target[index].float(),self.classes[index],self.ids[index]

@torch.inference_mode()
def predict(model,resident,local_indices):
    embeddings=[];states=[];model.eval()
    for start in range(0,len(local_indices),512):
        x,e,*_=resident.batch(local_indices[start:start+512]);emb,logits=model(x,e)
        embeddings.append(emb.cpu().numpy());states.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(embeddings),np.concatenate(states)

def evaluate(model,resident,split,prototypes,threshold=None):
    local=resident.split[split]
    return metrics(predict(model,resident,local),resident.arrays,resident.indices[local],prototypes,threshold)

class Pairs:
    """Training-only, source-aware positives and nearest-identity negatives."""
    def __init__(self,resident,stable):
        self.r=resident;train=resident.split['train'];train=train[resident.classes_cpu[train]==0]
        self.voices=np.unique(resident.ids_cpu[train]);self.banks={int(s):train[resident.ids_cpu[train]==s] for s in self.voices}
        centers=stable[self.voices];scores=centers@centers.T;np.fill_diagonal(scores,-10)
        self.neighbors={int(s):self.voices[np.argsort(scores[i])[-8:][::-1]] for i,s in enumerate(self.voices)}
        self.positive={};self.transcripts={};clip_idx,info=data.metadata();self.signature={}
        for s,bank in self.banks.items():
            for emotion in np.unique(resident.emotions_cpu[bank]):
                other=bank[resident.emotions_cpu[bank]!=emotion]
                self.positive[(s,int(emotion))]=other if len(other) else bank
            for i in bank:
                c=int(clip_idx[resident.indices[i]])
                if c<0:raise RuntimeError('Missing training source metadata')
                signature=info[c]['text_hash'];self.signature[int(i)]=signature
                self.transcripts.setdefault((s,signature),[]).append(int(i))
        self.matched=0;self.pairs=0
    def sample(self,indices,rng,hard=True):
        pos=indices.copy();neg=indices.copy()
        for j,index in enumerate(indices):
            s=int(self.r.ids_cpu[index])
            if s<0:continue
            candidates=self.positive[(s,int(self.r.emotions_cpu[index]))] if hard else self.banks[s]
            pos[j]=candidates[int(rng.integers(len(candidates)))];match=None
            neighbors=self.neighbors[s] if hard else self.voices[self.voices!=s]
            if hard:
                for other in neighbors:
                    same=self.transcripts.get((int(other),self.signature[int(index)]))
                    if same:match=same[int(rng.integers(len(same)))];self.matched+=1;break
            if match is None:
                other=int(neighbors[int(rng.integers(len(neighbors)))]);bank=self.banks[other];match=bank[int(rng.integers(len(bank)))]
            neg[j]=match;self.pairs+=1
        return pos,neg

def objective(m):return m['macro_speaker_accuracy']-.5*m['eer']+.1*m['source_utterance_teacher_cosine']

def regression(embedding,target,mean):
    target=F.normalize(target,dim=-1)
    raw=(1-(embedding*target).sum(-1)).mean()
    center=(1-(F.normalize(embedding-mean,dim=-1)*F.normalize(target-mean,dim=-1)).sum(-1)).mean()
    return .4*raw+.6*center+.05*F.mse_loss(embedding,target)*192

def train_one(resident,arm,seed,alpha,warm,mix_layers=False,pair_mode=None,epochs=8,pairs=None):
    guard();folder=WORK/'models';folder.mkdir(exist_ok=True);name=f'{arm}-s{seed}'
    done=folder/(name+'.json');bestfile=folder/(name+'-best.pt');resume=folder/(name+'-resume.pt')
    if done.exists():return json.loads(done.read_text())
    torch.manual_seed(seed);model=MixedStudent(mix_layers).to(resident.device)
    saved=torch.load(warm,map_location=resident.device,weights_only=False)
    missing,unexpected=model.load_state_dict(saved['model'],strict=False)
    if unexpected or any(k!='layer_logits' for k in missing):raise RuntimeError('Warm checkpoint topology mismatch')
    groups=[{'params':[p for n,p in model.named_parameters() if n!='layer_logits'],'lr':2e-4}]
    if mix_layers:groups.append({'params':[model.layer_logits],'lr':.003})
    optimizer=torch.optim.AdamW(groups,weight_decay=.003)
    train=resident.split['train'];prototypes=np.load(OLD/'prototypes.npy').astype(np.float32)
    prototypes/=np.maximum(np.linalg.norm(prototypes,axis=-1,keepdims=True),1e-8)
    stable=data.stable_targets();stable_tensor=torch.from_numpy(stable).to(resident.device)
    voices=np.unique(resident.ids_cpu[train]);voices=voices[voices>=0]
    proto=torch.from_numpy(prototypes).to(resident.device);mean=model.teacher_mean
    contrast=F.normalize(proto[voices]-mean,dim=-1)
    lookup=torch.full((500,),-1,dtype=torch.long,device=resident.device);lookup[torch.tensor(voices,device=resident.device)]=torch.arange(len(voices),device=resident.device)
    counts=np.bincount(resident.classes_cpu[train],minlength=3).clip(1)
    weights=torch.tensor(counts**-.5,dtype=torch.float32,device=resident.device);weights/=weights.mean()
    best=-1e9;best_epoch=0;history=[];first_epoch=0;examples=0;batch_size=512
    prior_matches=prior_pairs=0;pair_start=(pairs.matched,pairs.pairs) if pair_mode else (0,0)
    if resume.exists():
        z=torch.load(resume,map_location=resident.device,weights_only=False);model.load_state_dict(z['model']);optimizer.load_state_dict(z['optimizer'])
        first_epoch=z['epoch'];history=z['history'];best=z['best'];best_epoch=z['best_epoch'];examples=z['examples']
        prior_matches=z.get('matched_pairs',0);prior_pairs=z.get('total_pairs',0)
        torch.set_rng_state(z['torch_rng'].cpu());torch.cuda.set_rng_state(z['cuda_rng'].cpu())
    start=time.perf_counter()
    update(phase='Training 320-ms ablations',arm=arm,seed=seed,centroid_weight=alpha,epoch=first_epoch,
        validation_macro_accuracy=None,validation_eer=None)
    for epoch in range(first_epoch,epochs):
        model.train();order=np.random.default_rng(seed*100+epoch).permutation(train);rng=np.random.default_rng(seed*200+epoch)
        loss_sum=0.;epoch_start=time.perf_counter()
        for step,offset in enumerate(range(0,len(order),batch_size)):
            if step%100==0:guard()
            anchor=order[offset:offset+batch_size];n=len(anchor);chosen=anchor
            if pair_mode:
                pos,neg=pairs.sample(anchor,rng,hard=pair_mode=='hard');chosen=np.r_[anchor,pos,neg]
            x,e,target,activity,speaker=resident.batch(chosen);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                embedding,logits=model(x,e);valid=activity==0;reg=embedding.sum()*0
                if valid.any():
                    a=embedding[valid];b=target[valid];s=speaker[valid]
                    reg=(1-alpha)*regression(a,b,mean)+alpha*regression(a,stable_tensor[s],mean)
                    reg=reg+.2*F.cross_entropy(F.normalize(a-mean,dim=-1)@contrast.T/.12,lookup[s])
                loss=reg+.2*F.cross_entropy(logits,activity,weight=weights)
                if pair_mode:
                    active=activity[:n]==0
                    if active.any():
                        center=F.normalize(embedding-mean,dim=-1);a=center[:n][active];p=center[n:2*n][active];negative=center[2*n:][active]
                        positive=(a*p).sum(-1);negative_cos=(a*negative).sum(-1)
                        loss=loss+.1*(1-positive).mean()+.1*F.relu(negative_cos-positive+.15).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective; stop without overwriting baseline')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step()
            loss_sum+=float(loss.detach())*n;examples+=len(chosen)
            if step%200==0:update(phase='Training 320-ms ablations',arm=arm,seed=seed,epoch=epoch+1,batch=step,centroid_weight=alpha)
        validation=evaluate(model,resident,'validation',prototypes);score=objective(validation)
        history.append({'epoch':epoch+1,'loss':loss_sum/len(train),'seconds':time.perf_counter()-epoch_start,'validation':validation})
        if score>best:
            best=score;best_epoch=epoch+1
            checkpoint({'model':model.state_dict(),'arm':arm,'seed':seed,'alpha':alpha,'mix_layers':mix_layers,'pair_mode':pair_mode,
                'epoch':best_epoch,'validation':validation,'format_version':4},bestfile)
        checkpoint({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'history':history,'best':best,
            'best_epoch':best_epoch,'examples':examples,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
            'matched_pairs':prior_matches+pairs.matched-pair_start[0] if pair_mode else 0,
            'total_pairs':prior_pairs+pairs.pairs-pair_start[1] if pair_mode else 0},resume)
        if pair_mode:
            # Save a separate tiny counter record without duplicating optimizer weights.
            atomic(folder/(name+'-pairs.json'),{'matched_pairs':prior_matches+pairs.matched-pair_start[0],
                'total_pairs':prior_pairs+pairs.pairs-pair_start[1]})
        update(phase='Training 320-ms ablations',arm=arm,seed=seed,epoch=epoch+1,validation_macro_accuracy=validation['macro_speaker_accuracy'],validation_eer=validation['eer'])
        if epoch>=4 and epoch+1-best_epoch>=3:break
    saved=torch.load(bestfile,map_location=resident.device,weights_only=False);model.load_state_dict(saved['model']);model.eval()
    local=resident.split['validation'][:1];x,e,*_=resident.batch(local)
    with torch.inference_mode():
        for _ in range(10):model(x,e)
        torch.cuda.synchronize();t=time.perf_counter()
        for _ in range(100):model(x,e)
        torch.cuda.synchronize();latency=(time.perf_counter()-t)*10
    result={'arm':arm,'seed':seed,'alpha':alpha,'mix_layers':mix_layers,'pair_mode':pair_mode,'best_epoch':saved['epoch'],
        'parameters':sum(p.numel() for p in model.parameters()),'validation':saved['validation'],'history':history,
        'checkpoint':bestfile.name,'training_seconds_this_run':time.perf_counter()-start,'effective_examples_seen':examples,
        'committed_training_epoch_seconds':sum(h['seconds'] for h in history),
        'student_gpu_batch1_ms':latency,'layer_weights':model.layer_logits.softmax(0).detach().cpu().tolist() if mix_layers else [0,0,0,1]}
    if pair_mode:
        counts=folder/(name+'-pairs.json')
        counters=json.loads(counts.read_text()) if counts.exists() else {}
        result['full_transcript_negative_match_rate']=(prior_matches+pairs.matched-pair_start[0])/max(prior_pairs+pairs.pairs-pair_start[1],1)
        result['pair_counts']=counters
    atomic(done,result)
    # Completed best checkpoint + history are retained; only this redundant
    # optimizer resume cache is removed after verified successful publication.
    if resume.exists():resume.unlink()
    del model,optimizer;torch.cuda.empty_cache();return result

def final_tests(results,resident,stage):
    prototypes=np.load(OLD/'prototypes.npy').astype(np.float32);prototypes/=np.linalg.norm(prototypes,axis=-1,keepdims=True)
    for row in results[stage]:
        if row.get('test'):continue
        z=torch.load(WORK/'models'/row['checkpoint'],map_location=resident.device,weights_only=False)
        model=MixedStudent(row['mix_layers']).to(resident.device);model.load_state_dict(z['model']);model.eval()
        row['test']=evaluate(model,resident,'test',prototypes,row['validation']['far05_threshold'])
        del model
    report(results)

def aggregates(records):
    result={}
    for arm in sorted({r['arm'] for r in records}):
        items=[r for r in records if r['arm']==arm];values={}
        for split in ('validation','test'):
            if not all(split in r for r in items):continue
            values[split]={metric:{'mean':float(np.mean([r[split][metric] for r in items])),
                'std_across_seeds':float(np.std([r[split][metric] for r in items]))} for metric in
                ('macro_speaker_accuracy','eer','positive_cosine','impostor_cosine','source_utterance_teacher_cosine','activity_accuracy')}
        result[arm]={'seeds':len(items),**values}
    return result
