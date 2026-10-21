"""Validation-only predicted-to-predicted thresholds; disjoint source clips.

No test identities, private microphone audio or live enrolled embeddings are
read. Gallery averages several synthetic windows from one source UID and can
be optimistic relative to a real consecutive short utterance. Not authentication.
"""
import json,time
from pathlib import Path
import numpy as np
import torch
from speaker_runtime import LivePredictor
from speaker_v4 import data
from speaker_v4.common import FEATURES,atomic

ROOT=Path(__file__).resolve().parent

@torch.inference_mode()
def main():
    torch.set_num_threads(4);model=LivePredictor(None).fast
    old=data.arrays();keep=data.subset();indices=data.splits()['validation'];local=np.searchsorted(keep,indices)
    q=np.load(FEATURES/'q.npy',mmap_mode='r');s=np.load(FEATURES/'scale.npy',mmap_mode='r')
    if not np.array_equal(keep[local],indices):raise RuntimeError('Matched validation indices changed')
    vectors=[];states=[]
    for offset in range(0,len(indices),256):
        g=indices[offset:offset+256];l=local[offset:offset+256]
        x=np.concatenate([q[l].astype(np.float32)*s[l].astype(np.float32),
            (old['q'][g,-4:].astype(np.float32)*old['scale'][g,-4:].astype(np.float32))[:,None]],axis=1)
        v,a=model(torch.from_numpy(x),torch.from_numpy(old['energy'][g,-4:].astype(np.float32)))
        vectors.append(v.numpy());states.append(a.argmax(-1).numpy())
    vectors=np.concatenate(vectors);states=np.concatenate(states);ids=old['speaker'][indices];active=(old['activity'][indices]==0)&(states==0)
    clip_index,_=data.metadata();uids=clip_index[indices];gallery=[];probes=[];reference_uids={}
    for speaker in np.unique(ids[active]):
        eligible=np.flatnonzero(active&(ids==speaker));clips,counts=np.unique(uids[eligible],return_counts=True)
        clip=int(clips[counts.argmax()]);source=eligible[uids[eligible]==clip][:8]
        if len(source)<2:raise RuntimeError('Insufficient validation gallery windows')
        gallery.append(vectors[source].mean(0));probes.extend(eligible[uids[eligible]!=clip].tolist());reference_uids[str(int(speaker))]=clip
    gallery=np.stack(gallery);people=np.unique(ids[active]);probes=np.asarray(probes);y=ids[probes];result=[]
    for centered in (False,True):
        mean=model.teacher_mean.numpy() if centered else np.zeros(192,np.float32)
        a=vectors[probes]-mean;b=gallery-mean;a/=np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-8);b/=np.maximum(np.linalg.norm(b,axis=-1,keepdims=True),1e-8)
        scores=a@b.T;own=people[None,:]==y[:,None];genuine=scores[own];wrong=scores.copy();wrong[own]=-2
        maxima=wrong.max(1);threshold=float(np.quantile(maxima,.95));top=scores.argmax(1)
        accepted=(scores.max(1)>=threshold)&(np.sort(scores,axis=1)[:,-1]-np.sort(scores,axis=1)[:,-2]>=.04)
        result.append({'centered':centered,'threshold':threshold,'margin':.04,'validation_genuine_pass_rate':float((genuine>=threshold).mean()),
            'validation_max_impostor_pass_rate':float((maxima>=threshold).mean()),'validation_macro_identification':float(np.mean([(people[top][y==s]==s).mean() for s in people])),
            'validation_accept_correct':float((accepted&(people[top]==y)).mean()),'windows':len(probes),'voices':len(people)})
    selected=max(result,key=lambda r:r['validation_accept_correct']);output={**selected,'completed':time.time(),'alternatives':result,
        'protocol':'Validation identities only; predicted gallery/probes use disjoint source UIDs. 95th percentile maximum other-speaker cosine, margin .04.',
        'reference_source_ids':reference_uids,'training_or_test_identities_used':False,
        'limitations':'Repeated synthetic validation set; gallery averages augmented sampled windows, not exactly consecutive real-user enrollment. Session thresholds remain approximate and adjustable.'}
    atomic(ROOT/'calibration.json',output);print(json.dumps(output,indent=2))

if __name__=='__main__':main()
