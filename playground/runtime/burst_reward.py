"""Optional position-aware vocal-burst reward, with prefix-safe target maturity.

Times are seconds on actual PCM. An event can satisfy at most one request.
These are noisy model-based instrument readings, not verified human judgments.
"""
import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from metrics import tokens

LOCATOR_REPO='laion/vocalburst-locator'
LOCATOR_REVISION='3e4a8392fe0c2ee56e375b6244b033c96110287c'
DETECTOR_REPO='laion/vocal-burst-detector-commercial'
DETECTOR_REVISION='67bbf412eb85c49ba96548488461b693a7be7de3'
WHISPER_REVISION='973afd24965f72e36ca33b3055d56a652f456b4d'


class BurstTarget(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    label: str = Field(min_length=1,max_length=120)
    start: float = Field(ge=0,le=60)
    duration: float = Field(default=.3,ge=.05,le=10)


class BurstOptions(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    mode: Literal['off','location','typed'] = 'off'
    source: Literal['script','manual'] = 'script'
    timing: Literal['words','clock'] = 'words'
    match: Literal['family','exact'] = 'family'
    weight: float = Field(default=.25,ge=0,le=1)
    tolerance: float = Field(default=.3,ge=.04,le=2)
    targets: list[BurstTarget] = Field(default_factory=list,max_length=8)
    penalize_extra: bool = True

    @model_validator(mode='after')
    def manual_targets(self):
        if self.mode!='off' and self.source=='manual' and not self.targets:
            raise ValueError('Manual burst checking requires 1–8 targets')
        return self


def script_targets(spec):
    """Read the actual conditioning script; preserve word-boundary anchors."""
    import timed_script
    script=spec.get('tagged','')
    # The legacy renderer clamps requested burst durations to its training
    # range. Evaluation must instead read the numbers actually sent to Small,
    # including unmodified manual scripts.
    native_lengths=[]
    for cue in timed_script._CUE_RE.finditer(script):
        timed=timed_script._BURST_RE.fullmatch(cue.group())
        if timed:
            length=float(timed.group(2))
            if not math.isfinite(length) or not 0<length<=60:
                raise ValueError('Script burst durations must be finite and in (0, 60] seconds')
            native_lengths.append(length)
        elif timed_script._is_burst_label(cue.group(1)):
            native_lengths.append(timed_script.burst_length(cue.group(1))[1])
    lengths=iter(native_lengths)
    clock=0.; count=0; last_end=0.; pending=None; speech=[]; requests=[]
    for item in timed_script.parse(script):
        kind=item[0]
        if kind=='duration':pending=float(item[1])
        elif kind=='pause':clock+=float(item[1])
        elif kind=='speech':
            length=len(tokens(item[1]))
            duration=pending if pending is not None else max(.4,length*.36)
            speech.append({'word_index':count,'start':clock,'end':clock+duration})
            clock+=duration;last_end=clock;count+=length;pending=None
        elif kind=='burst':
            length=next(lengths)
            requests.append({'label':item[1],'start':clock,'duration':length,
                'word_index':count,'gap_before':max(0,clock-last_end)})
            clock+=length
    if len(requests)>8:raise ValueError('At most 8 burst requests per take')
    for request in requests:
        following=next((s for s in speech if s['word_index']>=request['word_index']),None)
        request['gap_after']=max(0,following['start']-request['start']-request['duration']) if following else 0.
    first=speech[0]['start'] if speech else 0.
    return requests,first


def word_anchors(reference,timestamps):
    """Levenshtein correspondence; only exact word matches become anchors."""
    ref=tokens(reference); hyp=[]; spans=[]
    for span in timestamps:
        for word in tokens(span['word']):hyp.append(word);spans.append(span)
    costs=np.zeros((len(ref)+1,len(hyp)+1),dtype=int)
    costs[:,0]=np.arange(len(ref)+1);costs[0,:]=np.arange(len(hyp)+1)
    for i in range(1,len(ref)+1):
        for j in range(1,len(hyp)+1):
            costs[i,j]=min(costs[i-1,j]+1,costs[i,j-1]+1,costs[i-1,j-1]+(ref[i-1]!=hyp[j-1]))
    mapping={};j=len(hyp)
    i=min(range(len(ref)+1),key=lambda endpoint:(costs[endpoint,j],abs(endpoint-j),-endpoint))
    while i or j:
        if i and j and costs[i,j]==costs[i-1,j-1]+(ref[i-1]!=hyp[j-1]):
            if ref[i-1]==hyp[j-1]:mapping[i-1]=spans[j-1]
            i-=1;j-=1
        elif i and costs[i,j]==costs[i-1,j]+1:i-=1
        else:j-=1
    return mapping


def expected_window(target,options,anchors,seconds,final):
    """Unseen future requests stay pending, never count as missing prematurely."""
    start=target['start'];end=start+target['duration'];kind='clock'
    if options.source=='script' and options.timing=='words':
        index=target['word_index'];left=anchors.get(index-1);right=anchors.get(index)
        if (index>0 and left is None) or (index==0 and right is None and not final and seconds<end+.12):
            return {'status':'pending' if not final else 'unresolved_words','start':start,'end':end,'basis':'words'}
        start=(left['end'] if left else 0.)+target.get('gap_before',0.)
        end=max(start,right['start']-target.get('gap_after',0.)) if right else start+target['duration']
        kind='words' if right else 'left_word_estimate'
        # Right word proves this inter-word slot has already passed. Without
        # it, wait until the requested sound could have finished at this cut.
        due=right is not None or seconds>=end+.12
    else:due=seconds>=end+.12
    return {'status':'due' if final or due else 'pending','start':start,'end':end,'basis':kind}


def event_utility(event,target,window,options):
    if window['status']!='due':return 0.,{}
    if window['basis']=='clock':distance=abs(event['start']-window['start'])
    else:
        midpoint=(event['start']+event['end'])/2
        distance=max(window['start']-midpoint,midpoint-window['end'],0.)
    timing=max(0.,1-distance/options.tolerance)
    actual=max(.02,event['end']-event['start'])
    # A midpoint inside the slot alone must not fully credit a long event
    # overlapping neighboring speech. Tolerance permits modest timestamp noise.
    overlap=1. if window['basis']=='clock' else min(1.,max(0.,
        min(event['end'],window['end']+options.tolerance)-
        max(event['start'],window['start']-options.tolerance))/actual)
    duration_fit=math.sqrt(min(actual,target['duration'])/max(actual,target['duration']))
    type_score=1.;type_supported=False;match_kind='location_only'
    if options.mode=='typed':
        from bestofn import _resolve_burst
        classes=event.get('classes',[]);families=event.get('families',[])
        exact,family=_resolve_burst(target['label'],classes,families)
        probabilities=event.get('probabilities',[])
        indexes=[i for i,f in enumerate(families) if family and f==family] if options.match=='family' else ([exact] if exact is not None else [])
        if indexes and probabilities:
            type_score=min(1.,sum(probabilities[i] for i in indexes));type_supported=True
            match_kind=options.match
        else:match_kind='unsupported_type_location_only'
    score=timing*overlap*duration_fit*float(event['confidence'])*type_score
    return float(np.clip(score,0,1)),{'timing':timing,'duration_fit':duration_fit,
        'slot_overlap':overlap,
        'type_score':type_score,'type_supported':type_supported,'match_kind':match_kind}


def evaluate_bursts(targets,events,options,reference,timestamps,seconds,final=False,first_speech=0.):
    from scipy.optimize import linear_sum_assignment
    stable=timestamps if final else [s for s in timestamps if s['end']<=seconds-.12]
    anchors=word_anchors(reference,stable)
    windows=[expected_window(t,options,anchors,seconds,final) for t in targets]
    active=[i for i,w in enumerate(windows) if w['status']!='pending']
    matrix=np.zeros((len(active),len(events)+len(active)))
    explanations={}
    for row,index in enumerate(active):
        for column,event in enumerate(events):
            matrix[row,column],explanations[row,column]=event_utility(event,targets[index],windows[index],options)
    assigned={};used=set()
    if active:
        rows,cols=linear_sum_assignment(-matrix)
        for row,column in zip(rows,cols):
            if column<len(events) and matrix[row,column]>0:
                assigned[active[row]]=(int(column),float(matrix[row,column]),explanations[row,column]);used.add(int(column))
    reports=[];values=[]
    for i,(target,window) in enumerate(zip(targets,windows)):
        report={'request':target,'expected_window':window,'status':window['status']}
        if i in assigned:
            column,value,parts=assigned[i]
            report.update(status='matched',event_index=column,score=value,**parts)
        elif i in active:report.update(status='missing' if window['status']=='due' else window['status'],score=0.)
        if i in active:values.append(report['score'])
        reports.append(report)
    score=float(np.mean(values)) if values else None
    extra=len(events)-len(used)
    penalty=min(.15,.15*extra/max(1,len(active))) if final and options.penalize_extra and active else 0.
    if score is not None:score=max(0.,score-penalty)
    speech_not_due=(not final and not stable and any(t.get('word_index')==0 for t in targets)
                    and seconds<=first_speech+options.tolerance)
    return {'score':score,'targets':reports,'events':events,'extra_events':extra,
            'extra_penalty':penalty,'active_targets':len(active),'pending_targets':len(targets)-len(active),
            'speech_not_due':speech_not_due,'mode':options.mode,'timing':options.timing,
            'type_match':options.match,'weight':options.weight,
            'note':'Noisy locator/classifier proxies. Unsupported types receive location-only checking.'}


class BurstReward:
    def __init__(self,models,spec,options):
        self.models=models;self.spec=spec;self.options=options
        if options.source=='manual':
            self.options=options.model_copy(update={'timing':'clock'})
            self.targets=[t.model_dump() for t in options.targets];self.first_speech=0.
        else:self.targets,self.first_speech=script_targets(spec)
        self.last_timing={'burst_seconds':0.}
        if self.targets:models.prepare(options.mode=='typed')

    def describe(self):
        return {'options':self.options.model_dump(),'targets':self.targets,
                'first_speech_seconds':self.first_speech,'active':bool(self.targets),
                'note':'No explicit burst targets; reward unchanged.' if not self.targets else
                    'Locator v2 timestamps + optional commercial class/family score.'}

    def score(self,audio16,timestamps,final):
        import time
        started=time.perf_counter()
        events=self.models.detect(audio16,self.options.mode=='typed') if self.targets else [[] for _ in audio16]
        result=[evaluate_bursts(self.targets,ev,self.options,self.spec['text'],spans,len(w)/16000,end,
                    self.first_speech) for w,ev,spans,end in zip(audio16,events,timestamps,final)]
        self.last_timing={'burst_seconds':time.perf_counter()-started}
        return result
