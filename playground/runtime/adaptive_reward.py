"""Prompt-adaptive, fixed-scale VoiceNet/EIV+ reward; pinned public metadata."""
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CATALOG = json.loads((Path(__file__).parent/'reward_catalog.json').read_text())
DIMENSIONS = {d['id']:d for d in CATALOG['dimensions']}
AUTOMATIC_DIMENSIONS = {key for key,d in DIMENSIONS.items()
                        if 'endpoint direction not fully specified' not in d['description']}


class RewardTerm(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    dimension: str
    objective: Literal['maximize','minimize','target'] = 'maximize'
    target: float | None = None
    weight: float = Field(default=1, gt=0, le=100)

    @model_validator(mode='after')
    def check_dimension(self):
        if self.dimension not in DIMENSIONS:
            raise ValueError('Unknown or unverified reward dimension')
        meta=DIMENSIONS[self.dimension]
        if self.objective=='target':
            if self.target is None or not meta['min']<=self.target<=meta['max']:
                raise ValueError('Target must lie on the dimension’s native scale')
        elif self.target is not None:
            raise ValueError('A target value is only valid for objective=target')
        return self


class RewardPlan(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    terms: list[RewardTerm] = Field(min_length=1,max_length=4)
    reason: str = Field(default='',max_length=2000)

    @model_validator(mode='after')
    def no_duplicates(self):
        if len({term.dimension for term in self.terms})!=len(self.terms):
            raise ValueError('Use each reward dimension only once')
        return self


def reward_catalog_context(automatic=False):
    """Native ranges and audible meanings, not transcript/content embeddings."""
    return '\n'.join(f"{d['id']} | {d['name']} | {d['min']}..{d['max']} | {d['description']}"
                     for d in CATALOG['dimensions'] if not automatic or d['id'] in AUTOMATIC_DIMENSIONS)


def adaptive_quality(raw,plan,genuineness,blend,style_share=.5):
    details=[]
    total_weight=sum(t.weight for t in plan.terms)
    for term in plan.terms:
        meta=DIMENSIONS[term.dimension]
        value=float(raw[term.dimension])
        if not math.isfinite(value):
            raise ValueError('Non-finite dimension score; refusing arbitrary ranking')
        normal=max(0.,min(1.,(value-meta['min'])/(meta['max']-meta['min'])))
        if term.objective=='maximize':utility=normal
        elif term.objective=='minimize':utility=1-normal
        else:
            want=(term.target-meta['min'])/(meta['max']-meta['min'])
            utility=max(0.,1-abs(normal-want))
        details.append({**term.model_dump(),'name':meta['name'],'raw':value,'normalized':normal,
                        'utility':utility,'normalized_weight':term.weight/total_weight,
                        'contribution':utility*term.weight/total_weight})
    style=sum(d['contribution'] for d in details)
    genuine=max(0.,min(1.,float(genuineness)/6))
    burst=max(0.,min(1.,float(blend)/10))
    quality=style_share*style+(1-style_share)*.5*(genuine+burst)
    return {'quality':quality,'style_score':style,'dimension_scores':details,
            'n_genuineness':genuine,'n_blend':burst,'style_share':style_share}
