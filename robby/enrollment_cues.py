"""Time-aligned enrollment cues and conservative foreground eligibility.

Pure NumPy; no enrollment, microphone storage, model calls or network access.
"""
import numpy as np
from identity import unit


def aligned_reference(records,start,end):
    """Average native 320-ms observations, not repeated diarizer subframes.

    Only the verified introduction interval is eligible. Loudness excludes
    quiet room pre-roll; it does not choose a desired voice embedding score.
    Require three actual encoder observations before replacing a fast anchor.
    """
    rows=[]
    for a,b,p,x in records:
        vector=p.get('raw_embedding',p.get('embedding'))
        if vector is not None and start<=(a+b)/2<=end and len(x):
            rows.append((float(np.sqrt(np.mean(x*x))),vector,p.get('long_embedding')))
    if len(rows)<3:return None,None
    floor=max(.003,max(r[0] for r in rows)*.25)
    voiced=[r for r in rows if r[0]>=floor]
    if len(voiced)<3:return None,None
    fast=unit(np.mean([unit(r[1]) for r in voiced],axis=0))
    long=[r[2] for r in voiced if r[2] is not None][-2:]
    return fast,unit(np.mean(long,axis=0)) if long else None


def foreground_context_allowed(records,levels):
    """Low diarizer coverage is not low foreground energy.

    This only permits an independent KNOWN-reference check; never new-user
    enrollment. Avoid binding a quiet background track from louder raw speech
    that belongs to a different activity channel.
    """
    background=[float(np.sqrt(np.mean(x*x))) for a,b,p,x in list(records)[-25:] if len(x)]
    owned=list(levels)[-8:]
    if len(background)<4 or not owned:return False
    floor=float(np.median(background))
    return float(np.median(owned))>=max(.008,2.5*floor)
