"""Conservative outside-word crop; same implementation as the listened-to baseline."""
import numpy as np
SR=48000
def crop(wave,report):
    if not report['alignment_ok']:raise ValueError('Refuse to cut without plausible word alignment')
    first,last=report['first_word_s'],report['last_word_s'];duration=len(wave)/SR
    a=max(0,first-.08);b=min(duration,last+.15)
    clip=wave[round(a*SR):round(b*SR)].copy()
    lead=first-a;tail=b-last
    fadein=min(.02,lead*.5);fadeout=min(.05,tail*.5)
    for length,reverse in [(int(fadein*SR),False),(int(fadeout*SR),True)]:
        if length>1:
            if reverse:clip[-length:]*=np.linspace(1,0,length,dtype=np.float32)
            else:clip[:length]*=np.linspace(0,1,length,dtype=np.float32)
    edit={'start_s':a,'end_s':b,'removed_head_s':a,'removed_tail_s':duration-b,
          'first_word_in_clip_s':lead,'last_word_in_clip_s':last-a,
          'lead_guard_s':lead,'tail_guard_s':tail,'fade_in_s':fadein,'fade_out_s':fadeout,
          'policy':'80 ms lead, 150 ms tail, fades outside aligned speech; preserve all internal timing'}
    return clip,edit
