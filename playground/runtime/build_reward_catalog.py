"""Regenerate public metadata from pinned upstream taxonomy/card, not weights.

Print JSON; source-file updates must be applied explicitly by the maintainer.
VoiceNet native definitions are abbreviated with attribution (CC BY 4.0).
"""
import ast
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

VN = 'cdafe5170561eaadb09f012ed5ae48e4f82b7b91'
EIV = '9ceb45912f4d555fc0210e92ec8459234b3992ec'
ATTRS = {
    'Age': (0,6,'Perceived age, infant to very old; not identity verification.'),
    'Gender': (-2,2,'Perceived presentation, masculine to feminine; not identity.'),
    'High-Pitched_vs._Low-Pitched': (0,4,'High-pitched at 0; low-pitched at 4.'),
    'Valence': (-3,3,'Negative to positive affect; 0 neutral.'),
    'Arousal': (0,4,'Very calm to very excited; midpoint 2.'),
    'Submissive_vs._Dominant': (-3,3,'Submission/authority axis; endpoint direction not fully specified in card.'),
    'Serious_vs._Humorous': (0,4,'Gravity/humor axis; endpoint direction not fully specified in card.'),
    'Confident_vs._Hesitant': (0,4,'Certainty/hesitation axis; endpoint direction not fully specified in card.'),
    'Soft_vs._Harsh': (-2,2,'Soothing soft at -2 to abrasive harsh at +2.'),
    'Monotone_vs._Expressive': (0,4,'Amount of dynamic pitch/rhythm variation.'),
    'Warm_vs._Cold': (-2,2,'Friendliness/distance axis; endpoint direction not fully specified in card.'),
}
UNKNOWN = {'Authenticity','Background_Noise','Recording_Quality','Vulnerable_vs._Emotionally_Detached'}
QUALITY = {'overall_quality': (1,3.7,'Overall Quality'), 'speech_quality': (1,2.4,'Speech Quality'),
           'background_quality': (1,4.3,'Background Quality'), 'content_enjoyment': (1.9,5.1,'Content Enjoyment')}

def main():
    path=hf_hub_download('laion/voicenet-dimension-predictors-commercial','dimensions.py',revision=VN,
                         cache_dir='/mnt/nvme/hf_cache/hub')
    tree=ast.parse(Path(path).read_text())
    dims=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='DIMENSIONS' for t in n.targets))
    rows=[]
    for key,d in dims.items():
        indexes=sorted({0,(d['n_levels']-1)//2,d['n_levels']-1})
        desc='; '.join(str(i)+': '+' '.join(d['levels'][i].split()[:16]) for i in indexes)
        rows.append({'id':'vn:'+key,'name':d['name'],'source':'voicenet','min':0,
                     'max':d['n_levels']-1,'description':desc})
    for file in HfApi().model_info('laion/Empathic-Insight-Voice-Plus',revision=EIV).siblings:
        filename=file.rfilename
        if not filename.endswith('_best.pth'):continue
        key=filename[len('model_'):-len('_best.pth')]
        if key in UNKNOWN:continue
        if key.startswith('score_'):
            lo,hi,name=QUALITY[key[6:]]
            desc='Higher is better. Expected empirical range, not hard bounds or calibrated human MOS.'
            group='quality'
        elif key in ATTRS:
            lo,hi,desc=ATTRS[key]; name=key.replace('_',' '); group='attribute'
        else:
            lo,hi=0,4; name=key.replace('_',' '); group='emotion'
            desc='Intensity of '+name+', absent at 0 to strong at 4; can exceed scale; not a probability.'
        rows.append({'id':'eiv:'+key,'name':name,'source':'eiv_plus','min':lo,'max':hi,
                     'description':desc,'group':group,'filename':filename})
    print(json.dumps({'attribution':'LAION VoiceNet taxonomy CC BY 4.0, abbreviated native level definitions; Empathic Insight Plus model-card ranges.',
        'voicenet_repo':'laion/voicenet-dimension-predictors-commercial','voicenet_revision':VN,
        'eiv_repo':'laion/Empathic-Insight-Voice-Plus','eiv_revision':EIV,
        'encoder_repo':'laion/BUD-E-Whisper','encoder_revision':'9400e6740ea103bf619a4ce66374576fab8a4e23',
        'excluded_unverified_scales':sorted(UNKNOWN),'dimensions':rows}))

if __name__=='__main__':main()
