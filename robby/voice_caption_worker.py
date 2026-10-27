"""Loopback-only GPU Humaneness Ears + pinned procedural caption rules.

One inference in flight, no queued microphone recordings. No ASR or generative
captioner is loaded. The main conversation can skip this optional side channel.
"""
import asyncio
import hashlib
import os
from pathlib import Path
import sys
import time

from aiohttp import web
import numpy as np
from voice_captioning import MODEL_REVISION,CAPTION_REVISION

ROOT=Path(__file__).resolve().parent
PROSODY={'AROU','ARSH','ATCK','CHNK','DARC','DFLU','EMPH','RANG','RESP','TEMP','VOLT','WHSP'}
EMOTIONS={'Amusement','Anger','Astonishment/Surprise','Calmness','Confusion','Curiosity',
    'Determination','Disappointment','Distress','Doubt','Excitement','Fear','Interest',
    'Joy','Relief','Sadness','Thankfulness/Gratitude','Hope/Optimism','Fatigue/Exhaustion'}

class Captioner:
    def __init__(self):
        source=Path(os.environ.get('ROBO_CAPTION_SOURCE',ROOT/'vendor/procedural-voice-captions'))
        import subprocess
        revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        if revision!=CAPTION_REVISION:raise ValueError('Caption source revision mismatch; run bootstrap_voice_captions.py')
        sys.path.insert(0,str(source))
        from humaneness_backend import HumanenessMedium
        from caption import load_baseline
        from humaneness_caption_settings import load_settings
        from bootstrap_voice_captions import weights
        model_root=os.environ.get('HUMANENESS_EARS_DIR') or weights()
        expected='00f30de0f7d656d2a1e86bdc629673ff8b576532e1263b02bcebdfc2763e54f1'
        with open(Path(model_root)/'medium/model.safetensors','rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=expected:raise ValueError('Medium checkpoint hash mismatch')
        import torch
        self.device=os.environ.get('ROBO_CAPTION_DEVICE','cuda:1' if torch.cuda.device_count()>1 else 'cuda:0')
        self.model=HumanenessMedium(model_root,device=self.device,threads=int(os.environ.get('ROBO_CAPTION_THREADS','4')))
        self.baseline=load_baseline();self.settings=load_settings()
        self.model.predict_batch([np.ones(16000,np.float32)*.01],include_frames=False)

    def predict(self,wave,impression=False):
        from caption import caption_detail,render_caption
        from humaneness_caption_settings import caption_view
        start=time.perf_counter()
        prediction=self.model.predict_batch([wave],event_threshold=.7,max_events=8,include_frames=False)[0]
        view=caption_view(prediction,self.baseline,{**self.settings,'omit_demographic_wording':not impression})
        from social_profiles import TRAITS
        traits={k:{'score':round(float(v),4),'meaning':TRAITS[k]} for k,v in view['dims'].items() if k in TRAITS} if impression else {}
        # This is a conversation aid, not a character/timbre or personality
        # caption: omit demographics, perceived honesty and quality judgments.
        view['dims']={k:v for k,v in view['dims'].items() if k in (PROSODY | ({'AGEV','GEND'} if impression else set()))}
        view['emo']={k:v for k,v in view['emo'].items() if k in EMOTIONS}
        view['content_enjoyment']=None
        detail=caption_detail(view,self.baseline,k_voicenet=2,k_emonet=1,
            always_on=('TEMP','AGEV','GEND') if impression else ('TEMP',),template='tags',synonym_seed=0,reliability_weighting=False)
        caption=render_caption(detail).strip()
        return {'status':'available','caption':'Possibly '+caption if caption else 'No distinctive delivery cue.',
            'uncertainty':'Uncalibrated acoustic estimates; noise, other speakers and acting may be misleading.',
            'voice_impression_only':impression,'window_s':round(len(wave)/16000,3),'inference_ms':round((time.perf_counter()-start)*1000,2),
            'vocal_traits':traits,
            'model':'laion/humaneness-ears-base-medium','model_revision':MODEL_REVISION,
            'rules_revision':CAPTION_REVISION,'device':self.device,'precision':'bfloat16' if self.device.startswith('cuda') else 'float32',
            'selected':{k:detail[k] for k in ('voicenet','emotions')},
            'normalization':'speech reference z-scores; not probabilities'}

async def caption(request):
    if request.app['busy']:raise web.HTTPServiceUnavailable(text='Caption worker busy; skip this optional hint')
    raw=await request.read()
    if len(raw)%4:raise web.HTTPBadRequest(text='Expected mono 16-kHz little-endian float32 PCM')
    wave=np.frombuffer(raw,dtype='<f4')
    if not 8000<=len(wave)<=128000 or not np.isfinite(wave).all():raise web.HTTPBadRequest(text='Expected finite 0.5–8-second audio')
    request.app['busy']=True
    try:return web.json_response(await asyncio.to_thread(request.app['captioner'].predict,wave,request.query.get('impression')=='1'))
    finally:request.app['busy']=False

async def startup(app):app['captioner']=await asyncio.to_thread(Captioner)
async def state(request):return web.json_response({'ready':True,'busy':request.app['busy'],
    'model_revision':MODEL_REVISION,'rules_revision':CAPTION_REVISION,'device':request.app['captioner'].device,'recordings_saved':False})
def main():
    app=web.Application(client_max_size=128000*4+1);app['busy']=False
    app.on_startup.append(startup);app.router.add_get('/state',state);app.router.add_post('/caption',caption)
    web.run_app(app,host='127.0.0.1',port=int(os.environ.get('ROBO_CAPTION_PORT','8802')))
if __name__=='__main__':main()
