"""Loopback CPU AudioSet candidates and coarse temporal windows, alongside Whisper.

EfficientAT MN04 is a clip tagger, not a calibrated spatial localizer. Window
scores provide approximate temporal evidence; they do not identify speakers or
prove a source. No audio is stored. One CPU job at a time, no stale-work queue.
"""
import asyncio,contextlib,io,os,sys,time
import numpy as np
from aiohttp import web
from bootstrap_sound_events import assets,REVISION,WEIGHT_SHA256
class EventTagger:
    def __init__(self):
        import torch
        root=assets();os.chdir(root);sys.path.insert(0,str(root))
        from models.mn.model import get_model
        from models.preprocess import AugmentMelSTFT
        from helpers.utils import labels
        torch.set_num_threads(2);self.labels=labels
        with contextlib.redirect_stdout(io.StringIO()):self.model=get_model(width_mult=.4,pretrained_name=None).eval()
        self.model.load_state_dict(torch.load(root/'mn04_as.pt',map_location='cpu',weights_only=True),strict=True)
        self.mel=AugmentMelSTFT(fmax=15000).eval();self.params=sum(p.numel() for p in self.model.parameters())
        self.predict(np.zeros(32000,np.float32))
    def predict(self,wave):
        import torch,torchaudio
        start=time.perf_counter();x=torch.from_numpy(wave.copy()).unsqueeze(0)
        # Stable official preprocessing at 32 kHz. The existing microphone is 16 kHz.
        with torch.inference_mode():
            x=torchaudio.functional.resample(x,16000,32000)
            full=torch.sigmoid(self.model(self.mel(x).unsqueeze(1))[0]).numpy()[0]
            n=40960;hop=20480;starts=list(range(0,max(1,x.shape[1]-n+1),hop))
            if x.shape[1]>n and starts[-1]!=x.shape[1]-n:starts.append(x.shape[1]-n)
            batch=torch.stack([torch.nn.functional.pad(x[0,i:i+n],(0,max(0,n-(x.shape[1]-i)))) for i in starts])
            frames=torch.sigmoid(self.model(self.mel(batch).unsqueeze(1))[0]).numpy()
        scores=np.maximum(full,frames.max(axis=0))
        def top(values,count=5):
            # Optional Ears impressions have their own toggle. Never infer age
            # or gender through the environmental event classifier.
            order=[i for i in np.argsort(values)[::-1] if self.labels[i] not in ('Male speech, man speaking','Female speech, woman speaking','Child speech, kid speaking')]
            return [{'label':self.labels[i],'score':round(float(values[i]),4)} for i in order[:count]]
        return {'status':'available','model':'EfficientAT mn04_as','revision':REVISION,'weight_sha256':WEIGHT_SHA256,
            'device':'cpu','parameters':self.params,'total_ms':round((time.perf_counter()-start)*1000,2),
            'window_s':round(len(wave)/16000,3),'candidates':top(scores,6),
            'music_score':round(float(scores[self.labels.index('Music')]),4),'speech_score':round(float(scores[self.labels.index('Speech')]),4),
            'frames':[{'start_s':round(i/32000,3),'end_s':round(min(i+n,x.shape[1])/32000,3),'candidates':top(v,3)} for i,v in zip(starts,frames)],
            'temporal_resolution_note':'Overlapping 1.28-second classification windows, 0.64-second stride; not precise event boundaries or spatial direction.',
            'uncertainty':'Multilabel model scores, not calibrated event probabilities.'}
async def classify(request):
    app=request.app
    if app['job'] and not app['job'].done():raise web.HTTPServiceUnavailable(text='Optional event tagger busy')
    raw=await request.read()
    if len(raw)%4:raise web.HTTPBadRequest(text='Expected float32 PCM')
    wave=np.frombuffer(raw,dtype='<f4')
    if not 8000<=len(wave)<=96000 or not np.isfinite(wave).all() or np.max(np.abs(wave))>1.01:raise web.HTTPBadRequest(text='Expected normalized 16-kHz mono, 0.5–6 s')
    if app['job'] and not app['job'].done():raise web.HTTPServiceUnavailable(text='Optional event tagger busy')
    app['job']=asyncio.create_task(asyncio.to_thread(app['tagger'].predict,wave.copy()))
    return web.json_response(await asyncio.shield(app['job']))
async def startup(app):app['tagger']=await asyncio.to_thread(EventTagger)
async def state(request):
    t=request.app['tagger'];return web.json_response({'ready':True,'busy':bool(request.app['job'] and not request.app['job'].done()),'model':'EfficientAT mn04_as','parameters':t.params,'device':'cpu','revision':REVISION,'recordings_saved':False})
if __name__=='__main__':
    app=web.Application(client_max_size=96000*4+1);app['job']=None;app.on_startup.append(startup)
    app.router.add_get('/state',state);app.router.add_post('/events',classify)
    web.run_app(app,host='127.0.0.1',port=int(os.environ.get('ROBO_EVENT_PORT','8804')))
