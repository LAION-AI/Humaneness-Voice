"""Own loopback-only, single-flight CPU Laya adapter. No transcripts on disk."""
import asyncio,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from aiohttp import web
from laya_policy import MODEL,REVISION,SOURCE_COMMIT,prepare,interpret

ROOT=Path(__file__).resolve().parent
STATE={'ready':False,'model':MODEL,'revision':REVISION,'source_commit':SOURCE_COMMIT,
    'policy_sha256':hashlib.sha256((ROOT/'laya_policy.py').read_bytes()).hexdigest(),'device':'cpu','requests':0,'error':None}

def load():
    os.environ.setdefault('USE_TF','0')
    source=ROOT/'vendor/laya'
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if commit!=SOURCE_COMMIT:raise RuntimeError('Laya source does not match the reviewed source commit')
    sys.path.insert(0,str(source))
    import torch,laya
    torch.set_num_threads(4)
    agent=laya.load(MODEL,revision=REVISION,device='cpu')
    # Warm actual paired questions before announcing availability.
    state,questions,_=prepare({'agent_speaking':True,'user_partial':'Stopp!','agent_full_reply':'A short story.'})
    agent.predict(state,questions)
    return agent

async def startup(app):
    async def warm():
        start=time.perf_counter()
        try:
            app['agent']=await asyncio.to_thread(load)
            STATE.update(ready=True,load_seconds=round(time.perf_counter()-start,3))
        except Exception as exc:STATE['error']=type(exc).__name__+': '+str(exc)[:200]
    app['warm_task']=asyncio.create_task(warm());app['busy']=False

async def health(request):return web.json_response(STATE)

async def decide(request):
    if request.remote not in ('127.0.0.1','::1') or request.headers.get('X-Forwarded-For'):raise web.HTTPForbidden()
    if not STATE['ready'] or request.app['busy']:raise web.HTTPServiceUnavailable(text='Local decision model not ready or busy')
    body=await request.json()
    if body.get('model')!=MODEL:raise web.HTTPBadRequest(text='Only the pinned multilingual Laya model is served')
    try:
        context=json.loads(body['messages'][-1]['content'])
        if not isinstance(context,dict):raise ValueError()
        state,questions,ids=prepare(context)
    except (TypeError,ValueError,KeyError,IndexError):raise web.HTTPBadRequest(text='Invalid bounded decision context')
    request.app['busy']=True;start=time.perf_counter()
    try:
        answers=(await asyncio.to_thread(request.app['agent'].predict,state,questions))['answers'] if questions else {}
        choice,scores=interpret(context,answers,ids)
        elapsed=round((time.perf_counter()-start)*1000,2)
        STATE.update(requests=STATE['requests']+1,last_ms=elapsed)
        return web.json_response({'model':MODEL,'system_fingerprint':REVISION,
            'choices':[{'message':{'role':'assistant','content':json.dumps(choice)}}],
            'local_decision':{'paired_probabilities':scores,'forward_ms':elapsed,'calibration':'development only; not calibrated identity or safety probabilities'}})
    finally:request.app['busy']=False

if __name__=='__main__':
    app=web.Application(client_max_size=32768);app.on_startup.append(startup)
    app.router.add_get('/health',health);app.router.add_post('/v1/chat/completions',decide)
    web.run_app(app,host='127.0.0.1',port=8797,access_log=None)
