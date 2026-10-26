"""FastAPI facade for Robby's local streaming pipeline and existing browser UI.

uvicorn bolt_api:app --host 127.0.0.1 --port 8798
The existing ASR/avatar websocket is proxied without changing its binary wire.
"""
import asyncio,base64,json,os,time
from contextlib import asynccontextmanager
import aiohttp,httpx,numpy as np
from fastapi import FastAPI,HTTPException,Request,WebSocket,WebSocketDisconnect
from fastapi.responses import StreamingResponse,JSONResponse
from pydantic import BaseModel,Field
from bolt_text import CATALOGUE,Sentence
from bolt_director import BoltDirector,MODEL_REPO,MODEL_REVISION,local_config

ASR_BASE=os.environ.get('ROBO_ASR_BASE','http://127.0.0.1:8793')
TTS_BASE=os.environ.get('ROBO_TTS_BASE','http://127.0.0.1:8792')

@asynccontextmanager
async def lifespan(app):
    app.state.http=httpx.AsyncClient(timeout=180,limits=httpx.Limits(max_connections=20,max_keepalive_connections=12))
    app.state.admission=asyncio.Semaphore(2)
    from llm_backends import refresh_providers
    try:await refresh_providers(app.state.http)
    except Exception:pass
    yield
    await app.state.http.aclose()

app=FastAPI(title='Robby local Nemotron sentence-stream API',version='1.0',lifespan=lifespan)

class Chat(BaseModel):
    text:str=Field(min_length=1,max_length=3000)
    speaker:dict=Field(default_factory=dict)
    history:list[dict]=Field(default_factory=list,max_length=24)
    voice:str=Field(default='emolia_c1699',max_length=80)
    chunk_frames:int=Field(default=20)
    reply_words:int|None=Field(default=None,ge=3,le=80)
    quick_replies:bool=True
    llm_model:str|None=Field(default=None,max_length=40)
    llm_provider:str|None=Field(default=None,max_length=100)

class Speech(BaseModel):
    text:str=Field(min_length=1,max_length=1200)
    cue:str=Field(default='contentment, soft',max_length=100,pattern=r'^[A-Za-z_ ,\-]+$')
    voice:str=Field(default='emolia_c1699',max_length=80)
    chunk_frames:int=20

def sse(event,payload):return 'event: '+event+'\ndata: '+json.dumps(payload,ensure_ascii=False)+'\n\n'

@app.get('/api/bolt/state')
async def state(request:Request):
    client=request.app.state.http;base,model=local_config()
    async def check(url):
        try:
            r=await client.get(url,timeout=3.);return r.status_code==200,r.json()
        except Exception as exc:return False,{'error':type(exc).__name__}
    llm,asr,tts=await asyncio.gather(check(base+'/health'),check(ASR_BASE+'/api/state'),check(TTS_BASE+'/api/state'))
    from llm_backends import default_selection,catalog,MODELS
    chosen=default_selection();remote=chosen['model']=='qwen';configured=catalog()['server_key_available']
    return {'model':MODELS[chosen['model']]['model'],'repository':None if remote else MODEL_REPO,'revision':None if remote else MODEL_REVISION,
        'engine':'OpenRouter · '+chosen['provider'] if remote else 'llama.cpp CUDA Q4_K_M','templates':len(CATALOGUE),
        'default_llm':chosen,'server_key_available':configured,
        'ready':(configured if remote else llm[0]) and asr[0] and asr[1].get('ready',False) and tts[0] and tts[1].get('ready',True),
        'llm':llm[1],'asr':asr[1],'tts_ready':tts[0],
        'streaming':'ASR sentences -> local LLM sentences -> TTS PCM chunks',
        'speaker_mode':'All intelligible voices; nearest voice prototype, optional names, session RAM'}

@app.get('/api/bolt/templates')
async def templates():return [{k:v for k,v in row.items() if k!='keywords'} for row in CATALOGUE]

@app.post('/api/bolt/llm')
async def llm(body:Chat,request:Request):
    from llm_backends import selection
    try:chosen=selection({'model':body.llm_model,'provider':body.llm_provider or ('local' if body.llm_model=='local' else 'cerebras/fp16')} if body.llm_model else None)
    except ValueError as exc:raise HTTPException(422,str(exc))
    async def events():
        metrics={}
        try:
            async with request.app.state.admission:
                async for sentence in BoltDirector().stream(request.app.state.http,body.text,body.speaker,body.history,metrics=metrics,reply_words=body.reply_words,backend=chosen):
                    yield sse('sentence',sentence.spec())
                yield sse('done',metrics)
        except Exception as exc:yield sse('error',{'error':type(exc).__name__})
    return StreamingResponse(events(),media_type='text/event-stream',headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})

@app.post('/api/bolt/chat')
async def chat(body:Chat,request:Request):
    if body.chunk_frames not in (2,5,10,20):raise HTTPException(422,'chunk_frames must be 2, 5, 10 or 20')
    from llm_backends import selection
    try:chosen=selection({'model':body.llm_model,'provider':body.llm_provider or ('local' if body.llm_model=='local' else 'cerebras/fp16')} if body.llm_model else None)
    except ValueError as exc:raise HTTPException(422,str(exc))
    async def events():
        from bolt_pipeline import pipeline
        # A dedicated client with the TTS base keeps URL handling local and
        # reuses keep-alive connections across all sentences in this reply.
        async with request.app.state.admission,httpx.AsyncClient(base_url=TTS_BASE,timeout=180) as client:
            try:
                async for kind,value in pipeline(client,body.text,body.speaker,body.history,body.voice,chunk_frames=body.chunk_frames,reply_words=body.reply_words,llm_selection=chosen,quick_replies=body.quick_replies):
                    if kind=='audio':
                        pcm=(np.clip(value,-1,1)*32767).astype('<i2')
                        value={'encoding':'pcm_s16le','sample_rate':24000,'channels':1,
                               'audio':base64.b64encode(pcm.tobytes()).decode(),'samples':len(pcm)}
                    yield sse(kind,value)
            except Exception as exc:yield sse('error',{'error':type(exc).__name__})
    return StreamingResponse(events(),media_type='text/event-stream',headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})

@app.post('/api/bolt/asr')
async def asr(request:Request):
    raw=await request.body()
    if not raw or len(raw)%4 or len(raw)>15*16000*4:raise HTTPException(422,'Send <=15 s mono 16-kHz float32 little-endian PCM')
    audio=np.frombuffer(raw,'<f4')
    if not np.isfinite(audio).all():raise HTTPException(422,'Finite samples required')
    start=time.perf_counter()
    from asr_catalog import MODELS,worker_url,default_model
    model=request.query_params.get('model',default_model())
    if model not in MODELS:raise HTTPException(422,'Unknown ASR model')
    url=ASR_BASE+'/internal/transcribe' if model=='nemotron' else worker_url()+'/transcribe/'+model
    response=await request.app.state.http.post(url,content=raw)
    if response.status_code!=200:raise HTTPException(response.status_code,'ASR is busy or unavailable')
    return {**response.json(),'model':model,'asr_ms':round((time.perf_counter()-start)*1000,2)}

@app.post('/api/bolt/tts')
async def tts(body:Speech,request:Request):
    if body.chunk_frames not in (2,5,10,20):raise HTTPException(422,'chunk_frames must be 2, 5, 10 or 20')
    async def events():
        from bolt_pipeline import pipeline
        async with request.app.state.admission,httpx.AsyncClient(base_url=TTS_BASE,timeout=180) as client:
            try:
                async for kind,value in pipeline(client,'',{},[],body.voice,chunk_frames=body.chunk_frames,preset=[Sentence(body.text,body.cue)]):
                    if kind=='audio':
                        pcm=(np.clip(value,-1,1)*32767).astype('<i2')
                        value={'encoding':'pcm_s16le','sample_rate':24000,'channels':1,
                               'audio':base64.b64encode(pcm.tobytes()).decode(),'samples':len(pcm)}
                    yield sse(kind,value)
            except Exception as exc:yield sse('error',{'error':type(exc).__name__})
    return StreamingResponse(events(),media_type='text/event-stream',headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})

@app.post('/api/bolt/audio-chat')
async def audio_chat(request:Request):
    """One complete <=15-s float32/16-kHz input, then streamed text and PCM.

    Live browser capture uses /ws; this endpoint is useful for API integrations
    and reproducible generation-time measurements from complete input audio.
    """
    recognized=await asr(request)
    response=await chat(Chat(text=recognized['text']),request)
    async def events():
        yield sse('asr',recognized)
        async for chunk in response.body_iterator:yield chunk
    return StreamingResponse(events(),media_type='text/event-stream',headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})

@app.websocket('/ws')
async def websocket(ws:WebSocket):
    if ws.query_params.get('synthetic_diagnostics'):
        await ws.close(code=1008);return
    await ws.accept()
    try:
        async with aiohttp.ClientSession() as session,session.ws_connect(ASR_BASE.replace('http','ws',1)+'/ws',max_msg_size=4*1024*1024) as upstream:
            async def browser_to_server():
                while True:
                    message=await ws.receive()
                    if message['type']=='websocket.disconnect':return
                    if message.get('bytes') is not None:await upstream.send_bytes(message['bytes'])
                    elif message.get('text') is not None:await upstream.send_str(message['text'])
            async def server_to_browser():
                async for message in upstream:
                    if message.type==aiohttp.WSMsgType.BINARY:await ws.send_bytes(message.data)
                    elif message.type==aiohttp.WSMsgType.TEXT:await ws.send_text(message.data)
                    elif message.type in (aiohttp.WSMsgType.CLOSE,aiohttp.WSMsgType.ERROR):return
            tasks=[asyncio.create_task(browser_to_server()),asyncio.create_task(server_to_browser())]
            try:await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in tasks:task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
    except (WebSocketDisconnect,ConnectionError,aiohttp.ClientError):pass
    finally:
        try:await ws.close()
        except (RuntimeError,WebSocketDisconnect):pass

@app.api_route('/{path:path}',methods=['GET','POST'])
async def browser_proxy(path:str,request:Request):
    if path.startswith('internal/'):raise HTTPException(404)
    if request.method=='POST':raise HTTPException(405)
    url=ASR_BASE+'/'+path
    if request.url.query:url+='?'+request.url.query
    response=await request.app.state.http.send(request.app.state.http.build_request('GET',url),stream=True)
    headers={k:v for k,v in response.headers.items() if k.lower() in ('content-type','cache-control','etag','last-modified','location')}
    async def content():
        try:
            async for chunk in response.aiter_bytes():yield chunk
        finally:await response.aclose()
    return StreamingResponse(content(),status_code=response.status_code,headers=headers)
