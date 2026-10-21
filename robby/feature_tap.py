"""Observe existing L15 results without changing TYGER inputs or decisions.

All callers share one serialized encoder. No second encoder, no extra
encoder pass, and no on-disk microphone features. Pilot requests reserve the
same session slot as the browser and are rejected while it is in use.
"""
import io
import asyncio
import numpy as np
from aiohttp import web


class ResearchLocked(web.HTTPClientError):
    status_code=423


async def reserve_research(ears):
    """Allow the native GPU queue a few ms to finish releasing a closed slot."""
    if ears is None:raise web.HTTPServiceUnavailable()
    for _ in range(10):
        if getattr(ears,'browser_waiting',0):raise ResearchLocked(text='Browser session has priority')
        session=ears.open_session(ears.defaults)
        if session is not None:return session
        await asyncio.sleep(.005)
    raise ResearchLocked(text='Live session or another research batch has priority')


def attach_feature_tap(ears):
    engine=ears.service.engine
    original_encode=engine.asr.encode
    original_process=engine.process
    capture={'states':None,'early':None,'owner':None}
    def encode(wavs):
        import threading
        hooks=[];buffers={k:[] for k in (4,8,12)};groups={}
        live=capture['owner']==threading.get_ident()
        if live:
            # Match NemotronAsr.encode's grouped batches, including its silence
            # cache. Do not assume that the agent channel was actually encoded.
            asr=engine.asr;owner=threading.get_ident()
            for i,w in enumerate(wavs):
                if len(w) in asr._zero_enc and not w.any():continue
                groups.setdefault(len(w),[]).append(i)
            def tap(layer):
                def hook(module,args,out):
                    if threading.get_ident()==owner:
                        x=out[0] if isinstance(out,tuple) else out
                        buffers[layer].append(x[:,-4:].detach().clone())
                return hook
            for k in buffers:hooks.append(asr.enc.layers[k-1].register_forward_hook(tap(k)))
        try:result=original_encode(wavs)
        finally:
            for hook in hooks:hook.remove()
        early=[None]*len(wavs)
        if live:
            import torch
            for g,indices in enumerate(groups.values()):
                if any(g>=len(buffers[k]) or len(buffers[k][g])!=len(indices) for k in buffers):
                    raise RuntimeError('Live intermediate-layer grouping mismatch')
                batch=torch.stack([buffers[k][g] for k in (4,8,12)],dim=1).float().cpu().numpy()
                for j,index in enumerate(indices):early[index]=batch[j]
        if live:
            capture['states']=result
            capture['early']=early
        return result
    def process(items):
        import threading
        capture['owner']=threading.get_ident()
        try:result=original_process(items)
        finally:capture['owner']=None
        states=capture.pop('states',None)
        early=capture.pop('early',None)
        if states is not None:
            from app.plugins.ears.tyger.rules import frame_energy
            for i,(session,user,agent,mode) in enumerate(items):
                h=states[2*i][-4:].detach().cpu().numpy().astype(np.float32)
                energy=frame_energy(user)[-len(h):]
                context=states[2*i][-38:].detach().cpu().numpy().astype(np.float32)
                context_db=frame_energy(session.win_u)[-len(context):]
                # Short growing encoder windows can contain one extra padded
                # state. Match the exact same suffix/energy rule as V3/V5.
                n=min(len(context),len(context_db));context=context[-n:];context_db=context_db[-n:]
                layers=None if early is None or early[2*i] is None else np.concatenate([early[2*i],h[None]],axis=0)
                result[session][0].append({'kind':'SPEAKER_FEATURES','features':h,'energy_db':energy,'audio16':user.copy(),
                    'layer_features':layers,'context_features':context,'context_energy_db':context_db,
                    'playback_rms':float(np.sqrt(np.mean(agent**2))),'playback_audio16':agent.copy()})
        return result
    engine.asr.encode=encode
    engine.process=process
    from gpu_guard import serialize_audio_gpu
    serialize_audio_gpu(engine)


def local_only(request):
    # A Cloudflare reverse proxy also connects over loopback: remote alone is
    # NOT sufficient to keep expensive development routes off the public URL.
    if (request.remote!='127.0.0.1' or request.host!='127.0.0.1:8793'
        or any(k in request.headers for k in ('CF-Ray','CF-Connecting-IP','Forwarded','X-Forwarded-For'))):
        raise web.HTTPForbidden()


async def transcribe_check(request,ears):
    """Private bounded ASR diagnostic; same frozen decoder, no LLM/TTS/write."""
    local_only(request)
    raw=await request.read()
    if not raw or len(raw)%4:raise web.HTTPBadRequest(text='float32 mono 16-kHz PCM required')
    audio=np.frombuffer(raw,'<f4')
    if len(audio)>30*16000 or not np.isfinite(audio).all():raise web.HTTPBadRequest()
    session=await reserve_research(ears)
    try:
        text=await asyncio.to_thread(ears.service.engine.asr.transcribe,[audio.copy()])
        return web.json_response({'text':text[0] if text else '', 'seconds':len(audio)/16000})
    finally:ears.close(session)


async def extract(request,ears):
    local_only(request)
    session=await reserve_research(ears)
    try:
        raw=await request.read()
        if not raw or len(raw)%4:raise web.HTTPBadRequest(text='float32 LE mono 16 kHz PCM required')
        audio=np.frombuffer(raw,'<f4')
        if len(audio)>30*16000 or not np.isfinite(audio).all():raise web.HTTPBadRequest(text='Finite audio <=30 seconds required')
        blocks=[]
        for i in range(0,len(audio),5120):
            hop=np.pad(audio[i:i+5120],(0,max(0,5120-len(audio[i:i+5120]))))
            events=await ears.feed(session,hop,np.zeros(5120,np.float32),'speaking')
            blocks.extend(e['features'] for e in events if e['kind']=='SPEAKER_FEATURES')
        features=np.concatenate(blocks)
        ends=np.arange(1,len(features)+1,dtype=np.float32)*.08
        keep=ends<=len(audio)/16000+1e-5
        buffer=io.BytesIO()
        np.savez_compressed(buffer,features=features[keep].astype(np.float16),frame_end_s=ends[keep],
            encoder_layer=np.int32(15),frame_ms=np.int32(80),hop_ms=np.int32(320))
        return web.Response(body=buffer.getvalue(),content_type='application/x-npz')
    finally:
        ears.close(session)


async def extract_context(request,ears):
    """Offline-only selected clock steps: batch four 8-s causal windows.

    Reserve the existing slot only for one short encoder batch. Browser sessions
    win between calls. No second model, no GRU state change, no future audio.
    """
    local_only(request)
    # Default remains byte/shape compatible with the immutable V3 cache.
    try: context_frames=int(request.query.get('frames','25'))
    except ValueError:raise web.HTTPBadRequest(text='frames must be 25 or 38')
    if context_frames not in (25,38):raise web.HTTPBadRequest(text='frames must be 25 or 38')
    session=await reserve_research(ears)
    try:
        raw=await request.read()
        if len(raw)%4:raise web.HTTPBadRequest()
        values=np.frombuffer(raw,'<f4')
        if not len(values) or len(values)%(8*16000) or len(values)>4*8*16000 or not np.isfinite(values).all():
            raise web.HTTPBadRequest(text='1–4 float32 mono 8-second windows required')
        waves=values.reshape(-1,128000)
        def work():
            import torch,time
            from app.plugins.ears.tyger.rules import frame_energy
            begin=time.perf_counter()
            hidden=ears.service.engine.asr.encode(list(waves))
            features=[];pools=[];energy=[]
            for h,w in zip(hidden,waves):
                x=h.detach().cpu().numpy().astype(np.float32)
                db=frame_energy(w)[-len(x):]
                n=min(len(db),len(x));x=x[-n:];db=db[-n:]
                voiced=x[db>-50]
                if not len(voiced):voiced=x
                if n<context_frames:raise RuntimeError('Encoder returned insufficient causal states')
                features.append(x[-context_frames:]);energy.append(db[-context_frames:])
                pools.append(np.stack([voiced.mean(0),voiced.std(0)]))
            torch.cuda.synchronize()
            return np.stack(features),np.stack(pools),np.stack(energy),time.perf_counter()-begin
        features,pools,energy,seconds=await asyncio.to_thread(work)
        result=io.BytesIO()
        np.savez_compressed(result,features=features.astype(np.float16),pool=pools.astype(np.float16),
            energy=energy.astype(np.float16),encoder_seconds=np.float32(seconds),frame_ms=np.int32(80))
        return web.Response(body=result.getvalue(),content_type='application/x-npz')
    finally:ears.close(session)


async def extract_layers(request,ears):
    """Private research tap: L4/L8/L12/L15, newest four 80-ms states.

    Temporary hooks observe this request's thread only. The native L15 stop
    hook, ASR decoding thread, TYGER inputs and decisions are not modified.
    """
    local_only(request)
    session=await reserve_research(ears)
    try:
        raw=await request.read()
        if len(raw)%4:raise web.HTTPBadRequest()
        values=np.frombuffer(raw,'<f4')
        if not len(values) or len(values)%128000 or len(values)>512000 or not np.isfinite(values).all():
            raise web.HTTPBadRequest(text='1–4 finite eight-second mono float32 windows required')
        waves=values.reshape(-1,128000)
        def work():
            import threading,time,torch
            begin=time.perf_counter();asr=ears.service.engine.asr;owner=threading.get_ident();captured={};hooks=[]
            def tap(layer):
                def hook(module,args,out):
                    if threading.get_ident()==owner:
                        x=out[0] if isinstance(out,tuple) else out
                        captured[layer]=x[:,-4:].detach().clone()
                return hook
            try:
                for layer in (4,8,12):hooks.append(asr.enc.layers[layer-1].register_forward_hook(tap(layer)))
                # Direct equal-length encode also captures genuine zero windows;
                # encode()'s L15-only silence cache cannot supply earlier layers.
                from app.plugins.ears.tyger.asr import _StopAtL15
                with torch.no_grad():
                    feats,mask=asr.features(list(waves));asr._tls.stop=True
                    try:asr.enc(input_features=feats,attention_mask=mask,num_lookahead_tokens=0)
                    except _StopAtL15:pass
                    finally:asr._tls.stop=False
                    captured[15]=asr._tls.h[:,-4:]
                x=torch.stack([captured[layer] for layer in (4,8,12,15)],dim=1).float().cpu().numpy()
                torch.cuda.synchronize();return x,time.perf_counter()-begin
            finally:
                for hook in hooks:hook.remove()
        x,seconds=await asyncio.to_thread(work)
        out=io.BytesIO();np.savez_compressed(out,features=x.astype(np.float16),layers=np.array([4,8,12,15]),encoder_seconds=seconds)
        return web.Response(body=out.getvalue(),content_type='application/x-npz')
    finally:ears.close(session)

