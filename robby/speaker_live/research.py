"""Loopback-only real native-hop prediction, public synthetic experiments only."""
import io
import asyncio
import numpy as np
from aiohttp import web
from feature_tap import local_only,reserve_research
from speaker_runtime import LivePredictor

async def extract_predictions(request,ears):
    local_only(request);raw=await request.read()
    if not raw or len(raw)%4:raise web.HTTPBadRequest()
    audio=np.frombuffer(raw,'<f4')
    if len(audio)>30*16000 or not np.isfinite(audio).all():raise web.HTTPBadRequest()
    session=await reserve_research(ears)
    predictor=LivePredictor('long');vectors=[];long=[];probs=[];energies=[]
    try:
        for at in range(0,len(audio),5120):
            hop=audio[at:at+5120];hop=np.pad(hop,(0,5120-len(hop)))
            events=await ears.feed(session,hop,np.zeros(5120,np.float32),'speaking')
            event=next(e for e in events if e['kind']=='SPEAKER_FEATURES')
            pred=await asyncio.to_thread(predictor.predict,event)
            vectors.append(pred['embedding'] if pred['embedding'] is not None else np.zeros(192,np.float32))
            long.append(pred['long_embedding'] if pred.get('long_embedding') is not None else np.zeros(192,np.float32))
            probs.append(pred.get('activity_probabilities',[0,1,0]));energies.append(event['energy_db'])
        buffer=io.BytesIO();np.savez_compressed(buffer,embedding=np.stack(vectors),long_embedding=np.stack(long),
            probabilities=np.array(probs),energy=np.array(energies),hop_seconds=np.float32(.32))
        return web.Response(body=buffer.getvalue(),content_type='application/x-npz')
    finally:ears.close(session)
