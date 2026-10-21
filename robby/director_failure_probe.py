"""Public synthetic prompt only. Diagnose provider failures without key output."""
import asyncio,json
import httpx
from conversation_director import CompactDirector
from visual_identity import VisualIdentity

class Probe(httpx.AsyncClient):
    async def post(self,*args,**kwargs):
        response=await super().post(*args,**kwargs)
        if response.status_code>=400:
            try:error=response.json().get('error',{})
            except Exception:error={}
            print(json.dumps({'status':response.status_code,'type':error.get('type'),'code':error.get('code'),'param':error.get('param'),'message':str(error.get('message',''))[:700]}),flush=True)
        return response

async def main():
    async with Probe() as client:
        for _ in range(5):
            try:
                result=await CompactDirector().generate(client,'Hey Rabi, ich bin Christoph. Schön dich kennenzulernen.',
                    {'speaker_id':'synthetic-a','display_name':'Christoph','named':True,'enrolled':True},[],
                    interaction_state={'enabled':True,'rules':[],'decision_model':'local Laya','decision_interval_ms':320})
                print(json.dumps({'ok':True,'director_ms':result['director_ms']}),flush=True)
            except Exception as exc:print(json.dumps({'ok':False,'error_type':type(exc).__name__}),flush=True)

if __name__=='__main__':asyncio.run(main())
