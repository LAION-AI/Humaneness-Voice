"""Private loopback IPC; embeddings and user audio never leave this host."""
import uuid
import httpx
import numpy as np

class DiarizationClient:
    def __init__(self):
        self.id=uuid.uuid4().hex;self.client=httpx.AsyncClient(base_url='http://127.0.0.1:8796',timeout=1.5)
    async def feed(self,audio):
        r=await self.client.post('/session/'+self.id,content=np.asarray(audio,'<f4').tobytes());r.raise_for_status();return r.json()
    async def close(self):
        try:await self.client.delete('/session/'+self.id)
        except httpx.HTTPError:pass
        await self.client.aclose()
