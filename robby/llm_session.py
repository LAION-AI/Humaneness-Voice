"""Per-conversation model/provider choice and optional encrypted personal key."""
import json
from llm_backends import selection,default_selection
from credential_store import store

class LLMSession:
    def __init__(self,conversation):
        self.c=conversation;self.selected=default_selection();self.credential=None
    def settings(self):
        return {'type':'llm_settings',**self.selected,'key_mode':'personal_session' if self.credential else 'server',
            'key_available':bool(self.credential) or store().configured(),
            'applies':'The next response; a response already speaking keeps its current model.'}
    def snapshot(self):return {'llm_selection':dict(self.selected),'llm_credential':self.credential}
    async def command(self,data):
        try:
            if data.startswith('llm:select:'):self.selected=selection(json.loads(data.split(':',2)[2]))
            elif data=='llm:key:clear':self.credential=None
            elif data.startswith('llm:key:'):self.credential=store().seal(data.split(':',2)[2])
            else:raise ValueError('Unknown language model setting')
        except (ValueError,TypeError,KeyError,json.JSONDecodeError):
            await self.c.ws.send_json({'type':'warning','text':'Language model setting was not accepted. Check the selected provider or personal key format.'})
        await self.c.ws.send_json(self.settings())
