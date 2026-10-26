"""Allowlisted language models, actual OpenRouter endpoint tags and transport.

Public metadata has no keys, credential ciphertexts or arbitrary endpoint URLs.
"""
import asyncio,json,os,re,time
from pathlib import Path
import httpx
from contextlib import asynccontextmanager
from credential_store import store

OPENROUTER='https://openrouter.ai/api/v1'
QWEN='qwen/qwen3.8-27b'
MODELS={'qwen':{'label':'Qwen 3.8 27B · OpenRouter','model':QWEN},
        'local':{'label':'Nemotron Bolt E4B · local GPU','model':'nemotron-bolt'}}
_providers={'cerebras/fp16':{'id':'cerebras/fp16','label':'Cerebras · FP16','parameters':['tools','tool_choice']}}
_updated=0.

async def refresh_providers(client=None):
    global _providers,_updated
    if time.monotonic()-_updated<300:return
    async def fetch(c):
        response=await c.get(OPENROUTER+'/models/'+QWEN+'/endpoints',timeout=10);response.raise_for_status()
        result={}
        for row in response.json()['data']['endpoints']:
            tag=row.get('tag','')
            if row.get('status',-1)<0 or not re.fullmatch(r'[a-z0-9_.-]+(?:/[a-z0-9_.-]+)*',tag):continue
            result[tag]={'id':tag,'label':row['provider_name']+' · '+row.get('quantization','unknown').upper(),
                'parameters':row.get('supported_parameters',[])}
        return result
    if client is not None:result=await fetch(client)
    else:
        async with httpx.AsyncClient() as c:result=await fetch(c)
    if result:_providers=result;_updated=time.monotonic()

def default_selection():
    return {'model':'qwen','provider':'cerebras/fp16'} if store().configured() else {'model':'local','provider':'local'}

def selection(value=None):
    value=value if value is not None else default_selection()
    if not isinstance(value,dict) or set(value)-{'model','provider'}:raise ValueError('Invalid language model selection')
    model=value.get('model');provider=value.get('provider','cerebras/fp16' if model=='qwen' else 'local')
    if model not in MODELS or (model=='qwen' and provider not in _providers) or (model=='local' and provider!='local'):
        raise ValueError('Language model or provider is not available')
    return {'model':model,'provider':provider}

def catalog():
    providers=sorted(_providers.values(),key=lambda r:(r['id']!='cerebras/fp16',r['label']))
    return {'models':[{'id':key,**spec} for key,spec in MODELS.items()],
        'providers':[{'id':r['id'],'label':r['label']} for r in providers],
        'default':default_selection(),'server_key_available':store().configured(),
        'privacy':'OpenRouter mode sends conversation text to OpenRouter and the selected provider. The server key is never sent to your browser.'}

def config(value=None,credential=None):
    chosen=selection(value)
    if chosen['model']=='local':
        from bolt_director import local_config
        base,model=local_config()
        return {'selection':chosen,'url':base+'/v1/chat/completions','model':model,'headers':{},'remote':False}
    vault=store();key=vault.open(credential) if credential else vault.owner()
    return {'selection':chosen,'url':OPENROUTER+'/chat/completions','model':QWEN,
        'headers':{'Authorization':'Bearer '+key,'X-OpenRouter-Title':'Robby Voice Demo'},'remote':True,
        'parameters':_providers[chosen['provider']]['parameters']}

def request_body(config,body):
    body=dict(body);body['model']=config['model']
    if not config['remote']:return body
    body.pop('chat_template_kwargs',None);body.pop('reasoning_effort',None);body.pop('parallel_tool_calls',None)
    if 'max_completion_tokens' in body:body['max_tokens']=body.pop('max_completion_tokens')
    body['reasoning']={'enabled':False,'exclude':True}
    body['provider']={'only':[config['selection']['provider']],'allow_fallbacks':False}
    if 'response_format' not in config['parameters']:body.pop('response_format',None)
    return body

def metrics(config):
    return {'llm_model':config['model'],'llm_provider':config['selection']['provider'],
        'llm_backend':'OpenRouter' if config['remote'] else 'Local GPU'}

def check_response(response):
    if getattr(response,'status_code',200)>=400:
        # No raw provider error body / headers / credential can enter the UI.
        raise RuntimeError('Language model request failed (HTTP '+str(response.status_code)+')')

def retry_delay(response):
    try:return min(2.,max(.5,float(response.headers.get('retry-after','.5'))))
    except (ValueError,AttributeError):return .5

@asynccontextmanager
async def completion_stream(client,cfg,body,metrics):
    for attempt in range(2):
        async with client.stream('POST',cfg['url'],json=request_body(cfg,body),headers=cfg['headers'],timeout=30.) as response:
            if cfg['remote'] and getattr(response,'status_code',200)==429 and not attempt:
                delay=retry_delay(response);metrics['llm_rate_limit_retries']=1
            else:
                check_response(response);yield response;return
        await asyncio.sleep(delay)
