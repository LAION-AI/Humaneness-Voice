"""Bounded, browser-scoped relevance context; never expose a memory capability."""
import hashlib,os,re
from pathlib import Path
from knowledge_memory import MemoryStore,cost,retrieve,normalize

PROFILE=('Robby is a warm, playful and attentive conversational robot. He helps the current user, '
         'follows their requests, and may notice relevant surroundings. Visual and audio impressions '
         'are uncertain; do not assume identity or inner feelings.')
MAX_BOUND=1000

def build_context(capability=None,live=None,root=None):
 context={'character':PROFILE,'recent_conversation':[],'relevant_memory':[]}
 live=live if isinstance(live,list) else []
 for row in live[-2:]:
  if not isinstance(row,dict) or row.get('role') not in ('user','assistant') or not isinstance(row.get('content'),str):continue
  item={'role':row['role'],'content':row['content'][:180]}
  if cost({**context,'recent_conversation':context['recent_conversation']+[item]})<=MAX_BOUND:context['recent_conversation'].append(item)
 enabled=isinstance(capability,str) and bool(re.fullmatch('[a-f0-9]{64}',capability))
 memory_used=False
 if enabled:
  base=Path(root or os.environ.get('ROBO_MEMORY_DIR',str(Path.home()/'.local/share/robby-memory'))).expanduser()
  workspace=base/hashlib.sha256(capability.encode()).hexdigest()
  # Do not create a workspace for a forged, disabled or stale capability.
  if base.is_dir() and not base.is_symlink() and workspace.is_dir() and not workspace.is_symlink():
   state,entries=MemoryStore(capability,base).snapshot()
   if not context['recent_conversation']:
    for row in state.get('history',[])[-2:]:
     if not isinstance(row,dict) or row.get('role') not in ('user','assistant') or not isinstance(row.get('content'),str):continue
     item={'role':row['role'],'content':row['content'][:180]}
     if cost({**context,'recent_conversation':context['recent_conversation']+[item]})<=MAX_BOUND:context['recent_conversation'].append(item)
   query=' '.join(x['content'] for x in context['recent_conversation'])
   if query:
    remaining=max(0,MAX_BOUND-cost(context)-48)
    rows,_=retrieve(entries,query,remaining,1)
    for row in rows:
     item={'title':row.get('title',''),'summary':row.get('summary','')}
     if cost({**context,'relevant_memory':context['relevant_memory']+[item]})<=MAX_BOUND:context['relevant_memory'].append(item)
   memory_used=bool(context['relevant_memory'])
 assert cost(context)<=MAX_BOUND
 return context,{'memory_used':memory_used,'recent_turns':len(context['recent_conversation']),'conservative_token_bound':cost(context)}

def flag_context_echo(annotation,context):
 """Flag literal overlaps; matching does not prove that an observation is false."""
 if not isinstance(annotation,dict) or not isinstance(context,dict):return annotation
 sources=[r.get('content','') for r in context.get('recent_conversation',[]) if isinstance(r,dict)]
 sources += [str(r.get('title',''))+' '+str(r.get('summary','')) for r in context.get('relevant_memory',[]) if isinstance(r,dict)]
 corpus=' '.join(normalize(s) for s in sources)
 out=dict(annotation)
 for key in ('observations','events','people'):
  if not isinstance(out.get(key),list):continue
  rows=[]
  for row in out[key]:
   if isinstance(row,dict):
    row=dict(row);description=normalize(row.get('description',''))
    if len(description)>=12 and description in corpus:row['possible_context_echo']=True
   rows.append(row)
  out[key]=rows
 return out
