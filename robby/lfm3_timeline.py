"""Validate and label approximate model-generated visual-event timestamps.

No interpolation, face recognition or temporal ground truth is performed here.
"""
import math,re

def normalize_relevance(value):
 if isinstance(value,bool):return None
 if isinstance(value,int) and value in (0,1,2):return value
 if isinstance(value,str) and value.strip() in ('0','1','2'):return int(value.strip())
 return None

def note_unreported_signal(annotation,quiet):
 """Do not silently present an empty model answer as detected silence."""
 if quiet or not isinstance(annotation,dict):return annotation
 obs=annotation.get('observations')
 if isinstance(obs,list) and not obs and not any(isinstance(annotation.get(k),str) and annotation[k].strip() for k in ('foreground','background','speech','music')):
  out=dict(annotation);out['observations']=[{'category':'signal','description':'Audio exceeds the silence threshold; the model provided no description.','relevance':None,'source':'signal_level_check'}];return out
 return annotation

def recover_repaired_video(candidate):
 """Recover only an unambiguous nested event list after syntactic repair."""
 if not isinstance(candidate,dict) or not any(k in candidate for k in ('foreground','background','people','observations','events')):return None
 out=dict(candidate)
 # Some models accidentally put top-level fields inside an observations
 # array. Lift only a single unambiguous carrier, preserving raw output.
 if isinstance(out.get('observations'),list):
  carriers=[r for r in out['observations'] if isinstance(r,dict) and not r.get('category') and not r.get('description') and any(k in r for k in ('people','events','changes'))]
  if len(carriers)==1:
   carrier=carriers[0]
   for key in ('people','events','changes'):
    if key not in out and key in carrier:out[key]=carrier[key]
   out['observations']=[r for r in out['observations'] if r is not carrier]
 if not isinstance(out.get('events'),list):
  found=[]
  def walk(v,depth=0):
   if depth>6:return
   if isinstance(v,dict):
    if isinstance(v.get('events'),list):found.append(v['events'])
    for value in v.values():walk(value,depth+1)
   elif isinstance(v,list):
    for value in v:walk(value,depth+1)
  walk(out)
  if len(found)==1:out['events']=found[0]
 if isinstance(out.get('people'),list):
  out['people']=[row for row in out['people'] if not isinstance(row,dict) or 'id' in row or 'description' in row]
 return out

def normalize_scores(annotation):
 if not isinstance(annotation,dict):return annotation
 out=dict(annotation)
 for key in ('people','events','observations'):
  if isinstance(out.get(key),list):
   out[key]=[{**row,'relevance':normalize_relevance(row.get('relevance'))} if isinstance(row,dict) else row for row in out[key][:100]]
   if key=='observations':
    before=len(out[key]);out[key]=[row for row in out[key] if not isinstance(row,dict) or not re.search(r'(?:^|[_\W])(?:memory|remembered|context)(?:$|[_\W])',str(row.get('category','')),re.I)]
    if before!=len(out[key]):out['context_only_items_filtered']=before-len(out[key])
    before=len(out[key]);out[key]=[row for row in out[key] if isinstance(row,dict) and isinstance(row.get('description'),str) and row['description'].strip()]
    if before!=len(out[key]):out['malformed_observations_filtered']=before-len(out[key])
 # Preserve model summaries as explicitly unscored observations when it did
 # not supply a matching rating; never infer a relevance value ourselves.
 observations=out.get('observations') if isinstance(out.get('observations'),list) else []
 categories={str(row.get('category','')).casefold() for row in observations if isinstance(row,dict)}
 for key in ('foreground','background','speech','music','changes','audio','visual','cross_modal'):
  value=out.get(key)
  if isinstance(value,str) and value.strip() and key not in categories and not re.match(r'^\s*(?:none|no |not |n/a|silence|absent|unknown)',value,re.I):
   observations.append({'category':key,'description':value,'relevance':None,'source':'unscored_summary'})
 if observations:out['observations']=observations
 return out

def number(value):
 if isinstance(value,bool):return None
 if isinstance(value,(int,float)):
  n=float(value);return n if math.isfinite(n) else None
 if not isinstance(value,str):return None
 s=value.strip().lower();s=re.sub(r'\s*(?:seconds?|secs?|s)$','',s).strip()
 try:
  if ':' in s:
   parts=s.split(':')
   if len(parts) not in (2,3):return None
   vals=[float(p) for p in parts]
   if any(not math.isfinite(x) or x<0 for x in vals):return None
   n=sum(v*60**i for i,v in enumerate(reversed(vals)))
  else:n=float(s)
  return n if math.isfinite(n) else None
 except ValueError:return None

def normalize_timeline(annotation,start,duration,fps=5):
 """Retain observations even when timing is missing or invalid; never invent times."""
 if not isinstance(annotation,dict):return annotation
 out=normalize_scores(annotation);events=out.get('events')
 if not isinstance(events,list):
  out['timeline_status']='Model did not return a structured event list.'
  return out
 normalized=[]
 ids={str(p.get('id')) for p in annotation.get('people',[]) if isinstance(p,dict) and p.get('id') is not None} if isinstance(annotation.get('people'),list) else set()
 for event in events[:100]:
  if not isinstance(event,dict):
   normalized.append({'description':str(event)[:1000],'timing_status':'unavailable'});continue
  row=dict(event);a=number(event.get('start_s'));b=number(event.get('end_s'))
  person=str(event.get('person_id') or '').strip()
  if person and not ids and re.match(r'^\s*(?:no|not a|without a)\s+(?:visible\s+)?(?:person|people|gesture|action)\b',str(event.get('description') or ''),re.I):
   continue
  if person and person not in ids:row['person_status']='unconfirmed_label'
  if a is not None and b is not None and 0<=a<=b<=duration+.05:
   # Slight rounding beyond the measured duration is clamped; larger errors
   # remain untimed rather than being silently assigned a false timestamp.
   a=min(a,duration);b=min(b,duration)
   row.update(start_s=round(a,3),end_s=round(b,3),absolute_start_s=round(start+a,3),absolute_end_s=round(start+b,3),timing_status='model_estimate')
  else:
   row['timing_status']='unavailable_or_out_of_range'
   row.pop('absolute_start_s',None);row.pop('absolute_end_s',None)
  normalized.append(row)
 out['events']=normalized
 out['timeline_status']=f'Model estimates, not verified frame timestamps; visual sampling is about {fps:g} fps.'
 return out
