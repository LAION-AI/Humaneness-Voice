"""Validate and label approximate model-generated visual-event timestamps.

No interpolation, face recognition or temporal ground truth is performed here.
"""
import math,re

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

def normalize_timeline(annotation,start,duration):
 """Retain observations even when timing is missing or invalid; never invent times."""
 if not isinstance(annotation,dict):return annotation
 out=dict(annotation);events=annotation.get('events')
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
 out['timeline_status']='Model estimates, not verified frame timestamps; visual sampling is about 2 fps.'
 return out
