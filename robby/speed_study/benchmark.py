"""Reproducible synthetic director comparison using the unmodified live prompt.

No private transcripts, microphone recordings or credentials are stored.
Each run keeps every attempt, including failures; local tools are validated but
not applied to a real session. Run from the app directory with its Python.
"""
import argparse, asyncio, json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from conversation_director import CompactDirector

IDENTITY = dict(enrolled=True, named=True, speaker_id='synthetic-speed-test', display_name='Selma')
STATE = dict(enabled=True, rules=[])
CASES = [
 ('arithmetic', 'Was ist 17 mal 4? Sag nur die Zahl.', [], 'number'),
 ('verbatim', 'Sage genau: Die rote Tür bleibt offen.', [], 'verbatim'),
 ('story', 'Erzähle eine Geschichte über einen Roboter und einen Schlüssel in genau zwei Sätzen. Keine Rückfrage.', [], 'story'),
 ('identity', 'Christoph meint, Robert sei dran. Aber wie heiße ich?', [], 'identity'),
 ('no_timer', 'Stelle bitte einen Timer auf fünf Minuten.', [], 'timer'),
 ('followup', 'Kürzer bitte.', [dict(role='user',content='Wie mache ich Tee?'),dict(role='assistant',content='Koche Wasser auf, gib den Teebeutel in eine Tasse, gieße das Wasser auf und lasse den Tee drei Minuten ziehen.')], 'short'),
 ('english_emotion', 'In English, sound delighted and tell me exactly: We finally did it!', [], 'english'),
 ('numeric_content', 'Bei wie viel Grad Celsius gefriert reines Wasser bei Normaldruck? Ein kurzer Satz.', [], 'freeze'),
 ('bananas_tool', 'Unterbrich mich bitte, wenn ich über Bananen rede, und sag dann Stopp, stopp!', [], 'install'),
 ('cancel_tool', 'Beende bitte das Spiel und lösche die Unterbrechungsregeln.', [], 'cancel'),
 ('no_accidental_tool', 'Bananen sind lecker. Nenne ein anderes gelbes Obst.', [], 'no_tool'),
 ('no_name', 'Sag bitte nicht ständig meinen Namen. Was ist 8 plus 9?', [], 'no_name'),
 ('history', 'Welche Farbe hatte die Tür, von der ich am Anfang gesprochen habe? Nur die Farbe.',
  [dict(role='user',content='Die Tür war violett. Merke dir das für dieses Gespräch.'),dict(role='assistant',content='Ja, die Tür war violett.')]+[dict(role=r,content=t) for _ in range(6) for r,t in [('user','Erzähl kurz etwas über Äpfel und Birnen.'),('assistant','Äpfel und Birnen wachsen an Bäumen. Es gibt viele verschiedene Sorten.')]], 'history'),
 ('bye', 'Bye bye!', [], 'bye'),
]

def quality(kind, result):
    text=result['text']; lower=text.casefold(); actions=result.get('interaction_actions',[])
    # These are explicit, inspectable checks, not a general quality score.
    checks={'valid_schema': True, 'acting_cue': bool(re.match(r'\([^)]*\)',result['script'])),
            'bounded_reply':len(text.split())<=120}
    expected={
      'number':bool(re.fullmatch(r'(68|achtundsechzig)[.!]?',text,re.I)),
      'verbatim':text=='Die rote Tür bleibt offen.',
      'story':len(re.findall(r'[.!?](?:\s|$)',text))==2 and 'schlüssel' in lower and '?' not in text,
      'identity':'selma' in lower and not any(n in lower for n in ('robert','christoph')),
      'timer':any(x in lower for x in ('kann','kein','nicht')) and not any(x in lower for x in ('timer läuft','timer ist gestellt','timer gestellt')),
      'short':len(text.split())<24 and '?' not in text,
      'english':text=='We finally did it!' and result['language']=='en' and any(x in (result['general']+result['script']).lower() for x in ('delight','joy','excit','happi','happy')),
      'freeze':bool(re.search(r'\b(?:0|null|zero)\b',lower)),
      'install':any(a['name']=='set_interruption_rule' and any('banan' in t.casefold() for t in a['arguments']['trigger_terms']) for a in actions),
      'cancel':any(a['name']=='clear_interruption_rules' for a in actions),
      'no_tool':not actions,
      'no_name':'selma' not in lower and bool(re.search(r'17|siebzehn',lower)),
      'history':bool(re.fullmatch(r'violett[.!]?',lower)),
      'bye':bool(re.search(r'bye|tschüss|wiedersehen|bis bald|take care|see you|mach.s gut',lower)),
    }
    checks['task']=expected[kind]
    if kind not in ('install','cancel'):checks['no_unrequested_action']=not actions
    return checks

class CandidateClient:
    def __init__(self, client, model, base=None, tuned=False):
        self.client=client; self.model=model; self.base=base; self.calls=[];self.tuned=tuned
    def stream(self,*args,**kwargs):return self.client.stream(*args,**kwargs)
    async def post(self,url,**kwargs):
        kwargs['json']={**kwargs['json'],'model':self.model}
        if self.tuned:
            kwargs['json']['reasoning_effort']='instant' if self.model.startswith('mercury') else 'none'
            kwargs['json']['response_format']={'type':'json_schema','json_schema':{'name':'robby_reply','strict':True,'schema':{
                'type':'object','properties':{'general':{'type':'string'},'script':{'type':'string'},'language':{'type':'string','enum':['de','en']}},
                'required':['general','script','language'],'additionalProperties':False}}}
            if self.model.startswith('mercury'):kwargs['json']['max_completion_tokens']=2048
        if self.base:
            url=self.base.rstrip('/')+'/v1/chat/completions'
            kwargs['headers']={}  # Never send the provider credential to a local server.
        start=time.perf_counter()
        response=await self.client.post(url,**kwargs)
        row=dict(ms=round(1000*(time.perf_counter()-start),2),status=response.status_code)
        try:
            packet=response.json();row['usage']=packet.get('usage');row['timings']=packet.get('timings')
            row['finish_reason']=packet.get('choices',[{}])[0].get('finish_reason')
            row['returned_model']=packet.get('model')
            if response.is_error:row['error']=str(packet.get('error',{}))[:800]
        except Exception:pass
        self.calls.append(row)
        return response

async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',default=['gpt-5.6-luna','mercury-2.5','deepseek-v4-flash'])
    parser.add_argument('--base');parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--tuned',action='store_true')
    parser.add_argument('--output',default='logs/director-speed-api.json');args=parser.parse_args()
    target=Path(args.output)
    report=dict(started=time.time(),scope='Synthetic task checks; same live system prompt, history and real validated tool schemas. Serial rotated model order. Includes all failures. Not a general intelligence benchmark.',repetitions=args.repetitions,models=args.models,local_base=args.base,tuned=args.tuned,rows=[])
    target.parent.mkdir(parents=True,exist_ok=True)
    async with httpx.AsyncClient(timeout=30) as client:
        for rep in range(args.repetitions):
            for ci,(label,prompt,history,kind) in enumerate(CASES):
                offset=(ci+rep)%len(args.models);models=args.models[offset:]+args.models[:offset]
                for model in models:
                    candidate=CandidateClient(client,model,args.base,args.tuned);start=time.perf_counter()
                    row=dict(model=model,repetition=rep,case=label,request=prompt)
                    try:
                        state=STATE if kind!='cancel' else dict(enabled=True,rules=[dict(condition='bananas',reply='Stopp, stopp!')])
                        result=await CompactDirector().generate(candidate,prompt,IDENTITY,history,state)
                        row.update(result=result,checks=quality(kind,result))
                        row['passed']=all(row['checks'].values())
                    except Exception as exc:
                        row.update(passed=False,error=type(exc).__name__+': '+str(exc)[:600])
                    row.update(total_ms=round(1000*(time.perf_counter()-start),2),calls=candidate.calls)
                    report['rows'].append(row);target.write_text(json.dumps(report,indent=2,ensure_ascii=False))
                    print(json.dumps({k:row[k] for k in ('model','repetition','case','total_ms','passed')}),flush=True)
    report['finished']=time.time();target.write_text(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':asyncio.run(main())
