"""Real provider + public tools, no microphone, all attempts retained."""
import asyncio,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
from conversation_director import CompactDirector

async def main():
    cases=[('news','Was sind gerade die neuesten Nachrichten bei Tagesschau und BBC? Fasse zwei Themen kurz zusammen.','search_news',{}),
        ('weather','Wie wird das Wetter heute und morgen in Berlin? Bitte schau nach.','get_weather',{}),
        ('wiki','Suche bei Wikipedia nach Bananen. Was ist eine Banane botanisch?','search_wikipedia',{}),
        ('where','Wie ist das Wetter bei mir?','ask_location',{}),
        ('guest','Kannst du mir eine kleine Geschichte erzählen?','ask_name',dict(enrolled=True,named=False,display_name='Guest 1',speaker_id='synthetic',needs_name=True))]
    rows=[]
    async with httpx.AsyncClient() as client:
        for label,text,expected,identity in cases:
            start=time.perf_counter();row=dict(case=label,request=text)
            try:
                result=await CompactDirector().generate(client,text,identity,[],dict(enabled=True,rules=[]))
                if expected=='ask_location':passed=not result['sources'] and all(x.get('error') for x in result['lookups']) and any(w in result['text'].lower() for w in ('stadt','ort','city','location'))
                elif expected=='ask_name':passed='?' in result['text'] and any(w in result['text'].lower() for w in ('name','heiß','heiss','nennen')) and len(result['text'].split())>=10
                else:passed=any(x['tool']==expected and not x.get('error') for x in result['lookups']) and bool(result['sources'])
                row.update(result=result,passed=passed)
            except Exception as e:row.update(error=type(e).__name__+': '+str(e),passed=False)
            row['ms']=(time.perf_counter()-start)*1000;rows.append(row)
            Path('logs/knowledge-director-live.json').write_text(json.dumps(dict(rows=rows,passed=all(r['passed'] for r in rows)),indent=2,ensure_ascii=False))
            print(json.dumps(row,ensure_ascii=False),flush=True)

if __name__=='__main__':asyncio.run(main())
