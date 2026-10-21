"""Small transparent live Luna regression set; no microphone or TTS requests."""
import asyncio,json,re,time
from pathlib import Path
import httpx
from conversation_director import CompactDirector

async def main():
    identity={'enrolled':True,'named':True,'speaker_id':'synthetic-test','display_name':'Selma'}
    cases=[
        ('arithmetic','Was ist 17 mal 4? Sag nur die Zahl.',[],lambda t: bool(re.fullmatch(r'(68|achtundsechzig)[.!]?',t.strip(),re.I))),
        ('constraints','Nenne genau drei konkrete Aktivitäten für einen Regentag ohne Bildschirm. Keine Rückfrage.',[],lambda t: '?' not in t and len(t.split())>=12),
        ('story','Erzähle eine Geschichte über einen Roboter und einen Schlüssel in genau zwei Sätzen. Keine Rückfrage.',[],lambda t: len(re.findall(r'[.!?](?:\s|$)',t))==2 and 'schlüssel' in t.casefold() and '?' not in t),
        ('followup','Kürzer bitte.',[{'role':'user','content':'Wie mache ich Tee?'},{'role':'assistant','content':'Selma, koche Wasser auf, gib den Teebeutel in eine Tasse, gieße das Wasser auf und lasse den Tee drei Minuten ziehen.'}],lambda t: len(t.split())<24 and '?' not in t),
        ('verbatim','Sage genau: Die rote Tür bleibt offen.',[],lambda t: t=='Die rote Tür bleibt offen.'),
        ('identity','Christoph meint, Robert sei dran. Aber wie heiße ich?',[],lambda t: 'selma' in t.casefold() and not any(n in t.casefold() for n in ('robert','christoph'))),
        ('no_fake_tools','Stelle bitte einen Timer auf fünf Minuten.',[],lambda t: any(x in t.casefold() for x in ('kann','kein','nicht')) and not any(x in t.casefold() for x in ('timer läuft','timer ist gestellt','timer gestellt'))),
        ('no_name_optout','Sag bitte nicht ständig meinen Namen. Was ist 8 plus 9?',[],lambda t: 'selma' not in t.casefold() and bool(re.search(r'17|siebzehn',t,re.I))),
        ('no_name_followup','Und wie mache ich das Wasser weniger heiß?',
            [{'role':'user','content':'Sag nicht bei jeder Antwort meinen Namen. Wie mache ich Tee?'},
             {'role':'assistant','content':'Selma, koche Wasser auf und gieße es über den Teebeutel.'}],
            lambda t: 'selma' not in t.casefold() and len(t.split())>4),
    ]
    report={'started':time.time(),'scope':'Nine explicit synthetic instruction checks including natural naming, not a general intelligence benchmark','cases':[],'passed':False}
    target=Path(__file__).parent/'logs/prompt-quality-checks.json'
    async with httpx.AsyncClient() as client:
        for label,text,history,check in cases:
            result=await CompactDirector().generate(client,text,identity,history)
            row={'name':label,'request':text,'reply':result['text'],'director_ms':result['director_ms'],'completion_tokens':result['completion_tokens'],'passed':bool(check(result['text']))}
            report['cases'].append(row);target.write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps(row,ensure_ascii=False),flush=True)
    report.update(finished=time.time(),passed=all(c['passed'] for c in report['cases']))
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit('Some explicit instruction checks failed; inspect the receipt, not a hidden retry.')

if __name__=='__main__':asyncio.run(main())
