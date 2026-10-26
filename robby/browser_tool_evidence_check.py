"""Separate tool/answer control: typed requests, real lookups and actual TTS.

This does not substitute for noisy spoken tool tests. Weather numeric evidence
is checked against the actual tool response; news/Wikipedia facts still require
human source review. Model text alone cannot invent a passing source receipt.
"""
import asyncio, json, os, re, time
from pathlib import Path
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parent


def numeric_mentions(text):
    """German integer/decimal mentions, including negative temperatures."""
    small={'null':0,'ein':1,'eins':1,'eine':1,'einen':1,'zwei':2,'drei':3,'vier':4,'fünf':5,'sechs':6,
        'sieben':7,'acht':8,'neun':9,'zehn':10,'elf':11,'zwölf':12,'dreizehn':13,'vierzehn':14,
        'fünfzehn':15,'sechzehn':16,'siebzehn':17,'achtzehn':18,'neunzehn':19,
        'zwanzig':20,'dreißig':30,'vierzig':40,'fünfzig':50,'sechzig':60,'siebzig':70,'achtzig':80,'neunzig':90}
    for unit in ('ein','zwei','drei','vier','fünf','sechs','sieben','acht','neun'):
        for tens in ('zwanzig','dreißig','vierzig','fünfzig','sechzig','siebzig','achtzig','neunzig'):
            small[unit+'und'+tens]=small[unit]+small[tens]
    small={word.casefold():value for word,value in small.items()}
    tokens=re.findall(r'-?\d+(?:[.,]\d+)?|[a-zäöüß]+',text.casefold())
    values=[];i=0
    while i<len(tokens):
        token=tokens[i];sign=-1 if i and tokens[i-1]=='minus' else 1
        value=float(token.replace(',','.')) if re.fullmatch(r'-?\d+(?:[.,]\d+)?',token) else small.get(token)
        if value is not None:
            if i+2<len(tokens) and tokens[i+1]=='komma' and tokens[i+2] in small:
                value+=small[tokens[i+2]]/10;i+=2
            values.append(sign*value)
        i+=1
    return values


async def main():
    from playwright.async_api import async_playwright
    url=os.environ.get('ROBBY_TEST_URL','http://127.0.0.1:8793').rstrip('/')
    report={'passed':False,'started':time.time(),'scope':'Typed browser control, actual LLM/tool/TTS; no microphone/identity claim. Weather numbers checked against received tool facts; textual factual accuracy still requires review.','cases':[]}
    tasks=[('wikipedia','Schau bei Wikipedia nach Pinguinen und erzähl mir zwei Fakten.','search_wikipedia','wikipedia.org',20),
           ('weather','Wie wird das Wetter morgen in Berlin? Schau nach und nenne die niedrigste und höchste Temperatur in Grad Celsius.','get_weather','open-meteo.com',15),
           ('news','Schau bei der Tagesschau nach und erzähl mir zwei aktuelle Nachrichten.','search_news','tagesschau.de',25)]
    errors=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=os.environ.get('ROBBY_TEST_CHROME','/usr/bin/google-chrome'),headless=True,
            args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        await page.add_init_script('localStorage.setItem("robby.cameraAuto","false");localStorage.setItem("robby.avatar-idle-enabled","false");localStorage.setItem("robby.avatar-pet-invites-enabled","false");')
        try:
            await page.goto(url+'/desk-robot/')
            await page.wait_for_function('window.robo&&robotFace.llmControls?.catalog&&!document.querySelector("#call").disabled',timeout=60000)
            await page.evaluate('''()=>{window.toolEvents=[];window.toolPCM=0;const old=robo.on.message,play=robo.play.bind(robo);
                robo.on.message=m=>{toolEvents.push(m);old(m)};robo.play=x=>{toolPCM++;play(x)}}''')
            await page.evaluate('robo.start(false)');await page.wait_for_function('toolEvents.some(e=>e.type==="ready")',timeout=20000)
            for name,prompt,tool,host,min_words in tasks:
                row={'id':name,'prompt':prompt,'passed':False}
                mark=await page.evaluate('({n:toolEvents.length,pcm:toolPCM,at:performance.now()})')
                try:
                    await page.fill('#message',prompt);await page.locator('#form').evaluate('f=>f.requestSubmit()')
                    await page.wait_for_function('(n)=>toolEvents.slice(n).some(e=>e.type==="turn_done")&&!robo.sources.length',arg=mark['n'],timeout=120000)
                    events=await page.evaluate('(n)=>toolEvents.slice(n)',mark['n'])
                    replies=[e for e in events if e['type']=='reply'];reply=replies[-1] if replies else {}
                    text=reply.get('text','');sources=reply.get('sources',[]);lookups=reply.get('lookups',[])
                    actual=[x for x in lookups if x['tool']==tool and not x.get('error')]
                    checks={'real_lookup_receipt':bool(actual),'result_hash':bool(actual) and all(len(x.get('result_sha256',''))==64 for x in actual),
                        'source':any((urlparse(s.get('url','')).hostname or '').removeprefix('www.').removeprefix('de.').removeprefix('en.')==host for s in sources),
                        'substantial_answer':len(text.split())>=min_words,'real_audio':await page.evaluate('toolPCM')>mark['pcm'],
                        'no_service_error':not any(e['type']=='error' for e in events)}
                    if name=='weather':
                        facts=actual[-1].get('weather_facts',{}) if actual else {};daily=facts.get('daily',{})
                        expected=[daily.get(k,[None,None])[1] if len(daily.get(k,[]))>1 else None for k in ('temperature_2m_min','temperature_2m_max')]
                        numbers=numeric_mentions(text)
                        checks['tomorrow_temperature_numbers']=all(v is not None and any(abs(v-x)<=.6 for x in numbers) for v in expected)
                        row.update(expected_tomorrow_temperatures_c=expected,numeric_mentions=numbers,weather_facts=facts,
                            numeric_check_limit='Numbers must occur within 0.6 C rounding tolerance; this does not verify every weather assertion or the grammatical min/max association.')
                    if name=='wikipedia':
                        checks['relevant_source_title']=any('pinguin' in s.get('title','').casefold() for s in sources)
                        row['factual_check_limit']='A relevant article title is required; facts, humor and causal explanations still need source review.'
                    row.update(checks=checks,text=text,sources=sources,lookups=lookups,events=events,passed=all(checks.values()))
                except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)[:350]
                report['cases'].append(row)
                (ROOT/'static/adversarial-tool-control.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
                print(json.dumps({k:row.get(k) for k in ('id','passed','checks','text','error')},ensure_ascii=False),flush=True)
            report['passed']=all(r['passed'] for r in report['cases']) and not errors
        finally:
            try:await page.evaluate('window.robo?.stop()')
            except Exception:pass
            await browser.close()
            report.update(finished=time.time(),page_errors=errors)
            (ROOT/'static/adversarial-tool-control.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':asyncio.run(main())
