"""Pinned local Laya pilot: synthetic text only, isolated CPU process."""
import json,os,sys,time
from pathlib import Path
os.environ.setdefault('USE_TF','0')
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor/laya'))
import torch,laya

MODEL='convaiinnovations/laya-multilingual'
REVISION='e4e9ddf21a7b1903b7acffd8814ad4307bf63a67'
QUESTION={'stop':{'type':'choice','instructions':'Should the agent stop speaking and listen to this user?',
    'criteria':{'A':'The user requests stop, pause, silence, or to let them speak; ignore preceding agent echo.',
                'B':'The user wants continuation, asks about/quotes the word stop, gives a backchannel, or does not ask to interrupt.'}}}

def main():
    torch.set_num_threads(4);start=time.perf_counter()
    agent=laya.load(MODEL,revision=REVISION,device='cpu')
    report={'model':MODEL,'revision':REVISION,'source_commit':'ffc142012af63d015fa19f05f29f9c5ff329a2cc',
        'load_seconds':time.perf_counter()-start,'device':'cpu','threads':4,'synthetic_text_only':True,'cases':[]}
    for text,expected in [('Stopp!',True),('Bitte sei mal leise.',True),('Abends stop',True),
                          ('Was bedeutet das Wort Stopp?',False),('Bitte erzähl weiter.',False),('Mhm, ja.',False)]:
        state={'user_speech':text,'agent_speaking':True,'agent_reply':'Eines Abends fand ein Roboter einen Stern im Garten.'}
        start=time.perf_counter();result=agent.predict(state,QUESTION)
        answer=result['answers']['stop'];row={'text':text,'expected_stop':expected,'answer':answer,
            'ms':round((time.perf_counter()-start)*1000,1)}
        row['passed']=(answer['choice']=='A')==expected;report['cases'].append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    report['passed']=all(r['passed'] for r in report['cases'])
    (Path(__file__).parent/'logs/laya-pilot.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)

if __name__=='__main__':main()
