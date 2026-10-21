"""Counterbalanced local decision pilot; retain earlier failed prompt receipts."""
import json,time
from pathlib import Path
from laya_probe import MODEL,REVISION,QUESTION
from laya_calibration import CASES
import laya,torch

def main():
    torch.set_num_threads(4);agent=laya.load(MODEL,revision=REVISION,device='cpu')
    first=QUESTION['stop'];second={**first,'criteria':{'A':first['criteria']['B'],'B':first['criteria']['A']}}
    rows=[]
    extra=[('Sei bitte still.',True),('Warte mal, ich möchte kurz etwas sagen.',True),('Please stop talking now.',True),
        ('Hör jetzt bitte auf.',True),('Nein, lass mich ausreden.',True),('Be quiet, please.',True),
        ('Ja, schön, mach weiter.',False),('Stop is the title of that song.',False),('Okay, spannend.',False),
        ('Don’t stop, I like this story.',False),('Bitte bleib nicht still, erzähl weiter.',False),('Ich höre nur zu.',False)]
    for index,(text,label) in enumerate(CASES+extra):
        state='AGENT: Eines Abends fand ein Roboter einen Stern im Garten.\nUSER: '+text
        start=time.perf_counter();answers=agent.predict(state,{'forward':first,'reverse':second})['answers']
        p=answers['forward']['probabilities']['A'];q=answers['reverse']['probabilities']['B']
        mean=(p+q)/2;row={'text':text,'stop':label,'p_forward':p,'p_reverse':q,'mean':mean,
            'ms':round((time.perf_counter()-start)*1000,1),'set':'new_evaluation' if index>=len(CASES) else 'development',
            'prediction':mean>=.60,'passed':(mean>=.60)==label}
        rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    report={'model':MODEL,'revision':REVISION,'device':'cpu','synthetic_only':True,'threshold':.60,'rows':rows}
    (Path(__file__).parent/'logs/laya-paired-pilot.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({name:sum(r['passed'] for r in rows if r['set']==name) for name in ('development','new_evaluation')}),flush=True)

if __name__=='__main__':main()
