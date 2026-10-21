"""Small development/held-out prompt check; not population calibration."""
import json,time
from pathlib import Path
from laya_probe import MODEL,REVISION,QUESTION
import laya,torch

CASES=[('Stopp!',True),('Bitte sei mal leise.',True),('Abends stop',True),
       ('Was bedeutet das Wort Stopp?',False),('Bitte erzähl weiter.',False),('Mhm, ja.',False),
       ('Halt, bitte hör auf.',True),('Hör bitte auf zu sprechen.',True),('Lass mich bitte kurz zu Wort kommen.',True),
       ('Could you please stop for a second?',True),('Stop, I need to ask something.',True),
       ('Ja, genau, ich höre zu.',False),('Tell me more.',False),('Das Wort Stopp kam in deiner Geschichte vor.',False),
       ('Bitte hör nicht auf, das ist spannend.',False),('Do not stop speaking.',False)]

def main():
    torch.set_num_threads(4);agent=laya.load(MODEL,revision=REVISION,device='cpu')
    templates=[QUESTION,
        {'stop':{'type':'choice','instructions':'Classify only the USER utterance. Does the user ask the assistant to stop or pause talking?',
            'criteria':{'A':'The user asks the assistant to stop or pause talking.',
                        'B':'The user does not ask the assistant to stop or pause talking.'}}},
        {'stop':{'type':'choice','instructions':'Which intention is expressed by the USER words?',
            'criteria':{'A':'Stop talking, pause, be quiet, or let me speak.',
                        'B':'Keep talking, agreement, acknowledgment, or merely discussing the word stop.'}}}]
    report={'model':MODEL,'revision':REVISION,'device':'cpu','synthetic_only':True,
        'development_indices':list(range(6)),'evaluation_indices':list(range(6,len(CASES))),
        'threshold_from_initial_pilot':.85,'templates':[]}
    for i,question in enumerate(templates):
        rows=[]
        for text,label in CASES:
            state='AGENT: Eines Abends fand ein Roboter einen Stern im Garten.\nUSER: '+text
            start=time.perf_counter();answer=agent.predict(state,question)['answers']['stop']
            p=answer['probabilities']['A'];rows.append({'text':text,'stop':label,'p_stop':p,
                'passed':(p>=.85)==label,'ms':round((time.perf_counter()-start)*1000,1)})
        row={'template':i,'question':question,'rows':rows,'development_correct':sum(r['passed'] for r in rows[:6]),
            'evaluation_correct':sum(r['passed'] for r in rows[6:])}
        report['templates'].append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('question','rows')}),flush=True)
    (Path(__file__).parent/'logs/laya-calibration.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
