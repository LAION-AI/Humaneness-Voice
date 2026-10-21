"""Fixed typed-question formulation ablation for a development semantic rule."""
import json,time
from pathlib import Path
from laya_probe import MODEL,REVISION
import torch,laya

def main():
    torch.set_num_threads(4);agent=laya.load(MODEL,revision=REVISION,device='cpu');rows=[]
    forms=[('Which description matches the user?', 'The user is talking about bananas.', 'The user is not talking about bananas.'),
        ('Does the user talk about bananas?', 'Yes, bananas are mentioned.', 'No, bananas are not mentioned.'),
        ('Classify the topic of the user utterance.', 'Bananas', 'Not bananas'),
        ('The rule is: interrupt when the user talks about bananas. Does it apply now?', 'The rule applies.', 'The rule does not apply.')]
    for i,(instructions,a,b) in enumerate(forms):
        questions={'forward':{'type':'choice','instructions':instructions,'criteria':{'A':a,'B':b}},
            'reverse':{'type':'choice','instructions':instructions,'criteria':{'A':b,'B':a}}}
        for text,label in [('Bananen sind gesund und ich esse sie jeden Tag.',True),('Äpfel sind gesund und knackig.',False)]:
            result=agent.predict('USER: '+text,questions)['answers']
            p=result['forward']['probabilities']['A'];q=result['reverse']['probabilities']['B']
            row={'form':i,'positive':label,'p':p,'q':q,'mean':(p+q)/2};rows.append(row);print(json.dumps(row),flush=True)
    (Path(__file__).parent/'logs/laya-rule-formulations.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
