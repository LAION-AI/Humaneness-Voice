"""Post-search emotion validation with a different encoder/model, never ranked."""
import json
import sys

import soundfile as sf
import torch
import torchaudio
from huggingface_hub import snapshot_download

import settings
sys.path.insert(0,str(settings.PARENT))
import config
from score_engine import VoiceScorer


class EmotionOnly(VoiceScorer):
    def __init__(self):
        self.device='cuda:1'
        self._load_emotion()


def main():
    root=settings.HERE/'search'
    config.WHISPER_DIR=snapshot_download('mkrausio/EmoWhisper-AnS-Small-v0.1',
        revision='e613edc6dfef73525a11faa84f1f94f10e5a1744',
        allow_patterns=['*.json','*.safetensors','*.bin','*.txt'])
    config.SCORE_ATTRS=[]
    scorer=EmotionOnly()
    target={'anger_en':'Anger','fear_en':'Fear','affection_de':'Affection','burst_en':'Anger'}
    results=[]
    for path in sorted(root.glob('quality-*.json')):
        row=json.loads(path.read_text())
        if row.get('error'):continue
        wave,sr=sf.read(root/'raw'/(row['id']+'.wav'),dtype='float32')
        with torch.inference_mode():
            audio=torchaudio.functional.resample(torch.tensor(wave,device=scorer.device),sr,16000)
            embed=scorer._whisper_embed(audio.cpu().numpy())
            values={name:float(head(embed).squeeze()) for name,head in scorer.emo.items()}
        order=sorted(values,key=lambda name:-values[name])
        rank=order.index(target[row['case']])+1
        results.append({'id':row['id'],'target':target[row['case']],'rank':rank,
            'rank_percentile':(len(order)-rank)/max(1,len(order)-1),'target_score':values[target[row['case']]],
            'all_emotions':values})
        print(row['id'],rank,flush=True)
    data={'encoder':'mkrausio/EmoWhisper-AnS-Small-v0.1',
        'encoder_revision':'e613edc6dfef73525a11faa84f1f94f10e5a1744','activation':'relu',
        'head_count':len(scorer.emo),'count':len(results),'results':results,
        'note':'These emotion heads/Whisper encoder were not used to select candidates. They are still learned proxies, not independent human ratings.'}
    (root/'emotion-check.json').write_text(json.dumps(data,indent=2))


if __name__=='__main__':main()
