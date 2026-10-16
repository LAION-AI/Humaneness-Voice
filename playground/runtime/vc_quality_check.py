"""Small numerical-optimization proxy check, not a VC emotion study."""
import json
from types import SimpleNamespace
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.nn import functional as F
import settings
from speaker_reward import SpeakerModels
from ears_reward import EarsModels
from dynamic_references import fixed_clip


def main():
    root=settings.HERE/'vc-performance'
    device='cuda:0';torch.set_num_threads(4)
    speaker=SpeakerModels(SimpleNamespace(retriever=SimpleNamespace(device=device)))
    target,_=speaker.target(fixed_clip(settings.DEFAULT_VOICE)[0],'ecapa')
    ears=EarsModels(device);ears.prepare()
    result={}
    for method in ('legacy','fp32','tf32','compiled','bf16'):
        files=[root/f'hardware-{method}-n8-{i}.wav' for i in range(8)]
        if not all(p.is_file() for p in files):continue
        with torch.inference_mode(),torch.cuda.device(device):
            audio=[torchaudio.functional.resample(torch.as_tensor(sf.read(p,dtype='float32')[0],device=device),48000,16000) for p in files]
            embeddings,_=speaker.embed(audio,'ecapa')
            similarities=(F.normalize(embeddings.float(),dim=-1)@target).cpu().tolist()
            scores=ears.predict(audio,'medium','ce')
        rows=[{'ecapa':similarities[i],**scores[i]['scores_raw']} for i in range(8)]
        result[method]={'rows':rows,'means':{k:float(np.mean([r[k] for r in rows])) for k in rows[0]}}
        print(method,result[method]['means'],flush=True)
    (root/'quality.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
