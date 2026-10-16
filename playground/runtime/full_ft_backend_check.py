"""Read-only Full-FT fast/legacy greedy prefix sanity; no production mutation."""
import json
import time

import torch
from types import SimpleNamespace
import torchaudio

import settings
settings.COMPILE_GLOBAL=False
from engine import SmallEngine
from full_ft_speaker_study import ROOT, design, write_json
from prompting import build, parse_literal


def main():
    torch.set_num_threads(4);torch.cuda.set_device(settings.DEVICE)
    engine=SmallEngine();engine.configure_adapters(__import__('dpo_adapters').AdapterSelection())
    from speaker_reward import SpeakerModels
    speakers=SpeakerModels(SimpleNamespace(retriever=SimpleNamespace(device=settings.SCORE_DEVICE)))
    speakers.prepare('ecapa')
    results=[]
    for voice in ('idris','selma'):
        row=next(r for r in design()['rows'] if r['voice']==voice and r['condition']=='baseline' and r['block']==0)
        spec=build(parse_literal(row['prompt'],'caption'),'caption','en',6.5,row['prompt'],reference_identity=True)
        reference=engine.reference(ROOT/'references'/f'{voice}-15-actual.wav')
        branches=[engine.pack(spec['prompt'],spec,reference,1),engine.pack(spec['unconditional'],spec,reference,1)]
        p={'max_new_tokens':40,'audio_temperature':0.,'audio_top_p':.95,'audio_top_k':50,
            'audio_repetition_penalty':1.,'text_temperature':0.,'text_top_p':1.,'text_top_k':50}
        outputs={}
        with torch.cuda.device(engine.device),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for backend in ('legacy','fast_eager','fast_graph'):
                torch.manual_seed(991);torch.cuda.manual_seed_all(991);tick=time.monotonic()
                iterator=engine._stream_frames_cfg(branches,p,chunk_frames=41,guidance=1.5) if backend=='legacy' else engine.fast.stream_frames(branches,p,chunk_frames=41,guidance=1.5,graph=backend=='fast_graph')
                frames=[]
                for frames in iterator:pass
                outputs[backend]=torch.stack(frames,dim=1).cpu()
                print(json.dumps({'voice':voice,'backend':backend,'frames':len(frames),'seconds':time.monotonic()-tick}),flush=True)
        similarities={}
        target,_=speakers.target(ROOT/'references'/f'{voice}-15-actual.wav','ecapa')
        with torch.cuda.device(engine.device),torch.inference_mode():
            for backend,codes in outputs.items():
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    wave=engine.proc.decode_audio_codes([codes[0]],return_stereo=False)[0].float().cpu().reshape(-1)
                audio=torchaudio.functional.resample(wave.to(settings.SCORE_DEVICE),48000,16000)
                vector,_=speakers.embed([audio],'ecapa')
                with torch.autocast('cuda',enabled=False):
                    similarities[backend]=float(torch.nn.functional.normalize(vector.float(),dim=-1)[0]@target.float())
        left=outputs['legacy'];right=outputs['fast_eager'];length=min(left.shape[1],right.shape[1])
        differences=(left[:,:length]!=right[:,:length]).nonzero()
        results.append({'voice':voice,'frames':{k:v.shape[1] for k,v in outputs.items()},
            'legacy_vs_fast_eager_exact':torch.equal(outputs['legacy'],outputs['fast_eager']),
            'eager_vs_graph_exact':torch.equal(outputs['fast_eager'],outputs['fast_graph']),
            'first_difference_row_frame_channel':differences[0].tolist() if len(differences) else None,
            'code_agreement':float((left[:,:length]==right[:,:length]).float().mean()),
            'prefix_speaker_cosine':similarities})
    report={'checkpoint':engine.checkpoint_info,'adapters':None,'temperature':0,'maximum_prefix_frames':40,
        'scope':'Two deterministic greedy prefix checks only; not stochastic-distribution or long-generation parity proof.',
        'results':results}
    write_json(ROOT/'backend-check.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
