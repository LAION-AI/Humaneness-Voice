"""Native one-copy GPU canary for production speaker steering.

Run only after coordinator grants the chosen GPU. No scorer models, HTTP calls
or live-process changes. Offline HF mode is enabled before imports; all model
assets must already be cached. Audio/codes and a JSON ledger are retained.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import soundfile as sf
import torch

import settings
from activation_steering import research_stream_frames
from prompting import build
from speaker_steering import resolve_speaker_steering


def weights_digest(model):
    """Hash actual parameter bytes, excluding mutable nonpersistent buffers."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        raw = parameter.detach().contiguous().cpu().view(torch.uint8).numpy()
        digest.update(memoryview(raw.reshape(-1)))
    return digest.hexdigest()


def codes_digest(codes):
    return hashlib.sha256(codes.contiguous().numpy().tobytes()).hexdigest()


def exact(left, right):
    return torch.equal(left, right)


def fixed_reference(voice):
    rows = json.loads((settings.HERE / "fixed-references/selection.json").read_text())["candidates"]
    return settings.HERE / next(row["wav"] for row in rows if row["voice"] == voice)


@torch.inference_mode()
def prefix(engine, specs, reference_path, temperature, *, graph=False, research_vector=None):
    with torch.cuda.device(engine.device), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = engine.reference(reference_path)
        branches = [engine.pack_many(specs, reference), engine.pack_many(specs, reference, neutral=True)]
        params = {"max_new_tokens":32,"audio_temperature":temperature,"audio_top_p":.95,
                  "audio_top_k":50,"audio_repetition_penalty":1.,"text_temperature":temperature,
                  "text_top_p":1.,"text_top_k":50}
        torch.manual_seed(72177); torch.cuda.manual_seed_all(72177)
        tick = time.monotonic()
        if research_vector is None:
            iterator = engine.fast.stream_frames(branches, params, 33, guidance=2., graph=graph)
        else:
            iterator = research_stream_frames(engine, branches, params, 33, guidance=2.,
                layer_vectors={16:research_vector}, factors=.75, collect_layers=(), graph=False)
        frames = []
        for frames in iterator: pass
        if not frames: raise AssertionError("Canary emitted no audio frames")
        torch.cuda.synchronize(engine.device)
        return torch.stack(frames, dim=1).cpu(), time.monotonic()-tick


def save_codes(folder, name, codes):
    np.savez_compressed(folder / (name + ".npz"), codes=codes.numpy())
    return {"name":name,"sha256":codes_digest(codes),"shape":list(codes.shape),
            "frames_per_row":codes[:, :, 0].ne(-1).sum(-1).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=settings.HERE / "test-results/speaker-steering-gpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"started_utc":datetime.now(timezone.utc).isoformat(),"device":args.device,
              "scope":"One native TTS/codec copy; no scorer loading, network API, live mutation or speaker-quality claim.",
              "checks":[],"artifacts":[],"passed":False}
    def checkpoint():
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    checkpoint()
    settings.DEVICE=args.device; settings.COMPILE_GLOBAL=False; settings.FAST_INFERENCE=True
    torch.set_num_threads(4); torch.cuda.set_device(args.device)
    from engine import SmallEngine
    from dpo_adapters import AdapterSelection
    engine = SmallEngine(); engine.configure_adapters(AdapterSelection())
    report["checkpoint"] = engine.checkpoint_info
    before = weights_digest(engine.model); report["parameter_sha256_before"] = before
    specifications = [build({'text':'We can talk about this tomorrow.','caption':'Calm natural conversation.'},
                             'caption','en',reference_identity=True),
                      build({'text':'Wir können morgen darüber sprechen.','caption':'Contained anger.'},
                             'caption','de',reference_identity=True)]
    voice_ids = {'mira':'emolia_c1699','selma':'emolia_c0542','idris':'anime_088'}
    references = {name:fixed_reference(voice) for name, voice in voice_ids.items()}
    directions = {}
    for name, voice in voice_ids.items():
        vector, info = resolve_speaker_steering(engine, voice, references[name])
        if vector is None or not info['active']:
            raise AssertionError(f"Canary requires valid {name} bank entry: {info}")
        directions[name] = (vector, info)
    report['banks'] = {name:info for name, (_,info) in directions.items()}
    try:
        # Remove/reinstall only while the transformer is eager. This compares
        # the true original model arithmetic, not just two inactive-buffer runs.
        engine.configure_speaker_steering(None, {'active':False})
        hook_state = engine.speaker_steering
        removed = []
        for temperature in (0., 1.):
            for handle in hook_state.handles: handle.remove()
            hook_state.handles.clear()
            try:
                plain, _ = prefix(engine, specifications, references['mira'], temperature)
            finally:
                hook_state.install(engine.model.transformer)
            dormant, _ = prefix(engine, specifications, references['mira'], temperature)
            assert exact(plain, dormant), 'Eager dormant hook changed acoustic tokens'
            engine.configure_speaker_steering(*directions['mira'])
            production, _ = prefix(engine, specifications, references['mira'], temperature)
            engine.configure_speaker_steering(None, {'active':False})
            research, _ = prefix(engine, specifications, references['mira'], temperature,
                                 research_vector=directions['mira'][0])
            assert exact(production, research), 'Production addition differs from research layer16 raw .75'
            report['checks'].append({'temperature':temperature,'eager_dormant_vs_removed_code_exact':True,
                                     'eager_production_vs_research_code_exact':True})
            for label, codes in (('plain',plain),('dormant',dormant),('production_mira',production),('research_mira',research)):
                report['artifacts'].append(save_codes(args.output, f'eager_t{temperature:g}_{label}', codes))
            checkpoint(); print(json.dumps(report['checks'][-1]), flush=True)

        # Production-style compile keeps the already-installed hook identities.
        engine.model.transformer.forward = torch.compile(engine.model.transformer.forward,
                                                          dynamic=True, mode='default')
        for temperature in (0., 1.):
            engine.configure_speaker_steering(None, {'active':False})
            off, seconds = prefix(engine, specifications, references['mira'], temperature)
            zero_report = {**directions['mira'][1], 'alpha':0., 'active':False}
            engine.configure_speaker_steering(directions['mira'][0], zero_report)
            zero, _ = prefix(engine, specifications, references['mira'], temperature)
            assert exact(off, zero), 'Compiled alpha0 is not an exact no-op'
            outputs = []
            for name in ('mira','selma','mira'):
                engine.configure_speaker_steering(*directions[name])
                codes, timing = prefix(engine, specifications, references['mira'], temperature)
                # Same reference for this switching-only test is intentional;
                # normal API resolution still requires exact matching ref SHA.
                outputs.append(codes)
                report['artifacts'].append(save_codes(args.output, f'compiled_t{temperature:g}_{name}_{len(outputs)}',codes))
            assert exact(outputs[0],outputs[2]), 'Voice switches accumulated or retained stale vector state'
            engine.configure_speaker_steering(None, {'active':False})
            off_again, _ = prefix(engine, specifications, references['mira'], temperature)
            assert exact(off,off_again), 'Compiled request cleanup changed the next disabled request'
            graph_off, _ = prefix(engine, specifications, references['mira'], temperature, graph=True)
            report['checks'].append({'temperature':temperature,'compiled_off_vs_alpha0_code_exact':True,
                'compiled_mira_selma_mira_no_accumulation':True,'compiled_off_after_cleanup_code_exact':True,
                'compiled_eager_local_vs_local_graph_code_exact':exact(off,graph_off),
                'compiled_off_first_call_seconds':seconds,
                'scope_note':'Eager-vs-compiled numerical equality is not required; compiled fusion may change stochastic trajectories.'})
            checkpoint(); print(json.dumps(report['checks'][-1]),flush=True)

        engine.configure_speaker_steering(*directions['mira'])
        chunks=[]
        waves=engine.generate(specifications[0], n=1, guidance=2., seed=7911, reference_path=references['mira'],
                              temperature=1., streaming=True, chunk_frames=20, max_frames=60,
                              on_audio=lambda chunk: chunks.append(np.asarray(chunk).copy()) or False)
        assert chunks and all(np.isfinite(chunk).all() and len(chunk)>0 for chunk in chunks)
        assert len(waves)==1 and np.isfinite(waves[0]).all()
        sf.write(args.output/'live-mira.wav', waves[0], 48000, subtype='FLOAT')
        report['checks'].append({'live_mira_1600ms_chunks':len(chunks),'chunk_samples':[len(c) for c in chunks],
                                 'audio_finite':True,'performance':engine.last_performance})
        checkpoint()
        from beam_search import ChunkSearch
        class WiringScorer:
            device='none (wiring-only CPU proxy)'
            last_timing={'total_seconds':0.,'asr_seconds':0.,'quality_seconds':0.}
            def score(self, waves, text, final):
                return [{'reward':float(len(wave)),'prefix_wer':0.} for wave in waves]
        commits=[]
        beam_wave, performance = ChunkSearch(engine,WiringScorer()).generate(specifications[0], n=4,beam_width=2,
            chunk_frames=20,max_seconds=3.2,guidance=2.,temperature=1.,seed=8123,
            reference_path=references['mira'],on_commit=lambda chunk,entry: commits.append(np.asarray(chunk).copy()) or False)
        assert len(beam_wave)>0 and np.isfinite(beam_wave).all()
        sf.write(args.output/'beam-mira.wav', beam_wave,48000,subtype='FLOAT')
        report['checks'].append({'beam_n4_top2_audio_finite':True,'committed_chunks':len(commits),
                                 'performance':performance,'quality_assessment':False})
        engine.configure_speaker_steering(None, {'active':False,'reason':'canary_complete'})
        after=weights_digest(engine.model)
        assert before==after,'Speaker steering changed model parameter bytes'
        report.update(parameter_sha256_after=after,parameters_unchanged=True,passed=True,
                      completed_utc=datetime.now(timezone.utc).isoformat())
        checkpoint(); print(json.dumps({'passed':True,'checks':len(report['checks']),'output':str(args.output)}),flush=True)
    finally:
        engine.configure_speaker_steering(None,{'active':False,'reason':'canary_finally'})
        checkpoint()


if __name__=='__main__': main()
