"""Load the actual Small export and reuse the existing channel-wise CFG loop."""
import json
import sys
import threading
import time
from contextlib import nullcontext

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download

import settings

sys.path.append(str(settings.PARENT))
from tts_engine import TTSEngine


def zero_temperature_sampler(sampler):
    """The upstream sampler requires positive T; T=0 means genuine argmax."""
    def sample(*args, **kwargs):
        if float(kwargs.get("temperature", 1)) == 0:
            kwargs["do_sample"] = False
        return sampler(*args, **kwargs)
    return sample


class SmallEngine(TTSEngine):
    def __init__(self):
        started = time.monotonic()
        self.device = settings.DEVICE
        self.lock = threading.Lock()
        self.refs = {}
        self.reference_info = {}
        self.last_codes = []
        self.last_performance = None

        root = snapshot_download(settings.REPO, revision=settings.REVISION,
            allow_patterns=["code/*.py", "assets/**", f"checkpoints/{settings.STAGE}/*", "README.md", "LICENSE"])
        from pathlib import Path
        root = Path(root)
        sys.path.insert(0, str(root / "code"))
        import moss_small
        from large_talker import build_fresh
        from packing import ScorePacker
        moss_small.SFT3 = str(root / "assets/sft3")
        moss_small.QWEN = str(root / "assets/qwen3")
        moss_small.load_qwen_backbone = lambda model, log=print: None
        schema = json.loads((root / "assets/score_schema.json").read_text())
        print(f"[small] loading {settings.STAGE} ({settings.REVISION[:8]})", flush=True)
        checkpoint=root/'checkpoints'/settings.STAGE/'model_bf16.pt'
        from rank1_adapters import BASE_SHA256, sha256
        expected_hash=settings.FULL_FT_SHA256 if settings.STAGE==settings.FULL_FT_STAGE else BASE_SHA256 if settings.STAGE=='S3' else None
        if expected_hash and sha256(checkpoint)!=expected_hash:
            raise ValueError('Small checkpoint integrity check failed')
        self.checkpoint_info={'stage':settings.STAGE,'revision':settings.REVISION,'sha256':expected_hash,
            'step':26774 if settings.STAGE==settings.FULL_FT_STAGE else None}
        if settings.STAGE==settings.FULL_FT_STAGE:
            metadata=json.loads((checkpoint.parent/'metadata.json').read_text())
            if metadata['sha256']!=expected_hash or metadata['final_step']!=26774:
                raise ValueError('Full-FT release metadata mismatch')
        self.model, cfg = build_fresh(schema, log=lambda _: None)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.tie_weights()
        self.model = self.model.to(self.device, dtype=torch.bfloat16).eval()
        from dpo_adapters import DPOBank, AdapterSelection
        self.adapter_bank = DPOBank(self.model, checkpoint, expected_base_sha256=expected_hash) if settings.ADAPTERS_ENABLED else None
        if self.adapter_bank:
            self.adapter_bank.apply(AdapterSelection())
        self.model._sample_next_token = zero_temperature_sampler(self.model._sample_next_token)
        del state
        _, _, Processor = moss_small.export_classes()
        self.proc = Processor.from_pretrained(moss_small.SFT3,
            codec_path="OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
            codec_weight_dtype="bf16" if settings.FAST_INFERENCE else "fp32", codec_compute_dtype="bf16")
        self.proc.audio_tokenizer = self.proc.audio_tokenizer.to(self.device).eval()
        self.packer = ScorePacker(self.proc, cfg, schema)
        self.sr = int(self.proc.model_config.sampling_rate)
        self.parameters = sum(p.numel() for p in self.model.parameters())
        from performance_core import FastGenerator
        self.fast = FastGenerator(self)
        from speaker_steering import install_speaker_steering
        self.speaker_steering = install_speaker_steering(self.model)
        self.speaker_steering_report = {'enabled':False,'active':False,'reason':'not_configured'}
        if settings.FAST_INFERENCE and settings.COMPILE_GLOBAL:
            # The global cache grows between frames; dynamic shapes avoid a
            # separate compile for every context length. Local decode instead
            # uses fixed-size CUDA graphs.
            from torch import _dynamo
            _dynamo.config.cache_size_limit = 64
            self.model.transformer.forward = torch.compile(
                self.model.transformer.forward, dynamic=True, mode="default")
        print(f"[small] ready: {self.parameters:,} parameters, {self.sr} Hz, {time.monotonic()-started:.1f}s", flush=True)

    def configure_adapters(self, selection):
        with self.lock:
            if hasattr(self, 'speaker_steering'):
                self.speaker_steering.configure(None)
                self.speaker_steering_report = {'enabled':False,'active':False,'reason':'adapter_configuration_changed'}
            if self.adapter_bank is None:
                if selection.active():
                    raise ValueError('DPO LoRAs require verified S3 or Ref15 Full FT')
                return {'selection': selection.model_dump(), 'active': [], 'switch_ms': 0}
            return self.adapter_bank.apply(selection)

    def configure_speaker_steering(self, vector, report):
        """Serialized per-request state; caller must reset in request finally."""
        with self.lock:
            self.speaker_steering.configure(vector, alpha=report.get('alpha', .75))
            self.speaker_steering_report = dict(report)
            return dict(self.speaker_steering_report)

    @torch.inference_mode()
    def warm_search(self):
        """Compile continuation/cache-fork paths before advertising ready."""
        from beam_search import ChunkSearch
        from prompting import build
        class WarmScorer:
            device = "none (startup warm-up)"
            last_timing = {"asr_seconds":0., "quality_seconds":0., "total_seconds":0.}
            def score(self, waves, text, final):
                return [{"reward":float(len(w)), "prefix_wer":0.} for w in waves]
        spec = build({"text":"Ready when you are.","caption":"Warm conversation."},"caption","en",seconds=6.6)
        path = settings.HERE / "performance/benchmark-reference.wav"
        for n in [4,8,12,16]:
            ChunkSearch(self,WarmScorer()).generate(spec,n=n,beam_width=1,chunk_frames=10,
                max_seconds=6.6,seed=991,reference_path=path if path.is_file() else None)
        self.last_codes=[]
        self.last_performance=None

    @torch.inference_mode()
    def warm_live_stream(self):
        """Warm the actual single/codec-stream context before advertising ready.

        Compiling inside this context can otherwise delay the first default
        live chunk substantially, even after the offline/search warm-ups.
        """
        from prompting import build
        path=settings.HERE/'performance/benchmark-reference.wav'
        spec=build({'text':'Ready when you are.','caption':'Warm, natural conversation.'},
            'caption','en',seconds=2.,reference_identity=path.is_file())
        self.generate(spec,reference_path=path if path.is_file() else None,
            streaming=True,chunk_frames=settings.DEFAULT_CHUNK_FRAMES,max_frames=20,
            guidance=settings.DEFAULT_GUIDANCE,
            on_audio=lambda chunk:False)
        self.last_codes=[];self.last_performance=None

    @staticmethod
    def _sample_continue_or_end(mdl, lh, p, stop_bias=None):
        # The old 4.55B model's stop bias is not a measured Small default.
        return mdl._sample_next_assistant_text_token(local_hidden_states=lh,
            do_sample=p["text_temperature"] > 0, temperature=p["text_temperature"], top_k=p["text_top_k"], top_p=p["text_top_p"])

    @torch.inference_mode()
    def reference(self, path):
        key = str(path)
        if key not in self.refs:
            # The pinned processor's path reader calls torchaudio.load, which
            # requires TorchCodec on newer builds. Use SoundFile and the public
            # waveform encoder instead, keeping every sample up to the cap.
            with sf.SoundFile(path) as audio:
                sr=audio.samplerate
                source_seconds=len(audio)/sr
                wave=audio.read(frames=int(settings.REFERENCE_SECONDS*sr),dtype='float32',always_2d=True)
            if not len(wave) or not np.isfinite(wave).all():
                raise ValueError('Reference audio is empty or contains invalid samples')
            wav=torch.from_numpy(wave.mean(axis=1).copy()).unsqueeze(0)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_available()):
                codes = self.proc.encode_audios_from_wav(wav,sr,n_vq=12)[0].cpu()
            # 12.5 Hz codec; allow up to ceil(cap/80 ms) frames without
            # imposing the old 37-frame guard. Actual codec rounding is reported.
            import math
            if codes.ndim!=2 or codes.shape[1]!=12 or not 0 < len(codes) <= math.ceil(settings.REFERENCE_SECONDS*12.5):
                raise ValueError('Unexpected reference codec frames')
            self.refs[key] = codes
            self.reference_info[key]={'source_seconds':source_seconds,'used_seconds':len(wave)/sr,
                'codec_frames':len(codes),'codec_seconds':len(codes)/12.5,
                'max_seconds':settings.REFERENCE_SECONDS,'truncated':source_seconds>settings.REFERENCE_SECONDS}
        return self.refs[key]

    def forget_reference(self, path):
        self.refs.pop(str(path), None)
        self.reference_info.pop(str(path), None)

    def pack(self, prompt, spec, reference, n):
        record = {"prompt": prompt, "text": spec["text"], "frames": spec["frames"], "lang": spec["lang"]}
        row = self.packer.pack_mode(record, [], "reference" if reference is not None else "instruction", reference, generation=True)
        batch = self.packer.collate([row] * n)
        return batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)

    def pack_many(self, specs, reference, neutral=False):
        from performance_core import combine_branches
        # The upstream collator right-pads, inappropriate for different
        # generation prefixes. Collate one row then explicitly left-pad.
        rows = [self.pack(s["unconditional"] if neutral else s["prompt"], s, reference, 1) for s in specs]
        return combine_branches(rows, self.model.config.pad_token_id, self.model.config.audio_pad_token_id)

    @torch.inference_mode()
    def generate_many(self, specs, guidance=1.5, seed=777, reference_path=None,
                      temperature=1.0, progress=None, streaming=False, batch_size=None, chunk_frames=10,
                      max_frames=None, on_audio=None, first_chunk_frames=None):
        from performance_core import StreamingCodec
        if on_audio is not None and (not streaming or len(specs) != 1):
            raise ValueError('Live audio callback requires one streaming take')
        waves, self.last_codes, batches = [], [], []
        with self.lock, torch.cuda.device(self.device), torch.autocast("cuda", dtype=torch.bfloat16,enabled=not getattr(self,'policy_fp32',False)):
            reference = self.reference(reference_path) if reference_path else None
            batch_size = min(batch_size or getattr(self, 'batch_size', settings.BATCH_SIZE),
                             getattr(self, 'max_batch_size', settings.BATCH_SIZE))
            for offset in range(0, len(specs), batch_size):
                group = specs[offset:offset + batch_size]
                count = len(group)
                branches = [self.pack_many(group, reference)]
                if guidance != 1 and any(s["prompt"] != s["unconditional"] for s in group):
                    branches.append(self.pack_many(group, reference, neutral=True))
                torch.manual_seed(seed + offset)
                torch.cuda.manual_seed_all(seed + offset)
                p = {"max_new_tokens": min(settings.MAX_FRAMES + 60, max(s["frames"] for s in group) + 60),
                     "audio_temperature": temperature, "audio_top_p": .95, "audio_top_k": 50,
                     "audio_repetition_penalty": 1., "text_temperature": 0. if temperature == 0 else 1.,
                     "text_top_p": 1., "text_top_k": 50}
                if max_frames is not None:
                    p["max_new_tokens"] = min(p["max_new_tokens"], max_frames)
                if first_chunk_frames is not None:p['first_chunk_frames']=first_chunk_frames
                torch.cuda.synchronize(self.device)
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                capture_before = self.fast.capture_seconds
                wall_start = time.perf_counter()
                start.record()
                decoder = self.proc.audio_tokenizer
                context = StreamingCodec(decoder, count, self.model.config.audio_pad_token_id) if streaming else nullcontext()
                outputs, frames, previous = [], [], 0
                first = None
                spans, segment_start = [], start
                callback_seconds = 0.
                with context as incremental:
                    iterator = (self.fast.stream_frames(branches, p, chunk_frames=chunk_frames if streaming else p["max_new_tokens"]+1,
                                guidance=guidance, graph=True) if settings.FAST_INFERENCE else
                                self._stream_frames_cfg(branches, p, chunk_frames=chunk_frames if streaming else p["max_new_tokens"]+1,
                                guidance=guidance))
                    for frames in iterator:
                        if streaming:
                            outputs.append(incremental.decode(torch.stack(frames[previous:], dim=1)))
                            previous = len(frames)
                            if first is None:
                                first = torch.cuda.Event(enable_timing=True)
                                first.record()
                            if on_audio is not None:
                                pcm, lengths = outputs[-1]
                                boundary = torch.cuda.Event(enable_timing=True)
                                boundary.record()
                                boundary.synchronize()
                                spans.append(segment_start.elapsed_time(boundary) / 1000)
                                tick = time.perf_counter()
                                chunk = pcm[0, :int(lengths[0])].float().cpu().numpy().copy()
                                if not np.isfinite(chunk).all():
                                    raise RuntimeError('Non-finite audio values in streaming chunk')
                                stop = bool(on_audio(chunk)) if len(chunk) else False
                                callback_seconds += time.perf_counter() - tick
                                segment_start = torch.cuda.Event(enable_timing=True)
                                segment_start.record()
                                if stop:
                                    iterator.close()
                                    break
                    if not frames:
                        raise RuntimeError("The model generated no audio frames.")
                    codes = torch.stack(frames, dim=1)
                    valid = codes[:, :, 0].ne(self.model.config.audio_pad_token_id)
                    if not streaming:
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            result = decoder.decode(codes.masked_fill(~valid.unsqueeze(-1), 0).permute(2, 0, 1).contiguous(),
                                padding_mask=valid, num_quantizers=12, return_dict=True, chunk_duration=8)
                        outputs.append((result.audio.float().mean(1), result.audio_lengths))
                end.record()
                end.synchronize()
                seconds = sum(spans) + segment_start.elapsed_time(end) / 1000 if on_audio else start.elapsed_time(end) / 1000
                durations = valid.sum(-1).cpu().tolist()
                batches.append({"batch_size": count, "device_seconds": seconds,
                    "audio_seconds": [f / 12.5 for f in durations],
                    "model_wall_seconds": time.perf_counter() - wall_start,
                    "first_audio_seconds": start.elapsed_time(first) / 1000 if first else seconds,
                    "graph_capture_seconds": self.fast.capture_seconds - capture_before})
                batches[-1]['live_callback_seconds'] = callback_seconds
                for i, c in enumerate(codes):
                    c = c[valid[i]]
                    if len(c) < 2:
                        raise RuntimeError("A take contains fewer than two audio frames.")
                    self.last_codes.append(c.cpu())
                    pieces = [pcm[i, :int(lengths[i])].cpu() for pcm, lengths in outputs if int(lengths[i]) > 0]
                    wave = torch.cat(pieces).numpy().reshape(-1)
                    if not np.isfinite(wave).all():
                        raise RuntimeError("Non-finite audio values in take.")
                    waves.append(wave)
                if progress:
                    progress(len(waves), len(specs))
        total_device = sum(b["device_seconds"] for b in batches)
        total_audio = sum(sum(b["audio_seconds"]) for b in batches)
        self.last_performance = {"device_seconds": total_device, "raw_audio_seconds": total_audio,
            "aggregate_rtf": total_device / total_audio, "audio_seconds_per_second": total_audio / total_device,
            "batches": batches, "streaming": streaming, "chunk_frames": chunk_frames if streaming else None,
            "backend": "fused CFG + local CUDA graphs + compiled global" if settings.FAST_INFERENCE and getattr(self, 'compile_global', settings.COMPILE_GLOBAL)
                else "fused CFG + local CUDA graphs" if settings.FAST_INFERENCE else "legacy",
            "includes": "TTS prefill, autoregressive decode and PCM codec; GPU event span including launch gaps",
            "excludes": "reference encoding, prompt packing, Luna, alignment, scoring, voice conversion, CPU audio copy, MP3, network",
            "graph_capture_seconds": sum(b["graph_capture_seconds"] for b in batches)}
        return waves

    @torch.inference_mode()
    def warm_up(self):
        from prompting import build
        spec = build({"text": "Ready when you are.", "caption": "Warm, natural conversation."}, "caption", "en", seconds=2.)
        path = settings.HERE / "performance/benchmark-reference.wav"
        references = [None]
        with self.lock, torch.cuda.device(self.device), torch.autocast("cuda", dtype=torch.bfloat16,enabled=not getattr(self,'policy_fp32',False)):
            if path.is_file():
                references.append(self.reference(path))
            for n in getattr(self, 'warm_batch_sizes', sorted({1, settings.DEFAULT_N, settings.BATCH_SIZE})):
                for reference in references:
                    branches = [self.pack(spec["prompt"], spec, reference, n),
                                self.pack(spec["unconditional"], spec, reference, n)]
                    params = {"max_new_tokens":20,"audio_temperature":1.,"audio_top_p":.95,"audio_top_k":50,
                        "audio_repetition_penalty":1.,"text_temperature":1.,"text_top_p":1.,"text_top_k":50}
                    frames=[]
                    for frames in self.fast.stream_frames(branches,params,chunk_frames=21,guidance=settings.DEFAULT_GUIDANCE,min_frames=20,graph=True):
                        pass
                    codes=torch.stack(frames,dim=1).permute(2,0,1).contiguous()
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        self.proc.audio_tokenizer.decode(codes,return_dict=True,chunk_duration=8)
            torch.cuda.synchronize(self.device)
        self.last_codes = []
        self.last_performance = None

    @torch.inference_mode()
    def generate(self, spec, n=1, guidance=1.5, seed=777, reference_path=None, temperature=1.0, progress=None, streaming=False, chunk_frames=10,
                 max_frames=None, on_audio=None, first_chunk_frames=None):
        batch_size, retries = min(n, getattr(self, 'batch_size', settings.BATCH_SIZE)), []
        while True:
            try:
                waves = self.generate_many([spec] * n, guidance, seed, reference_path, temperature, progress,
                                           streaming=streaming, batch_size=batch_size, chunk_frames=chunk_frames,
                                           max_frames=max_frames, on_audio=on_audio,first_chunk_frames=first_chunk_frames)
                self.last_performance["oom_batch_retries"] = retries
                return waves
            except torch.cuda.OutOfMemoryError:
                # The lock has been released by generate_many. Discard this
                # request's partial candidates and retry a smaller batch;
                # never reuse a half-finished codec state.
                self.fast.clear_graphs()
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise
                retries.append(batch_size)
                batch_size = max(1, batch_size // 2)

    @torch.inference_mode()
    def generate_legacy(self, spec, n=1, guidance=1.5, seed=777, reference_path=None, temperature=1.0, progress=None):
        waves, self.last_codes = [], []
        with self.lock, torch.autocast("cuda", dtype=torch.bfloat16):
            reference = self.reference(reference_path) if reference_path else None
            for offset in range(0, n, 4):
                count = min(4, n - offset)
                branches = [self.pack(spec["prompt"], spec, reference, count)]
                if guidance != 1.0 and spec["prompt"] != spec["unconditional"]:
                    branches.append(self.pack(spec["unconditional"], spec, reference, count))
                torch.manual_seed(seed + offset)
                torch.cuda.manual_seed_all(seed + offset)
                p = {"max_new_tokens": min(settings.MAX_FRAMES + 60, spec["frames"] + 60),
                     "audio_temperature": temperature, "audio_top_p": 0.95,
                     "audio_top_k": 50, "audio_repetition_penalty": 1.0,
                     "text_temperature": 0.0 if temperature == 0 else 1.0, "text_top_p": 1.0, "text_top_k": 50}
                frames = []
                for frames in self._stream_frames_cfg(branches, p, chunk_frames=p["max_new_tokens"] + 1, guidance=guidance):
                    pass
                if not frames:
                    raise RuntimeError("The model generated no audio frames.")
                codes = torch.stack(frames, dim=1)
                for c in codes:
                    c = c[c[:, 0] != self.model.config.audio_pad_token_id]
                    if len(c) < 2:
                        raise RuntimeError("A take contains fewer than two audio frames.")
                    self.last_codes.append(c.cpu())
                    decoded = self.proc.decode_audio_codes([c], return_stereo=False)[0]
                    wave = decoded.float().cpu().numpy().reshape(-1)
                    if not np.isfinite(wave).all():
                        raise RuntimeError("Non-finite audio values in take.")
                    waves.append(wave)
                if progress:
                    progress(len(waves), n)
        return waves


class Scorers:
    """Use the same proven Judge/Parakeet/VoiceCLAP ranker, without its LoRAs."""
    def __init__(self):
        from asr_engine import ParakeetASR
        from bestofn import Judge
        from retrieval import Retriever
        self.asr = ParakeetASR(device=settings.SCORE_DEVICE)
        self.retriever = Retriever(device=settings.SCORE_DEVICE)
        self.retriever._load()
        self.judge = Judge(device=settings.SCORE_DEVICE, asr=self.asr)
        if not self.judge.ok or not self.retriever.ok:
            raise RuntimeError("Best-of-N scorers could not be loaded.")
        self.judge.attach_clap(self.retriever)
        if not self.judge._load_heads():
            raise RuntimeError("Burst detector heads are unavailable.")
        # Confirmed numerically by search_validation.py: both packaged heads
        # use exactly this VoiceCLAP tower. Share its weights instead of keeping
        # three resident copies. The old scoring interface remains unchanged.
        self.judge.gen.embedder = self.retriever.model
        self.judge.bl.embedder = self.retriever.model
        from dimension_predictors import DimensionPredictors
        self.dimensions = DimensionPredictors(self.retriever.device)
        from burst_models import BurstModels
        self.bursts = BurstModels(self.retriever)
        from speaker_reward import SpeakerModels
        self.speakers = SpeakerModels(self)
        from ears_reward import EarsModels
        self.ears = EarsModels(settings.SCORE_DEVICE)
        self.ears.prepare()

    def score(self, waves, spec):
        from bestofn import rank
        scores = self.judge.score(waves, 48000, spec["text"], general=spec["general"],
            script=spec["tagged"], tagged=spec["tagged"])
        from metrics import word_errors
        for score in scores:
            score["legacy_word_distance"] = score["wer"]
            score.update(word_errors(spec["text"], score["hyp"]))
        rank(scores)
        return scores
