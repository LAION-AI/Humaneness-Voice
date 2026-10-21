"""Optional CPU ECAPA teacher baseline; no training, GPU or network access.

This is NOT the Nemotron-feature TYGER V2 student. All audio/embeddings remain
in memory, and speaker identity is an approximate cue, never authentication.
Configure ROBO_ECAPA_ROOT to an existing pinned SpeechBrain snapshot, or use
an already populated HF_HUB_CACHE. Importing this module loads no model.
"""
from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import threading
import time

import numpy as np

SAMPLE_RATE = 16000
EMBEDDING_DIM = 192
TEACHER_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
REPO = "speechbrain/spkrec-ecapa-voxceleb"
REQUIRED_FILES = ("hyperparams.yaml", "embedding_model.ckpt",
                  "mean_var_norm_emb.ckpt", "classifier.ckpt", "label_encoder.txt")


def preserve_encoder_precision():
    """SpeechBrain's import quirk must not enable TF32 in the EARS process."""
    disabled={x.strip() for x in os.environ.get('SB_DISABLE_QUIRKS','').split(',') if x.strip()}
    disabled.add('allow_tf32')
    os.environ['SB_DISABLE_QUIRKS']=','.join(sorted(disabled))


def resolve_model_root(model_root=None):
    """Resolve local files only; never call snapshot_download or trust a remote ID."""
    explicit = model_root or os.environ.get("ROBO_ECAPA_ROOT")
    if explicit:
        root = Path(explicit).expanduser().resolve()
    else:
        from huggingface_hub.constants import HF_HUB_CACHE
        cache = Path(os.environ.get("HF_HUB_CACHE", HF_HUB_CACHE)).expanduser()
        root = cache / "models--speechbrain--spkrec-ecapa-voxceleb" / "snapshots" / TEACHER_REVISION
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("ECAPA is offline-only: set ROBO_ECAPA_ROOT to a complete "
                                "local snapshot; missing " + ", ".join(missing))
    return root


def _audio(values):
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Expected finite mono float audio at 16000 Hz")
    return np.ascontiguousarray(x)


def enrollment_reference(audio16,seconds=3.04):
    """Bound a verified introduction to its most audible contiguous context.

    This changes only the voice anchor, never recognized words or turn timing.
    Energy, not a speaker score or target identity, selects the window. Quiet
    room pre-roll must not dominate a new person's embedding. No padding or
    waiting is used for utterances shorter than the requested context.
    """
    x=_audio(audio16)
    if not .64<=seconds<=15:raise ValueError('Reference context must be 0.64–15 seconds')
    n=round(seconds*SAMPLE_RATE)
    if len(x)<=n:return x.copy()
    starts=np.unique(np.r_[np.arange(0,len(x)-n+1,1280),len(x)-n]).astype(int)
    power=np.r_[0.,np.cumsum(x.astype(np.float64)**2)]
    at=int(starts[np.argmax(power[starts+n]-power[starts])])
    return x[at:at+n].copy()


class EcapaTeacher:
    """Thread-safe lazy model; embed() returns a unit-length float32 [192].

    A minimum of 640 ms is accepted, but clean full utterances are preferable.
    The RMS check rejects silence, not noise/overlap; use upstream activity cues.
    Calls are serialized to avoid competing CPU inference on the shared model.
    No torch thread setting or other process-global runtime option is changed.
    """
    def __init__(self, model_root=None, min_seconds=.64, min_rms=1e-5):
        if min_seconds < .64 or min_rms < 0:
            raise ValueError("Need at least 640 ms and a nonnegative silence threshold")
        self.model_root = model_root
        self.min_samples = int(np.ceil(min_seconds * SAMPLE_RATE))
        self.min_rms = min_rms
        self._model = None
        self._lock = threading.RLock()

    @property
    def loaded(self):
        return self._model is not None

    def _load(self):
        if self._model is None:
            preserve_encoder_precision()
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import FetchConfig, LocalStrategy
            root = resolve_model_root(self.model_root)
            # NO_LINK leaves the local snapshot alone, even when caches are read-only.
            # allow_network=False covers pretrainer collection as well as YAML fetch.
            self._model = EncoderClassifier.from_hparams(
                source=str(root), overrides={"pretrained_path": str(root)},
                savedir=None, local_strategy=LocalStrategy.NO_LINK,
                fetch_config=FetchConfig(allow_network=False, allow_updates=False),
                run_opts={"device": "cpu"})
            self._model.eval()
        return self._model

    def embed(self, audio16):
        x = _audio(audio16)
        if len(x) < self.min_samples:
            raise ValueError("Need at least 640 ms of speech; prefer a clean full utterance")
        if float(np.sqrt(np.mean(x.astype(np.float64)**2))) < self.min_rms:
            raise ValueError("Silent/near-silent audio has no reliable speaker embedding")
        import torch
        with self._lock, torch.inference_mode():
            model = self._load()
            result = model.encode_batch(torch.from_numpy(x.copy())[None]).reshape(-1)
            vector = result.detach().cpu().float().numpy().copy()
        norm = float(np.linalg.norm(vector))
        if vector.shape != (EMBEDDING_DIM,) or not np.isfinite(vector).all() or norm < 1e-8:
            raise ValueError("Teacher returned an invalid speaker embedding")
        return (vector / norm).astype(np.float32, copy=False)


_shared = {}
_shared_lock = threading.Lock()


def get_cached_teacher(model_root=None):
    """Share one lazy CPU model per local snapshot in this process."""
    root = resolve_model_root(model_root)
    with _shared_lock:
        if str(root) not in _shared:
            _shared[str(root)] = EcapaTeacher(root)
        return _shared[str(root)]


class StreamingTeacher:
    """Bounded per-conversation audio context over a shared teacher.

    Input cadence may be 20 ms or 320 ms. First estimate needs 1.6 s of accepted
    speech; subsequent estimates are computed every 640 ms by default. Set
    update_seconds=1.0 for a slower CPU cadence. Call from a worker thread, not
    the microphone/event-loop callback. Never publish/persist these buffers.

    speech=True must come from a separate activity cue, not speaker similarity.
    speech=False and overlap=True clear context immediately. With speech=None,
    energy alone cannot establish speech, so no identity is emitted. Confidence
    is a context/eligibility indicator, NOT a calibrated identity probability.
    """
    def __init__(self, teacher=None, min_context_seconds=1.6, update_seconds=.64,
                 max_context_seconds=4., min_rms=.003):
        if min_context_seconds < 1.6 or update_seconds <= 0 or max_context_seconds < min_context_seconds or min_rms < 0:
            raise ValueError("Need >=1.6 s context, positive cadence and sufficient buffer")
        self.teacher = teacher if teacher is not None else get_cached_teacher()
        self.min_samples = int(np.ceil(min_context_seconds * SAMPLE_RATE))
        self.update_samples = int(np.ceil(update_seconds * SAMPLE_RATE))
        self.max_samples = int(np.ceil(max_context_seconds * SAMPLE_RATE))
        self.min_rms = min_rms
        self.reset()

    def reset(self):
        self._chunks = deque()
        self._samples = 0
        self._since_update = 0
        self._estimated = False

    def _reply(self, kind, **fields):
        return {"embedding": None, "kind": kind, "confidence": 0.,
                "context_ms": self._samples * 1000 / SAMPLE_RATE,
                "source": "ecapa_teacher_cpu", "fresh": False, **fields}

    def update(self, audio16, speech=None, overlap=False):
        x = _audio(audio16)
        if not len(x):
            return self._reply("warming")
        rms = float(np.sqrt(np.mean(x.astype(np.float64)**2)))
        active = isinstance(speech, (bool, np.bool_)) and bool(speech)
        silent = isinstance(speech, (bool, np.bool_)) and not bool(speech)
        if overlap or not active or rms < self.min_rms:
            self.reset()
            return self._reply("overlap" if overlap else "silence" if silent or rms < self.min_rms else "uncertain")
        retained = x[-self.max_samples:].copy()
        self._chunks.append(retained)
        self._samples += len(retained)
        self._since_update += len(x)
        while self._samples > self.max_samples:
            extra = self._samples - self.max_samples
            first = self._chunks.popleft()
            trim = min(extra, len(first))
            if trim < len(first):
                self._chunks.appendleft(first[trim:].copy())
            self._samples -= trim
        if self._samples < self.min_samples:
            return self._reply("warming")
        if self._estimated and self._since_update < self.update_samples:
            # Do not repeat the last voiceprint and accidentally count it as new evidence.
            return self._reply("waiting")
        started = time.perf_counter()
        vector = self.teacher.embed(np.concatenate(self._chunks))
        self._since_update = 0
        self._estimated = True
        return self._reply("speech", embedding=vector, confidence=.8, fresh=True,
                           latency_ms=(time.perf_counter()-started)*1000)
