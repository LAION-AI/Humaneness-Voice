"""Speaker similarity of a finished take, measured live.

ECAPA cosine against the corpus anchor, the same metric the manual uses for its
thresholds: regenerate below 0.58, repair below 0.45, reject below 0.40.  It is
computed after the audio has already been streamed, so it never delays playback
— it arrives with the closing event and is shown in the debug panel.

Its point here is comparison: the speaker adapter and the voice converter are two
different answers to the same drift, and this is the number that tells them apart.
"""
import threading
import time

import numpy as np


class SpeakerSim:
    def __init__(self, anchor_path, device="cpu"):
        import torch
        import warnings
        warnings.filterwarnings("ignore")
        from speechbrain.inference.speaker import EncoderClassifier
        self.torch = torch
        self.device = device
        self.lock = threading.Lock()
        t0 = time.time()
        self.enc = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/moss_eval/ecapa", run_opts={"device": device})
        self.anchors = {}                       # path -> embedding
        self.default_path = anchor_path
        self.anchor = self._embed_path(anchor_path)
        self.anchors[anchor_path] = self.anchor
        print(f"[sim] ready in {time.time()-t0:.1f}s", flush=True)

    # ---------------------------------------------------------------- embed
    def _embed(self, wav, sr):
        import torchaudio
        t = self.torch.from_numpy(np.asarray(wav, dtype=np.float32))[None]
        if sr != 16000:
            t = torchaudio.functional.resample(t, sr, 16000)
        with self.torch.no_grad():
            e = self.enc.encode_batch(t.to(self.device)).squeeze()
        return self.torch.nn.functional.normalize(e, dim=-1)

    def _embed_path(self, path):
        import subprocess, tempfile
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", path, "-ac", "1",
                            "-ar", "16000", f.name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            w, sr = sf.read(f.name, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        return self._embed(w, sr)

    # ----------------------------------------------------------------- score
    def anchor_for(self, path=None):
        """Embedding of one speaker's own recording, cached.

        Scoring a different profile against the corpus anchor is meaningless —
        it correctly reports that a different person is a different person — so
        each profile is measured against its own reference.
        """
        if not path:
            return self.anchor
        e = self.anchors.get(path)
        if e is None:
            try:
                e = self.anchors[path] = self._embed_path(path)
            except Exception as ex:
                print(f"[sim] anchor failed {path}: {ex}", flush=True)
                return self.anchor
        return e

    def score(self, wav, sr, anchor_path=None):
        """Cosine to that speaker's anchor, or None if the clip is too short."""
        if wav is None or len(wav) < sr * 0.5:
            return None
        try:
            with self.lock:
                a = self.anchor_for(anchor_path)
                e = self._embed(wav, sr)
                return round(float((a * e).sum()), 4)
        except Exception as ex:
            print(f"[sim] score failed: {ex}", flush=True)
            return None
