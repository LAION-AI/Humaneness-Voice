"""Streaming voice conversion with Chatterbox VC.

Why this one.  MeanVC2 was tried first and measurably damaged German: word
recall through speech recognition fell from 1.000 to 0.889, which is what an
"English accent" sounds like when you measure it — its content bottleneck is an
ASR trained elsewhere and it reshapes unfamiliar phonetics.  Chatterbox keeps
recall at 1.000 for the same identity gain, runs at RTF 0.08, and outputs
24 kHz instead of 16 kHz.  It is also the model that produced the
`voice_converted` half of the reference corpus, so it pulls towards exactly the
speaker our references already carry.

Streaming.  The target embedding is computed once, then each chunk of generated
audio is converted as it appears.  A chunk is converted together with a slice of
the audio before it, which is then cut away — flow matching conditioned on a
cold start at a chunk boundary is what produces the stutter — and consecutive
outputs are joined with a short equal-power crossfade to hide any residual
drift.  Conversion is far faster than playback, so this stays ahead.
"""
import os
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

import config


class VCStream:
    """Per-request state: nothing is shared between concurrent turns."""

    def __init__(self, sr=48000):
        self.sr = sr
        self.ctx = np.zeros(0, np.float32)     # source audio before this chunk
        self.tail = np.zeros(0, np.float32)    # converted tail held for crossfade


class VoiceConverter:
    OUT_SR = 24000

    def __init__(self, device=None):
        from chatterbox_vc import VoiceConverter as _CB
        self.device = device or config.VC_DEVICE
        self.lock = threading.Lock()
        self.ready = False
        t0 = time.time()
        print(f"[vc] loading Chatterbox VC on {self.device} ...", flush=True)
        self.cb = _CB(device=self.device)
        self.ready = True
        print(f"[vc] ready in {time.time()-t0:.1f}s", flush=True)

    @staticmethod
    def _normalise(w, target_dbfs=-16.0, peak_ceiling=0.97):
        """Bring a clip to a sane level without clipping it."""
        rms = float(np.sqrt((w ** 2).mean())) or 1e-9
        g = 10 ** ((target_dbfs - 20 * np.log10(rms)) / 20)
        peak = float(np.abs(w).max()) or 1e-9
        return (w * min(g, peak_ceiling / peak)).astype(np.float32)

    def prepare_target(self, src_path, out_path=None):
        """Turn the raw anchor recording into a good conversion target.

        The corpus anchor is a quiet 16 kHz mp3 (-21 dBFS, nothing above 8 kHz),
        which is a poor thing to ask a 24 kHz converter to imitate.  So: level it,
        restore it with Sidon (which returns 48 kHz and puts real energy back into
        the 8-12 kHz band), then level it again.  Sidon is loaded only for this and
        released straight afterwards — it is worth ~1 GB of VRAM otherwise.
        """
        import soundfile as sf

        out_path = out_path or os.path.join(tempfile.gettempdir(),
                                            "moss_vc_target.wav")
        raw = os.path.join(tempfile.gettempdir(), "moss_anchor16.wav")
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", src_path,
                        "-ac", "1", "-ar", "16000", raw],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not os.path.exists(raw):
            return None
        w, _ = sf.read(raw, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        w = self._normalise(w)

        ck = config.SIDON_CKPTS
        if not os.path.exists(os.path.join(ck, "feature_extractor_cuda.pt")):
            print("[vc] Sidon checkpoints missing, using the levelled anchor",
                  flush=True)
            sf.write(out_path, w, 16000)
            return out_path

        t0 = time.time()
        env = dict(os.environ)
        # its TorchScript constants are pinned to cuda:0, so give the helper a
        # process where the card we want to use *is* cuda:0
        env["CUDA_VISIBLE_DEVICES"] = config.SIDON_GPU
        env["MOSS_SIDON_CKPTS"] = ck
        env["MOSS_SIDON_SRC"] = config.SIDON_SRC
        env.pop("LD_LIBRARY_PATH", None)
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "sidon_restore.py")
        p = subprocess.run([sys.executable, script, src_path, out_path],
                           env=env, capture_output=True, text=True, timeout=600)
        if p.returncode == 0 and os.path.exists(out_path):
            print(f"[vc] anchor levelled + restored with Sidon in "
                  f"{time.time()-t0:.1f}s — {p.stdout.strip()}", flush=True)
            return out_path
        print(f"[vc] Sidon failed, using the levelled anchor: "
              f"{(p.stderr or '').strip()[-200:]}", flush=True)
        sf.write(out_path, w, 16000)
        return out_path

    def use_target(self, src_path):
        """Switch the conversion target, preparing each speaker only once."""
        if not hasattr(self, "_targets"):
            self._targets = {}
        if getattr(self, "_current", None) == src_path:
            return True
        tgt = self._targets.get(src_path)
        if tgt is None:
            key = os.path.basename(os.path.dirname(src_path)) + "_" + \
                os.path.basename(src_path)
            tgt = self.prepare_target(
                src_path, os.path.join(tempfile.gettempdir(),
                                       "moss_vc_target_" + key + ".wav"))
            if not tgt:
                return False
            self._targets[src_path] = tgt
        with self.lock:
            self.cb.set_target_voice(tgt)
        self._current = src_path
        self.target = tgt
        return True

    def set_target(self, wav_path):
        """The voice everything is converted towards — the corpus anchor,
        levelled and bandwidth-restored first."""
        tgt = self.prepare_target(wav_path)
        if not tgt or not os.path.exists(tgt):
            return False
        with self.lock:
            self.cb.set_target_voice(tgt)
        self.target = tgt
        self._current = wav_path
        return True

    def new_stream(self, sr=48000):
        return VCStream(sr)

    # ------------------------------------------------------------- convert
    def convert_chunk(self, pcm48, st):
        """One streamed chunk in, the converted chunk out, same sample rate."""
        if not self.ready or pcm48 is None or len(pcm48) < 2400:
            return pcm48
        import soundfile as sf
        import torch
        import torchaudio

        n_ctx = int(config.VC_CONTEXT_S * st.sr)
        joined = np.concatenate([st.ctx, pcm48]) if st.ctx.size else pcm48
        try:
            with self.lock:
                with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                    sf.write(f.name, joined, st.sr)
                    out = self.cb._model.generate(audio=f.name)
            o = out.squeeze().detach().cpu().float()
            o48 = torchaudio.functional.resample(
                o[None], self.OUT_SR, st.sr).squeeze(0).numpy().astype(np.float32)
        except Exception as e:
            print(f"[vc] convert failed: {e}", flush=True)
            return pcm48

        # cut away the context we prepended.  Chatterbox preserves timing, so the
        # ratio is stable; the crossfade below absorbs the sample-level slack.
        if st.ctx.size:
            drop = int(round(len(o48) * st.ctx.size / len(joined)))
            o48 = o48[drop:]
        st.ctx = joined[-n_ctx:] if len(joined) >= n_ctx else joined

        xf = int(config.VC_CROSSFADE_S * st.sr)
        if st.tail.size and o48.size > xf > 0:
            n = min(xf, st.tail.size, o48.size)
            # equal-power, so the join keeps a constant loudness
            t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
            o48[:n] = st.tail[-n:] * np.cos(t) + o48[:n] * np.sin(t)
        if o48.size > xf > 0:
            st.tail = o48[-xf:].copy()
            o48 = o48[:-xf]        # hold the seam back for the next chunk
        return o48

    def flush(self, st):
        """Whatever is still held back for the seam."""
        t, st.tail = st.tail, np.zeros(0, np.float32)
        return t
