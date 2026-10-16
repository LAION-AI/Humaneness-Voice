"""Speech input via NVIDIA Parakeet TDT 0.6B v3.

Replaces the Gemma audio encoder, which worked but cost 0.6-1.1 s per clip
because every transcription went through the same 12B decoder that also has to
write the reply.  Parakeet is a dedicated 0.6B transducer: measured 75-107 ms
for 5-12 s of audio here, and it returns properly cased German with umlauts
rather than the lower-case stream Gemma produced.

Runs on the second card so it never competes with the voice model for VRAM.
"""
import re
import subprocess
import tempfile
import threading
import time

import numpy as np
import torch

import config

_BLANK = re.compile(r"<blank>")


class ParakeetASR:
    def __init__(self, model=None, device=None):
        from transformers import ParakeetForTDT, ParakeetProcessor
        model = model or config.ASR_MODEL
        self.device = device or config.ASR_DEVICE
        t0 = time.time()
        print(f"[asr] loading {model} on {self.device} ...", flush=True)
        self.proc = ParakeetProcessor.from_pretrained(model)
        self.model = ParakeetForTDT.from_pretrained(
            model, dtype=torch.bfloat16).to(self.device).eval()
        self.lock = threading.Lock()
        print(f"[asr] ready in {time.time()-t0:.1f}s", flush=True)

    @staticmethod
    def _to_wav16k(raw_bytes):
        """Whatever MediaRecorder produced -> 16 kHz mono float32."""
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".bin") as fin, \
                tempfile.NamedTemporaryFile(suffix=".wav") as fout:
            fin.write(raw_bytes)
            fin.flush()
            p = subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", fin.name,
                                "-ac", "1", "-ar", "16000", fout.name],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if p.returncode != 0:
                raise ValueError("could not decode audio")
            w, sr = sf.read(fout.name, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        return w

    @torch.inference_mode()
    def transcribe(self, audio):
        """audio: raw container bytes or a float32 16 kHz array -> text."""
        if isinstance(audio, (bytes, bytearray)):
            audio = self._to_wav16k(audio)
        if audio is None or len(audio) < 1600:      # under 0.1 s is not speech
            return ""
        with self.lock:
            inp = self.proc(np.asarray(audio, dtype=np.float32),
                            sampling_rate=16000, return_tensors="pt")
            inp = {k: (v.to(self.device).to(torch.bfloat16)
                       if v.dtype.is_floating_point else v.to(self.device))
                   for k, v in inp.items()}
            out = self.model.generate(**inp)
            text = self.proc.batch_decode(out.sequences)[0]
        # the transducer's blank symbol leaks into the decoded string
        return re.sub(r"\s{2,}", " ", _BLANK.sub(" ", str(text))).strip()
