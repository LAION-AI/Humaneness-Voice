#!/usr/bin/env python3
"""Find where the scripted words actually start and end, and trim to them.

The problem this solves: the model spends the duration it is given.  It does not
overrun and it does not stop early — it *fills*.  So when a take improvises, the
filler sits inside the requested duration rather than past it, and no clock-based
cut can find it (docs/EXPERIMENTS.md §8).  Something has to notice that the words
that were asked for have been said.

Forced alignment is the right tool rather than recognition: we already know which
words were supposed to be spoken, so this is an alignment problem.  An aligner
cannot hallucinate a word that was never in the script, which a transcriber can.

Two edges are handled:

  lead-in   If the script does not open on a vocal burst, everything before the
            first word is throat-clearing the director did not ask for.  Fade in
            so that full level is reached exactly at the first word's onset.
            When the script DOES open on a burst, nothing is touched — the burst
            is the performance.

  tail      After the last scripted word ends, fade out over ~150 ms and stop.
            Anything the model was about to add is not in the script.

The aligner is `romara-labs/mms-300m-1130-forced-aligner-ONNX` for the emissions
(Wav2Vec2 CTC, 31 tokens, 20 ms per frame at 16 kHz) and
`torchaudio.functional.forced_align` for the Viterbi pass — a tested kernel
rather than a hand-rolled trellis, which is where this kind of code goes wrong.

LICENCE NOTE: that aligner is **CC BY-NC 4.0**, unlike everything else this
server loads.  It is downloaded at runtime and not redistributed here, so it does
not change this repository's licence, but a commercial deployment must either
switch it off (`MOSS_ALIGN=0`, or the checkbox) or replace it.  The rest of the
stack was chosen for commercial usability on purpose; this one component is not.
"""
import os
import re
import threading
import unicodedata

import numpy as np

import config

FRAME_S = 0.02           # 320-sample stride at 16 kHz
_SR = 16000
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
# ß has no NFKD decomposition and would become <unk>; the rest of German folds
# to ASCII by stripping combining marks
_PRE = {"ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "å": "a", "đ": "d", "ł": "l",
        "þ": "th", "ð": "d"}


def script_words(plain_text):
    """The words the aligner will be asked to find, in order."""
    return _WORD.findall(str(plain_text or ""))


def opens_on_burst(tagged):
    """Does the rendered script put a vocal burst before its first word?

    In this format a round bracket WITH a number is a burst and one without is a
    direction, and `[N.N seconds duration]` precedes every speech segment.  So a
    burst tag appearing before the first duration tag is a burst before the first
    word — and then the lead-in must be left alone.
    """
    s = str(tagged or "")
    m_dur = re.search(r"\[[0-9.]+\s*seconds\s+duration\]", s, re.I)
    cut = m_dur.start() if m_dur else len(s)
    return bool(re.search(r"\([^)]*,\s*[0-9.]+\s*seconds?\)", s[:cut], re.I))


def closing_burst_s(tagged):
    """Seconds of vocal burst the script asks for AFTER its last word.

    A burst is performance, not filler, and the trim must not eat it.  The
    leading case is handled by `opens_on_burst`; this is the same rule at the
    other end, and it was found by looking at the data rather than the summary:
    a line ending on `(breathy giggle)` had 1.03 s removed, which read as a
    successful trim on every metric — extra words 1 -> 0, word error 0.10 -> 0.00
    — while actually deleting the giggle the director wrote.

    The tags carry their own lengths, so the allowance is exact rather than
    guessed.
    """
    s = str(tagged or "")
    last = None
    for m in re.finditer(r"\[[0-9.]+\s*seconds\s+duration\]", s, re.I):
        last = m
    tail = s[last.end():] if last else s
    total = 0.0
    for m in re.finditer(r"\([^)]*,\s*([0-9.]+)\s*seconds?\)", tail, re.I):
        try:
            total += float(m.group(1))
        except ValueError:
            pass
    return total


class QwenAligner:
    """Qwen3-ForcedAligner-0.6B, the permissively licensed alternative.

    Apache-2.0 rather than CC BY-NC, which is the reason it exists here: it is
    the only component of this stack that had a non-commercial licence, and the
    rest was chosen for commercial usability on purpose.

    Measured against the MMS aligner on the same clips: word ends agree to
    0.057 s on average (0.22 s worst) on German and to within a frame or two on
    English, and it is about twice as fast — 38 ms against 61 ms for 10 s of
    English, 68 against 135 for 14 s of German — because it is one
    non-autoregressive forward pass rather than an emission pass plus a Viterbi
    pass.  It costs 1.84 GB of VRAM loaded against roughly 0.4 GB for the ONNX
    CTC model, which is why it goes on the second card by default.

    It returns no confidence, so `score` here is a STRUCTURAL plausibility check
    rather than a model confidence: whether the words came back in the number
    asked for, in order, inside the audio, and without an implausibly long one.
    That is weaker evidence than the CTC path's per-token posterior, and the
    honest consequence is that the score threshold catches gross failures here
    but not subtle ones.
    """

    REPO = "Qwen/Qwen3-ForcedAligner-0.6B-hf"

    def __init__(self, device=None, repo=None):
        self.ok = False
        self.lock = threading.Lock()
        self.device = device or config.ALIGN_QWEN_DEVICE
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoProcessor
            repo = repo or config.ALIGN_QWEN_REPO or self.REPO
            self.proc = AutoProcessor.from_pretrained(repo)
            self.model = AutoModelForTokenClassification.from_pretrained(
                repo, dtype=torch.bfloat16).to(self.device).eval()
            self.ok = True
            print(f"[align] {repo} on {self.device}, apache-2.0, "
                  f"{sum(p.numel() for p in self.model.parameters())/1e9:.2f}B params",
                  flush=True)
        except Exception as e:
            print(f"[align] Qwen aligner unavailable ({type(e).__name__}: {e}); "
                  f"falling back", flush=True)

    def align(self, wav16k, words, language=None):
        import torch
        if not self.ok or not words:
            return None
        transcript = " ".join(str(w) for w in words)
        lang = language or ("German" if _looks_german(transcript) else "English")
        try:
            with self.lock:
                ins, wl = self.proc.prepare_forced_aligner_inputs(
                    audio=np.asarray(wav16k, dtype=np.float32).reshape(-1),
                    transcript=transcript, language=lang)
                ins = ins.to(self.model.device, self.model.dtype)
                with torch.inference_mode():
                    out = self.model(**ins)
                ts = self.proc.decode_forced_alignment(
                    logits=out.logits, input_ids=ins["input_ids"], word_lists=wl,
                    timestamp_token_id=self.model.config.timestamp_token_id)[0]
        except Exception as e:
            print(f"[align] qwen align failed: {type(e).__name__}: {e}", flush=True)
            return None
        if not ts:
            return None
        dur = len(wav16k) / float(_SR)
        sane = (len(ts) == len(words))
        prev = -1.0
        for t in ts:
            a, b = float(t["start_time"]), float(t["end_time"])
            if a < prev - 1e-3 or b < a or b > dur + 0.25 \
                    or (b - a) > config.ALIGN_QWEN_MAX_WORD_S:
                sane = False
                break
            prev = a
        score = 1.0 if sane else 0.0
        return [(t["text"], float(t["start_time"]), float(t["end_time"]), score)
                for t in ts]


_DE = re.compile(r"[äöüßÄÖÜ]|\b(?:und|nicht|noch|ist|das|die|der|ein|eine|mit|"
                 r"auf|für|über|wie|was|dass|weil|ich|du|wir|sie|aber|schon)\b")


def _looks_german(text):
    return len(_DE.findall(str(text or ""))) >= 2


def make_aligner():
    """The configured aligner, with a fallback when the preferred one is missing."""
    order = [b.strip() for b in str(config.ALIGN_BACKEND).split(",") if b.strip()]
    for b in order:
        a = QwenAligner() if b == "qwen" else Aligner()
        if a.ok:
            return a
        print(f"[align] backend {b!r} did not load, trying the next", flush=True)
    return None


class Aligner:
    """The CTC emission model, loaded once.

    CC BY-NC 4.0 -- see the module docstring.  `make_aligner()` prefers the
    Apache-2.0 Qwen model; this remains available and is the lighter of the two.
    """

    def __init__(self, root=None, device=None):
        self.ok = False
        self.lock = threading.Lock()
        self.root = root or config.ALIGN_DIR
        vocab_p = os.path.join(self.root, "vocab.json")
        want_gpu = (device or config.ALIGN_DEVICE) != "cpu"
        model_p = os.path.join(self.root,
                               "model.fp32.onnx" if want_gpu else "model.q8.onnx")
        if not (os.path.exists(vocab_p) and os.path.exists(model_p)):
            print(f"[align] no aligner at {self.root}; end-trimming is off",
                  flush=True)
            return
        try:
            import json
            import onnxruntime as ort
            self.vocab = json.load(open(vocab_p, encoding="utf-8"))
            self.blank = int(self.vocab.get("<blank>", 0))
            self.unk = int(self.vocab.get("<unk>", 3))
            provs = (["CUDAExecutionProvider", "CPUExecutionProvider"] if want_gpu
                     else ["CPUExecutionProvider"])
            so = ort.SessionOptions()
            so.log_severity_level = 3
            self.sess = ort.InferenceSession(model_p, so, providers=provs)
            self.provider = self.sess.get_providers()[0]
            self.ok = True
            print(f"[align] {os.path.basename(model_p)} on {self.provider}, "
                  f"{len(self.vocab)} tokens, {FRAME_S*1000:.0f} ms per frame",
                  flush=True)
        except Exception as e:
            print(f"[align] unavailable ({type(e).__name__}: {e}); "
                  f"end-trimming is off", flush=True)

    # ---------------------------------------------------------------- text
    def _tokens(self, words):
        """Per-word token id lists, with the characters the model cannot see dropped."""
        out = []
        for w in words:
            s = unicodedata.normalize("NFKD", str(w).lower())
            s = "".join(_PRE.get(c, c) for c in s)
            s = "".join(c for c in s if unicodedata.category(c) != "Mn")
            s = s.replace("’", "'")
            ids = [self.vocab[c] for c in s if c in self.vocab]
            out.append(ids)
        return out

    # ---------------------------------------------------------------- audio
    def emissions(self, wav16k):
        import torch
        x = np.asarray(wav16k, dtype=np.float32).reshape(-1)
        m = float(x.mean())
        sd = float(np.sqrt(((x - m) ** 2).mean() + 1e-7))
        x = ((x - m) / sd)[None, :]
        with self.lock:
            logits = self.sess.run(
                ["logits"], {"input_values": x,
                             "attention_mask": np.ones(x.shape, dtype=np.int64)})[0]
        lp = torch.log_softmax(torch.from_numpy(logits[0]).float(), dim=-1)
        return lp

    def align(self, wav16k, words):
        """[(word, start_s, end_s, score)] or None when it cannot be done."""
        import torch
        import torchaudio.functional as AF
        if not self.ok or not words:
            return None
        per_word = self._tokens(words)
        flat, owner = [], []
        for wi, ids in enumerate(per_word):
            for t in ids:
                flat.append(t)
                owner.append(wi)
        if not flat:
            return None
        lp = self.emissions(wav16k)
        if lp.shape[0] < len(flat):
            # fewer frames than target tokens: alignment is not defined
            return None
        try:
            labels, scores = AF.forced_align(
                lp[None], torch.tensor([flat], dtype=torch.int32),
                blank=self.blank)
        except Exception as e:
            print(f"[align] forced_align failed: {type(e).__name__}: {e}",
                  flush=True)
            return None
        labels = labels[0].tolist()
        scores = scores[0].exp().tolist()
        # walk the frame labels, attributing each non-blank run to its token
        spans, ti = [], 0
        f = 0
        n = len(labels)
        while f < n and ti < len(flat):
            if labels[f] == self.blank:
                f += 1
                continue
            start = f
            sc = []
            while f < n and labels[f] != self.blank:
                sc.append(scores[f])
                f += 1
                if ti + 1 < len(flat) and f < n and labels[f] != self.blank \
                        and labels[f] != labels[f - 1]:
                    break
            spans.append((owner[ti], start, f, float(np.mean(sc)) if sc else 0.0))
            ti += 1
        if not spans:
            return None
        out = []
        for wi, w in enumerate(words):
            mine = [s for s in spans if s[0] == wi]
            if not mine:
                out.append((w, None, None, 0.0))
                continue
            out.append((w, mine[0][1] * FRAME_S, mine[-1][2] * FRAME_S,
                        float(np.mean([m[3] for m in mine]))))
        return out


# ------------------------------------------------------------------ editing
def _fade_in(x, sr, upto_s, ramp_s):
    """Silence before `upto_s - ramp_s`, then a linear ramp to full at `upto_s`."""
    n = len(x)
    end = int(max(0.0, upto_s) * sr)
    if end <= 0:
        return x
    end = min(end, n)
    ramp = min(int(ramp_s * sr), end)
    y = x.copy()
    if end - ramp > 0:
        y[:end - ramp] = 0.0
    if ramp > 0:
        y[end - ramp:end] *= np.linspace(0.0, 1.0, ramp, dtype=np.float32)
    return y


def _fade_out(x, sr, from_s, ramp_s):
    """Linear ramp to zero starting at `from_s`, and nothing after it."""
    n = len(x)
    start = int(max(0.0, from_s) * sr)
    if start >= n:
        return x
    ramp = min(int(ramp_s * sr), n - start)
    y = x[:start + ramp].copy()
    if ramp > 0:
        y[start:start + ramp] *= np.linspace(1.0, 0.0, ramp, dtype=np.float32)
    return y


def edit_plan(spans, dur_s, allow_lead, min_score=None, closing_burst=0.0):
    """Where to fade in and where to cut, from an alignment.

    Returns (fade_in_upto_s or None, cut_at_s or None, note).  Both may be None:
    a take whose last word runs to the end of the audio needs no tail, and a
    take that opens on a burst needs no lead-in.
    """
    ms = min_score if min_score is not None else config.ALIGN_MIN_SCORE
    got = [s for s in spans if s[1] is not None] if spans else []
    if not got:
        return None, None, "alignment found none of the scripted words"
    lead = None
    if allow_lead:
        w0 = got[0]
        if w0[3] >= ms and w0[1] > config.ALIGN_LEAD_MIN_S:
            lead = float(w0[1])
    cut = None
    last = got[-1]
    if last[3] < ms:
        return lead, None, (f"last word {last[0]!r} aligned at score "
                            f"{last[3]:.2f} < {ms:.2f}; tail left alone")
    # a scripted burst after the last word is performance: leave room for it,
    # plus a little, since the model rarely hits a burst length exactly
    allow = 0.0
    if closing_burst > 0:
        allow = closing_burst * config.ALIGN_BURST_SLACK + config.ALIGN_TAIL_PAD_S
    tail_room = dur_s - float(last[2]) - allow
    if tail_room > config.ALIGN_TAIL_MIN_S:
        cut = float(last[2]) + config.ALIGN_TAIL_PAD_S + allow
    note = ("" if allow <= 0 else
            f"kept {allow:.2f}s for the {closing_burst:.2f}s of burst the script "
            f"puts after its last word")
    return lead, cut, note


def trim(wav, sr, tagged, plain, aligner, allow_lead=None, note=None):
    """Trim one finished take.  Returns (audio, report)."""
    rep = {"applied": False, "lead_s": None, "cut_s": None, "removed_s": 0.0,
           "words": 0, "note": ""}
    if aligner is None or not aligner.ok or wav is None or len(wav) < sr // 4:
        rep["note"] = "aligner unavailable"
        return wav, rep
    words = script_words(plain)
    if not words:
        rep["note"] = "no scripted words"
        return wav, rep
    if allow_lead is None:
        allow_lead = not opens_on_burst(tagged)
    dur = len(wav) / float(sr)
    wav16 = _to16k(wav, sr)
    spans = aligner.align(wav16, words)
    if spans is None:
        rep["note"] = "alignment failed"
        return wav, rep
    lead, cut, why = edit_plan(spans, dur, allow_lead,
                               closing_burst=closing_burst_s(tagged))
    rep.update(words=len([s for s in spans if s[1] is not None]), note=why)
    y = wav
    if lead is not None:
        y = _fade_in(y, sr, lead, config.ALIGN_LEAD_RAMP_S)
        rep["lead_s"] = round(lead, 3)
    if cut is not None:
        before = len(y) / float(sr)
        y = _fade_out(y, sr, cut, config.ALIGN_TAIL_RAMP_S)
        rep["cut_s"] = round(cut, 3)
        rep["removed_s"] = round(before - len(y) / float(sr), 3)
    rep["applied"] = bool(lead is not None or cut is not None)
    rep["spans"] = [(w, None if a is None else round(a, 3),
                     None if b is None else round(b, 3), round(s, 3))
                    for w, a, b, s in spans]
    return y, rep


def _to16k(x, sr):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if sr == _SR:
        return x
    if sr % _SR == 0:                      # 48000 -> 16000, the usual case
        k = sr // _SR
        n = (len(x) // k) * k
        return x[:n].reshape(-1, k).mean(1)
    idx = np.linspace(0, len(x) - 1, int(len(x) * _SR / sr), dtype=np.float32)
    return np.interp(idx, np.arange(len(x), dtype=np.float32), x).astype(np.float32)


class StreamGuard:
    """The same two edits, applied to audio that is already being played.

    A stream cannot retract what it has emitted, so this keeps a lookahead
    buffer: audio is held back for `ALIGN_LOOKAHEAD_S` before being handed on.
    While it is held, the aligner is run on everything generated so far — not on
    every chunk, which would be far more alignment than generation, but every
    `ALIGN_EVERY_S` of new audio.  Once the last scripted word is found to have
    ended, the tail is faded and the rest is dropped.

    With the aligner missing or switched off this class is a passthrough that
    still buffers nothing, so the streaming path is unchanged.
    """

    def __init__(self, aligner, sr, tagged, plain, enabled=True, expect_s=None):
        self.sr = sr
        self.a = aligner
        self.words = script_words(plain)
        self.on = bool(enabled and aligner is not None and aligner.ok
                       and self.words)
        self.allow_lead = self.on and not opens_on_burst(tagged)
        cb = closing_burst_s(tagged)
        self.tail_allow = (cb * config.ALIGN_BURST_SLACK + config.ALIGN_TAIL_PAD_S
                           if cb > 0 else 0.0)
        self.buf = np.zeros(0, np.float32)
        self.emitted = 0            # samples handed on
        self.total = 0              # samples seen
        self.done = False           # tail found; drop everything further
        self.lead_done = not self.allow_lead
        self.last_check = 0
        self.report = {"applied": False, "lead_s": None, "cut_s": None,
                       "removed_s": 0.0, "checks": 0, "note": ""}
        self.look = int(config.ALIGN_LOOKAHEAD_S * sr)
        self.every = int(config.ALIGN_EVERY_S * sr)
        # The first word often starts later than the ordinary lookahead: measured
        # offline, onsets run to 1.3 s.  Deciding the lead-in at 0.5 s asked the
        # aligner to fit the whole script into half a second, which fails, so the
        # fade silently never fired in the stream while firing 28 times in 36
        # takes offline.  Hold back further for the first decision only, and ask
        # about the first few words rather than all of them.
        self.lead_scan = int(config.ALIGN_LEAD_SCAN_S * sr)
        # Do not look for the end of the line before most of the requested
        # duration exists.  Forced alignment always places every target
        # somewhere, so against a prefix it will happily report the last word as
        # finished when half the line is still unspoken -- and that cut would
        # truncate real speech rather than filler.  The duration budget is
        # honoured to within 0.02 s, so it is a reliable gate.
        self.expect = float(expect_s or 0.0)
        self.all = [] if self.on else None

    def _align_now(self, words=None):
        w = np.concatenate(self.all) if self.all else np.zeros(0, np.float32)
        if len(w) < self.sr // 4:
            return None
        self.report["checks"] += 1
        return self.a.align(_to16k(w, self.sr), words or self.words)

    def feed(self, chunk):
        """Take one generated chunk, return what may be played now."""
        c = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if not self.on:
            return c
        if self.done:
            return np.zeros(0, np.float32)
        self.all.append(c)
        self.total += len(c)
        self.buf = np.concatenate([self.buf, c]) if len(self.buf) else c

        # the lead-in: decided once, from a longer first stretch and only about
        # the opening words
        if not self.lead_done and self.total >= self.lead_scan:
            spans = self._align_now(self.words[:config.ALIGN_LEAD_WORDS])
            if spans:
                got = [s for s in spans if s[1] is not None]
                if got and got[0][3] >= config.ALIGN_MIN_SCORE \
                        and got[0][1] > config.ALIGN_LEAD_MIN_S:
                    self.buf = _fade_in(self.buf, self.sr, got[0][1],
                                        config.ALIGN_LEAD_RAMP_S)
                    self.report["lead_s"] = round(float(got[0][1]), 3)
                    self.report["applied"] = True
            self.lead_done = True

        # the tail: once the last word has ended inside what we have generated
        ready = (self.expect <= 0
                 or self.total / float(self.sr) >= self.expect * config.ALIGN_TAIL_AFTER)
        if ready and self.total - self.last_check >= self.every:
            self.last_check = self.total
            spans = self._align_now()
            got = [s for s in spans if s[1] is not None] if spans else []
            if got and len(got) == len(self.words) \
                    and got[-1][3] >= config.ALIGN_MIN_SCORE:
                cut = (float(got[-1][2]) + config.ALIGN_TAIL_PAD_S
                       + self.tail_allow)
                gen = self.total / float(self.sr)
                if cut < gen - config.ALIGN_TAIL_MIN_S:
                    keep = int(cut * self.sr) - self.emitted
                    out = self.buf[:max(0, keep)]
                    ramp = int(config.ALIGN_TAIL_RAMP_S * self.sr)
                    ramp = min(ramp, len(out))
                    if ramp > 0:
                        out = out.copy()
                        out[len(out) - ramp:] *= np.linspace(
                            1.0, 0.0, ramp, dtype=np.float32)
                    self.done = True
                    self.report.update(
                        applied=True, cut_s=round(cut, 3),
                        removed_s=round(gen - cut, 3))
                    self.emitted += len(out)
                    self.buf = np.zeros(0, np.float32)
                    return out

        # ordinary case: hand on everything except the lookahead.  Nothing at all
        # goes out until the lead-in has been decided, or the fade would be
        # applied to audio the listener has already heard.
        keep_back = self.look if self.lead_done else max(self.look, self.lead_scan)
        if len(self.buf) > keep_back:
            out = self.buf[:len(self.buf) - keep_back]
            self.buf = self.buf[len(self.buf) - keep_back:]
            self.emitted += len(out)
            return out
        return np.zeros(0, np.float32)

    def flush(self):
        """Whatever is still held back, at the end of generation."""
        if not self.on or self.done:
            return np.zeros(0, np.float32)
        out, self.buf = self.buf, np.zeros(0, np.float32)
        self.emitted += len(out)
        return out
