"""Reuse the old demo's word alignment and fades, with Small edge-cue protection."""
import logging
import re
import sys
import numpy as np

import settings

if str(settings.PARENT) not in sys.path:
    sys.path.append(str(settings.PARENT))
import align_engine
import timed_script

log = logging.getLogger("small.alignment")
_TAG = re.compile(r"\([^)]*\)|\[[^]]*\]")


def edge_cues(tagged):
    """Only tags outside the spoken words protect the leading/trailing edges.

    Unlike the legacy heuristic, this works without duration tags and does not
    mistake a mid-sentence burst for a leading or closing sound. Explicit edge
    trailing silence is also intentional. Leading silence needs no exception:
    the legacy fade silences the lead without removing samples or its duration.
    """
    tagged = str(tagged or "")
    masked = _TAG.sub(lambda m: " " * len(m[0]), tagged)
    speech = list(re.finditer(r"\S+", masked))
    if not speech:
        return True, ""

    def protects(tag):
        if tag.startswith("["):
            return "pause" in tag.lower()
        body = tag[1:-1]
        return bool(re.search(r",\s*[0-9.]+\s*seconds?", body, re.I)
                    or timed_script._is_burst_label(body))

    prefix = tagged[:speech[0].start()]
    suffix = tagged[speech[-1].end():]
    allow_lead = not any(m[0].startswith("(") and protects(m[0]) for m in _TAG.finditer(prefix))
    closing = []
    for m in _TAG.finditer(suffix):
        tag = m[0]
        if not protects(tag):
            continue
        number = re.search(r"([0-9.]+)\s*seconds?", tag, re.I)
        seconds = float(number[1]) if number else (0.5 if tag.startswith("[") else 0.28)
        # Legacy trim's tail rule retains twice the requested length plus air.
        closing.append(f"(requested edge sound or silence, {seconds} seconds)")
    return allow_lead, "[0 seconds duration] speech " + " ".join(closing)


class AudioFinisher:
    def __init__(self):
        # Deliberately choose the same Apache-2.0 Qwen backend, not the old
        # optional non-commercial MMS fallback.
        self.aligner = align_engine.QwenAligner(device=settings.ALIGN_DEVICE)

    @property
    def available(self):
        return self.aligner is not None and self.aligner.ok

    def finish(self, waves, spec, enabled=True):
        allow_lead, edges = edge_cues(spec.get("edge_script", spec.get("tagged", "")))
        output, reports = [], []
        for wave in waves:
            raw_seconds = len(wave) / 48000
            if not enabled:
                report = {"applied": False, "note": "disabled", "spans": []}
                edited = wave
            else:
                try:
                    edited, report = align_engine.trim(wave, 48000, edges,
                        spec["text"], self.aligner, allow_lead=allow_lead)
                except Exception:
                    log.exception("Alignment failed; leaving this take unchanged")
                    edited, report = wave, {"applied": False, "note": "alignment failed", "spans": []}
            report.update(enabled=enabled, available=self.available,
                          raw_seconds=round(raw_seconds, 3), seconds=round(len(edited) / 48000, 3),
                          lead_ramp_s=align_engine.config.ALIGN_LEAD_RAMP_S,
                          tail_ramp_s=align_engine.config.ALIGN_TAIL_RAMP_S,
                          score_kind="structural plausibility, not model confidence")
            output.append(edited)
            reports.append(report)
        return output, reports


class LiveCutoff:
    """Buffered Parakeet-timestamp fades; never align unfinished target words.

Tail cuts require exact recognized target words inside the committed PCM.
Keep 1 s lookahead, initially 1.3 s for lead detection. Final forced-aligned
audio supplies whatever remains held back at EOS. No prefix retraction.
"""
    def __init__(self,spec,enabled=True):
        from metrics import tokens
        self.words=tokens(spec['text']);self.enabled=enabled
        self.allow_lead,edges=edge_cues(spec.get('edge_script',spec.get('tagged','')))
        closing=align_engine.closing_burst_s(edges)
        self.tail_allow=closing*align_engine.config.ALIGN_BURST_SLACK+align_engine.config.ALIGN_TAIL_PAD_S if closing else 0.
        self.buffer=np.zeros(0,np.float32);self.total=0;self.emitted=0
        self.lead_done=not self.allow_lead;self.done=False
        self.report={'enabled':enabled,'backend':'Parakeet v3 word timestamps + final Qwen forced alignment',
                     'lead_s':None,'cut_s':None,'stopped':False,'lookahead_seconds':1.,'emitted_seconds':0.}

    def feed(self,chunk,score):
        from metrics import tokens
        chunk=np.asarray(chunk,dtype=np.float32).reshape(-1)
        if not self.enabled:
            self.total+=len(chunk);self.emitted+=len(chunk)
            return chunk
        if self.done:return np.zeros(0,np.float32)
        self.buffer=np.concatenate((self.buffer,chunk));self.total+=len(chunk)
        timestamped=score.get('timestamps') or []
        seconds=self.total/48000
        if not self.lead_done and seconds>=align_engine.config.ALIGN_LEAD_SCAN_S:
            if timestamped and score.get('prefix_wer',1)<=.25:
                first=float(timestamped[0]['start'])
                if first>align_engine.config.ALIGN_LEAD_MIN_S and first<seconds:
                    self.buffer=align_engine._fade_in(self.buffer,48000,first,align_engine.config.ALIGN_LEAD_RAMP_S)
                    self.report['lead_s']=first
            self.lead_done=True
        recognized=tokens(score.get('stable_hyp',''))
        if recognized==self.words and timestamped:
            last=float(timestamped[-1]['end'])
            cut=last+align_engine.config.ALIGN_TAIL_PAD_S+self.tail_allow
            end=cut+align_engine.config.ALIGN_TAIL_RAMP_S
            if end<=seconds and cut>=self.emitted/48000:
                self.buffer=align_engine._fade_out(self.buffer,48000,cut-self.emitted/48000,align_engine.config.ALIGN_TAIL_RAMP_S)
                self.done=True;self.report.update(cut_s=cut,stopped=True)
        keep_back=0 if self.done else int(48000*(1. if self.lead_done else 1.3))
        count=max(0,len(self.buffer)-keep_back)
        output=self.buffer[:count].copy();self.buffer=self.buffer[count:]
        self.emitted+=count;self.report['emitted_seconds']=self.emitted/48000
        return output

    def finish(self,aligned_wave):
        if not self.enabled or self.done:return np.zeros(0,np.float32)
        if len(aligned_wave)<self.emitted:
            # A timestamp estimate may arrive too late to retract played PCM.
            # Fade the held raw tail instead of ending playback abruptly.
            output=align_engine._fade_out(self.buffer,48000,0.,align_engine.config.ALIGN_TAIL_RAMP_S)
            self.report.update(note='Final alignment cannot retract already streamed PCM; held tail faded',
                               fallback_tail_fade=True)
        else:
            output=np.asarray(aligned_wave[self.emitted:],dtype=np.float32).copy()
        self.emitted+=len(output);self.buffer=np.zeros(0,np.float32)
        self.report['emitted_seconds']=self.emitted/48000
        return output
