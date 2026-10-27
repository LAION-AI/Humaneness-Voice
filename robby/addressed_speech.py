"""Conservative session-only background salience, not an intent classifier.

Open conversation accepts new voices. Only a conjunction of weak relative
level, no confident known voice and no explicit address can reject a turn.
No demographic inference, voiceprint persistence, or noisy SNR-only veto.
"""
import math,re
import numpy as np

class AddressedSpeechGate:
    def __init__(self):self.enabled=True
    def evaluate(self,text,embedding,users,level,noise_db=None):
        refs=[u.get('level',0.) for u in users.values() if u.get('level',0.)>0]
        reference=float(np.median(refs)) if refs else 0.
        similarity=-1.
        if embedding is not None:
            v=np.asarray(embedding,np.float32);v=v/max(1e-8,float(np.linalg.norm(v)))
            similarity=max((float(v@ref) for u in users.values() for ref in u['vectors']),default=-1.)
        from sentence_session import name_claim
        explicit=bool(name_claim(text) or re.search(
            r'\brobb(?:y|ie|i)\b|\b(?:kannst|könntest|magst|möchtest|würdest) du\b|'
            r'\b(?:erzähl|erzähle|sag|sagen|erklär|erkläre|zeig|zeige) (?:bitte )?mir\b|'
            r'\b(?:can|could|would|will) you\b|\b(?:tell|show) me\b',text,re.I))
        ratio=level/reference if reference else None
        # A constant room baseline is descriptive only. AGC/device changes and
        # source direction make it unsuitable as the sole rejection criterion.
        snr=20*math.log10(max(level,1e-7))-noise_db if noise_db is not None else None
        quiet=bool(reference and ratio<.28)
        accepted=not (self.enabled and quiet and similarity<.62 and not explicit)
        return {'accepted':accepted,'source':'background_salience','relative_level':round(ratio,3) if ratio is not None else None,
            'known_voice_similarity':round(similarity,4),'explicit_address':explicit,
            'estimated_level_above_floor_db':round(snr,1) if snr is not None else None,
            'reason':'Very quiet, unmatched speech without an explicit address' if not accepted else 'Open speaker turn; no combined background evidence',
            'confidence_is_calibrated':False}
