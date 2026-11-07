"""Keep selected TTS weights and latency-critical speech finishing on their GPU."""
import gc
import torch

import settings


def place_auxiliaries(state, model_id):
    """Called only while the server's exclusive generation/switch gate is held.

    Large RL FP32 needs all auxiliaries offloaded. Large 1.7 BF16 has enough
    headroom for the ASR tail check and aligner; keep those fast to avoid gaps
    between spoken sentences. Its optional Ears ranking remains on CPU.
    """
    offload = model_id == 'large_rl'
    ears_offload = model_id in ('large_rl','large17')
    scorers = state.get('scorers')
    if scorers is not None:
        asr = scorers.asr
        destination = 'cpu' if offload else settings.SCORE_DEVICE
        asr.model.to(destination)
        asr.device = destination
        ears = scorers.ears
        ears_destination = 'cpu' if ears_offload else settings.SCORE_DEVICE
        with ears.lock:
            for model, _, _ in ears.models.values(): model.to(ears_destination)
            ears.device = ears_destination
    finisher = state.get('finisher')
    if finisher is not None and finisher.available:
        destination = 'cpu' if offload else settings.ALIGN_DEVICE
        finisher.aligner.model.to(destination)
        finisher.aligner.device = destination
    gc.collect()
    with torch.cuda.device(settings.ACTING_DEVICE): torch.cuda.empty_cache()
    state['auxiliary_placement'] = ('CPU ASR/Ears/alignment; GPU VoiceCLAP' if offload else
        'GPU ASR/alignment/VoiceCLAP; CPU Ears' if ears_offload else 'GPU')
