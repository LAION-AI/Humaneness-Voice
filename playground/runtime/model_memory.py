"""Keep the Large RL FP32 policy on the TTS GPU, without using another service's GPU."""
import gc
import torch

import settings


def place_auxiliaries(state, model_id):
    """Called only while the server's exclusive generation/switch gate is held.

    Offload the independent ASR, Ears and aligner for Large RL. VoiceCLAP,
    codec and the FP32 policy retain their original GPU implementations.
    All three auxiliaries support CPU inference; scores/formulas stay intact.
    """
    offload = model_id == 'large_rl'
    scorers = state.get('scorers')
    if scorers is not None:
        asr = scorers.asr
        destination = 'cpu' if offload else settings.SCORE_DEVICE
        asr.model.to(destination)
        asr.device = destination
        ears = scorers.ears
        with ears.lock:
            for model, _, _ in ears.models.values(): model.to(destination)
            ears.device = destination
    finisher = state.get('finisher')
    if finisher is not None and finisher.available:
        destination = 'cpu' if offload else settings.ALIGN_DEVICE
        finisher.aligner.model.to(destination)
        finisher.aligner.device = destination
    gc.collect()
    with torch.cuda.device(settings.ACTING_DEVICE): torch.cuda.empty_cache()
    state['auxiliary_placement'] = 'CPU ASR/Ears/alignment; GPU VoiceCLAP' if offload else 'GPU'
