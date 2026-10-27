"""Single-flight loopback GPU sound captioner. No microphone files are written.

The request is optional: overload returns 503, never queues stale observations.
Inference continues safely to completion if a caller disconnects; its slot remains
reserved until the CUDA job really finishes. Main ASR never awaits this worker.
"""
import asyncio
import os
import re
import time
import numpy as np
from aiohttp import web
from bootstrap_sound_caption import assets, REPO, REVISION, PROCESSOR_REVISION

class Captioner:
    def __init__(self):
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        torch.set_num_threads(2)
        self.device = os.environ.get('ROBO_SOUND_DEVICE', 'cuda:0')
        self.precision = os.environ.get('ROBO_SOUND_PRECISION', 'float16')
        self.dtype = getattr(torch, self.precision) if self.device.startswith('cuda') else torch.float32
        model_path, processor_path = assets()
        self.processor = WhisperProcessor.from_pretrained(processor_path, local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path,
            local_files_only=True, dtype=self.dtype, attn_implementation='sdpa').eval().to(self.device)
        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.language = None
        self.model.generation_config.task = None
        self.max_tokens = int(os.environ.get('ROBO_SOUND_MAX_TOKENS', '48'))
        self.predict(np.zeros(16000, np.float32))

    def predict(self, wave):
        import torch
        start = time.perf_counter()
        feats = self.processor.feature_extractor(wave, sampling_rate=16000,
            return_tensors='pt').input_features.to(self.device, dtype=self.dtype)
        prep = time.perf_counter()
        with torch.inference_mode():
            # Call the base GenerationMixin: this fine-tune is an audio captioner,
            # not Whisper's ASR language/timestamp-segmentation pipeline.
            from transformers.generation.utils import GenerationMixin
            from transformers import StoppingCriteria, StoppingCriteriaList
            processor=self.processor
            class TwoSentences(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    value=processor.decode(input_ids[0],skip_special_tokens=True)
                    return bool(len(re.findall(r'[.!?](?:\s|$)',value))>=2)
            events=[torch.cuda.Event(enable_timing=True) for _ in range(3)] if self.device.startswith('cuda') else None
            if events:events[0].record()
            encoded=self.model.get_encoder()(feats)
            if events:events[1].record()
            ids = GenerationMixin.generate(self.model, encoder_outputs=encoded,
                decoder_input_ids=torch.tensor([[self.model.config.decoder_start_token_id]], device=self.device),
                max_new_tokens=self.max_tokens, num_beams=1, do_sample=False,
                use_cache=True,stopping_criteria=StoppingCriteriaList([TwoSentences()]))
            if events:events[2].record();events[2].synchronize()
        raw = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        # Prefer completed sentences to a token-budget-truncated clause.
        sentences = re.findall(r'[^.!?]+[.!?]', raw)
        caption = ' '.join(s.strip() for s in sentences[:2]) if sentences else raw
        return {'status': 'available', 'caption': caption[:450],
            'uncertainty': 'Fallible sound hypothesis, not a transcript or confirmed event.',
            'window_s': round(len(wave)/16000, 3), 'tokens': int(ids.shape[-1]-1),
            'preprocess_ms': round((prep-start)*1000, 2),
            'encoder_gpu_ms': round(events[0].elapsed_time(events[1]),2) if events else None,
            'decoder_gpu_ms': round(events[1].elapsed_time(events[2]),2) if events else None,
            'inference_ms': round((time.perf_counter()-prep)*1000, 2),
            'total_ms': round((time.perf_counter()-start)*1000, 2),
            'model': REPO, 'revision': REVISION, 'device': self.device, 'precision': self.precision}

async def caption(request):
    if request.app['job'] and not request.app['job'].done():
        raise web.HTTPServiceUnavailable(text='Optional sound worker busy')
    raw = await request.read()
    if len(raw)%4: raise web.HTTPBadRequest(text='Expected little-endian float32 PCM')
    wave = np.frombuffer(raw, dtype='<f4')
    if not 8000 <= len(wave) <= 96000 or not np.isfinite(wave).all():
        raise web.HTTPBadRequest(text='Expected finite 16-kHz mono audio, 0.5–6 seconds')
    if np.max(np.abs(wave))>1.01: raise web.HTTPBadRequest(text='PCM must be normalized')
    # Claim the slot before yielding; two completed HTTP bodies cannot race it.
    if request.app['job'] and not request.app['job'].done():
        raise web.HTTPServiceUnavailable(text='Optional sound worker busy')
    task = asyncio.create_task(asyncio.to_thread(request.app['captioner'].predict, wave.copy()))
    request.app['job'] = task
    return web.json_response(await asyncio.shield(task))

async def startup(app): app['captioner'] = await asyncio.to_thread(Captioner)
async def state(request):
    c = request.app['captioner']
    return web.json_response({'ready': True, 'busy': bool(request.app['job'] and not request.app['job'].done()),
        'model': REPO, 'revision': REVISION, 'processor_revision': PROCESSOR_REVISION,
        'device': c.device, 'precision': c.precision, 'max_tokens': c.max_tokens,
        'recordings_saved': False})

def main():
    app = web.Application(client_max_size=96000*4+1); app['job'] = None
    app.on_startup.append(startup)
    app.router.add_get('/state', state); app.router.add_post('/caption', caption)
    web.run_app(app, host='127.0.0.1', port=int(os.environ.get('ROBO_SOUND_PORT', '8803')))

if __name__ == '__main__': main()
