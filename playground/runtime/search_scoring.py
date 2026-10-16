"""Batched, prefix-aware scoring for causal chunk search; no adapters/API calls."""
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tokenizers.decoders import DecodeStream

from metrics import tokens, word_errors


def prefix_errors(reference, hypothesis, final=False):
    """Free unspoken target suffix, but not internal deletion/substitution errors.

    Choose the target-prefix endpoint with smallest *edit count*, not smallest
    WER (the latter can reward declaring more nonexistent progress).
    Empty speech has gate zero. This is a search surrogate, not final WER.
    """
    if final:
        result = word_errors(reference, hypothesis)
        return {**result, 'matched_prefix_words': result['reference_words'], 'prefix_wer': result['wer']}
    ref, hyp = tokens(reference), tokens(hypothesis)
    if not hyp:
        return {'prefix_wer': 1., 'matched_prefix_words': 0, 'reference_words': len(ref)}
    candidates = [(word_errors(' '.join(ref[:i]), ' '.join(hyp)), i) for i in range(1, len(ref)+1)]
    if not candidates:
        return {'prefix_wer': 1., 'matched_prefix_words': 0, 'reference_words': 0}
    result, endpoint = min(candidates, key=lambda item: (
        item[0]['substitutions']+item[0]['deletions']+item[0]['insertions'],
        abs(item[1]-len(hyp)), -item[1]))
    return {**result, 'matched_prefix_words': endpoint, 'prefix_wer': result['wer']}


def search_reward(score):
    # Fixed scales preserve comparability over time and across beam parents.
    clap = float(np.clip((score['clap']+1)/2, 0, 1))
    genuine = float(np.clip(score['genuineness']/6, 0, 1))
    blend = float(np.clip(score['blend']/10, 0, 1))
    quality = .5*clap + .25*genuine + .25*blend
    gate = max(0., 1-min(1., max(0., score['prefix_wer'])))
    return {'reward': quality*gate, 'quality': quality, 'gate': gate,
            'n_clap': clap, 'n_genuineness': genuine, 'n_blend': blend}


def token_spans_to_words(spans):
    """Merge incremental SentencePiece decode pieces into timestamped words."""
    words = []
    current = None
    for piece in spans:
        text = piece['token']
        for match in re.finditer(r'\s+|\S+', text):
            fragment = match.group()
            if fragment.isspace():
                if current is not None:
                    words.append(current)
                    current = None
            else:
                if current is None:
                    current = {'word': '', 'start': piece['start'], 'end': piece['end']}
                current['word'] += fragment
                current['end'] = max(current['end'], piece['end'])
    if current is not None:
        words.append(current)
    return words


def style_query(spec):
    from bestofn import direction_text
    text = direction_text(spec.get('general',''), spec.get('tagged',''))
    text = text.replace(spec.get('text','') or '\0', '')
    text = re.sub(r'TRANSCRIPT\s*:.*', '', text, flags=re.I|re.S)
    text = re.sub(r'["“][^"”]*["”]', '', text)
    return ' '.join(text.split())


class SearchScorer:
    """Reuse loaded Parakeet + commercial VoiceCLAP tower and both small heads.

    All candidates share one batched audio embedding pass, replacing the old
    three encodes per candidate. No WAV temp files or HTTP scorer services.
    Genuineness uses raw unit embeddings, blend uses their L2 normalization;
    these follow the shipped heads' preprocessing exactly.
    """
    def __init__(self, scorers, direction, reward_plan=None, style_share=.5, burst_options=None, speaker_reward=None, ranking=None):
        self.scorers = scorers
        self.asr, self.retriever, self.judge = scorers.asr, scorers.retriever, scorers.judge
        self.device = self.retriever.device
        # Never fall back to the literal transcript as a style query.
        spec = direction
        direction = style_query(direction)
        self.direction = direction
        self.reward_plan = reward_plan
        self.style_share = style_share
        self.speaker_reward = speaker_reward
        self.ranking = ranking if ranking is not None and ranking.mode!='legacy' else None
        if self.ranking:
            if reward_plan or speaker_reward or (burst_options and burst_options.mode!='off'):
                raise ValueError('Ears ranking cannot include hidden adaptive/speaker/burst weights')
            scorers.ears.load(self.ranking.model)
        self.want = self.judge.want_vector(direction, '') if direction and reward_plan is None else None
        if reward_plan is not None:
            self.dimensions = scorers.dimensions
            self.dimensions.prepare(reward_plan)
        self.burst_reward = None
        if burst_options is not None and burst_options.mode!='off':
            from burst_reward import BurstReward
            self.burst_reward = BurstReward(scorers.bursts,spec,burst_options)
        self.last_timing = {}

    @torch.inference_mode()
    def transcribe_batch(self, audio16):
        asr = self.asr
        with asr.lock:
            inp = asr.proc([w.float().cpu().numpy() for w in audio16], sampling_rate=16000,
                           padding=True, return_tensors='pt')
            inp = {k: v.to(asr.device, dtype=torch.bfloat16) if v.dtype.is_floating_point else v.to(asr.device)
                   for k, v in inp.items()}
            output = asr.model.generate(**inp, max_new_tokens=2048)
            ids, durations = output.sequences.cpu(), output.durations.cpu()
            texts = asr.proc.batch_decode(ids, skip_special_tokens=True)
            offsets = durations.cumsum(-1)-durations
            from transformers.models.parakeet.processing_parakeet import ParakeetProcessorKwargs
            audio_kwargs = asr.proc._merge_kwargs(ParakeetProcessorKwargs,
                tokenizer_init_kwargs=asr.proc.tokenizer.init_kwargs)['audio_kwargs']
            frame_rate = asr.proc.feature_extractor.hop_length / 16000 * audio_kwargs['subsampling_factor']
            all_words = []
            for row, row_starts, row_durations in zip(ids, offsets, durations):
                stream = DecodeStream(skip_special_tokens=True)
                spans = []
                for index, token in enumerate(row):
                    if int(token) in {asr.proc.tokenizer.pad_token_id, asr.proc.blank_token_id}:
                        continue
                    piece = stream.step(asr.proc.tokenizer._tokenizer, int(token))
                    if piece is not None:
                        spans.append({'token':piece, 'start':float(row_starts[index])*frame_rate,
                            'end':float(row_starts[index]+row_durations[index])*frame_rate})
                all_words.append(token_spans_to_words(spans))
        return [re.sub(r'\s+', ' ', str(t).replace('<blank>', ' ')).strip() for t in texts], all_words

    @torch.inference_mode()
    def score(self, waves, text, final=None):
        started = time.perf_counter()
        final = final or [False]*len(waves)
        audio16 = [torchaudio.functional.resample(torch.as_tensor(w if len(w) else np.zeros(4800), dtype=torch.float32, device=self.device),
                    48000, 16000) for w in waves]
        # A completely stopped/empty row still gets a minimal ASR buffer.
        audio16 = [F.pad(w, (0, max(0,1600-len(w)))) for w in audio16]
        texts, timestamps = self.transcribe_batch(audio16)
        asr_end = time.perf_counter()
        batch = torch.stack([F.pad(w, (0, max(len(x) for x in audio16)-len(w))) for w in audio16])
        embeddings = self.retriever.model.encode_waveform(batch, sample_rate=16000).float()
        genuine = self.judge.gen._predict(embeddings)
        unit = F.normalize(embeddings, dim=-1)
        blend = self.judge.bl.mlp((unit-self.judge.bl.mu)/self.judge.bl.sd).reshape(-1).clamp(0,10)
        centered = F.normalize(embeddings-torch.as_tensor(self.retriever.qmu, device=self.device), dim=-1)
        clap = centered @ torch.as_tensor(self.want, device=self.device) if self.want is not None else torch.zeros(len(waves),device=self.device)
        genuine, blend, clap = genuine.cpu().tolist(), blend.cpu().tolist(), clap.cpu().tolist()
        raw_dimensions = self.dimensions.predict(audio16, embeddings, self.reward_plan) if self.reward_plan else None
        ears_reports = self.scorers.ears.predict(audio16,self.ranking.model,self.ranking.metric) if self.ranking else None
        speaker_reports = None
        if self.speaker_reward:
            window=getattr(self.speaker_reward,'window_seconds',None)
            speech=[bool(t.strip()) and (not window or any(
                word['end']>len(wave)/48000-window and word['start']<len(wave)/48000
                for word in stamps)) for t,wave,stamps in zip(texts,waves,timestamps)]
            speaker_reports=self.speaker_reward.score(audio16,embeddings,speech)
        burst_reports = self.burst_reward.score(audio16,timestamps,final) if self.burst_reward else None
        scored = []
        for i, wave in enumerate(waves):
            seconds = len(wave)/48000
            # At intermediate cuts, ignore words too close to the right edge.
            # Final EOS/budget endpoints use full transcript and full WER.
            stable = ' '.join(x['word'] for x in timestamps[i] if x['end'] <= seconds-.12)
            hypothesis = texts[i] if final[i] else stable
            item = {'hyp':texts[i], 'stable_hyp':hypothesis, 'timestamps':timestamps[i],
                    'sec':seconds, 'clap':clap[i], 'genuineness':genuine[i], 'blend':blend[i],
                    **prefix_errors(text, hypothesis, final[i])}
            item.update(search_reward(item))
            if self.reward_plan:
                from adaptive_reward import adaptive_quality
                item.update(adaptive_quality(raw_dimensions[i],self.reward_plan,genuine[i],blend[i],self.style_share))
                item['reward'] = item['quality']*item['gate']
                item['reward_model'] = 'adaptive_dimensions'
            if speaker_reports is not None:
                from speaker_reward import combine_identity
                combine_identity(item, speaker_reports[i], self.speaker_reward.options.weight, bool(self.reward_plan))
            if burst_reports is not None:
                report=burst_reports[i]
                item['burst_alignment']=report
                item['base_quality']=item['quality']
                if report['score'] is not None:
                    weight=self.burst_reward.options.weight
                    item['quality']=(1-weight)*item['quality']+weight*report['score']
                # Intentional leading non-speech must not fail ASR before
                # speech is due. Never relax final full-transcript WER.
                if report['speech_not_due'] and not hypothesis:
                    item['gate']=1.
                    item['gate_note']='Requested leading non-speech; speech is not due yet'
                item['reward']=item['quality']*item['gate']
            if ears_reports is not None:
                from ears_reward import combine_ears
                report=ears_reports[i]
                if not len(wave):report={**report,'available':False,'raw':None,'normalized':None}
                combine_ears(item,report,self.ranking)
            scored.append(item)
        self.last_timing = {'asr_seconds':asr_end-started, 'quality_seconds':time.perf_counter()-asr_end,
                            'total_seconds':time.perf_counter()-started}
        if self.reward_plan:
            self.last_timing.update(self.dimensions.last_timing)
        if self.burst_reward:
            self.last_timing.update(self.burst_reward.last_timing)
        if self.speaker_reward:
            self.last_timing.update(self.speaker_reward.last_timing)
        if self.ranking:
            self.last_timing.update(self.scorers.ears.last_timing)
        return scored
