"""Small, reproducible reference shortlist; never scan/generate every audio.

Read existing VoiceNet/naturalness metadata, shortlist one clip per voice,
then measure Ears Medium on the exact normalized <=15s conditioning audio.
Only deliberate reruns replace the manifest; runtime never re-ranks references.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import settings
import dynamic_references as refs
import config
from ears_reward import EarsModels
from small_app import normalize, encode_mp3

ROOT = settings.HERE / 'fixed-references'


def shortlist():
    out = []
    for voice, profile in refs.profiles().items():
        eligible = []
        source = Path(config.REF3_DIR) / f'meta_{voice}.jsonl'
        for line in source.read_text().splitlines():
            row = json.loads(line)
            if not 10 <= row['dur'] <= 20:
                continue
            if row['block'] == 'emotion':
                allowed = row.get('emotion') in ('Concentration', 'Contentment', 'Contemplation', 'Interest') and row.get('condition') == 'B'
            elif row['block'] == 'voicenet':
                allowed = (row.get('dim') in ('S_CASU', 'S_CONV', 'SMTH', 'WARM', 'CLRT', 'RCQL') and row.get('level') == 'moderately_high') or (row.get('dim') in ('S_DRAM', 'S_RANT', 'S_WHIS', 'TENS', 'AROU') and row.get('level') == 'moderately_low')
            else:
                allowed = False
            if not allowed:
                continue
            vn = json.loads(row['voicenet'])
            if any(vn.get(k, 6) > cap for k, cap in [('AROU', 3.5), ('S_RANT', 3), ('S_DRAM', 3.8), ('S_WHIS', 3)]):
                continue
            if row['spk_sim'] < .28:
                continue
            merit = row['genuineness'] + .6*vn['RCQL'] + .4*vn['CLRT'] + .3*vn['S_CONV'] - .4*vn['S_RANT'] - .3*vn['S_DRAM'] - .3*vn['S_WHIS']
            eligible.append((merit, row, vn))
        if eligible:
            merit, row, vn = max(eligible, key=lambda x: x[0])
            path = (Path(config.REF3_DIR) / row['path']).resolve()
            assert path.is_relative_to(Path(config.REF3_DIR).resolve()) and path.is_file()
            out.append({'voice': voice, 'name': profile['name'], 'source': str(path),
                'key': row['key'], 'source_seconds': row['dur'], 'preselection_merit': merit,
                'preselection_count': len(eligible), 'existing_genuineness': row['genuineness'],
                'existing_speaker_similarity': row['spk_sim'],
                'voicenet': {k: vn[k] for k in ('AROU','S_RANT','S_DRAM','S_WHIS','S_CONV','RCQL','CLRT')},
                'description': 'Low-arousal conversational/clear speech, selected with existing proxy filters.'})
        else:
            out.append({'voice': voice, 'name': profile['name'], 'source': profile['anchor'],
                'key': 'original-profile-anchor', 'source_seconds': sf.info(profile['anchor']).duration,
                'preselection_count': 0, 'description': 'Original fixed profile anchor; no longer clip passed the conservative filters.'})
    return out


def main():
    import torch
    torch.set_num_threads(4)
    ROOT.mkdir(exist_ok=True)
    (ROOT/'audio').mkdir(exist_ok=True)
    candidates = shortlist()
    assert len(candidates) == 10
    waves = []
    for item in candidates:
        wav = ROOT/'audio'/f"{item['voice']}.wav"
        item['seconds'] = normalize(Path(item['source']).read_bytes(), wav)
        encode_mp3(wav, wav.with_suffix('.mp3'))
        item['wav'] = str(wav.relative_to(settings.HERE))
        item['mp3'] = str(wav.with_suffix('.mp3').relative_to(settings.HERE))
        item['sha256'] = hashlib.sha256(wav.read_bytes()).hexdigest()
        wave, rate = sf.read(wav, dtype='float32')
        waves.append(resample_poly(wave, 1, 3).astype(np.float32))
        assert rate == 48000 and .24 <= item['seconds'] <= 15.001
    model = EarsModels()
    reports = model.predict(waves, 'medium', 'ce')
    for item, report in zip(candidates, reports):
        item['ears'] = report
        print(json.dumps({'name': item['name'], 'seconds': item['seconds'], 'ce': report['raw'],
            'pq': report['scores_raw']['audiobox:PQ'], 'key': item['key']}), flush=True)
    default = next(x for x in candidates if x['voice'] == settings.DEFAULT_VOICE)
    # Keep the requested Selma default; promote five other distinct voices by CE.
    selected = [default] + sorted((x for x in candidates if x is not default),
        key=lambda x: x['ears']['raw'], reverse=True)[:5]
    manifest = {'version': 1, 'date': '2026-10-09', 'reference_cap_seconds': 15,
        'default': settings.DEFAULT_VOICE, 'selected': [x['voice'] for x in selected],
        'method': 'One metadata-filtered candidate per existing voice, ten Ears Medium evaluations. Preserve Selma; add top five other CE scores. No exhaustive audio scan, no human listening study, no TTS experiment. Conditioning audio is normalized/capped before scoring. Never pad or loop short anchors.',
        'limitations': 'CE, naturalness, style and historical speaker metrics are model proxies; neutral delivery and cloning are not guaranteed. Source clips may be synthetic. Selection is not a validation of CFG2 with these six references.',
        'candidates': candidates}
    assert all(math.isfinite(x['ears']['raw']) for x in candidates)
    (ROOT/'selection.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False))
    print('Selected:', ', '.join(x['name'] for x in selected), flush=True)


if __name__ == '__main__':
    main()
