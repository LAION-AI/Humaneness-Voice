"""Fixed, measured references by default; legacy retrieval for explicit studies.

Uses the already-loaded commercial VoiceCLAP text tower and original centered
condition centroids, not old speaker LoRAs or spoken-text semantic matching.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import settings
sys.path.append(str(settings.PARENT))
import config
import voice_profiles
from retrieval import split_direction, looks_german


@lru_cache(maxsize=1)
def profiles():
    return voice_profiles.discover()


@lru_cache(maxsize=1)
def fixed_manifest():
    data = json.loads((settings.HERE / 'fixed-references/selection.json').read_text())
    if len(data['selected']) != 6 or len(set(data['selected'])) != 6:
        raise ValueError('Expected six distinct fixed reference presets')
    for item in data['candidates']:
        for key in ('wav', 'mp3'):
            path = (settings.HERE / item[key]).resolve()
            if not path.is_relative_to((settings.HERE / 'fixed-references/audio').resolve()) or not path.is_file():
                raise ValueError('Missing or invalid fixed reference file')
        if hashlib.sha256((settings.HERE / item['wav']).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Fixed reference changed; rebuild the manifest deliberately')
    return data


def fixed_clip(voice, playback=False):
    if voice not in profiles():
        raise ValueError('Unknown reference voice')
    item = next(x for x in fixed_manifest()['candidates'] if x['voice'] == voice)
    return settings.HERE / item['mp3' if playback else 'wav'], item


@lru_cache(maxsize=10)
def clips(voice):
    if voice not in profiles():
        raise ValueError('Unknown reference voice')
    out = {}
    for root, name in ((Path(config.REF3_DIR), f'index_{voice}.json'),
                       (Path(config.REF_DIR), f'index_profile_{voice}.json')):
        file = root / name
        if not file.is_file():
            continue
        for gid, candidates in json.loads(file.read_text()).items():
            if not gid.startswith(voice + '|') or gid in out:
                continue
            for relative in candidates if isinstance(candidates, list) else [candidates]:
                path = (root / relative).resolve()
                if path.is_relative_to(root.resolve()) and path.is_file():
                    out[gid] = path
                    break
    return out


def catalog():
    data = fixed_manifest()
    selected = data['selected']
    order = selected + [x for x in profiles() if x not in selected]
    voices = []
    for id_ in order:
        p = profiles()[id_]
        _, item = fixed_clip(id_)
        voices.append({key: p[key] for key in ('id', 'name', 'short', 'identity', 'accent')} | {
            'dynamic': False, 'dynamic_available': bool(clips(id_)), 'recommended': id_ in selected, 'reference_seconds': round(item['seconds'], 2),
            'content_enjoyment': round(item['ears']['raw'], 3), 'reference_sha256': item['sha256'],
            'preview_url': f'/api/voices/reference/{id_}', 'reference_policy': 'fixed'})
    return {'default': settings.DEFAULT_VOICE, 'voices': voices, 'presets': selected,
            'retriever': 'VoiceCLAP commercial (optional)', 'reference_policy': 'fixed', 'reference_seconds': settings.REFERENCE_SECONDS,
            'score_note': 'Humaneness Ears Medium Content Enjoyment proxy, not a human rating.'}


def choose(retriever, spec, voice, *, fixed=True):
    if voice not in profiles():
        raise ValueError('Unknown reference voice')
    if fixed:
        path, item = fixed_clip(voice)
        return path, {'mode': 'profile', 'voice': voice, 'name': item['name'],
            'language': spec['lang'], 'reference_language': 'de' if item['key'] != 'original-profile-anchor' else 'original',
            'dynamic': False, 'policy': 'fixed', 'retriever': None, 'fallback': None,
            'gid': item['key'], 'reference_sha256': item['sha256'],
            'seconds': item['seconds'], 'content_enjoyment': item['ears']['raw'],
            'note': 'Fixed clip of this speaker; unchanged across emotion, language and direction.'}
    profile = profiles()[voice]
    general, cues = split_direction(spec.get('general', ''), spec.get('tagged', ''))
    direction = ' '.join(x for x in (general, cues) if x).strip()
    from prompting import performance_direction
    direction=performance_direction(direction)
    # The old text tower is English. Do not rank English embeddings against
    # German acting cues; use the same speaker's anchor and disclose fallback.
    fallback = ('No English delivery instruction' if not direction else
                'VoiceCLAP text encoder expects English directions' if looks_german(direction) else
                'No matching catalog reference')
    report = {'mode': 'profile', 'voice': voice, 'name': profile['name'],
              'query': direction, 'language': spec['lang'], 'retriever': 'VoiceCLAP commercial',
              'dynamic': False, 'fallback': fallback}
    paths = clips(voice)
    if direction and not looks_german(direction) and retriever is not None and retriever.ok:
        indices = [i for i in retriever.by_voice.get(voice, [])
            if retriever.cond[i]['voice'] == voice and retriever.cond[i]['lang'] == spec['lang']
            and retriever.cond[i]['block'] not in ('burst_isolated', 'character')
            and retriever.cond[i]['gid'] in paths]
        if indices:
            with retriever.lock:
                q = retriever.embed(direction) - retriever.qmu
                q /= max(float(np.linalg.norm(q)), 1e-8)
                eq = retriever.embed(cues) - retriever.qmu if cues else q.copy()
                eq /= max(float(np.linalg.norm(eq)), 1e-8)
            top_emotion = retriever.emotions[int(np.argmax(retriever.A @ eq))] if len(retriever.emotions) else ''
            similarity = retriever.C[indices] @ q
            scores = similarity + np.array([config.RETRIEVAL_EMO_BONUS
                if retriever.cond[i]['emotion'] == top_emotion else 0. for i in indices])
            intensity = 'A' if any(word in direction.lower() for word in ('intense', 'extreme', 'unrestrained', 'overwhelming')) else (
                'B' if any(word in direction.lower() for word in ('restrained', 'contained', 'subtle', 'quiet')) else None)
            scores += np.array([-config.RETRIEVAL_LEVEL_PENALTY
                if intensity and retriever.cond[i].get('condition') in ('A','B')
                and retriever.cond[i]['condition'] != intensity else 0. for i in indices])
            best = int(np.argmax(scores))
            condition = retriever.cond[indices[best]]
            report.update(dynamic=True, policy='dynamic', fallback=None, gid=condition['gid'],
                emotion=condition['emotion'], dimension=condition.get('dim'), level=condition.get('level') or condition.get('condition'),
                audio_similarity=float(similarity[best]), retrieval_score=float(scores[best]))
            return paths[condition['gid']], report
    path,item=fixed_clip(voice)
    report.update(policy='fixed_fallback',gid=item['key'],seconds=item['seconds'],
                  reference_sha256=item['sha256'])
    return path, report
