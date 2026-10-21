"""Bounded, reproducible public-synthetic TYGER V2 pilot preparation.

Never downloads a complete ~2 GB tar: only explicit HTTP byte ranges. Camera,
microphone and enrolled-user data are not inputs to this script.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

import numpy as np
import requests
from scipy.signal import fftconvolve, butter, sosfilt

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get('ROBO_PILOT_CACHE', str(Path.home()/'.cache'/'robo-pilot')))
REPO = 'laion/laion-voice-profiles-annotated'
REVISION = '88ea857d6a6f20839674dfde30936d8fd6b23c62'
TEACHER_REVISION = '0f99f2d0ebe89ac095bcc5903c4dd8f72b367286'
BASE = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/'
SR = 16000


def guard():
    for root in (CACHE, HERE):
        root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free < 5 * 1024**3:
            raise RuntimeError(f'Less than 5 GiB free on {root}; stop without deleting user files')
    total = sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file())
    if total > 1024**3:
        raise RuntimeError('Pilot source-cache hard budget 1 GiB reached')


def get(url, **kwargs):
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=90, **kwargs)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def pcm(mp3):
    result = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', 'pipe:0',
        '-ac', '1', '-ar', str(SR), '-f', 'f32le', 'pipe:1'], input=mp3,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return np.frombuffer(result.stdout, '<f4').copy()


def profile_plan(train=24, val=4, test=4):
    raw = get(BASE + 'stats/stats_profiles.csv').content
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    voices = sorted({r['voice'] for r in rows if r['origin'] == 'original'})
    if len(voices) != 500:
        raise ValueError('Expected published 500 identities')
    ordered = sorted(voices, key=lambda v: hashlib.sha256(('tyger-v2-split-v1:' + v).encode()).hexdigest())
    split = {v: 'validation' if i < 10 else 'test' if i < 20 else 'train'
             for i, v in enumerate(ordered)}
    selected = []
    for name, count in [('train', train), ('validation', val), ('test', test)]:
        candidates = [v for v in ordered if split[v] == name]
        selected += [{'voice': v, 'split': name, 'catalog_index': voices.index(v)}
                     for v in candidates[:count]]
    for i, row in enumerate(selected):
        row['speaker_idx'] = i
    atomic_json(HERE / 'pilot' / 'global_split.json', {
        'scheme': 'sha256(tyger-v2-split-v1:voice), first10 validation, next10 test, remaining480 train',
        'counts': {s: sum(v == s for v in split.values()) for s in set(split.values())},
        'speakers': split, 'selected': selected})
    return selected


def source_voice(row):
    """Read small prefixes in two acting-condition shards; source bytes cached only if selected."""
    guard()
    out = CACHE / 'sources' / row['voice']
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'clips.json'
    if manifest.exists():
        return {**row, 'clips': json.loads(manifest.read_text())}
    choices = []
    for local_shard in (1, 2):
        shard = row['catalog_index'] * 4 + local_shard
        path = f'data/origin=original/vprof_base-{shard:05d}.tar'
        # Query contains range to prevent a CDN from reusing a different partial response.
        length = 8 * 1024**2
        with requests.get(BASE + path + f'?pilot_range=0-{length-1}',
                headers={'Range': f'bytes=0-{length-1}'}, stream=True, timeout=90) as response:
            if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith('bytes 0-'):
                raise RuntimeError('Server did not honor bounded Range; refuse full tar download')
            content = response.raw.read(length)
        tf = tarfile.open(fileobj=io.BytesIO(content), mode='r:')
        audio = {}
        try:
            for member in tf:
                if member.offset_data + member.size > len(content):
                    break
                if member.name.endswith('.mp3'):
                    audio[member.name[:-4]] = tf.extractfile(member).read()
                elif member.name.endswith('.json'):
                    record = json.load(tf.extractfile(member))
                    uid = member.name[:-5]
                    if uid not in audio:
                        continue
                    if not uid.startswith(row['voice'] + '__'):
                        raise ValueError(f'Catalog shard mapping mismatch for {row["voice"]}: {uid}')
                    meta = record.get('src_meta', {})
                    quality = record.get('quality', {})
                    if not (6 <= record.get('dur_s', 0) <= 25 and len(record.get('words', [])) >= 8):
                        continue
                    if float(meta.get('wer', 1)) > .3 or record.get('n_bursts', 0) > 1:
                        continue
                    ranking = float(quality.get('overall_quality', 0)) + float(meta.get('spk_sim', 0))
                    choices.append({'uid': uid, 'mp3': audio.pop(uid), 'record': record,
                        'source_tar': path, 'source_range_bytes': length, 'ranking': ranking,
                        'condition_shard': local_shard})
        except (tarfile.ReadError, EOFError):
            pass  # The bounded prefix intentionally ends before the archive does.
    # Two clean long enrollment clips and four different prediction clips, spanning both shards.
    selected = []
    for shard in (1, 2):
        pool = sorted([x for x in choices if x['condition_shard'] == shard],
                      key=lambda x: (-x['ranking'], -x['record']['dur_s'], x['uid']))
        selected += pool[:3]
    if len(selected) < 4:
        raise RuntimeError(f'Insufficient usable clips in bounded prefixes for {row["voice"]}: {len(selected)}')
    clips = []
    # Reserve one enrollment utterance from each condition; never use it in a soundscape.
    for i, x in enumerate(selected):
        name = hashlib.sha256(x['uid'].encode()).hexdigest()[:16] + '.mp3'
        (out / name).write_bytes(x['mp3'])
        record = x['record']
        clips.append({'uid': x['uid'], 'path': str(out / name),
            'role': 'teacher' if i in (0, 3) else 'prediction', 'duration_s': record['dur_s'],
            'words': record['words'], 'bursts': record.get('bursts'),
            'lang': record.get('lang'), 'condition': x['uid'].split('__')[2:-1],
            'source_tar': x['source_tar'], 'downloaded_range_bytes': x['source_range_bytes'],
            'sha256': hashlib.sha256(x['mp3']).hexdigest(),
            'published_wer': record.get('src_meta', {}).get('wer'),
            'published_spk_sim': record.get('src_meta', {}).get('spk_sim')})
    atomic_json(manifest, clips)
    print(json.dumps({'source_ready': row['voice'], 'split': row['split'], 'clips': len(clips)}), flush=True)
    return {**row, 'clips': clips}


def teachers(speakers, output_root=None):
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    torch.set_num_threads(6)
    model_root = os.environ.get('ROBO_ECAPA_ROOT',
        '/mnt/nvme/hf_cache/hub/models--speechbrain--spkrec-ecapa-voxceleb/snapshots/' + TEACHER_REVISION)
    if not Path(model_root).exists():
        from huggingface_hub import snapshot_download
        model_root = snapshot_download('speechbrain/spkrec-ecapa-voxceleb',
            revision=TEACHER_REVISION, cache_dir=str(CACHE/'hub'), token=False,
            allow_patterns=['hyperparams.yaml', '*.ckpt', 'label_encoder.txt'])
    model = EncoderClassifier.from_hparams(source=model_root, overrides={'pretrained_path': model_root},
        savedir=str(CACHE / 'ecapa'), run_opts={'device': 'cpu'})
    output_root = Path(output_root or HERE/'pilot')
    output_root.mkdir(parents=True, exist_ok=True)
    vectors, details, probe_vectors, probe_speakers, probe_uids = [], [], [], [], []
    with torch.inference_mode():
        for row in speakers:
            v = []
            for clip in row['clips']:
                audio = pcm(Path(clip['path']).read_bytes())
                # Teacher sees complete long clean utterances, not 320-ms regression windows.
                e = model.encode_batch(torch.from_numpy(audio)[None]).reshape(192).float().numpy()
                e /= max(np.linalg.norm(e), 1e-8)
                if clip['role'] == 'teacher':
                    v.append(e)
                else:
                    probe_vectors.append(e)
                    probe_speakers.append(row['speaker_idx'])
                    probe_uids.append(clip['uid'])
            avg = np.mean(v, axis=0)
            avg /= max(np.linalg.norm(avg), 1e-8)
            vectors.append(avg)
            details.append({'speaker_idx': row['speaker_idx'], 'voice': row['voice'],
                'teacher_pair_cosine': float(np.dot(v[0], v[1])) if len(v) > 1 else None,
                'teacher_utterances': len(v),
                'teacher_seconds': sum(c['duration_s'] for c in row['clips'] if c['role'] == 'teacher')})
            print(json.dumps({'teacher_ready': row['voice'], **details[-1]}), flush=True)
        # Fixed public profile reference only: never user-uploaded or microphone audio.
        public_url = 'http://127.0.0.1:8792/api/voices/reference/emolia_c1699'
        reference = get(public_url).content
        audio = pcm(reference)
        e = model.encode_batch(torch.from_numpy(audio)[None]).reshape(192).float().numpy()
        e /= max(np.linalg.norm(e), 1e-8)
        artifact = HERE / 'artifacts' / 'robo_reference.npz'
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(artifact, embedding=e, source_url=public_url,
            sha256=hashlib.sha256(reference).hexdigest(), voice='emolia_c1699',
            duration_s=len(audio)/SR, teacher_revision=TEACHER_REVISION)
    np.save(output_root / 'teacher_embeddings.npy', np.stack(vectors).astype(np.float32))
    np.savez_compressed(output_root / 'probe_teacher_embeddings.npz',
        embeddings=np.stack(probe_vectors).astype(np.float32),
        speaker_idx=np.asarray(probe_speakers, np.int16), uid=np.asarray(probe_uids))
    atomic_json(output_root / 'teacher_quality.json', details)


def source_signal(clip):
    audio = pcm(Path(clip['path']).read_bytes())
    active = np.zeros(len(audio), bool)
    for w in clip['words']:
        start = max(0, int((w['s'] - .035) * SR))
        end = min(len(audio), int((w['e'] + .06) * SR))
        active[start:end] = True
    return audio, active


def conversation(speakers, anchor, seed, duration=24):
    rng = np.random.default_rng(seed)
    same_split = [s for s in speakers if s['split'] == anchor['split']]
    count = int(rng.integers(1, min(5, len(same_split)) + 1))
    others = [x for x in same_split if x['speaker_idx'] != anchor['speaker_idx']]
    selected = [anchor] + list(rng.choice(others, size=min(count-1, len(others)), replace=False))
    n = duration * SR
    tracks = [np.zeros(n, np.float32) for _ in selected]
    active_tracks = [np.zeros(n, bool) for _ in selected]
    events = []
    # One conversation clock: mostly alternating turns, with genuine interruptions
    # rather than placing every speaker continuously on top of every other one.
    t = float(rng.uniform(.1, 1.3))
    for turn in range(12):
        who = turn % len(selected) if turn < len(selected) else int(rng.integers(len(selected)))
        person = selected[who]
        clips = [x for x in person['clips'] if x['role'] == 'prediction']
        clip = clips[int(rng.integers(len(clips)))]
        audio, speech = source_signal(clip)
        length = min(len(audio), int(rng.uniform(2.0, 4.5) * SR))
        src = int(rng.integers(max(1, len(audio) - length + 1)))
        offset = int(t * SR)
        take = min(length, n - offset)
        if take <= 0:
            break
        gain = float(rng.uniform(.6, 1.1))
        tracks[who][offset:offset+take] += audio[src:src+take] * gain
        active_tracks[who][offset:offset+take] |= speech[src:src+take]
        events.append({'speaker_idx': person['speaker_idx'], 'uid': clip['uid'],
            'onset_s': offset/SR, 'offset_s': (offset+take)/SR, 'source_start_s': src/SR})
        # All boundaries intentionally sample arbitrary positions, not 320-ms multiples.
        t += take / SR + float(rng.uniform(-.4, .9))
    clean = np.sum(tracks, axis=0)
    scenario = ['clean', 'noise', 'reverb', 'telephone', 'echo'][seed % 5]
    audio = clean.copy()
    if scenario == 'noise':
        noise = rng.normal(size=n).astype(np.float32)
        noise = sosfilt(butter(2, 3000, fs=SR, output='sos'), noise)
        rms = max(float(np.sqrt(np.mean(clean**2))), .01)
        snr_db = float(rng.uniform(8, 25))
        audio += noise / max(np.std(noise), 1e-8) * rms * 10 ** (-snr_db/20)
    elif scenario == 'reverb':
        ir = np.zeros(int(.35*SR), np.float32)
        ir[0] = 1
        for delay, gain in [(.017,.25),(.043,.18),(.081,.12),(.14,.07)]:
            ir[int((delay+rng.uniform(0,.003))*SR)] = gain
        audio = fftconvolve(audio, ir)[:n].astype(np.float32)
    elif scenario == 'telephone':
        audio = sosfilt(butter(3, [300, 3400], fs=SR, btype='bandpass', output='sos'), audio).astype(np.float32)
        audio = np.round(np.clip(audio, -1, 1) * 127) / 127
    playback = np.zeros(n, np.float32)
    if scenario == 'echo':
        # Public synthetic source only; distinct reference, exact known causal delay.
        playback = tracks[0].copy()
        delay = int(rng.uniform(.025, .15) * SR)
        audio[delay:] += playback[:-delay] * float(rng.uniform(.03, .15))
        events.append({'type': 'agent_echo', 'delay_s': delay/SR,
                       'note': 'Reference is public synthetic audio; identity is one scenario speaker in pilot'})
    peak = max(float(np.max(np.abs(audio))), 1)
    audio = audio / peak * .95
    activity = np.stack(active_tracks)
    return audio.astype('<f4'), playback, activity, selected, events, scenario


def extract(speakers, repeats=2, endpoint='http://127.0.0.1:8793/internal/features', output_root=None):
    output_root = Path(output_root or HERE/'pilot')
    dest = output_root / 'conversations'
    dest.mkdir(parents=True, exist_ok=True)
    # A resume or a held-out-first extraction must never temporarily erase
    # completed examples from the manifest while other consumers evaluate them.
    records = [json.loads(p.read_text()) for p in sorted(dest.glob('*.json'))]
    session = requests.Session()
    for anchor in speakers:
        for repeat in range(repeats):
            seed = 892000 + anchor['speaker_idx']*101 + repeat
            path = dest / f'{anchor["split"]}-{anchor["speaker_idx"]:03d}-{repeat}.npz'
            metadata_path = path.with_suffix('.json')
            if path.exists() and metadata_path.exists():
                continue
            guard()
            audio, playback, activity, selected, events, scenario = conversation(speakers, anchor, seed)
            start = time.perf_counter()
            for attempt in range(30):
                response = session.post(endpoint, data=audio.tobytes(),
                    headers={'Content-Type': 'application/octet-stream'}, timeout=90)
                if response.status_code != 423:
                    break
                time.sleep(2)
            response.raise_for_status()
            with np.load(io.BytesIO(response.content), allow_pickle=False) as features:
                x, end_s = features['features'].copy(), features['frame_end_s'].copy()
            ids = np.full(len(x), -1, np.int16)
            fraction = np.zeros(len(x), np.float32)
            ambiguity = np.zeros(len(x), bool)
            for f, end in enumerate(end_s):
                b, e = max(0, int(round((float(end)-.08)*SR))), min(activity.shape[1], int(round(float(end)*SR)))
                if e <= b:
                    continue
                occupancy = activity[:, b:e].mean(axis=1)
                overlap = (activity[:, b:e].sum(axis=0) >= 2).mean()
                best = int(np.argmax(occupancy))
                fraction[f] = occupancy[best]
                if overlap >= .2 or np.sum(occupancy > .3) >= 2:
                    ids[f], ambiguity[f] = -2, True
                elif occupancy[best] >= .5:
                    ids[f] = selected[best]['speaker_idx']
            valid = (ids >= 0) & ~ambiguity
            np.savez_compressed(path, features=x.astype(np.float16), frame_end_s=end_s,
                speaker_idx=ids, valid_mask=valid, speech_fraction=fraction,
                ambiguity=ambiguity)
            record = {'path': str(path.relative_to(output_root)), 'split': anchor['split'],
                'speaker_ids': [s['speaker_idx'] for s in selected], 'seed': seed,
                'scenario': scenario, 'duration_s': len(audio)/SR, 'frames': len(x),
                'valid_frames': int(valid.sum()), 'overlap_frames': int(ambiguity.sum()),
                'silence_frames': int((ids == -1).sum()), 'feature_seconds': time.perf_counter()-start,
                'events': events, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'playback_reference_note': 'Exact procedural waveform available by replaying seed; encoder pilot uses microphone only; turn-taking unchanged'}
            atomic_json(metadata_path, record)
            records.append(record)
            atomic_json(output_root/'manifest.json', manifest(speakers, records))
            print(json.dumps({'conversation_ready': str(path.name), 'frames': len(x),
                              'valid': int(valid.sum()), 'feature_seconds': record['feature_seconds']}), flush=True)
    atomic_json(output_root/'manifest.json', manifest(speakers, records))


def manifest(speakers, records):
    return {'schema_version': 1, 'dataset_repo': REPO, 'dataset_revision': REVISION,
        'source_license': 'CC-BY-4.0', 'attribution': 'Christoph Schuhmann and LAION',
        'teacher_repo': 'speechbrain/spkrec-ecapa-voxceleb', 'teacher_revision': TEACHER_REVISION,
        'encoder_layer': 15, 'feature_dim': 1024, 'frame_ms': 80, 'decision_hop_ms': 320,
        'sample_rate': SR, 'augmentation_version': 'procedural-v1', 'numpy_version': np.__version__,
        'activity_rule': {'word_padding_before_s': .035, 'word_padding_after_s': .06,
                          'dominant_occupancy_min': .5, 'overlap_occupancy_min': .2},
        'teachers': 'teacher_embeddings.npy', 'speakers': speakers, 'conversations': records,
        'limitations': ['Pilot uses two acting-condition shard prefixes per identity, not full emotion matrix',
          'Teacher IDs are synthetic profile labels, not verified biological identities',
          'Source forced alignments and model quality scores may be imperfect',
          'Procedural impulse responses/noise only, no third-party music',
          'Echo source shares one conversation identity in pilot; distinct Robo self-voice eval remains required',
          'No private microphone recordings, enrollment embeddings, or real camera frames included']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['sources','teacher','features','all'], default='all')
    p.add_argument('--train-speakers', type=int, default=24)
    p.add_argument('--val-speakers', type=int, default=4)
    p.add_argument('--test-speakers', type=int, default=4)
    p.add_argument('--repeats', type=int, default=2)
    args = p.parse_args()
    started = time.time()
    guard()
    plan = profile_plan(args.train_speakers, args.val_speakers, args.test_speakers)
    if args.stage in ('sources','all'):
        with ThreadPoolExecutor(max_workers=3) as pool:
            speakers = list(pool.map(source_voice, plan))
        atomic_json(HERE/'pilot'/'speakers.json', speakers)
    else:
        speakers = json.loads((HERE/'pilot'/'speakers.json').read_text())
    if args.stage in ('teacher','all'):
        teachers(speakers)
    if args.stage in ('features','all'):
        extract(speakers, args.repeats)
    print(json.dumps({'stage_complete': args.stage, 'elapsed_s': time.time()-started}), flush=True)


if __name__ == '__main__':
    main()
