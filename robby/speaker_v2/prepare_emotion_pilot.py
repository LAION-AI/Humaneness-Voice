"""Broaden the immutable first pilot using <500 MB of additional HTTP ranges.

Parquet column-range reads locate emotion groups. Approximate tar positions use
cumulative audio sizes, then checksum-valid 512-byte TAR headers resynchronize.
Full tar archives are never downloaded. V1 teacher references remain unchanged.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import threading
import time

import numpy as np
import pyarrow.parquet as pq
import requests

try:
    from . import prepare_pilot as core
except ImportError:  # Direct script execution.
    import prepare_pilot as core

OUTPUT = core.HERE/'pilot_v2'
LIMIT = 500_000_000
budget_lock = threading.Lock()
charged_bytes = 0
budget_ledger = None


def byte_range(path, start, length):
    global charged_bytes
    with budget_lock:
        if charged_bytes + length > LIMIT:
            raise RuntimeError('Additional download hard limit 500 MB reached')
        charged_bytes += length
        if budget_ledger is not None:
            core.atomic_json(budget_ledger, {'reserved_download_bytes': charged_bytes,
                'hard_limit_bytes': LIMIT,
                'note': 'Conservative reserved byte count, including retry-run allowance; may exceed actual bytes received'})
    url = core.BASE + path + f'?tyger_v2_range={start}-{start+length-1}'
    for attempt in range(6):
        with requests.get(url, headers={'Range': f'bytes={start}-{start+length-1}'},
                stream=True, timeout=90) as r:
            if r.status_code in (429, 502, 503, 504) and attempt < 5:
                time.sleep(min(30, float(r.headers.get('Retry-After', 2**attempt))))
                continue
            if r.status_code != 206 or not r.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                raise RuntimeError(f'Range not honored(status{r.status_code}); refusing whole archive download')
            total = int(r.headers['Content-Range'].split('/')[-1])
            data = r.raw.read(length)
        return data, total
    raise RuntimeError('Bounded Range retry exhausted')


class ColumnRangeFile(io.RawIOBase):
    """Minimal seekable range reader: only two Parquet columns, not full indexes."""
    def __init__(self, path):
        self.path, self.position, self.cache = path, 0, {}
        _, self.size = byte_range(path, 0, 1)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        return self.position

    def read(self, n=-1):
        if n < 0:
            n = self.size-self.position
        begin, end = self.position, min(self.size, self.position+n)
        if begin == end:
            return b''
        key = (begin, end)
        if key not in self.cache:
            self.cache[key] = byte_range(self.path, begin, end-begin)[0]
        self.position = end
        return self.cache[key]


def tar_pairs(blob, voice):
    audio, records = {}, {}
    at = 0
    while at+512 <= len(blob):
        try:
            header = tarfile.TarInfo.frombuf(blob[at:at+512], 'utf-8', 'surrogateescape')
        except (tarfile.HeaderError, ValueError):
            at += 512
            continue
        if not header.name.startswith(voice+'__') or header.size < 0:
            at += 512
            continue
        end = at+512+header.size
        if end > len(blob):
            break
        payload = blob[at+512:end]
        if header.name.endswith('.mp3'):
            audio[header.name[:-4]] = payload
        elif header.name.endswith('.json'):
            try:
                records[header.name[:-5]] = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        at = at+512+((header.size+511)//512)*512
    return [(uid, audio[uid], records[uid]) for uid in sorted(audio.keys() & records.keys())]


def sources(row):
    core.guard()
    directory = core.CACHE/'sources_v2'/row['voice']
    directory.mkdir(parents=True, exist_ok=True)
    saved = directory/'clips.json'
    if saved.exists():
        return {**row, 'clips': json.loads(saved.read_text())}
    indexes = {}
    for local in (0, 1):
        shard = row['catalog_index']*4+local
        remote = ColumnRangeFile(f'index/origin=original/part-{shard:06d}.parquet')
        index = pq.read_table(remote, columns=['uid', 'audio_bytes'])
        ids = index['uid'].to_pylist()
        lengths = index['audio_bytes'].to_numpy().astype(np.float64)+20_000
        fraction = np.cumsum(lengths)/lengths.sum()
        indexes[local] = (ids, fraction)
    requests_plan = []
    for emotion in ('Anger','Amusement','Elation','Sadness'):
        found = None
        for local, (ids, fractions) in indexes.items():
            matches = [i for i, uid in enumerate(ids) if '__E__'+emotion+'__' in uid]
            if matches:
                found = (local, float(fractions[matches[len(matches)//2]]))
                break
        if found:
            requests_plan.append((found[0], found[1], 3*1024**2, emotion))
    # Neutral-ish ordinary speech condition, away from V1's prefix recording IDs.
    requests_plan.append((2, .015, 1024**2, 'neutral_proxy'))
    selected = [copy.deepcopy(c) for c in row['clips'] if c['role'] == 'teacher']
    old_ids = {c['uid'] for c in row['clips']}
    actual_coverage = []
    for local, fraction, length, requested_emotion in requests_plan:
        shard = row['catalog_index']*4+local
        path = f'data/origin=original/vprof_base-{shard:05d}.tar'
        _, size = byte_range(path, 0, 1)
        start = int(size*fraction)//512*512
        content, _ = byte_range(path, start, length)
        candidates = []
        for uid, mp3, record in tar_pairs(content, row['voice']):
            if uid in old_ids or not (4 <= record.get('dur_s', 0) <= 25):
                continue
            if len(record.get('words') or []) < 6 or (record.get('n_bursts') or 0) > 1:
                continue
            meta = record.get('src_meta', {})
            if float(meta.get('wer', 1)) > .35:
                continue
            match = '__E__'+requested_emotion+'__' in uid
            ranking = float(record.get('quality', {}).get('overall_quality', 0))
            candidates.append((match, ranking, uid, mp3, record))
        for _, _, uid, mp3, record in sorted(candidates, key=lambda x: (-x[0],-x[1],x[2]))[:2]:
            name = hashlib.sha256(uid.encode()).hexdigest()[:16]+'.mp3'
            (directory/name).write_bytes(mp3)
            clip = {'uid': uid, 'path': str(directory/name), 'role': 'prediction',
                'duration_s': record['dur_s'], 'words': record['words'], 'bursts': record.get('bursts'),
                'lang': record.get('lang'), 'condition': uid.split('__')[2:-1],
                'source_tar': path, 'source_range': [start, start+length-1],
                'downloaded_range_bytes': length, 'requested_emotion': requested_emotion,
                'sha256': hashlib.sha256(mp3).hexdigest(),
                'published_wer': record.get('src_meta', {}).get('wer'),
                'published_spk_sim': record.get('src_meta', {}).get('spk_sim')}
            selected.append(clip)
            actual_coverage.append('/'.join(clip['condition']))
    if len(selected) < 6:
        raise RuntimeError(f'Only {len(selected)-2} new prediction clips for {row["voice"]}')
    core.atomic_json(saved, selected)
    print(json.dumps({'emotion_source_ready': row['voice'], 'conditions': actual_coverage,
                      'clips': len(selected)}), flush=True)
    return {**row, 'clips': selected}


def main():
    global charged_bytes, budget_ledger
    started = time.time()
    budget_ledger = core.CACHE/'emotion_v2_download_ledger.json'
    if budget_ledger.exists():
        charged_bytes = json.loads(budget_ledger.read_text())['reserved_download_bytes']
    else:
        charged_bytes = int(os.environ.get('ROBO_V2_PREVIOUS_DOWNLOAD_BYTES', '0'))
    original = json.loads((core.HERE/'pilot'/'speakers.json').read_text())
    with ThreadPoolExecutor(max_workers=3) as workers:
        speakers = list(workers.map(sources, original))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    core.atomic_json(OUTPUT/'speakers.json', speakers)
    core.atomic_json(OUTPUT/'retrieval.json', {'additional_requested_bytes': charged_bytes,
        'hard_limit_bytes': LIMIT, 'source_seconds': time.time()-started,
        'method': 'Two-column Parquet HTTP Range + estimated tar offset + checksum-validated header resynchronization',
        'source_cache': str(core.CACHE/'sources_v2')})
    core.teachers(speakers, output_root=OUTPUT)
    core.extract(speakers, repeats=2, output_root=OUTPUT)
    manifest = json.loads((OUTPUT/'manifest.json').read_text())
    manifest['pilot_revision'] = 'emotion-diverse-v2'
    manifest['limitations'][0] = 'Emotion-position HTTP ranges broaden coverage but do not cover the full842-conditionmatrix; see actual source condition audit'
    manifest['source_v1_teacher_targets_retained'] = True
    manifest['limitations'].append('Emotion-position midpoint usually lands in German C-condition takes; see language audit, do not claim balanced English emotional robustness')
    manifest['limitations'].append('BKGN/very_high read clips are not clean neutral controls; whispering not covered')
    core.atomic_json(OUTPUT/'manifest.json', manifest)
    # Immutable evaluation view, reconstructed from per-conversation records,
    # never by racing a manifest while another extraction is appending to it.
    heldout = core.HERE/'pilot_v2_heldout'
    heldout.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(p.read_text()) for g in ('validation-*.json','test-*.json')
            for p in sorted((OUTPUT/'conversations').glob(g))]
    view = core.manifest(speakers, rows)
    view['pilot_revision'] = 'emotion-v2-heldout-immutable'
    core.atomic_json(heldout/'manifest.json', view)
    import shutil
    for name in ('teacher_embeddings.npy','teacher_quality.json','probe_teacher_embeddings.npz'):
        shutil.copy2(OUTPUT/name, heldout/name)
    if not (heldout/'conversations').exists():
        (heldout/'conversations').symlink_to('../pilot_v2/conversations', target_is_directory=True)
    print(json.dumps({'emotion_v2_complete': True, 'elapsed_s': time.time()-started}), flush=True)


if __name__ == '__main__':
    main()
