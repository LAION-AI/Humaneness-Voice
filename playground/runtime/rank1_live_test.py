"""Small isolated-session HTTP generation smoke after GPU startup, not a study."""
import argparse
import hashlib
import http.cookiejar
import json
import os
from pathlib import Path
import time
import urllib.request


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default=f'http://127.0.0.1:{os.environ.get("SMALL_PORT", "8792")}')
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    def get(path):
        with opener.open(args.base + path, timeout=600) as response:
            return response.read()
    state = json.loads(get('/api/state'))
    if not state['ready'] or state['busy']:
        print(json.dumps({'status': 'skipped', 'reason': 'Demo not ready or another user is active'}), flush=True)
        return
    rows = []
    voice = {'voice': {'id': 'profile/anime_000', 'weight': 1}}
    emotion = {'emotion': {'id': 'emotion/Affection', 'weight': 1}}
    style = {'style': {'id': 'vn/S_DRAM_high', 'weight': 1}}
    mix = {**voice, **emotion, 'style': {'id': 'vn/S_DRAM_high', 'weight': .5}}
    cases = [('base-before', {}, 'single'), ('voice', voice, 'single'),
             ('emotion', emotion, 'single'), ('style', style, 'single'),
             ('mix', mix, 'single'), ('base-after', {}, 'single'),
             ('voice-best-of-2', voice, 'best'), ('mix-winner-2', mix, 'search_winner'),
             ('style-top2-1600', style, 'search_beam')]
    for label, picks, mode in cases:
        # Yield to a human's already-running turn, rather than displacing it.
        if json.loads(get('/api/state'))['busy']:
            rows.append({'case': label, 'skipped': 'another user is active'})
            break
        body = {'message': 'CAPTION: Warm, natural conversation.\nTRANSCRIPT: "Ready when you are."',
            'format': 'caption', 'language': 'en', 'input_mode': 'manual', 'generation': mode,
            'n': 2, 'search_n': 2, 'search_chunk_frames': 20 if mode == 'search_beam' else 10,
            'search_seconds': 3.2, 'guidance': 1.5, 'temperature': 1., 'seed': 777,
            'seconds': 2., 'align': False, 'adapters': picks}
        request = urllib.request.Request(args.base + '/api/turn', data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'}, method='POST')
        started = time.perf_counter()
        with opener.open(request, timeout=600) as response:
            lines = response.read().decode().splitlines()
        events = [json.loads(line[6:]) for line in lines if line.startswith('data: ')]
        errors = [event['message'] for event in events if event['event'] == 'error']
        if errors:
            raise RuntimeError(label + ': ' + '; '.join(errors))
        result = next(event for event in events if event['event'] == 'result')
        expected = {slot: picks.get(slot) for slot in ['voice', 'emotion', 'style']}
        assert result['adapters']['selection'] == expected, label
        active = result['adapters']['active']
        assert len(active) == len(picks), label
        clips = []
        for candidate in result['candidates']:
            encoded = get(candidate['url'])
            assert len(encoded) > 1000 and candidate['sec'] > 0, label
            path = ROOT / 'test-results' / f'rank1-live-{label}-{candidate["index"]}.mp3'
            path.write_bytes(encoded)
            clips.append({'seconds': candidate['sec'], 'wer_proxy': candidate['wer'],
                'mp3_sha256': hashlib.sha256(encoded).hexdigest(), 'file': path.name})
        row = {'case': label, 'generation': mode, 'adapters': result['adapters'],
            'clips': clips, 'wall_seconds': time.perf_counter() - started}
        rows.append(row)
        print(json.dumps({'passed': label, 'clips': len(clips)}), flush=True)
    lookup = {row['case']: row for row in rows if 'clips' in row}
    restored = (lookup['base-before']['clips'][0]['mp3_sha256'] == lookup['base-after']['clips'][0]['mp3_sha256']) if 'base-after' in lookup else None
    report = {'verification': 'Actual HTTP TTS and scorers in an isolated cookie session; short smoke only, not emotion/identity efficacy or throughput study',
        'completed': len(lookup), 'planned': len(cases), 'base_audio_restored_identically': restored,
        'cases': rows}
    (ROOT / 'test-results/rank1-live.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'cases'}), flush=True)
    if restored is False:
        raise RuntimeError('Base-before and base-after MP3 differ; investigate deterministic reset')


if __name__ == '__main__':
    main()
