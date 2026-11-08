"""Check the actual two-episode, pure 50/50 speaker-selection artifacts."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf
from recipe import HERE, WORKSPACE, sha, dump
from speaker_scoring import speaker_reward

p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args()
out=Path(a.output).resolve()
m=json.loads((out/'manifest.json').read_text());runs=Path(m['generated_from'])
n=m['configuration']['n']
assert m['episodes'] and 1<=n<=16
assert not m['configuration']['voice_conversion']
assert 'no multiplier' in m['configuration']['wer_role']
turn_count=0;reviews=[]
for e in m['episodes']:
    path=out/e['file'];assert sha(path)==e['sha256']
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)],text=True))
    s=probe['streams'][0];assert s['codec_name']=='mp3' and s['channels']==1 and int(s['sample_rate'])==48000
    rows=json.loads(path.with_suffix('.audit.json').read_text());assert rows
    target=[t for lang,t in m['configuration'].get('effective_jobs',[]) if lang==e['language']]
    if target:assert [r['turn'] for r in rows]==target
    for r in rows:
        p=r['performance'];scores=p['final_scores'];assert len(scores)==n
        expected=0 if n==1 else max(range(n),key=lambda i:(scores[i]['reward'],-i));assert p['winner']==expected
        for score in scores:
            assert score['reference_speaker']==r['turn']['speaker']
            if n==1:assert score['ranking_skipped'] and score['reward'] is None
            else:assert abs(score['reward']-speaker_reward(score['ears']['genuineness_0_6'],score['ecapa_cosine']))<1e-9
        folder=Path(r['source_folder']);raw,sr=sf.read(folder/'raw.flac',dtype='float32')
        chosen,sr2=sf.read(folder/f'candidate-{expected}.flac',dtype='float32')
        assert sr==sr2==48000 and np.array_equal(raw,chosen)
        assert np.isfinite(raw).all() and np.sqrt(np.mean(raw*raw))>.001
        from recipe import specification
        assert r['prompt']==specification(r['turn'],r['language'])['prompt']
        if r['audit']['review_required']:reviews.append([r['language'],r['turn']['index'],r['audit']['wer']])
        turn_count+=1
    timeline=json.loads(path.with_suffix('.timeline.json').read_text())
    for a,b in zip(timeline,timeline[1:]):assert abs(b['speech_start_s']-a['speech_end_s']-.4)<1/48000
    batch=json.loads((runs/'production-batches.json').read_text())
    assert abs(e['inference_seconds']-sum(b['batch_wall_seconds'] for b in batch if b['language']==e['language']))<1e-6
if (runs/'token-regression.json').exists():
    assert json.loads((runs/'token-regression.json').read_text())['shared_and_full_prefill_codes_equal']
    assert (runs/'codec-regression.json').exists()
elif n in (1,2,4):assert all(json.loads((runs/'offline-regression.json').read_text()).values())
controls=json.loads((runs/'reference-controls.json').read_text())
assert controls['dimensions']==192 and min(np.diag(controls['cosine']))>.999
result=dict(passed=True,episodes=len(m['episodes']),turns=turn_count,candidates=turn_count*n,review_turns=reviews,
    checks=['pure 50/50 argmax; no WER gate','correct reference per speaker','selected PCM equals candidate',
        'exact prompt and preserved transcript','MP3 hashes/format','400ms gaps','batch timing counted once',
        'recorded prefill/token and decoder correctness checks'])
dump(out/'validation.json',result);print(json.dumps(result,indent=2))
