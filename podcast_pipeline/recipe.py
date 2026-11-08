"""Pure configuration, scoring and reproducibility helpers (no GPU imports)."""
import hashlib
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
SR = 48000
FRAME_SECONDS = .08
GENERAL = (
    'A warm, emotionally thoughtful two-host podcast conversation. '
    'Very authentic, very genuine. '
    'Natural, intimate, spontaneous spoken delivery with clear words, subtle emotion, '
    'relaxed conversational pacing and brief meaningful pauses. '
    'Be sincere, without an exaggerated announcer voice or forced laughter. '
    'Perform only the current speaker turn and speak exactly the SCRIPT. '
    'Keep the same speaker identity as the reference audio.'
)
MODEL_REPO = 'laion/Humanness-Voice-Podcast'
MODEL_REVISION = 'a1f8946bd654db1c19b6c476ed6b4acde384e882'

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def tokens(text):
    return re.findall(r'\w+', text.casefold())

def edit_row(reference, hypothesis):
    # Last DP row gives edit distance to every reference prefix.
    row = list(range(len(reference)+1))
    for i, word in enumerate(hypothesis, 1):
        nxt = [i]
        for j, target in enumerate(reference, 1):
            nxt.append(min(nxt[-1]+1, row[j]+1, row[j-1]+(word != target)))
        row = nxt
    return row

def errors(reference, hypothesis, final=True):
    ref, hyp = tokens(reference), tokens(hypothesis)
    row = edit_row(ref, hyp)
    if final:
        endpoint = len(ref)
    elif not hyp:
        return {'wer': 1., 'matched_words': 0, 'hypothesis_words': 0,
                'reference_words': len(ref), 'edit_count': len(ref)}
    else:
        endpoint = min(range(1, len(ref)+1),
                       key=lambda i: (row[i], abs(i-len(hyp)), -i)) if ref else 0
    return {'wer': row[endpoint]/max(1, endpoint), 'matched_words': endpoint,
            'hypothesis_words': len(hyp), 'reference_words': len(ref),
            'edit_count': row[endpoint]}

def reward(genuineness, wer):
    if not all(math.isfinite(x) for x in (genuineness, wer)) or wer < 0:
        raise ValueError('Invalid score')
    return max(0., min(1., genuineness/6.))/(1.+wer)

def chunk_choice(scores, text, finished, previous_progress=0, stagnant=0):
    """Completion is a property of the reward winner, never a priority class.

    A matched word count alone can hide a substituted ending. Require the last
    three target words too, while allowing earlier ASR spelling errors.
    """
    ref=tokens(text);tail=ref[-min(3,len(ref)):]
    complete={i for i,s in enumerate(scores) if ref and s['matched_words']==len(ref)
        and s['hypothesis_words']>=len(ref) and s['wer']<=.12 and tokens(s['asr'])[-len(tail):]==tail}
    eligible=[i for i,s in enumerate(scores) if not finished[i] or i in complete] or list(range(len(scores)))
    plausible=[i for i in eligible if scores[i]['matched_words']<len(ref) or i in complete]
    if plausible:eligible=plausible
    progressing=[i for i in eligible if scores[i]['matched_words']>previous_progress]
    guarded=stagnant>=2 and bool(progressing)
    if guarded:eligible=progressing
    winner=max(eligible,key=lambda i:(scores[i]['reward'],-i))
    return winner,winner in complete,guarded

def specification(turn, language):
    if turn.get('sentences'):
        assert ' '.join(s['text'] for s in turn['sentences']) == turn['text']
        script = '\n'.join(f"(Very genuine, very natural, very authentic; {s['emotion']}) {s['text']}" for s in turn['sentences'])
    else:
        script = f"({turn['direction']}) {turn['text']}"
    return {'prompt': f"GENERAL: {GENERAL}\nSCRIPT:\n{script}",
            'unconditional': 'GENERAL: Keep the same speaker identity as the reference audio. '
                'Plain neutral conversational delivery.\nSCRIPT:\n'+turn['text'],
            'text': turn['text'], 'lang': language, 'frames': 320, 'free_timing': True}

def references(config):
    result = {}
    for speaker, item in config.items():
        path = (HERE/item['path']).resolve()
        if sha(path) != item['sha256']:
            raise ValueError('Reference changed: '+str(path))
        result[speaker] = path
    return result
