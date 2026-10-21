"""Validate public-synthetic input before claiming an ASR/speaker test passed.

The malformed original Selma follow-up is retained as a nonverbal/background
negative, not counted as intelligible same-speaker speech. No private audio.
"""
import json,re,time
from pathlib import Path
import requests
from speaker_v4.common import atomic
from speaker_turns import ROBOT_CALL,word_distance,words
from .fixtures import ROOT
from .live_test import audio

def normalized(text):
    text=re.sub(r'\baufzu(reden|sprechen)\b',r'auf zu \1',text,flags=re.I)
    text=re.sub(r'\bstopp\b','stop',text,flags=re.I)
    return words(ROBOT_CALL.sub('Robby',text))

def main():
    rows=[];negatives={'selma-followup','christoph-pause','christoph-pause-clear'}
    for path in sorted(ROOT.glob('*.json')):
        if path.name=='manifest.json':continue
        row=json.loads(path.read_text());wave=audio(row['name'])
        for _ in range(60):
            try:response=requests.post('http://127.0.0.1:8793/internal/transcribe',data=wave.astype('<f4').tobytes(),timeout=30)
            except requests.ConnectionError:time.sleep(.25);continue
            if response.status_code in (423,503):time.sleep(.25);continue
            response.raise_for_status();break
        else:raise RuntimeError('Close the live browser/test slot before fixture QA')
        text=response.json()['text'];reference=normalized(row['text']);hypothesis=normalized(text)
        wer=word_distance(reference,hypothesis)/max(1,len(reference))
        rows.append({'name':row['name'],'expected':row['text'],'recognized':text,'clean_wer':wer,
            'valid_intelligible_fixture':wer<=.25,'retained_nonverbal_negative':row['name'] in negatives,
            'sha256':row['sha256'],'seconds':len(wave)/16000})
    report={'public_synthetic_only':True,'finished':time.time(),'criterion':'Clean ASR WER <= 0.25; normalize Robby spelling, Stop/Stopp orthography and exact ASR-joined aufzusprechen/aufzureden.',
        'fixtures':rows,'invalid_retained_as_negatives':sorted(negatives),'no_identity_scores_used_for_selection':True}
    folder=Path('/mnt/nvme/robo-speaker-studies/speaker-live-results');atomic(folder/'fixture-clean-qa.json',report)
    assert all(r['valid_intelligible_fixture'] or r['retained_nonverbal_negative'] for r in rows),'An unvalidated speech fixture cannot be used as positive input'
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
