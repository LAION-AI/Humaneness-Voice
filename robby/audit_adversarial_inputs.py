"""Paired clean-input audit, separate from end-to-end conversation acceptance.

Run only when the shared native ASR slot is free. Never select/regenerate takes
based on this audit. Same TTS profile is a source label, not proof that the
resulting synthetic timbre is invariant. Teacher cosine is not a probability.
"""
import argparse, datetime, fcntl, hashlib, json, os
from pathlib import Path
import httpx
import torch
import numpy as np
from adversarial_audio import word_error_rate
from generate_adversarial_fixtures import decode, fixture_root
from speaker_teacher import get_cached_teacher, enrollment_reference

ROOT=Path(__file__).resolve().parent


def main(args):
    lock=None
    if os.environ.get('ROBBY_PROTOCOL_LOCK_HELD')!='1':
        p=ROOT/'logs/voice-assistant-runs/.lock';p.parent.mkdir(parents=True,exist_ok=True);lock=p.open('w')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another protocol runner holds the audio-test lock')
    torch.set_num_threads(2)
    manifest=json.loads((args.fixtures/'manifest.json').read_text())
    rows={r['name']:r for r in manifest['cases']};teacher=get_cached_teacher()
    anchors={}
    for name in ('christoph-intro','selma-intro','background'):
        row=rows[name];p=args.fixtures/row['file']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Changed fixture '+name)
        anchors[row['voice']]=teacher.embed(enrollment_reference(decode(p)))
    report={'passed':False,'finished':False,'scope':'Clean-input ASR and synthetic-voice consistency audit; not a browser, intent, real-human speaker or independent listening test.',
        'fixture_manifest_sha256':hashlib.sha256((args.fixtures/'manifest.json').read_bytes()).hexdigest(),
        'cases':[],'voice_threshold_for_diagnostics':args.threshold,
        'all_source_files_retained':True,'embeddings_exported':False,'attempts_per_clip':1}
    out=ROOT/'static/adversarial-input-audit.json'
    with httpx.Client(timeout=90) as client:
        for name,row in rows.items():
            p=args.fixtures/row['file']
            if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Changed fixture '+name)
            wave=decode(p);record={'name':name,'source_profile':row['voice'],'authored_script':row['text'],'sha256':row['sha256']}
            try:
                v=teacher.embed(enrollment_reference(wave));anchor=anchors.get(row['voice'])
                record['clean_ecapa_similarity_to_intro']=round(float(v@anchor),5) if anchor is not None else None
                response=client.post(args.url.rstrip('/')+'/internal/transcribe',content=wave.astype('<f4').tobytes())
                response.raise_for_status();record['transcript']=response.json()['text']
                record['wer']=word_error_rate(row['text'],record['transcript'])
                record['flags']=[]
                if record['wer']>.35:record['flags'].append('clean ASR/script mismatch: listen before attributing to noise')
                if anchor is not None and record['clean_ecapa_similarity_to_intro']<args.threshold:
                    record['flags'].append('same source profile, weak clean voice similarity: source synthesis / reference / teacher confound')
                record['processed']=True
            except Exception as exc:
                record.update(processed=False,error=type(exc).__name__+': '+str(exc)[:300])
            report['cases'].append(record);out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
            print(json.dumps(record,ensure_ascii=False),flush=True)
            if not record['processed']:break
    report['finished']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    report['passed']=len(report['cases'])==len(rows) and all(r['processed'] for r in report['cases'])
    report['all_inputs_unflagged']=report['passed'] and not any(r.get('flags') for r in report['cases'])
    report['meaning_of_passed']='The audit completed; inspect flags. This does not certify that every fixture is valid speech or a perfectly constant voice.'
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://127.0.0.1:8793')
    p.add_argument('--fixtures',type=Path,default=fixture_root());p.add_argument('--threshold',type=float,default=.28)
    main(p.parse_args())
