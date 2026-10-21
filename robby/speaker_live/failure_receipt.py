"""Preserve an unsafe rejected development replay; public synthetic fixtures only."""
import hashlib,json
from pathlib import Path

def main():
    root=Path(__file__).resolve().parent.parent
    path=Path('/mnt/nvme/robo-speaker-studies/speaker-live-results/inflight-1792538549.jsonl')
    events=[json.loads(line) for line in path.read_text().splitlines()]
    assert events[0].get('public_synthetic_only') is True
    kept=[e for e in events if (e['type']=='endpoint' and e['at_s']>45) or
        (e['type']=='synthetic_barge_verification' and e.get('accepted'))]
    report={'public_synthetic_only':True,'physical_microphone_used':False,
        'status':'Rejected development revision, NOT the current default',
        'finding':'Similar background timbre could pass a teacher-only gate. The later relative foreground guard prevents quiet background from renewing/reassigning known activity or triggering an interruption.',
        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'events':kept,
        'limit':'One reused synthetic background voice; not an estimated real-world error rate.'}
    (root/'logs/failure-before-foreground-guard.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'rejected_revision_events':len(kept),'private_recordings_published':0}))

if __name__=='__main__':main()
