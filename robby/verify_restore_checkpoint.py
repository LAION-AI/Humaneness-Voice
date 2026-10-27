"""Verify the pre-proactive baseline source without models, secrets or network."""
import hashlib,json
from pathlib import Path

def main():
    folder=Path(__file__).resolve().parent;root=folder.parent
    manifest=json.loads((folder/'restore-before-proactive-2026-10-27.json').read_text())
    failures=[]
    for name,expected in manifest['source_sha256'].items():
        path=root/name
        if not path.is_file():failures.append({'path':name,'error':'missing'})
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=expected:failures.append({'path':name,'error':'changed'})
    print(json.dumps({'passed':not failures,'baseline_commit':manifest['code_commit'],
        'files_checked':len(manifest['source_sha256']),'failures':failures},indent=2))
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
