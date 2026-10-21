"""Fail-closed staged-file audit; reports paths/rules, NEVER matched secrets.

Run from the destination Git repository. This is a transparent pattern/allowlist
check, not a claim that every possible secret encoding can be detected.
"""
import hashlib,json,re,subprocess,sys
from pathlib import PurePosixPath

PATTERNS={
 'hf_token':rb'hf_[A-Za-z0-9]{20,}',
 'github_token':rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,})',
 'provider_secret':rb'(?:sk-(?:proj-|ant-)?[A-Za-z0-9_\-]{24,}|AIza[A-Za-z0-9_\-]{30,}|AKIA[A-Z0-9]{16})',
 'private_key':rb'-----BEGIN [A-Z ]*PRIVATE KEY-----',
 'literal_bearer':rb'Bearer [A-Za-z0-9_\-.]{24,}',
 'literal_credential':rb'''(?i)(?:api[_-]?key|access[_-]?token|password)\s*[=:]\s*["'][A-Za-z0-9_\-.]{24,}["']''',
}
def main():
    names=subprocess.check_output(['git','diff','--cached','--name-only','--diff-filter=ACM','-z']).decode().split('\0')
    failures=[];files=[]
    for name in filter(None,names):
        path=PurePosixPath(name);raw=subprocess.check_output(['git','show',':'+name])
        if (path.name in ('config.local.json','.env','server.json') or
            any(p in ('.runtime','logs','uploads','node_modules','vendor','__pycache__') for p in path.parts) or
            path.suffix.lower() in ('.wav','.mp3','.mp4','.pt','.pth','.npz','.npy','.safetensors','.gguf','.zip','.png','.jpg')):
            failures.append({'file':name,'rule':'private_or_binary_artifact_path'})
        for label,pattern in PATTERNS.items():
            if re.search(pattern,raw):failures.append({'file':name,'rule':label})
        try:raw.decode('utf-8')
        except UnicodeDecodeError:failures.append({'file':name,'rule':'non_utf8_binary'})
        files.append({'file':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    print(json.dumps({'passed':not failures,'staged_files':len(files),'bytes':sum(f['bytes'] for f in files),
        'checks':list(PATTERNS)+['private_or_binary_artifact_path','utf8_only'],
        'failures':failures,'files':files},indent=2))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
