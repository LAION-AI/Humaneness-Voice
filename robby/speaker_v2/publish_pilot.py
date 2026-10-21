"""Audit-first optional HF publication. Default is LOCAL dry run: no credentials/network.

Never prints tokens, headers, credential-helper output, or remote exception text.
Only explicit --publish may contact HF authentication or mutate remote state.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

SECRET = re.compile(rb'(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)')
FORBIDDEN = {'logs','runtime','sessions','enrolled','microphone','__pycache__','.git','.env'}


def audit_release(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'MANIFEST.json').read_text())
    allowed = set(manifest['files']) | {'MANIFEST.json', 'tyger-v2-public-pilot.zip'}
    actual = set()
    total = 0
    for p in root.rglob('*'):
        if p.is_symlink():
            raise ValueError('Release symlinks are forbidden')
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if set(p.relative_to(root).parts) & FORBIDDEN:
            raise ValueError('Forbidden release path')
        if rel not in allowed:
            raise ValueError('Unexpected file outside the audited allowlist')
        actual.add(rel)
        total += p.stat().st_size
        if rel == 'tyger-v2-public-pilot.zip':
            continue
        content = p.read_bytes()
        if SECRET.search(content):
            raise ValueError('Potential credential/private-key material in release')
        if rel != 'MANIFEST.json':
            entry = manifest['files'][rel]
            if hashlib.sha256(content).hexdigest() != entry['sha256'] or len(content) != entry['bytes']:
                raise ValueError('Release hash/size mismatch')
    if set(manifest['files']) - actual:
        raise ValueError('Missing audited release file')
    if total > 200_000_000:
        raise ValueError('Release including ZIP exceeds 200 MB')
    archive = root/'tyger-v2-public-pilot.zip'
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            expected = set(manifest['files']) | {'MANIFEST.json'}
            if set(z.namelist()) != expected:
                raise ValueError('ZIP entries do not match audited allowlist')
            for name in z.namelist():
                if name.startswith('/') or '..' in Path(name).parts:
                    raise ValueError('Unsafe ZIP entry')
                if (z.getinfo(name).external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('ZIP symlinks are forbidden')
                content = z.read(name)
                expected_hash = hashlib.sha256((root/name).read_bytes()).hexdigest()
                if hashlib.sha256(content).hexdigest() != expected_hash:
                    raise ValueError('ZIP payload differs from audited release')
    return {'audit_passed': True, 'files': len(manifest['files']), 'total_bytes_including_zip': total,
            'dataset_files': sum(k.startswith('dataset/') for k in manifest['files']),
            'model_files': sum(k.startswith('model/') for k in manifest['files']),
            'zip_present': archive.exists(), 'private_recordings': 0,
            'third_party_encoder_or_teacher_weights': 0}


def configured_token():
    from huggingface_hub import get_token
    token = get_token()
    if token:
        return token
    # Credential remains in process memory. Never echo its output or persist it.
    result = subprocess.run(['git','credential','fill'],
        input='protocol=https\nhost=huggingface.co\n\n', text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
        env={**__import__('os').environ, 'GIT_TERMINAL_PROMPT':'0'})
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if line.startswith('password='):
                return line.partition('=')[2]
    raise PermissionError('No configured HF credential available')


def validate_write_identity(identity, repo_ids):
    # Fail closed if OAuth/resource-group metadata does not prove org write access.
    org = next((o for o in identity.get('orgs', []) if o.get('name','').lower() == 'laion'), {})
    if org.get('roleInOrg') not in {'write','admin'}:
        raise PermissionError('whoami does not prove LAION write/admin membership')
    access = identity.get('auth', {}).get('accessToken', {})
    if access.get('role') == 'write':
        return
    if access.get('role') != 'fineGrained':
        raise PermissionError('Configured token does not prove repository write permission')
    scopes = access.get('fineGrained', {}).get('scoped', [])
    permitted = set()
    for scope in scopes:
        entity = scope.get('entity', {})
        name = str(entity.get('name', '')).lower()
        rights = {str(p).lower() for p in scope.get('permissions', [])}
        if not rights & {'repo.write','repos.write','write-repos'}:
            continue
        if name == 'laion':
            permitted.update(repo_ids)
        permitted.update(r for r in repo_ids if name == r.lower())
    if not set(repo_ids) <= permitted:
        raise PermissionError('whoami does not prove fine-grained write scope for both target repos')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=Path(__file__).parent/'release')
    parser.add_argument('--dataset-repo', default='laion/tyger-v2-speaker-pilot')
    parser.add_argument('--model-repo', default='laion/tyger-v2-speaker-predictor-pilot')
    parser.add_argument('--publish', action='store_true', help='Explicitly authorize remote creation/upload after checks')
    parser.add_argument('--allow-existing', action='store_true', help='Explicitly permit updating matching existing repos; never deletes remote files')
    args = parser.parse_args()
    result = audit_release(args.release)
    for repo in (args.dataset_repo, args.model_repo):
        if not re.fullmatch(r'laion/[A-Za-z0-9][A-Za-z0-9_.-]*', repo):
            raise ValueError('Target must be an exact LAION repository ID')
    if args.dataset_repo == args.model_repo:
        raise ValueError('Use distinct dataset and model repository names')
    result.update(dry_run=not args.publish, network_used=False,
                  dataset_repo=args.dataset_repo, model_repo=args.model_repo)
    if not args.publish:
        print(json.dumps(result, indent=2))
        return
    from huggingface_hub import HfApi
    token = configured_token()
    # Do not send a configured credential to an HF_ENDPOINT override.
    api = HfApi(endpoint='https://huggingface.co', token=token)
    try:
        identity = api.whoami(token=token)
        validate_write_identity(identity, (args.dataset_repo, args.model_repo))
    except Exception:
        raise PermissionError('HF authentication/write validation failed; no repository created or uploaded') from None
    targets = [(args.dataset_repo,'dataset','dataset'), (args.model_repo,'model','model')]
    # All read-only target checks happen before either repository is changed.
    existing = {}
    try:
        for repo, kind, _ in targets:
            existing[repo] = api.repo_exists(repo, repo_type=kind)
            if existing[repo] and not args.allow_existing:
                raise PermissionError('A target exists; explicit --allow-existing is required')
    except PermissionError:
        raise
    except Exception:
        raise PermissionError('HF target preflight failed; no repository created or uploaded') from None
    completed = []
    try:
        for repo, kind, folder in targets:
            if not existing[repo]:
                api.create_repo(repo, repo_type=kind, private=False, exist_ok=False)
            api.upload_folder(repo_id=repo, repo_type=kind,
                folder_path=str(args.release/folder),
                commit_message='Publish audited public-synthetic TYGER V2 pilot and failed-quality-gate results')
            completed.append(repo)
    except Exception:
        # Do not print external exceptions, request headers, or credential data.
        print(json.dumps({'publication_complete':False, 'completed_repos':completed,
                          'message':'HF upload failed; inspect repository state before retrying'}))
        raise RuntimeError('HF upload failed; no private local recordings were included') from None
    print(json.dumps({'publication_complete':True, 'repos':completed, 'network_used':True}))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Even unusual credential-library failures must not leak remote headers
        # or secret-bearing exception messages. No traceback/exception text.
        print(json.dumps({'ok':False,'error_type':type(exc).__name__,
            'message':'Publication/audit failed; credential and HTTP-response details are intentionally suppressed.'}), file=sys.stderr)
        raise SystemExit(1)
