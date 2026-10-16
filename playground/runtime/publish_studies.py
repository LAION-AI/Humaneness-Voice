"""Publish an explicit public-artifact allowlist; credentials live only in memory.

Interactive commands: inspect, prepare, space, model, status, quit.
No login(), credential cache, environment token, or token-bearing CLI argument.
"""
import getpass
import hashlib
import json
from pathlib import Path
import re
import shutil

from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
from huggingface_hub.utils import disable_progress_bars

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "publication" / "space"
MODEL = "laion/Humaneness-Voice-Small"
SPACE = "laion/humaneness-voice-small-studies"
PERFORMANCE = ["baseline", "optimizations", "compile", "sweep", "validation", "latency", "long"]


def prepare():
    for folder in [PUBLIC / "performance", PUBLIC / "study" / "audio"]:
        folder.mkdir(parents=True, exist_ok=True)
    realization = ROOT / "test-results/realization-check.html"
    if realization.is_file():
        shutil.copyfile(realization, PUBLIC / "realization-check.html")
    for report in ["performance", "study"]:
        html = (ROOT / report / "index.html").read_text()
        html = html.replace('href="/">Voice demo', 'href="../index.html">Study home')
        html = html.replace('href="/performance/"', 'href="../performance/index.html"')
        html = html.replace('href="/study/"', 'href="../study/index.html"')
        (PUBLIC / report / "index.html").write_text(html)
    for name in PERFORMANCE:
        shutil.copyfile(ROOT / "performance" / f"{name}.json", PUBLIC / "performance" / f"{name}.json")
    for name in ["design", "summary", "results", "scorer-environment"]:
        data = json.loads((ROOT / "study" / f"{name}.json").read_text())
        if name == "scorer-environment":
            data.pop("encoder_path", None)
            data["encoder_repo"] = "mkrausio/EmoWhisper-AnS-Small-v0.1"
            data["encoder_revision"] = "e613edc6dfef73525a11faa84f1f94f10e5a1744"
        (PUBLIC / "study" / f"{name}.json").write_text(json.dumps(data, indent=2))
    for path in sorted((ROOT / "study" / "audio").glob("*.mp3")):
        shutil.copyfile(path, PUBLIC / "study" / "audio" / path.name)
    if (ROOT / 'search/index.html').is_file():
        (PUBLIC / 'search/audio').mkdir(parents=True, exist_ok=True)
        for name in ['index.html','design.json','results.json','validation.json','emotion-check.json']:
            shutil.copyfile(ROOT / 'search' / name, PUBLIC / 'search' / name)
        for path in sorted((ROOT / 'search/audio').glob('*.mp3')):
            shutil.copyfile(path, PUBLIC / 'search/audio' / path.name)
    (PUBLIC / 'code').mkdir(exist_ok=True)
    sources=['beam_search.py','search_scoring.py','metrics.py','performance_core.py','search_benchmark.py',
             'search_validation.py','search_emotion_check.py','search_report.py','engine.py','small_app.py',
             'index.html','prompting.py','alignment.py','storage.py','director.py','actor_prompt.py',
             'adaptive_reward.py','dimension_predictors.py','reward_catalog.json','build_reward_catalog.py',
             'adaptive_reward_validation.py','adaptive_reward_browser_test.py',
             'burst_reward.py','burst_models.py','burst_reward_validation.py','burst_reward_browser_test.py',
             'ui_controller_test.cjs','ui_layout_browser_test.py']
    for name in sources:
        shutil.copyfile(ROOT / name, PUBLIC / 'code' / name)
    manifest = []
    for path in sorted(PUBLIC.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(PUBLIC).as_posix()
        allowed = relative in {"README.md", "index.html", "code/README.md", "realization-check.html"} or relative in {'code/'+name for name in sources} or re.fullmatch(
            r"(?:performance/(?:index\.html|(?:baseline|optimizations|compile|sweep|validation|latency|long)\.json)|study/(?:index\.html|(?:design|summary|results|scorer-environment)\.json|audio/[A-Za-z0-9_.-]+\.mp3))", relative)
        allowed = allowed or re.fullmatch(r'search/(?:index\.html|(?:design|results|validation|emotion-check)\.json|audio/[A-Za-z0-9_.-]+\.mp3)', relative)
        if not allowed:
            raise ValueError("Unexpected publication file: " + relative)
        contents = path.read_bytes()
        if path.suffix in {".json", ".html", ".md", ".py"} and re.search(rb"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}", contents):
            raise ValueError("Credential-like content detected")
        manifest.append({"path": relative, "bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()})
    (PUBLIC / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"public_files": len(manifest)+1, "mp3_files": sum(x['path'].endswith('.mp3') for x in manifest),
                      "total_bytes": sum(x['bytes'] for x in manifest)}), flush=True)


def main():
    disable_progress_bars()
    apis = []
    for number in [1, 2]:
        token = getpass.getpass(f"Token {number} (not saved): ")
        apis.append(HfApi(token=token))
        del token
    selected = None
    for number, api in enumerate(apis, 1):
        try:
            who = api.whoami()
            print(json.dumps({"credential": number, "user": who['name'],
                "role": who.get('auth', {}).get('accessToken', {}).get('role'),
                "organizations": [{"name": x['name'], "role": x.get('roleInOrg')} for x in who.get('orgs', [])]}), flush=True)
            if selected is None and any(x['name'] == 'laion' and x.get('roleInOrg') in {'write', 'admin'} for x in who.get('orgs', [])):
                selected = api
        except Exception as exc:
            print(json.dumps({"credential": number, "error_type": type(exc).__name__}), flush=True)
    api = selected or apis[0]
    print("Commands ready; credential values will never be printed.", flush=True)
    while True:
        try:
            command = input("publish> ").strip()
            if command == "quit":
                break
            if command == "inspect":
                info = api.model_info(MODEL)
                parent = info.sha
                readme = Path(hf_hub_download(MODEL, "README.md", revision=parent, token=api.token))
                dest = ROOT / "publication" / "model-README.md"
                dest.parent.mkdir(exist_ok=True)
                shutil.copyfile(readme, dest)
                print(json.dumps({"model": MODEL, "parent_commit": parent, "readme_copy": str(dest)}), flush=True)
            elif command == "prepare":
                prepare()
            elif command == "space":
                api.create_repo(SPACE, repo_type="space", space_sdk="static", private=False, exist_ok=True)
                files = sorted(p for p in PUBLIC.rglob("*") if p.is_file())
                # Small commits avoid the Hub's per-commit regular-file limit.
                for offset in range(0, len(files), 150):
                    batch = files[offset:offset+150]
                    commit = api.create_commit(repo_id=SPACE, repo_type="space",
                        operations=[CommitOperationAdd(path_in_repo=p.relative_to(PUBLIC).as_posix(), path_or_fileobj=str(p)) for p in batch],
                        commit_message=f"Publish study artifacts {offset+1}-{offset+len(batch)} of {len(files)}")
                    print(json.dumps({"uploaded": offset+len(batch), "total": len(files), "commit": commit.oid}), flush=True)
            elif command == "model":
                commit = api.create_commit(repo_id=MODEL, repo_type="model", parent_commit=parent,
                    operations=[CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(ROOT / "publication" / "model-README.md"))],
                    commit_message="Document pilot CFG/temperature recommendations and link public study reports")
                print(json.dumps({"model_commit": commit.oid}), flush=True)
            elif command == "status":
                info = api.space_info(SPACE)
                print(json.dumps({"space": SPACE, "host": info.host, "sha": info.sha,
                    "runtime": info.runtime.raw if info.runtime else None}), flush=True)
            elif command.startswith("use "):
                api = apis[int(command.split()[1])-1]
                print("Selected credential " + command.split()[1], flush=True)
            else:
                print("Unknown command", flush=True)
        except EOFError:
            break
        except Exception as exc:
            response = getattr(exc, "response", None)
            print(json.dumps({"error_type": type(exc).__name__, "http_status": getattr(response, "status_code", None)}), flush=True)


if __name__ == "__main__":
    main()
