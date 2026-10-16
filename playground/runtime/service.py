"""Start/stop only this demo's own processes, with persistent logs and PID files."""
import argparse
import os
from pathlib import Path
import signal
import subprocess

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
PYTHON = "/home/c4r33u19/moss15v2/.venv/bin/python"


def pid_for(name):
    path = LOGS / f"{name}.pid"
    if not path.exists():
        return None
    pid = int(path.read_text().strip())
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return None
    if not command:
        return None
    expected = {"app": b"small_app:app", "tunnel": b"cloudflared", "waiter": b"wait_for_gpus.py"}[name]
    if expected not in command:
        raise RuntimeError(f"PID {pid} does not belong to {name}; refusing to signal it")
    return pid


def start(name):
    LOGS.mkdir(exist_ok=True)
    if pid := pid_for(name):
        print(f"{name} already running, PID {pid}")
        return
    environment = dict(os.environ)
    environment.pop("LD_LIBRARY_PATH", None)
    environment.setdefault("HF_HOME", "/mnt/nvme/hf_cache")
    environment["PYTHONUNBUFFERED"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    port = environment.get("SMALL_PORT", "8792")
    command = ([PYTHON, "-m", "uvicorn", "small_app:app", "--host", "127.0.0.1", "--port", port,
                "--timeout-graceful-shutdown", "5"]
               if name == "app" else ["/home/c4r33u19/.local/bin/cloudflared", "tunnel", "--url",
                                       f"http://127.0.0.1:{port}", "--no-autoupdate", "--protocol", "http2"])
    with (LOGS / f"{name}.log").open("ab") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=log, start_new_session=True)
    (LOGS / f"{name}.pid").write_text(str(process.pid))
    print(f"{name}: PID {process.pid}; logs/{name}.log")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "status", "tunnel", "wait-start"])
    args = parser.parse_args()
    if args.action == 'wait-start':
        LOGS.mkdir(exist_ok=True)
        if pid := pid_for('waiter'):
            print(f'GPU watcher already running, PID {pid}')
            return
        with (LOGS / 'waiter.log').open('ab') as log:
            process = subprocess.Popen([PYTHON, str(ROOT / 'wait_for_gpus.py'), '--smoke-test'], cwd=ROOT,
                env=dict(os.environ, PYTHONUNBUFFERED='1'), stdin=subprocess.DEVNULL,
                stdout=log, stderr=log, start_new_session=True)
        (LOGS / 'waiter.pid').write_text(str(process.pid))
        print(f'GPU watcher: PID {process.pid}; logs/waiter.log')
    elif args.action in {"start", "tunnel"}:
        start("app" if args.action == "start" else "tunnel")
    elif args.action == "status":
        for name in ["app", "tunnel", "waiter"]:
            print(name, pid_for(name) or "stopped")
    else:
        for name in ["waiter", "tunnel", "app"]:
            if pid := pid_for(name):
                os.kill(pid, signal.SIGTERM)
                print(f"Sent SIGTERM to {name}, PID {pid}")


if __name__ == "__main__":
    main()
