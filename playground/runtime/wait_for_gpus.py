"""Wait for genuinely idle requested GPUs, then start only this demo.

No foreign processes are signalled. Require multiple clean observations and
take an advisory singleton lock so two watchers cannot launch two servers.
Run foreground, or use `service.py wait-start` for persistent logs/PID.
"""
import argparse
import csv
import fcntl
import json
import os
import signal
import subprocess
import time
import urllib.request

from service import ROOT, LOGS, pid_for, start


def emit(event, **data):
    print(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'event': event, **data}), flush=True)


def gpu_status():
    def query(option):
        output = subprocess.check_output(['nvidia-smi', option, '--format=csv,noheader,nounits'], text=True, timeout=15)
        return list(csv.reader(output.strip().splitlines(), skipinitialspace=True))
    devices = {int(index): {'uuid': uuid, 'memory_mib': int(memory), 'utilization_percent': int(utilization)}
        for index, uuid, memory, utilization in query('--query-gpu=index,uuid,memory.used,utilization.gpu')}
    processes = [{'uuid': uuid, 'pid': int(pid), 'memory_mib': int(memory)}
        for uuid, pid, memory in query('--query-compute-apps=gpu_uuid,pid,used_gpu_memory')]
    return devices, processes


def available(devices, processes, indexes, own_pid=None, memory_limit=256, utilization_limit=5):
    for index in indexes:
        if index not in devices:
            return False
        gpu = devices[index]
        active = [process for process in processes if process['uuid'] == gpu['uuid']]
        if any(process['pid'] != own_pid for process in active):
            return False
        own_memory = sum(process['memory_mib'] for process in active if process['pid'] == own_pid)
        if gpu['memory_mib'] - own_memory > memory_limit or gpu['utilization_percent'] > utilization_limit:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--interval', type=float, default=15.)
    parser.add_argument('--consecutive', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=3600., help='seconds; 0 waits indefinitely')
    parser.add_argument('--smoke-test', action='store_true', help='After startup, exercise real LoRA generation in an isolated session')
    args = parser.parse_args()
    indexes = [int(value) for value in args.gpus.split(',')]
    if args.interval <= 0 or args.consecutive < 1 or args.timeout < 0 or not indexes:
        parser.error('Positive interval/consecutive, nonnegative timeout and GPU indices required')
    LOGS.mkdir(exist_ok=True)
    with (LOGS / 'waiter.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            emit('another_watcher_is_running')
            return
        from dpo_adapters import prefetch
        emit('prefetch_adapters_cpu_only')
        emit('adapters_cached', counts=prefetch()['counts'])
        started, streak = time.monotonic(), 0
        while not args.timeout or time.monotonic() - started < args.timeout:
            try:
                devices, processes = gpu_status()
                own = pid_for('app')
                idle = available(devices, processes, indexes, own_pid=own)
                if idle and own:
                    port = os.environ.get('SMALL_PORT', '8792')
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/state', timeout=5) as response:
                        state = json.load(response)
                    if state.get('busy') or not state.get('ready'):
                        idle = False
                    else:
                        # Only the validated demo process; preserve its tunnel.
                        os.kill(own, signal.SIGTERM)
                        emit('stopping_idle_demo', pid=own)
                        idle = False
                streak = streak + 1 if idle else 0
                emit('gpu_check', devices=devices, process_count=len(processes), idle=idle,
                    consecutive=streak, required=args.consecutive)
                if streak >= args.consecutive:
                    # Recheck immediately, then start with service's own PID guard.
                    latest, jobs = gpu_status()
                    if not available(latest, jobs, indexes):
                        streak = 0
                        continue
                    start('app')
                    start('tunnel')
                    emit('demo_started', pid=pid_for('app'), tunnel_pid=pid_for('tunnel'))
                    # Startup warming can take minutes. Persist its actual
                    # health outcome; never report a launched PID as ready.
                    health_started = time.monotonic()
                    port = os.environ.get('SMALL_PORT', '8792')
                    while time.monotonic() - health_started < 1200:
                        if not pid_for('app'):
                            emit('demo_exited_during_startup')
                            raise SystemExit(1)
                        try:
                            with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/state', timeout=5) as response:
                                state = json.load(response)
                            emit('startup_health', ready=state.get('ready'), phase=state.get('phase'), error=state.get('error'))
                            if state.get('error'):
                                raise SystemExit(1)
                            if state.get('ready'):
                                emit('demo_ready', pid=pid_for('app'), port=port)
                                if args.smoke_test:
                                    from service import PYTHON
                                    result = subprocess.run([PYTHON, str(ROOT / 'dpo_live_test.py')], timeout=1800)
                                    emit('post_start_smoke_finished', exit_code=result.returncode)
                                return
                        except OSError:
                            emit('startup_health', phase='HTTP listener starting')
                        time.sleep(args.interval)
                    emit('startup_timeout', seconds=1200)
                    raise SystemExit(1)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                streak = 0
                emit('check_failed_no_launch', error_type=type(exc).__name__, message=str(exc))
            time.sleep(args.interval)
        emit('timeout_no_processes_killed', timeout_seconds=args.timeout)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
