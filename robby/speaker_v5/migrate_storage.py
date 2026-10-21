"""Balance only these studies' files across root and the separate NVMe volume.

Frozen V3 pooling array and persistent V4/V5 outputs are copied and SHA256
verified before compatibility symlinks replace their old locations. Only
verified redundant leaf copies are deleted. No live demo or TTS process stops.
"""
import json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
from speaker_v4.common import atomic,digest

APP=Path(__file__).resolve().parents[1]
CACHE=Path.home()/'.cache'
DEST=Path('/mnt/nvme/robo-speaker-studies')

def owned(pid,module):
    try:
        p=Path('/proc',str(pid));args=(p/'cmdline').read_bytes().split(b'\0')
        return (p/'cwd').resolve()==APP and module.encode() in args
    except (FileNotFoundError,ProcessLookupError):return False

def stop(record,module):
    path=APP/record
    if not path.exists():return
    pid=json.loads(path.read_text()).get('pid',0)
    if not owned(pid,module):
        if Path('/proc',str(pid),'cmdline').exists():raise RuntimeError('PID ownership mismatch; no process stopped')
        return
    os.kill(pid,signal.SIGTERM)
    for _ in range(200):
        if not owned(pid,module):return
        time.sleep(.05)
    raise RuntimeError('Owned study did not stop; not force-killed, no files moved')

def manifest(source):
    files=[source] if source.is_file() else sorted(p for p in source.rglob('*') if p.is_file())
    if source.is_symlink() or (source.is_dir() and any(p.is_symlink() for p in source.rglob('*'))):
        raise RuntimeError('Unexpected source symlink; inspect before migration')
    return {'.' if p==source else p.relative_to(source).as_posix():
        {'bytes':p.stat().st_size,'sha256':digest(p)} for p in files}

def verify(source,expected):
    if manifest(source)!=expected:raise RuntimeError('Hash verification failed; original copies retained')

def redirect_and_remove_copy(source,target,expected):
    backup=source.with_name(source.name+'.nvme-verified-backup')
    if backup.exists():raise RuntimeError('Unexpected previous backup; not overwritten')
    verify(source,expected);verify(target,expected)
    source.rename(backup)
    try:
        source.symlink_to(target,target_is_directory=target.is_dir())
        # Follow this one new compatibility pointer; verify its target directly.
        if source.resolve()!=target.resolve():raise RuntimeError('Redirect target mismatch')
        verify(target,expected)
    except BaseException:
        if source.is_symlink():source.unlink()
        backup.rename(source);raise
    # Never recursively delete a study tree: exact checksummed leaf copies only.
    if backup.is_file():
        if digest(backup)!=expected['.']['sha256']:raise RuntimeError('Backup changed; retained')
        backup.unlink()
    else:
        verify(backup,expected)
        for name in expected:(backup/name).unlink()
        for directory in sorted((p for p in backup.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):directory.rmdir()
        backup.rmdir()

def start(module,record,log,visible,args=()):
    path=APP/record
    if path.exists():
        pid=json.loads(path.read_text()).get('pid',0)
        if owned(pid,module):raise RuntimeError('A study process is already active; no duplicate started')
    with (APP/log).open('ab') as output:
        child=subprocess.Popen([sys.executable,'-u','-m',module,*args],cwd=APP,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':visible},stdout=output,stderr=output,
            stdin=subprocess.DEVNULL,start_new_session=True)
    print('Started',module,'PID',child.pid,flush=True);return child

def main():
    if not Path('/mnt/nvme').is_mount() or Path('/mnt/nvme').stat().st_dev==CACHE.stat().st_dev:
        raise RuntimeError('Separate NVMe mount not confirmed')
    if DEST.exists():raise RuntimeError('Destination already exists; inspect its manifest, do not overwrite')
    sources=[CACHE/'robo-v3/arrays/pool.npy',CACHE/'robo-v4',CACHE/'robo-v5']
    for p in sources:
        if not p.exists() or p.is_symlink():raise RuntimeError('Missing source or already migrated: '+str(p))
    baseline=json.loads((CACHE/'robo-v3/models/results.json').read_text())
    if not baseline.get('finished'):raise RuntimeError('V3 pooling array is not a frozen completed-study artifact')
    size=sum(p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) for p in sources)
    if shutil.disk_usage('/mnt/nvme').free-size<4*1024**3:raise RuntimeError('Migration would breach 4-GiB NVMe reserve')
    # Stop the queue and CPU watcher before its producer; never stop Robo/TTS.
    stop('speaker_v5/runner.json','speaker_v5.run')
    stop('speaker_v4/report_watcher.json','speaker_v4.finalize')
    stop('speaker_v4/runner.json','speaker_v4.run')
    DEST.mkdir(mode=0o700)
    targets=[DEST/'v3-frozen-pool.npy',DEST/'robo-v4',DEST/'robo-v5'];entries=[]
    before={p:shutil.disk_usage(p).free for p in ('/','/mnt/nvme','/mnt/raid')}
    for source,target in zip(sources,targets):
        print('Copy and hash-verify',source,'→',target,flush=True)
        expected=manifest(source)
        if source.is_file():shutil.copy2(source,target)
        else:shutil.copytree(source,target)
        verify(target,expected)
        entries.append({'source':str(source),'target':str(target),'files':expected})
    state={'started':time.time(),'status':'all copies verified; originals still retained','entries':entries,
        'free_bytes_before':before,'recovery':'Persistent NVMe copies; original paths preserved by compatibility symlinks. RAM feature caches remain separately regenerable.'}
    atomic(DEST/'migration.json',state)
    for entry in entries:redirect_and_remove_copy(Path(entry['source']),Path(entry['target']),entry['files'])
    state.update(status='NVMe active; only verified redundant original copies removed',completed=time.time(),
        free_bytes_after={p:shutil.disk_usage(p).free for p in before})
    atomic(DEST/'migration.json',state)
    worker=start('speaker_v4.run','speaker_v4/runner.json','speaker_v4/study.log','0')
    for _ in range(60):
        if worker.poll() is not None:raise RuntimeError('V4 resume failed; copied files remain safe on NVMe')
        progress=json.loads((APP/'speaker_v4/progress.json').read_text())
        if not progress.get('error') and progress.get('updated',0)>time.time()-10:break
        time.sleep(.5)
    else:raise RuntimeError('No fresh V4 progress; watcher and queue not started')
    start('speaker_v4.finalize','speaker_v4/report_watcher.json','speaker_v4/report_watcher.log','',('--wait',))
    start('speaker_v5.run','speaker_v5/runner.json','speaker_v5/study.log','0')
    print(json.dumps({'migration':str(DEST/'migration.json'),'free_bytes':state['free_bytes_after'],'all_hashes_verified':True}),flush=True)

if __name__=='__main__':main()
