"""Hash-verified migration of THIS study's regenerable cache to roomy tmpfs.

Stops/restarts only the validated study worker. Models, source dataset and live
services are not migration targets. Original disk copies are removed only after
verified copies exist. Persistent manifest records exact hashes and recovery.
"""
import json,os,shutil,signal,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
from .common import ROOT,WORK,FEATURES,atomic,digest

def main():
    if FEATURES.is_symlink():raise RuntimeError('Cache already redirected; inspect migration manifest before any further move')
    if not (WORK/'protocol.json').exists() or FEATURES.resolve()!=ROOT.parent/'.cache/speaker-v4':
        raise RuntimeError('Not the validated V4 cache')
    files=list(FEATURES.iterdir());allowed={'q.npy','scale.npy','filled.npy','provenance.json'}
    if any(p.is_symlink() or not p.is_file() or p.name not in allowed for p in files):raise RuntimeError('Unexpected cache contents; refusing migration')
    size=sum(p.stat().st_size for p in files)
    mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    if size>1_600_000_000 or shutil.disk_usage('/dev/shm').free<size+8*1024**3 or mem['MemAvailable']<size+8*1024**3:
        raise RuntimeError('Insufficient RAM budget for bounded cache migration')
    pid=json.loads((ROOT/'runner.json').read_text())['pid'];process=Path('/proc',str(pid))
    if b'speaker_v4.run' not in (process/'cmdline').read_bytes() or (process/'cwd').resolve()!=ROOT.parent:
        raise RuntimeError('Study PID ownership mismatch')
    os.kill(pid,signal.SIGTERM)
    for _ in range(100):
        if not process.exists() or not (process/'cmdline').read_bytes():break
        time.sleep(.05)
    else:raise RuntimeError('Worker did not stop; no cache moved')
    target=Path(tempfile.mkdtemp(prefix='robo-v4-layers-',dir='/dev/shm'));manifest={}
    for source in files:
        destination=target/source.name;shutil.copyfile(source,destination)
        checksum=digest(source)
        if checksum!=digest(destination):raise RuntimeError('Cache-copy hash mismatch; originals retained')
        manifest[source.name]={'bytes':source.stat().st_size,'sha256':checksum}
    for name in ('q.npy','scale.npy','filled.npy'):
        a=np.load(FEATURES/name,mmap_mode='r');b=np.load(target/name,mmap_mode='r')
        if a.shape!=b.shape or a.dtype!=b.dtype:raise RuntimeError('Cache schema mismatch')
    backup=FEATURES.with_name('speaker-v4-verified-disk-copy-'+str(int(time.time())))
    if backup.exists():raise RuntimeError('Migration backup collision; originals retained')
    atomic(WORK/'cache_migration.json',{'source':str(FEATURES),'temporary_target':str(target),'files':manifest,
        'bytes':size,'status':'copies hash-verified; originals still present','recovery':'Regenerate temporary layers from preserved V3 sources and deterministic seeds after reboot'})
    FEATURES.rename(backup);FEATURES.symlink_to(target,target_is_directory=True)
    for item in files:
        if digest(FEATURES/item.name)!=manifest[item.name]['sha256']:raise RuntimeError('Redirect verification failed; disk backup retained')
    # Exact validated leaf files only; never recursive deletion or user data.
    for name in manifest:(backup/name).unlink()
    backup.rmdir()
    state=json.loads((WORK/'cache_migration.json').read_text());state.update(status='active RAM cache; verified redundant disk copy removed',completed=time.time())
    atomic(WORK/'cache_migration.json',state)
    with (ROOT/'study.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u','-m','speaker_v4.run'],cwd=ROOT.parent,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},
            stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    print(json.dumps({'cache_bytes_moved':size,'all_hashes_verified':True,'temporary_ram_path':str(target),'resumed_worker':child.pid}))

if __name__=='__main__':main()
