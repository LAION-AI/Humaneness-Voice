"""Download only pinned inference files, without changing shared dependencies."""
import concurrent.futures,json,time
from asr_catalog import MODELS,cache_root

def prepare(key):
    from huggingface_hub import snapshot_download
    spec=MODELS[key];start=time.perf_counter()
    folder=snapshot_download(spec['repo'],revision=spec['revision'],allow_patterns=spec['files'],
        cache_dir=str(cache_root()/'hub'),token=False,max_workers=2)
    print(json.dumps({'model':key,'path':folder,'seconds':round(time.perf_counter()-start,2)}),flush=True)
    return key,folder

if __name__=='__main__':
    root=cache_root();root.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        result=dict(pool.map(prepare,[k for k,s in MODELS.items() if s['kind']!='native']))
    (root/'assets.json').write_text(json.dumps(result,indent=2))
