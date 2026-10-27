"""Download only pinned EfficientAT inference sources and one small checkpoint."""
import hashlib,json,os
from pathlib import Path
import httpx
REVISION='a425fdce92572e602a1d5634799bd9f1f2efa806'
ROOT=Path(os.environ.get('ROBO_EVENT_CACHE','/mnt/nvme/robby-sound-events'))
FILES=['LICENSE','models/mn/model.py','models/mn/block_types.py','models/mn/utils.py','models/mn/attention_pooling.py','models/preprocess.py','helpers/utils.py','metadata/class_labels_indices.csv']
WEIGHT_URL='https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn04_as_mAP_432.pt'
WEIGHT_SHA256='899a8c6217063f6941793102a2780b3dd788ab11fce468d9cf5f0577c453321c'
def assets():
    ROOT.mkdir(parents=True,exist_ok=True);manifest={}
    with httpx.Client(timeout=60,follow_redirects=True,trust_env=False) as c:
        for name,url in [(f,f'https://raw.githubusercontent.com/fschmid56/EfficientAT/{REVISION}/{f}') for f in FILES]+[('mn04_as.pt',WEIGHT_URL)]:
            path=ROOT/name
            if not path.is_file():
                r=c.get(url);r.raise_for_status();path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_bytes(r.content);tmp.replace(path)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if name=='mn04_as.pt' and WEIGHT_SHA256 and digest!=WEIGHT_SHA256:raise ValueError('EfficientAT checkpoint checksum mismatch')
            manifest[name]={'url':url,'bytes':path.stat().st_size,'sha256':digest}
    (ROOT/'manifest.json').write_text(json.dumps({'revision':REVISION,'files':manifest},indent=2))
    return ROOT
if __name__=='__main__':
    root=assets();print((root/'manifest.json').read_text())
