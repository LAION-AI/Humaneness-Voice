"""Download pinned browser dependencies, not webcam data or biometric profiles.

The upstream .cjs MIME is application/node and cannot be imported by a browser
worker with nosniff. Serve the identical bytes with a JavaScript extension.
"""
from pathlib import Path
import urllib.request
import hashlib,json,concurrent.futures

def main():
    folder=Path(__file__).parent/'static/deps';folder.mkdir(exist_ok=True)
    base='https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/'
    pinned='https://raw.githubusercontent.com/justadudewhohacks/face-api.js/8609b25a15f417398a4d2b95dc91985cd1e0c0c3/'
    assets=[(base+'vision_bundle.cjs','mediapipe-vision-0.10.21.js'),
        (base+'vision_bundle.cjs.map','vision_bundle_cjs.js.map'),(base+'README.md','mediapipe-README.txt'),
        ('https://www.apache.org/licenses/LICENSE-2.0.txt','mediapipe-LICENSE.txt'),
        ('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task','face_landmarker.task'),
        (pinned+'dist/face-api.min.js','face-api-0.22.2.js'),(pinned+'LICENSE','face-api-LICENSE.txt')]
    for name in ('vision_wasm_internal.js','vision_wasm_internal.wasm','vision_wasm_nosimd_internal.js','vision_wasm_nosimd_internal.wasm'):
        assets.append((base+'wasm/'+name,'wasm/'+name))
    for name in ('face_recognition_model-weights_manifest.json','face_recognition_model-shard1','face_recognition_model-shard2'):
        assets.append((pinned+'weights/'+name,'face-api/'+name))
    def fetch(item):
        source,name=item;path=folder/name;path.parent.mkdir(exist_ok=True)
        if not path.exists():
            last=None
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(source,timeout=45) as response:raw=response.read(20_000_001)
                    if len(raw)>20_000_000:raise ValueError('Asset exceeds bounded download size')
                    path.write_bytes(raw);last=None;break
                except Exception as exc:last=exc
            if last:raise last
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if name=='mediapipe-vision-0.10.21.js' and digest!='bf58c9b3f592304422947b51e1f7de360c67d4dfeaa900064352da0c5a7c9115':raise ValueError('MediaPipe bundle hash mismatch')
        print(name,path.stat().st_size,flush=True)
        return {'file':name,'source':source,'sha256':digest,'bytes':path.stat().st_size}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:records=list(executor.map(fetch,assets))
    (folder/'assets-manifest.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()
