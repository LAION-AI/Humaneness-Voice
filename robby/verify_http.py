"""Read-only public HTTPS checks; development/GPU routes must stay private."""
import argparse,hashlib,io,json,time,zipfile
from pathlib import Path
import requests

def main(base):
    base=base.rstrip('/');rows=[]
    for path in ('/','/desk-robot/','/expressions/','/static/desk-robot3d.js','/static/report.html',
            '/static/turn-latency.html','/speaker-training/320ms-study/','/speaker-training/context-study/'):
        response=requests.get(base+path,timeout=20)
        row={'path':path,'status':response.status_code,'content_type':response.headers.get('Content-Type','')}
        rows.append(row)
        assert response.status_code==200,row
        if path=='/desk-robot/':assert 'data-robot-view="desk3d"' in response.text
    for path in ('/internal/features','/internal/transcribe','/internal/sync-check'):
        response=requests.post(base+path,timeout=15);rows.append({'path':path,'method':'POST','status':response.status_code})
        assert response.status_code==403,rows[-1]
    response=requests.get(base+'/ws?synthetic_diagnostics=1',timeout=15)
    rows.append({'path':'/ws?synthetic_diagnostics=1','status':response.status_code})
    assert response.status_code==403,rows[-1]
    for private in ('/config.local.json','/logs/server.log'):
        response=requests.get(base+private,timeout=15)
        rows.append({'path':private,'status':response.status_code});assert response.status_code==404,rows[-1]
    response=requests.get(base+'/downloads/robby-source.zip',timeout=30);assert response.status_code==200
    local=Path(__file__).parent/'releases/robby-source.zip'
    digest=hashlib.sha256(response.content).hexdigest()
    assert digest==hashlib.sha256(local.read_bytes()).hexdigest(),'Hosted ZIP differs from local audited source'
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names=archive.namelist();assert not any(n.startswith('logs/') or n=='config.local.json' for n in names)
        manifest=json.loads(archive.read('SOURCE_MANIFEST.json'))
        for name,entry in manifest['files'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==entry['sha256']
    rows.append({'path':'/downloads/robby-source.zip','status':200,'sha256':digest,'manifest_files':len(manifest['files'])})
    state=requests.get(base+'/api/state',timeout=15).json()
    assert state.get('ready') and state.get('identity_available'),{k:state.get(k) for k in ('ready','identity_available')}
    result={'passed':True,'checked':time.time(),'non_mutating':True,'physical_microphone_used':False,
        'url':base,'checks':rows,'identity_models':state.get('identity_models')}
    path=Path(__file__).resolve().parent/'logs/http-checks.json';path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com')
    main(p.parse_args().url)
