"""Pin the exact public synthetic test inputs; never scans arbitrary recordings.

The optional 2-MB archive contains only explicitly named, authored TTS clips.
Metadata is allowlisted; private microphone recordings/voiceprints are excluded.
"""
import argparse, hashlib, json, os, subprocess, zipfile
from pathlib import Path
from run_voice_assistant_checks import FIXTURES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',default=os.environ.get('ROBBY_TEST_FIXTURES','/mnt/nvme/robo-speaker-studies/speaker-live-fixtures'))
    p.add_argument('--pack',action='store_true');a=p.parse_args()
    root=Path(__file__).parent;folder=Path(a.fixtures);rows=[];payload={}
    for name in FIXTURES:
        path=folder/(name+'.mp3');raw=path.read_bytes()
        meta=json.loads((folder/(name+'.json')).read_text())
        if not ('synthe' in meta.get('source','').lower()):raise ValueError('Fixture must be explicitly synthetic: '+name)
        pcm=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(path),'-ar','16000','-ac','1','-f','f32le','pipe:1'],capture_output=True,check=True).stdout
        row={k:meta[k] for k in ('voice','text','seed','model') if k in meta}
        row.update(name=name,file=path.name,bytes=len(raw),seconds_16k=len(pcm)/4/16000,
            sha256=hashlib.sha256(raw).hexdigest(),decoded_f32le_sha256=hashlib.sha256(pcm).hexdigest())
        rows.append(row);payload[path.name]=raw
    version=subprocess.run(['ffmpeg','-version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    report={'schema_version':1,'synthetic_only':True,'private_microphone_recordings':0,
        'speaker_voiceprints':0,'sample_rate':16000,'channels':1,'decoded_format':'float32 little endian',
        'ffmpeg':version,'normalization':'Each decoded source is scaled to full-clip RMS 0.055 by speaker_live.live_test.audio.',
        'attribution':'LAION Humaneness Voice; synthetic speech generated from laion/laion-voice-profiles-annotated reference profiles (CC BY 4.0). Texts authored for this regression suite. No private recordings.',
        'source_dataset':'https://huggingface.co/datasets/laion/laion-voice-profiles-annotated',
        'reproduction':'Restore this exact pack for identical source inputs. Some older clips used a service-default seed; regenerating them is a new fixture version, not bit-exact reproduction.',
        'cases':rows}
    raw=json.dumps(report,indent=2,ensure_ascii=False).encode()
    (root/'static/voice-test-fixtures.json').write_bytes(raw)
    if a.pack:
        target=root/'releases/robby-synthetic-fixtures.zip';target.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
            for name,data in sorted(payload.items()):archive.writestr(name,data)
            archive.writestr('manifest.json',raw)
        print(json.dumps({'file':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))
    else:print('Pinned',len(rows),'synthetic fixture hashes in static/voice-test-fixtures.json')

if __name__=='__main__':main()
