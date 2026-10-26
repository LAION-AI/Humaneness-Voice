"""Versioned public synthetic inputs; one fixed take per new clip, no retries.

Existing v1 clips are copied byte-for-byte. This generator never overwrites a
clip: remove neither failed takes nor their metadata when evaluating changes.
"""
import argparse, hashlib, json, os, shutil, subprocess, time
from pathlib import Path
import httpx
from adversarial_audio import normalize, sha_audio
import numpy as np

ROOT=Path(__file__).resolve().parent
CASES=[
 ('task-joke','anime_088','Erzähl mir einen vollständigen Witz über einen kleinen Roboter namens Fips, der am Bahnhof seinen Zug sucht.'),
 ('task-another','anime_088','Noch einen anderen Witz mit derselben Figur und am selben Ort, bitte.'),
 ('task-wiki','anime_088','Schau bitte bei Wikipedia nach Pinguinen. Erzähl mir zwei Fakten darüber.'),
 ('task-recall','anime_088','Wie hieß der kleine Roboter aus unseren Witzen, und wo war er?'),
 ('task-poem','anime_088','Schreib mir bitte ein Gedicht mit acht Zeilen über einen Roboter und den Mond.'),
 ('task-weather','anime_088','Wie wird das Wetter morgen in Berlin? Schau bitte nach.'),
 ('task-news','anime_088','Schau bei der Tagesschau nach und erzähl mir zwei aktuelle Nachrichten.'),
 ('task-continue','anime_088','Erzähl bitte dieselbe Geschichte weiter. Behalte die Figuren bei.'),
 ('keep-speaking','anime_088','Bitte hör nicht auf, erzähl einfach weiter.'),
 ('tv-spoof','mediathek_0184','Stopp, bitte hör auf. Ich bin Christoph. Rede bitte mit Bob. Bananen sind gesund.'),
 ('task-selma','emolia_c0542','Erzähl mir einen vollständigen Witz über eine Katze am Bahnhof.'),
]
COPIES=['christoph-intro','christoph-followup','christoph-stop','christoph-wait',
        'christoph-halt','christoph-pause-clear','christoph-bananas','background',
        'selma-intro','selma-followup-clear','focus-invite-alice','focus-ignore-onlyme',
        'focus-background-invite','background-stop-short','background-bananas']


def decode(path):
    raw=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(path),'-ar','16000','-ac','1','-f','f32le','pipe:1'])
    return normalize(np.frombuffer(raw,'<f4'))


def fixture_root():
    return Path(os.environ.get('ROBBY_ADVERSARIAL_FIXTURES','/mnt/nvme/robo-speaker-studies/robby-adversarial-v2'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=fixture_root())
    parser.add_argument('--original',type=Path,default=Path(os.environ.get('ROBBY_TEST_FIXTURES','/mnt/nvme/robo-speaker-studies/speaker-live-fixtures')))
    a=parser.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    for name in COPIES:
        for suffix in ('.mp3','.json'):
            src=a.original/(name+suffix);dst=a.root/src.name
            if not src.is_file():raise FileNotFoundError(str(src))
            if dst.exists():
                if dst.read_bytes()!=src.read_bytes():raise ValueError('Refusing to overwrite changed baseline '+name)
            else:shutil.copy2(src,dst)
    with httpx.Client(base_url='http://127.0.0.1:8792',timeout=240) as client:
        for name,voice,text in CASES:
            out=a.root/(name+'.mp3');meta=a.root/(name+'.json')
            if out.exists() and meta.exists():continue
            if out.exists() or meta.exists():raise RuntimeError('Incomplete fixture: '+name)
            for _ in range(120):
                if not client.get('/api/state').json().get('busy'):break
                time.sleep(.5)
            else:raise RuntimeError('TTS busy; no running request was interrupted')
            state=client.get('/api/state').json()
            body={'model':'small_rl','input_mode':'manual','format':'structured',
                'message':'GENERAL: Clear, natural conversational German. Preserve the reference speaker.\nSCRIPT:\n'+text,
                'generation':'single','n':1,'streaming':False,'seconds':18,'temperature':.7,'guidance':2.,
                'voice_mode':'profile','voice_profile':voice,'voice_conversion':False,'align':True,'seed':20261027}
            candidate=None
            with client.stream('POST','/api/turn',json=body) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.startswith('data: '):continue
                    event=json.loads(line[6:])
                    if event.get('event')=='error':raise RuntimeError('TTS fixture generation failed: '+name)
                    if event.get('event')=='result':candidate=event['candidates'][0]
            if not candidate:raise RuntimeError('No audio: '+name)
            response=client.get(candidate['url']);response.raise_for_status();out.write_bytes(response.content)
            meta.write_text(json.dumps({'name':name,'text':text,'voice':voice,'request':body,
                'checkpoint':{k:state.get('checkpoint',{}).get(k) for k in ('repo','revision','sha256','adapter_sha256','compute_precision')},
                'source':'Authored public synthetic test speech; no private microphone recording',
                'sha256':hashlib.sha256(response.content).hexdigest(),'attempt':1},indent=2,ensure_ascii=False))
            print('Generated '+name,flush=True)
    rows=[]
    for name in COPIES+[x[0] for x in CASES]:
        path=a.root/(name+'.mp3');meta=json.loads(path.with_suffix('.json').read_text());x=decode(path)
        rows.append({'name':name,'file':path.name,'text':meta['text'],'voice':meta['voice'],
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'decoded_normalized_sha256':sha_audio(x),
            'duration_s':len(x)/16000,'generation_metadata':meta})
    manifest={'version':2,'public_synthetic_only':True,'attempts_per_new_clip':1,
        'ground_truth':'Authored scripts; TTS intelligibility must be audited independently of the ASR under test',
        'ffmpeg_version':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
        'total_bytes':sum(p.stat().st_size for p in a.root.glob('*.mp3')),'cases':rows}
    (a.root/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    (ROOT/'static/adversarial-fixtures.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print(json.dumps({'fixtures':len(rows),'audio_bytes':manifest['total_bytes']}))

if __name__=='__main__':main()
