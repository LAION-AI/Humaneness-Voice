"""Generate only three cached, public synthetic permission-test utterances."""
import hashlib,json,time
from pathlib import Path
import httpx
from speaker_live.fixtures import ROOT
CASES=[('focus-invite-alice','anime_088','Das ist meine Freundin Alice. Rede bitte auch mit ihr.'),
       ('focus-ignore-onlyme','anime_088','Ignoriere die Stimmen im Hintergrund. Rede bitte nur mit mir.'),
       ('focus-background-invite','mediathek_0184','Das ist mein Kumpel Bob. Rede bitte auch mit ihm.')]
def main():
    with httpx.Client(base_url='http://127.0.0.1:8792',timeout=180) as client:
        for name,voice,text in CASES:
            out=ROOT/(name+'.mp3');meta=ROOT/(name+'.json')
            if out.exists() and meta.exists():continue
            for _ in range(60):
                if not client.get('/api/state').json().get('busy'):break
                time.sleep(.5)
            body={'model':'small_rl','input_mode':'manual','format':'structured',
                'message':'GENERAL: Clear, natural conversational German. Preserve the reference speaker.\nSCRIPT:\n'+text,
                'generation':'single','n':1,'streaming':False,'seconds':12,'temperature':.7,'guidance':2.,
                'voice_mode':'profile','voice_profile':voice,'voice_conversion':False,'align':True,'seed':20261026}
            candidate=None
            with client.stream('POST','/api/turn',json=body) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.startswith('data: '):continue
                    event=json.loads(line[6:])
                    if event.get('event')=='error':raise RuntimeError('Synthetic fixture generation failed')
                    if event.get('event')=='result':candidate=event['candidates'][0]
            if not candidate:raise RuntimeError('No synthetic audio')
            response=client.get(candidate['url']);response.raise_for_status();out.write_bytes(response.content)
            meta.write_text(json.dumps({'text':text,'voice':voice,'seed':20261026,'source':'public synthetic test audio; no microphone',
                'sha256':hashlib.sha256(response.content).hexdigest()},indent=2,ensure_ascii=False))
            print('Generated',name,len(response.content),'bytes',flush=True)
if __name__=='__main__':main()
