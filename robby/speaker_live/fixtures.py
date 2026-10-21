"""Generate only explicit public-synthetic regression fixtures via shared TTS."""
import hashlib,json,time
from pathlib import Path
import httpx
from speaker_v4.common import atomic

ROOT=Path('/mnt/nvme/robo-speaker-studies/speaker-live-fixtures')
CASES=[('christoph-intro','anime_088','Hey Robbie, ich bin Christoph. Schön, dich kennenzulernen.'),
       ('christoph-followup','anime_088','Kannst du mir eine kleine Geschichte erzählen?'),
       ('robert-intro','k395_age3_bg1','Hey Robby, ich bin Robert. Wie geht es dir heute?'),
       ('robert-followup','k395_age3_bg1','Erzähl mir bitte einen kurzen Witz.'),
       ('selma-intro','emolia_c0542','Hey Robbie, ich bin Selma. Schön, dass du da bist.'),
       ('selma-followup','emolia_c0542','Was können wir heute zusammen machen?'),
       ('selma-followup-clear','emolia_c0542','Was können wir heute zusammen machen?'),
       ('background','mediathek_0184','Morgen wird es sonnig. Im Hintergrund läuft ein Bericht über das Wetter und die Nachrichten.'),
       ('robo-reply','emolia_c1699','Christoph, schön dich zu hören. Was möchtest du heute machen?'),
       ('christoph-stop','anime_088','Stopp.'),
       ('christoph-halt','anime_088','Halt, bitte hör auf.'),
       ('christoph-quiet','anime_088','Bitte sei mal leise.'),
       ('christoph-pause','anime_088','Pause, bitte.'),
       ('christoph-pause-clear','anime_088','Pause, bitte.'),
       ('christoph-stop-speaking','anime_088','Bitte hör auf zu sprechen.'),
       ('robert-stop','k395_age3_bg1','Stopp, bitte hör auf.'),
       ('background-stop','mediathek_0184','Stopp, bitte hör auf.'),
       ('robert-stop-short','k395_age3_bg1','Stopp.'),
       ('background-stop-short','mediathek_0184','Stopp.')]
CASES.extend([
    ('christoph-wait','anime_088','Warte bitte.'),
    ('christoph-listen','anime_088','Hör bitte auf.'),
    ('christoph-long-reading','anime_088','Schau bitte bei Wikipedia nach Bananen und lies mir eine ausführliche Erklärung in ungefähr hundert Wörtern vor.'),
    ('christoph-game','anime_088','Wir spielen ein Spiel. Wenn ich über Bananen rede, dann unterbrich mich bitte sofort und sage Stopp, stopp.'),
    ('christoph-bananas','anime_088','Bananen sind gesund, und Bananen haben viele wunderbare Eigenschaften. Ich könnte noch lange über Bananen reden. Man könnte den ganzen Tag Bananen essen, und ich möchte jetzt immer weiter über Bananen und ihre gelbe Farbe sprechen.'),
    ('christoph-apples','anime_088','Äpfel sind gesund und knackig. Ich esse gerne einen roten Apfel am Nachmittag. Es gibt viele unterschiedliche Apfelsorten, und auch Birnen und anderes Obst schmecken mir gut.'),
    ('background-bananas','mediathek_0184','Bananen sind gesund, und Bananen haben viele wunderbare Eigenschaften. Ich könnte noch lange über Bananen reden. Man könnte den ganzen Tag Bananen essen und immer weiter davon erzählen.'),
    ('christoph-stop-repeat','anime_088','Stopp, stopp, bitte hör jetzt auf!')])

def main():
    ROOT.mkdir(exist_ok=True);records=[]
    with httpx.Client(base_url='http://127.0.0.1:8792',timeout=240) as client:
        for name,voice,text in CASES:
            path=ROOT/(name+'.mp3');meta=ROOT/(name+'.json')
            if path.exists() and meta.exists():records.append(json.loads(meta.read_text()));continue
            for _ in range(120):
                if not client.get('/api/state').json().get('busy'):break
                time.sleep(1)
            body={'model':'small_rl','input_mode':'manual','format':'structured',
                'message':'GENERAL: Clear, natural conversational delivery. Preserve the reference speaker.\nSCRIPT:\n'+text,
                'generation':'single','n':1,'streaming':False,'seconds':9.6,'temperature':.7,'guidance':2.,
                'voice_mode':'profile','voice_profile':voice,'voice_conversion':False,'align':True}
            if name in ('christoph-game','christoph-bananas','christoph-apples','background-bananas'):
                body['seconds']=20.;body['seed']=20261022
                body['message']='GENERAL: Clear, natural conversational German. Preserve the reference speaker.\nSCRIPT:\n'+text
            if name=='selma-followup-clear':
                body['seed']=20261020
                body['message']='GENERAL: Clear, natural conversational German. Preserve the reference speaker.\nSCRIPT:\n[3.2 seconds duration] (warm, clear and natural) '+text
            if name=='christoph-pause-clear':
                body['seed']=20261021
                body['message']='GENERAL: Clear, firm and intelligible German speech. Preserve the reference speaker.\nSCRIPT:\n[2.0 seconds duration] (firm, clearly articulated) '+text
            candidate=None
            print('Generating synthetic fixture',name,flush=True)
            with client.stream('POST','/api/turn',json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith('data: '):continue
                    event=json.loads(line[6:])
                    if event.get('event')=='error':raise RuntimeError('Synthetic fixture generation failed')
                    if event.get('event')=='result':candidate=event['candidates'][0]
            if candidate is None:raise RuntimeError('TTS did not return fixture')
            audio=client.get(candidate['url']);audio.raise_for_status();path.write_bytes(audio.content)
            item={'name':name,'voice':voice,'text':text,'file':path.name,'sha256':hashlib.sha256(audio.content).hexdigest(),
                'source':'newly synthesized regression fixture, not a human microphone recording','model':'small_rl','seed':body.get('seed','service default')}
            atomic(meta,item);records.append(item)
    atomic(ROOT/'manifest.json',{'public_synthetic_only':True,'cases':records});print('Synthetic fixtures ready:',len(records),flush=True)

if __name__=='__main__':main()
