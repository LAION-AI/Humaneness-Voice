"""Assemble checked turns on the word clock; export matched mono MP3 episodes."""
import json,math,subprocess,re,hashlib,html,shutil
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'podcast-episodes'
SR=48000

def assemble(rows):
    # Execute the newly published pipeline's actual word-clock reassembler.
    import importlib.util
    source=Path(__file__).resolve().parent/'published_audio.py'
    spec=importlib.util.spec_from_file_location('published_podcast_audio',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    units=[];lengths=[]
    for i,row in enumerate(rows):
        wave,sr=sf.read(ROOT/row['edited'],dtype='float32')
        assert sr==SR and wave.ndim==1 and np.isfinite(wave).all()
        edge=row['edit']
        units.append((i,wave,{'first_word_s':edge['first_word_in_clip_s'],
                             'last_word_s':edge['last_word_in_clip_s']}))
        lengths.append(len(wave))
    whole,maps=module.assemble(units,gap_ms=400)
    timeline=[]
    for i,(row,m) in enumerate(zip(rows,maps)):
        timeline.append({'index':i,'speaker':row['turn']['speaker'],'text':row['turn']['text'],
            'clip_start_s':m['start_s'],'clip_end_s':m['start_s']+lengths[i]/SR,
            'speech_start_s':m['first_word_s'],'speech_end_s':m['last_word_s'],
            'source':row['edited']})
    gaps=[timeline[i]['speech_start_s']-timeline[i-1]['speech_end_s'] for i in range(1,len(timeline))]
    assert all(abs(g-.4)<1/SR for g in gaps),gaps
    return whole,timeline


def stamp(seconds):
    ms=round(seconds*1000);return f'{ms//3600000:02d}:{ms//60000%60:02d}:{ms//1000%60:02d},{ms%1000:03d}'

def export(wave,path,title,language):
    source=path.with_suffix('.source.flac');sf.write(source,wave,SR,subtype='PCM_24')
    common=['ffmpeg','-nostdin','-hide_banner','-y','-i',str(source)]
    p=subprocess.run(common+['-af','loudnorm=I=-18:TP=-1:LRA=11:print_format=json','-f','null','-'],capture_output=True,text=True,check=True)
    match=re.findall(r'\{\s*"input_i".*?\}',p.stderr,re.S);assert match
    measured=json.loads(match[-1]);f='loudnorm=I=-18:TP=-1:LRA=11:linear=true:print_format=json'
    for key,src in [('measured_I','input_i'),('measured_TP','input_tp'),('measured_LRA','input_lra'),('measured_thresh','input_thresh'),('offset','target_offset')]:f+=':'+key+'='+measured[src]
    p=subprocess.run(common+['-af',f,'-ar','48000','-ac','1','-c:a','libmp3lame','-b:a','192k',
          '-metadata','title='+title,'-metadata','artist=Mira and Idris (synthetic voices)',
          '-metadata','album=Small Questions, Big Feelings — Podcast model experiment',
          '-metadata','language='+language,str(path)],capture_output=True,text=True,check=True)
    # Decode the entire deliverable; this checks that an apparently complete file is playable.
    subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(path),'-f','null','-'],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(path)],text=True))
    assert probe['streams'][0]['codec_name']=='mp3' and probe['streams'][0]['channels']==1
    return {'file':path.name,'seconds':float(probe['format']['duration']),'bytes':path.stat().st_size,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'normalization_first_pass':measured,'sample_rate':48000,'channels':1,'bitrate':192000,'decoded_without_errors':True}
