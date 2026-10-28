"""Reproduce three-model, 5-fps, one-call and private-context checks on authored media."""
import json,secrets,shutil,subprocess,tempfile
from pathlib import Path
import httpx,soundfile as sf
from environment_fixtures import music_fixture
from knowledge_memory import MemoryStore
from run_voice_assistant_checks import deployed_source_hash
ROOT=Path(__file__).resolve().parent

def main():
 token=secrets.token_hex(32);store=MemoryStore(token);quote='Bob likes jazz.'
 try:
  store.update({'entries':[{'title':'Bob','kind':'person','triggers':['Bob'],'facts':[{'key':'interests','value':'jazz','evidence':quote,'related':None}]}]},quote,0)
  with tempfile.TemporaryDirectory(prefix='lfm3-check-',dir=ROOT/'.runtime') as tmp:
   tmp=Path(tmp);sf.write(tmp/'music.wav',music_fixture(3),16000)
   subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','lavfi','-i','testsrc2=size=320x240:rate=10:duration=3','-i',str(tmp/'music.wav'),'-c:v','libx264','-preset','ultrafast','-pix_fmt','yuv420p','-c:a','aac','-shortest',str(tmp/'fixture.mp4')],check=True)
   with (tmp/'fixture.mp4').open('rb') as f:
    r=httpx.post('http://127.0.0.1:8810/analyze',files={'media':('fixture.mp4',f,'video/mp4')},data={'window':'3','stride':'1','context_enabled':'true','memory_capability':token,'recent_context':json.dumps([{'role':'user','content':'Bob likes jazz music'}])},timeout=240)
    r.raise_for_status();d=r.json();d.pop('playback_mp4_base64',None)
  s=d['segments'][0]
  checks={'private_context_used':d['relevance_context']['memory_used'],'context_bound':d['relevance_context']['conservative_token_bound']<=1000,'capability_not_returned':token not in json.dumps(d),
   'audio_one_call':s['audio']['model_calls']==1,'vision_five_fps':s['video']['sampling_fps']==5 and s['video']['encoded_frames_with_padding']>=14,
   'omni_one_call':s['omni']['model_calls']==1,'omni_grid_nine':s['omni'].get('contact_sheet_frames')==9,'omni_output':s['omni']['parse_ok']}
  report={'source_sha256':deployed_source_hash(),'checks':checks,'passed':all(checks.values()),'result':d,'limitations':['Synthetic input; not a real-world accuracy evaluation.','Model output and relevance can be biased by context.']}
  (ROOT/'static/lfm3-three-model-repro.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({'passed':report['passed'],'checks':checks,'wall_ms':d['wall_ms']}))
  if not report['passed']:raise SystemExit(1)
 finally:
  # This workspace was created with a fresh random test capability above.
  store.reset();shutil.rmtree(store.path)
if __name__=='__main__':main()
