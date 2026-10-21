"""Synthetic MP3 transport parity and bounded CPU decode microbenchmark."""
import asyncio, io, json, subprocess, sys, tempfile, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from tts_bridge import decode_mp3_ffmpeg as decode_mp3

async def main():
    rows=[]
    with tempfile.TemporaryDirectory(prefix='robby-decode-') as temp:
        for duration in (.6,1.6,3.2,10.):
            sr=48000;t=np.arange(round(duration*sr))/sr
            wave=(.15*np.sin(2*np.pi*237*t)+.05*np.sin(2*np.pi*701*t)).astype('float32')
            path=Path(temp)/'fixture.mp3'
            subprocess.run(['ffmpeg','-v','error','-y','-f','f32le','-ar',str(sr),'-ac','1','-i','pipe:0','-b:a','96k',str(path)],input=wave.tobytes(),check=True)
            raw=path.read_bytes()
            # Seekable ffmpeg honors the exact same encoder padding metadata
            # as soundfile. The old pipe decoder retains some trailing padding.
            seek=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-f','f32le','-ac','1','-ar','48000','pipe:1']),dtype='<f4')
            fast,rate=sf.read(io.BytesIO(raw),dtype='float32')
            old=await decode_mp3(raw)
            assert rate==48000 and len(fast)==len(seek)==len(wave)
            error=float(abs(fast-seek).max());assert error<2e-6,error
            prefix_error=float(abs(old[:len(fast)]-fast).max());assert prefix_error<2e-6,prefix_error
            old_ms=[];fast_ms=[]
            for _ in range(30):
                start=time.perf_counter();await decode_mp3(raw);old_ms.append((time.perf_counter()-start)*1000)
                start=time.perf_counter();await asyncio.to_thread(sf.read,io.BytesIO(raw),dtype='float32');fast_ms.append((time.perf_counter()-start)*1000)
            rows.append(dict(duration_s=duration,sample_rate=rate,max_pcm_error=error,old_prefix_error=prefix_error,
              trailing_padding_removed_samples=len(old)-len(fast),ffmpeg_ms=old_ms,soundfile_worker_ms=fast_ms))
    Path('logs/audio-decoder-speed.json').write_text(json.dumps(dict(passed=True,synthetic_only=True,rows=rows),indent=2))
    print(json.dumps([{k:v for k,v in r.items() if not isinstance(v,list)}|dict(old_median_ms=float(np.median(r['ffmpeg_ms'])),new_median_ms=float(np.median(r['soundfile_worker_ms']))) for r in rows],indent=2))

if __name__=='__main__':asyncio.run(main())
