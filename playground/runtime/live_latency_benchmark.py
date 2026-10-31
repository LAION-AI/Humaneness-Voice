"""Live TTS latency: time to first PCM and playback gaps through the running server.

Sends the same streaming request Robby sends (POST /api/turn, pcm_f32, live_only)
and simulates a listener that starts playing at the first chunk. A "gap" is any
moment where the next chunk arrives after the previous audio has finished.

    python live_latency_benchmark.py                      # server default chunking
    python live_latency_benchmark.py --first-chunk 2 5     # compare first-chunk sizes
"""
import argparse,base64,json,statistics,time
import httpx

PROMPT=("GENERAL: Use the same speaker as the reference audio. Preserve that speaker's voice identity. "
        "Use warm, subtle delivery, clear articulation and natural spontaneous phrasing.\nSCRIPT:\n"
        "(warm, playful) Mir geht es heute ganz wunderbar, meine Schaltkreise summen voller Freude!")

def request(client,first_chunk,seed,voice):
    body={'model':'small_rl','input_mode':'manual','format':'structured','message':PROMPT,'language':'de',
          'generation':'single','n':1,'batch_size':1,'streaming':True,'stream_transport':'pcm_f32','live_only':True,
          'chunk_frames':20,'cancel_on_disconnect':True,'temperature':.8,'guidance':2.,'voice_mode':'profile',
          'voice_profile':voice,'voice_conversion':False,'speaker_steering':False,'dynamic_reference':False,
          'align':True,'seed':seed}
    if first_chunk:body['first_chunk_frames']=first_chunk
    start=time.perf_counter();timeline=[];samples=0
    with client.stream('POST','/api/turn',json=body) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith('data: '):continue
            event=json.loads(line[6:])
            if event.get('event')=='error':raise RuntimeError(event)
            if event.get('event')=='audio_chunk' and event.get('pcm_f32'):
                samples+=len(base64.b64decode(event['pcm_f32']))//4
                timeline.append((time.perf_counter()-start,samples/48000))
    return timeline

def gaps(timeline):
    """Total silence a listener hears after playback started at the first chunk."""
    playing_until=timeline[0][0];played=0.;silence=0.
    for arrived,cumulative in timeline:
        if arrived>playing_until:silence+=arrived-playing_until;playing_until=arrived
        playing_until+=cumulative-played;played=cumulative
    return silence

def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base',default='http://127.0.0.1:8792')
    parser.add_argument('--voice',default='emolia_c1699')
    parser.add_argument('--first-chunk',type=int,nargs='*',default=[0],help='0 = server default')
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    with httpx.Client(base_url=args.base,timeout=300) as client:
        request(client,0,1,args.voice)  # warm
        for first in args.first_chunk:
            runs=[request(client,first,777+i,args.voice) for i in range(args.repeats)]
            print(json.dumps({'first_chunk_frames':first or 'server default',
                'first_pcm_s':round(statistics.median(r[0][0] for r in runs),3),
                'gap_s':round(statistics.median(gaps(r) for r in runs),3),
                'total_s':round(statistics.median(r[-1][0] for r in runs),3),
                'audio_s':round(statistics.median(r[-1][1] for r in runs),3),
                'timeline':[(round(t,2),round(a,2)) for t,a in runs[0]]}),flush=True)

if __name__=='__main__':main()
