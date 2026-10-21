"""Real single-stream GPU cancellation check; abort only this owned HTTP turn."""
import asyncio,json,time
from pathlib import Path
import httpx
from conversation_director import validate
from final_verify import runtime_signature

async def main():
    report={'passed':False,'synthetic_only':True,'runtime_source_sha256':runtime_signature()}
    text=('A rainbow appears when sunlight enters tiny drops of water in the air. '
          'The light bends and separates into colors, reflects inside each drop, and bends again as it leaves. '
          'Different colors travel at slightly different angles. You usually see a rainbow with the sun behind you '
          'and rain ahead. Every observer sees light from different drops, so each rainbow depends on where you stand. '
          'The familiar arc is part of a circle that the ground normally hides.')
    spec=validate({'general':'Calm, gentle explanation.','script':'(interest, unhurried) '+text,'language':'en'})
    body={'model':'small_rl','message':spec['manual_prompt'],'input_mode':'manual','format':'structured',
          'generation':'single','n':1,'streaming':True,'chunk_frames':20,'cancel_on_disconnect':True,
          'temperature':.7,'guidance':2.,'voice_mode':'profile','voice_profile':'emolia_c1699',
          'voice_conversion':False,'align':False}
    try:
        async with httpx.AsyncClient(base_url='http://127.0.0.1:8792',timeout=90) as c:
            state=(await c.get('/api/state')).json()
            assert state.get('ready') and not state['busy'],'Shared service not idle; no request started'
            assert state.get('cooperative_disconnect_cancellation'),'Backend update not loaded'
            async with c.stream('POST','/api/turn',json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith('data: '):continue
                    event=json.loads(line[6:])
                    assert event.get('event')!='error','Generation error'
                    if event.get('event')=='audio_chunk':
                        report['first_chunk_seconds']=event.get('seconds');break
                else:raise AssertionError('No real streamed chunk before cancellation')
                assert (await c.get('/api/state')).json()['busy'],'Generation already ended; invalid cancellation test'
                start=time.monotonic()
            for _ in range(100):
                if not (await c.get('/api/state')).json()['busy']:break
                await asyncio.sleep(.05)
            else:raise AssertionError('Canceled worker did not release the model within five seconds')
            report['disconnect_to_idle_s']=time.monotonic()-start
            report['passed']=runtime_signature()==report['runtime_source_sha256']
    except Exception as exc:report['error']=str(exc)
    finally:
        Path('logs/shared-cancellation-live.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
    if not report['passed']:raise SystemExit('Scoped cancellation not verified')

if __name__=='__main__':asyncio.run(main())
