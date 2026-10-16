"""Real HTTPS/GPU VC smoke across live, offline, best-of-N and branching search.

Integration, not an identity/emotion improvement study. No user uploads altered.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
URL = 'https://switch-surprised-clock-sparc.trycloudflare.com'


def main():
    report = {'verified_at': datetime.now(timezone.utc).isoformat(), 'base': URL,
              'scope': 'Real HTTPS Chromium, GPU TTS + VC + ASR/rewards; integration only', 'cases': []}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage'])
        context = browser.new_context(viewport={'width': 1440, 'height': 1050})
        page = context.new_page(); errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        assert page.goto(URL).status == 200
        page.wait_for_function('ready && !running && !!smallLoras', timeout=240000)
        page.evaluate('document.querySelectorAll(".settings-card").forEach(el=>el.open=true)')
        assert not page.locator('#vc').is_checked() and page.locator('#vc').is_enabled()
        page.locator('#vc').check()
        page.locator('#voice-choice').select_option('none')
        assert page.locator('#vc').is_disabled() and not page.locator('#vc').is_checked()
        page.locator('#voice-choice').select_option('emolia_c0542')
        page.locator('#vc').check()
        page.locator('#inputmode').select_option('manual')
        page.locator('#language').select_option('en')
        page.evaluate('''() => {
            const oldChunk=LivePlayer.prototype.chunk;
            LivePlayer.prototype.chunk=async function(event){
                await oldChunk.call(this,event);
                window.chunkEvents.push({...event,beforeResult:!window.actualResult,
                    decodedSeconds:this.buffers.at(-1).duration,contextState:this.ctx.state});
            };
            const oldTakes=takes;
            takes=function(box,result){window.actualResult=result;return oldTakes(box,result);};
        }''')
        prompt = ('CAPTION: Use the same speaker as the reference audio. Preserve that speaker\'s voice identity. '
                  'Warm, relieved conversational speech with a brief pause between sentences.\n'
                  'TRANSCRIPT: "You came back. I thought I would never see you again."')
        cases = [('single-live', 'single', 'streaming', 20),
                 ('single-offline', 'single', 'offline', 20),
                 ('best-two-offline', 'best', 'offline', 20),
                 ('search-winner-two-800', 'search_winner', 'streaming', 10),
                 ('search-top-two-four-1600', 'search_beam', 'streaming', 20)]
        for label, generation, streaming, frames in cases:
            page.wait_for_function('ready && !running', timeout=240000)
            page.evaluate('window.chunkEvents=[];window.actualResult=null')
            page.locator('#generation').select_option(generation)
            if generation.startswith('search_'):
                page.locator('#searchn').select_option('4' if generation == 'search_beam' else '2')
                page.locator('#searchchunk').select_option(str(frames))
                page.locator('#searchseconds').fill('6.4')
            else:
                page.locator('#streaming').select_option(streaming)
                if generation == 'best': page.locator('#n').fill('2')
            page.locator('#q').fill(prompt)
            page.locator('#send').click()
            page.wait_for_function('actualResult!==null || !!document.querySelector(".error")', timeout=300000)
            assert not page.locator('.error').count(), page.locator('.error').all_text_contents()
            page.wait_for_function('!running', timeout=10000)
            result = page.evaluate('actualResult'); chunks = page.evaluate('chunkEvents')
            vc = result['voice_conversion']
            assert vc['enabled'] and vc['native_sample_rate'] == 24000
            assert vc['conversion_calls'] > 0 and vc['conversion_seconds'] > 0
            assert vc['scoring_audio'] == 'converted output, never raw TTS'
            assert result['reference']['mode'] == 'profile' and result['reference']['dynamic']
            assert 37 < result['reference']['tts_context']['codec_frames'] <= 188
            assert 0 < vc['reference_used_seconds'] <= 10
            assert len(result['candidates']) == (2 if generation == 'best' else 1)
            is_live = label == 'single-live' or generation.startswith('search_')
            if is_live:
                assert chunks and chunks[0]['beforeResult']
                assert page.evaluate('livePlayback.played && livePlayback.buffers.every(b=>b.numberOfChannels===1)')
                assert all(c['contextState'] == 'running' for c in chunks)
                assert abs(sum(c['decodedSeconds'] for c in chunks)-sum(c['seconds'] for c in chunks)) < .003
            else: assert not chunks
            if generation.startswith('search_'):
                perf = result['search_performance']
                assert perf['chunk_ms'] == frames*80
                assert perf['voice_conversion_seconds'] > 0
                assert all(step['voice_conversion_seconds'] > 0 for step in perf['timeline'])
                assert all(len(step['scores']) == perf['n'] for step in perf['timeline'])
                assert 'Chatterbox conversion before all candidate rewards' in perf['includes']
            for kind, url in [('take', result['candidates'][0]['url'])] + ([('chunk', chunks[0]['url'])] if chunks else []):
                response = context.request.get(URL+url)
                assert response.ok and response.headers['content-type'].startswith('audio/mpeg')
                path = ROOT/'test-results'/f'chatterbox-{label}-{kind}.mp3'
                path.write_bytes(response.body())
                stream = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
                    '-of', 'json', str(path)]))['streams'][0]
                assert stream['channels'] == 1 and int(stream['bit_rate']) == 96000
                foreign = browser.new_context()
                assert foreign.request.get(URL+url).status == 404
                foreign.close()
            record = {'case': label, 'chunks_before_result': len(chunks), 'result': result,
                      'chunks': chunks, 'passed': True}
            report['cases'].append(record)
            (ROOT/'test-results/chatterbox-live.json').write_text(json.dumps(report, indent=2))
            print(json.dumps({'case': label, 'chunks': len(chunks), 'vc_calls': vc['conversion_calls'],
                'vc_seconds': vc['conversion_seconds'], 'total_seconds': result['total_ms']/1000,
                'wer': result['candidates'][0]['wer'], 'passed': True}), flush=True)
        for width in [390, 320]:
            page.set_viewport_size({'width': width, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        assert not errors, errors
        report['javascript_errors'] = errors
        report['passed'] = True
        (ROOT/'test-results/chatterbox-live.json').write_text(json.dumps(report, indent=2))
        browser.close()


if __name__ == '__main__': main()
