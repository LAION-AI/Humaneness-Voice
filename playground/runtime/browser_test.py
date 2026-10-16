"""Real Chromium interaction: reference upload, CFG, manual prompt, candidates."""
import argparse
import json
from pathlib import Path
import subprocess

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8792")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--chunk-frames", type=int, default=10)
    args = parser.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/snap/bin/chromium", headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        response = page.goto(args.url)
        assert response.status == 200
        page.wait_for_function("!document.getElementById('send').disabled", timeout=360000)
        assert page.locator("#cfg").is_checked()
        assert page.locator("#g").input_value() == "1.5"
        assert page.locator("#temperature").get_attribute("min") == "0"
        assert page.locator("#align").is_checked()
        assert page.locator("#alignstatus").inner_text() == ""
        assert page.locator("#reference").input_value() == ""
        page.locator("#ref-toggle").click()
        page.locator("#fileA").set_input_files(ROOT / "test-results/english_caption_cfg1-take1.wav")
        page.wait_for_function("document.getElementById('reference').options.length === 2")
        assert page.locator("#reference").input_value()
        assert page.locator("#previewA").is_visible()
        page.locator("#fileB").set_input_files(ROOT / "test-results/transcript_reference-take1.wav")
        page.wait_for_function("document.getElementById('reference').options.length === 3")
        page.locator("#reference").select_option(index=1)
        page.locator("#inputmode").select_option("manual")
        page.locator("#format").select_option("caption")
        page.locator("#generation").select_option("best")
        page.locator("#n").fill(str(args.n))
        page.locator("#streaming").select_option("streaming" if args.streaming else "offline")
        if args.streaming:
            page.locator("#chunkframes").select_option(str(args.chunk_frames))
        else:
            assert page.locator("#chunkframes").is_disabled()
        page.evaluate("() => {window._originalTakes=takes; takes=(box,result)=>{window._testResult=result;window._originalTakes(box,result);};}")
        page.locator("#temperature").evaluate("(el,value)=>{el.value=value;el.dispatchEvent(new Event('input',{bubbles:true}));}", args.temperature)
        page.locator("#seconds").fill("4")
        page.locator("#autoplay").uncheck()
        page.locator("#q").fill('CAPTION: Warm, quiet conversation.\nTRANSCRIPT: "Ich habe dir das noch nie gesagt."')
        page.locator("#send").click()
        page.wait_for_selector(".take", timeout=180000)
        page.wait_for_function("!document.getElementById('send').disabled", timeout=180000)
        result=page.evaluate("window._testResult")
        performance=result['performance']
        assert performance['streaming']==args.streaming
        assert performance['device_seconds']>0 and performance['raw_audio_seconds']>0
        assert abs(performance['aggregate_rtf']-performance['device_seconds']/performance['raw_audio_seconds'])<1e-9
        assert sum(b['batch_size'] for b in performance['batches'])==args.n
        assert page.locator('#m_gpu').inner_text()!='—' and page.locator('#m_rtf').inner_text()!='—'
        (ROOT/'test-results/browser-performance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        assert page.locator(".take").count() == args.n
        assert page.locator(".take.best").count() == 1
        assert page.locator(".take .alignment").count() == args.n
        assert "Audio:" in page.locator(".take .alignment pre").first.text_content()
        assert page.locator(".msg.bot .bubble").inner_text() == "Ich habe dir das noch nie gesagt."
        assert not page.locator(".error").count()
        assert page.locator(".take audio").first.get_attribute("src").startswith("/api/audio/")
        audio = page.request.get(args.url + page.locator(".take audio").first.get_attribute("src"))
        assert audio.ok and audio.headers["content-type"] == "audio/mpeg"
        artifact = ROOT / "test-results/browser-take.mp3"
        artifact.write_bytes(audio.body())
        info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(artifact)]))["streams"][0]
        assert info["codec_name"] == "mp3" and info["channels"] == 1 and info["bit_rate"] == "96000"
        duration = page.locator(".take audio").first.evaluate("audio => new Promise((resolve,reject) => {audio.onloadedmetadata=()=>resolve(audio.duration);audio.onerror=()=>reject(Error('Cannot decode MP3'));audio.load();})")
        assert duration > 0
        page.locator(".msg.bot details").first.locator("summary").click()
        page.locator(".msg.bot details button").first.click()
        assert page.locator("#inputmode").input_value() == "manual"
        assert "CAPTION:" in page.locator("#q").input_value()
        page.screenshot(path=str(ROOT / "test-results/desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert page.locator("#q").is_visible()
        page.screenshot(path=str(ROOT / "test-results/mobile.png"), full_page=True)
        # Resetting history leaves user references intact and selectable.
        page.locator("#reset").click()
        assert page.locator("#reference option").count() == 3
        page.locator("#deleteA").click()
        page.wait_for_function("document.getElementById('reference').options.length === 2")
        page.locator("#deleteB").click()
        page.wait_for_function("document.getElementById('reference').options.length === 1")
        assert not errors, errors
        print(f"PASS: Chromium desktop/mobile, references A/B, CFG 1.5 default, temperature {args.temperature}, mono MP3 96kbps playback, Best-of-{args.n}, alignment, prompt editing, reset, deletion; no JS errors")
        browser.close()


if __name__ == "__main__":
    main()
