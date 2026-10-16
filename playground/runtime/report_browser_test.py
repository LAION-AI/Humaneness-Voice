"""Read-only report/UI rendering tests; mock turns do not occupy a GPU."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
URL = "https://switch-surprised-clock-sparc.trycloudflare.com"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/snap/bin/chromium", headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width":1440,"height":1000})
        errors=[]
        page.on("pageerror",lambda e:errors.append(str(e)))
        for report in ["performance","study"]:
            response=page.goto(URL+"/"+report+"/")
            assert response.status==200
            assert page.locator("h1").is_visible()
            assert page.evaluate("getComputedStyle(document.body).color")=='rgb(236, 236, 241)', 'Dark-background report needs readable foreground colors'
            if report=="study":
                page.locator("#filter").fill("Fear caption g1.5 t1-s777 main")
                assert page.locator("#takes details").count()==1
                page.locator("#takes summary").click()
                audio=page.locator("#takes audio").first
                audio.evaluate("e => e.load()")
                page.wait_for_function("document.querySelector('#takes audio').readyState >= 1")
                assert audio.evaluate("e => e.duration")>0
            page.evaluate("document.activeElement.blur(); window.scrollTo(0,0)")
            page.screenshot(path=str(ROOT/"test-results"/(report+"-desktop.png")),full_page=False)
            page.set_viewport_size({"width":390,"height":844})
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            page.screenshot(path=str(ROOT/"test-results"/(report+"-mobile.png")),full_page=False)
            page.set_viewport_size({"width":1440,"height":1000})
        # Exercise the actual updated UI/JS while the GPU is busy with study
        # work. No fake metric is saved into the measured result files.
        page.route(URL+"/",lambda route:route.fulfill(content_type="text/html",body=(ROOT/"index.html").read_text()))
        original=(ROOT.parent/"index.html").read_text()
        css=original.split('<style>',1)[1].split('</style>',1)[0]
        page.route("**/legacy.css",lambda route:route.fulfill(content_type="text/css",body=css))
        state={"ready":True,"busy":False,"stage":"S3","director":"gpt-5.6-luna","director_ready":True,"alignment_available":True,"references":[]}
        page.route("**/api/state",lambda route:route.fulfill(json=state))
        take={"index":0,"rank":0,"sec":2.,"wer":0.,"reward":1.,"genuineness":2.,"blend":2.,"clap":.2,
              "hyp":"Ready when you are.","url":"/study/audio/main-g1.5-t1-s777-Affection-caption.mp3"}
        performance={"device_seconds":.5,"raw_audio_seconds":2.,"aggregate_rtf":.25,"audio_seconds_per_second":4.,
            "batches":[{"batch_size":1,"device_seconds":.5,"audio_seconds":[2.]}],"backend":"mock display fixture",
            "streaming":True,"includes":"TTS + codec","excludes":"network"}
        result={"event":"result","candidates":[take],"winner":0,"generation":"single","guidance":1.5,"cfg_effective":True,
                "llm_ms":0,"generation_ms":500,"scoring_ms":100,"alignment_ms":20,"total_ms":620,"performance":performance}
        requests=[]
        def respond(route):
            requests.append(route.request.post_data_json)
            route.fulfill(content_type="text/event-stream",body='data: '+json.dumps(result)+'\n\n')
        page.route("**/api/turn",respond)
        assert page.goto(URL).status==200
        page.wait_for_function("!document.getElementById('send').disabled")
        page.locator("#autoplay").uncheck()
        page.locator("#q").fill("A controlled display test")
        page.locator("#streaming").select_option("streaming")
        page.locator("#temperature").fill("0")
        page.locator("#send").click()
        page.wait_for_function("document.getElementById('m_rtf').textContent==='0.250'")
        assert page.locator("#m_gpu").inner_text()=="0.50 s"
        assert page.locator("#m_throughput").inner_text()=="4.00"
        assert requests[0]['streaming'] is True and requests[0]['temperature']==0
        page.set_viewport_size({"width":390,"height":844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert not errors,errors
        browser.close()
        print("PASS: HTTPS HTML reports, mobile layout, filter/audio playback, GPU RTF display, streaming request, temperature zero; no GPU use")


if __name__=="__main__":
    main()
