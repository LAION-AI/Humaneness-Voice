"""Check real UI with intercepted busy/error responses; do not occupy the GPU."""
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
URL = "https://switch-surprised-clock-sparc.trycloudflare.com"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/snap/bin/chromium", headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        errors = []
        requests = []
        status = {"ready": True, "busy": True, "job_phase": "Regie", "job_seconds": 32,
            "stage": "S3", "director": "gpt-5.6-luna", "director_ready": True,
            "alignment_available": True, "references": []}
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("**/api/state", lambda route: route.fulfill(json=status))
        def turn(route):
            requests.append(route.request.post_data_json)
            if len(requests) == 1:
                route.fulfill(content_type="text/event-stream", body='data: {"event":"status","phase":"Regie"}\n\ndata: {"event":"error","message":"Luna hat nicht rechtzeitig geantwortet. Bitte erneut senden."}\n\n')
            else:
                status["busy"] = True
                route.fulfill(status=409, json={"detail": "Es läuft gerade eine Generierung. Bitte kurz warten."})
        page.route("**/api/turn", turn)
        assert page.goto(URL).status == 200
        page.wait_for_function("document.getElementById('busy').textContent.includes('Regie (32 s)')")
        assert page.locator("#send").is_disabled()
        page.locator("#q").fill("Sag etwas Tröstendes.")
        page.locator("#q").press("Enter")
        assert not requests, "Do not send while another session occupies the server"
        status["busy"] = False
        page.evaluate("state()")
        page.wait_for_function("!document.getElementById('send').disabled")
        assert page.locator("#align").is_checked()
        assert page.locator("#format").input_value() == "caption"
        assert page.locator("#temperature").input_value() == "1"
        page.locator("#format").select_option("structured")
        page.locator("#temperature").fill("0.5")
        page.locator("#g").fill("3")
        page.locator("#cfg").uncheck()
        page.locator("#recommended").click()
        assert page.locator("#format").input_value() == "caption"
        assert page.locator("#temperature").input_value() == "1"
        assert page.locator("#g").input_value() == "1.5"
        assert page.locator("#cfg").is_checked()
        page.locator("#temperature").fill("0")
        page.locator("#send").click()
        page.wait_for_selector(".error")
        page.wait_for_function("!document.getElementById('send').disabled")
        assert page.locator(".msg.bot .bubble").first.inner_text() == "Anfrage nicht abgeschlossen."
        assert page.locator("#q").input_value() == "Sag etwas Tröstendes."
        assert requests[0]["align"] and requests[0]["guidance"] == 1.5 and requests[0]["temperature"] == 0
        assert requests[0]["format"] == "caption"
        page.locator("#send").click()
        page.wait_for_function("document.querySelectorAll('.error').length===2")
        page.wait_for_function("document.getElementById('send').disabled")
        assert page.locator(".msg.bot .bubble").last.inner_text() == "Anfrage nicht abgeschlossen."
        assert "Server belegt" in page.locator("#busy").inner_text()
        assert "Regie wird vorbereitet" not in page.locator("#chat").inner_text()
        status["busy"] = False
        page.evaluate("state()")
        page.wait_for_function("!document.getElementById('send').disabled")
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert page.locator("#q").is_visible()
        assert not errors, errors
        print("PASS: public HTTPS UI, busy session/Enter guard, timeout display, 409 display, retry, CFG/fades default, mobile; no GPU generation")
        browser.close()


if __name__ == "__main__":
    main()
