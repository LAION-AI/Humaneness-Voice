"""Lightweight status page while isolated GPU benchmarks own the TTS GPU."""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from small_app import performance_report, performance_data, emotion_report, study_data, study_audio

app = FastAPI()
app.add_api_route("/performance/", performance_report)
app.add_api_route("/performance/{filename}", performance_data)
app.add_api_route("/study/", emotion_report)
app.add_api_route("/study/{filename}", study_data)
app.add_api_route("/study/audio/{filename}", study_audio)


@app.get("/api/state")
def state():
    return {"ready": False, "busy": False, "error": None, "phase": "Chunk-search GPU / quality study in progress",
            "references": [], "director_ready": True, "stage": "S3", "director": "gpt-5.6-luna"}


@app.get("/", response_class=HTMLResponse)
def index():
    return '<!doctype html><html lang="en"><meta name="viewport" content="width=device-width"><title>Chunk-search study</title><body style="background:#101014;color:#ddd;font:18px system-ui;max-width:650px;margin:12vh auto;padding:24px"><h1>Chunk-search experiments in progress</h1><p>The demo is temporarily paused for isolated GPU measurements. The same HTTPS URL will return to the demo afterwards.</p><p><a style="color:#9cf" href="https://huggingface.co/spaces/laion/humaneness-voice-small-studies">Completed GPU / emotion reports on Hugging Face</a></p></body></html>'
