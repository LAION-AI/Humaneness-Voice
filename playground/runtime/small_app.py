"""Separate Small demo; the original server files are unchanged."""
import asyncio
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Literal
import uuid

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
import numpy as np
from pydantic import BaseModel, Field, model_validator
import soundfile as sf

import settings
from director import Director
from prompting import build, parse_literal
from storage import AudioStore
from adaptive_reward import CATALOG, RewardPlan
from burst_reward import BurstOptions
from speaker_reward import SpeakerOptions, SpeakerReward, UnavailableSpeakerReward, ReferenceTooShort, catalog as speaker_catalog
from dpo_adapters import AdapterSelection, catalog as adapter_catalog, validate_selection
import dynamic_references
from voice_conversion import ConversionManager
from ears_reward import RankingOptions, catalog as ears_catalog
from turn_cancellation import DisconnectCancellation, TurnCancelled

log = logging.getLogger("small")
STATE = {"engine": None, "scorers": None, "finisher": None, "error": None, "phase": "starting", "busy": False,
         "job_phase": None, "job_started": None}
store = AudioStore()
sessions = {}
director = Director()
vc_manager = ConversionManager()


def initialize():
    from engine import Scorers
    from acting_engine import ActingEngine
    STATE["phase"] = "loading default TTS checkpoint"
    STATE["engine"] = ActingEngine(settings.DEFAULT_MODEL)
    if settings.FAST_INFERENCE:
        STATE["phase"] = "warming compiled GPU decoder and CUDA graphs"
        STATE["engine"].warm_up()
        # Large search graphs are allocated only on explicit use; avoid
        # warming the old Small 8/12-row banks on a 24-GB card.
        STATE['phase']='warming optional single-take live streaming'
        STATE['engine'].warm_live_stream()
        if settings.DEFAULT_MODEL=='small_rl':
            STATE['phase']='warming two-stage continuous speech'
            from live_batch import warm_up
            warm_up(STATE['engine'])
    STATE["phase"] = "loading Parakeet and quality scorers"
    STATE["scorers"] = Scorers()
    vc_manager.bind_condition_scorer(STATE['scorers'].ears)
    from alignment import AudioFinisher
    STATE["phase"] = "loading Qwen word aligner"
    STATE["finisher"] = AudioFinisher()
    STATE["phase"] = "ready"
    # Independent tensor-only CPU prefetch; no CUDA model/RNG changes and no
    # delay to the default model's ready state. Errors remain visible in catalog.
    import threading
    from checkpoint_ram import RAM
    threading.Thread(target=RAM.preload_experimental,name='experimental-checkpoint-ram',daemon=True).start()


async def initialize_async():
    try:
        await asyncio.to_thread(initialize)
    except Exception as exc:
        log.exception("Model startup failed")
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        STATE["phase"] = "error"


async def cleanup():
    while True:
        await asyncio.sleep(60)
        for item in store.cleanup():
            if STATE["engine"]:
                STATE["engine"].forget_reference(item.path)
        cutoff = time.time() - settings.TTL
        for sid in list(sessions):
            if sessions[sid] < cutoff and not any((x.active or x.kind == 'reference') and x.owner == sid for x in store.items.values()):
                sessions.pop(sid, None)


@asynccontextmanager
async def lifespan(app):
    boot = asyncio.create_task(initialize_async())
    janitor = asyncio.create_task(cleanup())
    yield
    janitor.cancel()
    await asyncio.gather(janitor, return_exceptions=True)
    await boot
    await director.close()


app = FastAPI(title="Humaneness Voice Small", lifespan=lifespan)


def switch_acting_model(model_id):
    """Caller holds STATE.busy across both model switching and generation."""
    from acting_engine import ActingEngine
    old=STATE['engine']
    if getattr(old,'model_id',None)==model_id:return
    STATE['phase']='loading '+model_id
    # Finish verification / CPU prefetch before releasing the current GPU
    # model. A failed download must not evict the working default.
    from checkpoint_ram import RAM
    try:RAM.get(model_id)
    except Exception:
        STATE['phase']='ready' if old is not None else 'error'
        raise
    processor=old.unload() if old is not None else None
    STATE['engine']=None
    try:
        from model_memory import place_auxiliaries
        place_auxiliaries(STATE,model_id)
        engine=ActingEngine(model_id,processor=processor)
        STATE['engine']=engine
        if settings.FAST_INFERENCE:
            STATE['phase']='warming '+model_id+' streaming'
            engine.warm_up();engine.warm_live_stream()
        STATE['error']=None;STATE['phase']='ready'
    except Exception as exc:
        STATE['error']=f'{type(exc).__name__}: {exc}';STATE['phase']='error'
        raise


@app.get('/api/models/catalog')
def model_catalog():
    from acting_models import catalog
    return catalog()


class ModelChoice(BaseModel):
    model: Literal['large_acting','small_acting','small_alpha','large_alpha','small_rl','large_rl']


def candidate_batch_size(body):
    from acting_models import MODELS
    return (min(body.n,MODELS[body.model]['max_batch_size'])
            if body.generation in ('best','all') else body.batch_size)


@app.post('/api/models/select')
async def select_model(body: ModelChoice):
    if STATE['busy'] or STATE['phase'] not in ('ready','error'):
        raise HTTPException(409,'Wait for the current request before switching models.')
    STATE['busy']=True;STATE['job_phase']='Switching model';STATE['job_started']=time.monotonic()
    try:
        await asyncio.to_thread(switch_acting_model,body.model)
        return {'model':body.model,'checkpoint':STATE['engine'].checkpoint_info}
    finally:
        STATE['busy']=False;STATE['job_phase']=None;STATE['job_started']=None


@app.get('/api/adapters/catalog')
def get_adapter_catalog():
    data=adapter_catalog()
    return {**data,'enabled':False,'base_label':getattr(STATE['engine'],'release',{}).get('label','Acting'),
            'note':'Legacy S3 adapters are not compatible with the Acting checkpoints.'}


@app.get('/lora-ui.js')
def lora_ui():
    return FileResponse(settings.HERE / 'lora_ui.js', media_type='text/javascript')


@app.get('/acting-ui.js')
def acting_ui():
    return FileResponse(settings.HERE / 'acting_ui.js', media_type='text/javascript')


@app.get('/api/voices/catalog')
def voice_catalog():
    return dynamic_references.catalog()


@app.get('/api/voices/reference/{voice}')
def voice_reference(voice: str):
    try:
        path, _ = dynamic_references.fixed_clip(voice, playback=True)
    except (ValueError, StopIteration):
        raise HTTPException(404, 'Unknown reference voice')
    # Only the curated public corpus is exposed, never private user uploads.
    return FileResponse(path, media_type='audio/mpeg')


@app.get('/api/ranking/catalog')
def ranking_catalog():
    return ears_catalog()


@app.get('/api/speaker/catalog')
def speaker_reward_catalog():
    return speaker_catalog()


@app.get('/api/speaker-steering/catalog')
def speaker_steering_catalog():
    return {'default_enabled':False,'available':False,'voices':[],
            'reason':'Old checkpoint vectors are incompatible with Acting releases.'}


async def clear_speaker_steering(engine, reason):
    """Clear request-scoped steering before/after every serialized job."""
    if hasattr(engine, 'configure_speaker_steering'):
        await asyncio.to_thread(engine.configure_speaker_steering, None,
            {'requested': False, 'enabled': False, 'active': False, 'reason': reason})


async def prepare_speaker_steering(engine, body, reference):
    """Only the selected catalog identity may supply a steering vector."""
    report = {'requested': body.speaker_steering, 'enabled': False, 'active': False,
              'experimental': True, 'layer': 16, 'alpha': .75,
              'method': 'quality_high_minus_low_contrast', 'normalization': 'raw',
              'vector_source': 'quality_selected_takes_not_reference_no_reference_delta',
              'voice': body.voice_profile if body.voice_mode == 'profile' else None}
    if not hasattr(engine, 'configure_speaker_steering'):
        return {**report, 'reason': 'engine_unavailable'}
    if getattr(engine,'model_id',None):
        return await asyncio.to_thread(engine.configure_speaker_steering,None,
            {**report,'reason':'acting_checkpoint_no_steering'})
    from speaker_steering import resolve_speaker_steering
    vector, resolved = await asyncio.to_thread(resolve_speaker_steering, engine,
        body.voice_profile if body.voice_mode == 'profile' else None,
        reference.path if reference else None, enabled=body.speaker_steering,
        custom=body.voice_mode == 'custom')
    report.update(resolved)
    # Explicit opt-in LoRAs change the activation distribution; do not reuse
    # base-only vectors silently even if a simplified engine has no bank state.
    if body.adapters.active() and body.speaker_steering:
        vector = None
        report.update(enabled=False, active=False, reason='lora_active')
    return await asyncio.to_thread(engine.configure_speaker_steering, vector, report)


@app.get('/api/reward/catalog')
def reward_catalog():
    return CATALOG


class RewardBrief(BaseModel):
    description: str = Field(min_length=1,max_length=8000)


@app.post('/api/reward/plan')
async def preview_reward(body: RewardBrief):
    if STATE['busy']:
        raise HTTPException(409,'Generation in progress. Retry reward selection afterward.')
    try:
        plan=await asyncio.wait_for(director.reward_plan(body.description),timeout=35)
        return plan.model_dump()
    except asyncio.TimeoutError:
        raise HTTPException(504,'Luna reward selection timed out. Select dimensions manually.')
    except (ValueError,RuntimeError) as exc:
        raise HTTPException(502,str(exc))


@app.middleware("http")
async def session(request, call_next):
    sid = request.cookies.get("small_session")
    fresh = sid not in sessions and not store.references(sid)
    if fresh:
        sid = uuid.uuid4().hex
    sessions[sid] = time.time()
    request.state.sid = sid
    response = await call_next(request)
    response.set_cookie("small_session", sid, httponly=True, samesite="lax", max_age=365*86400,
                        secure=request.headers.get("x-forwarded-proto") == "https")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/", response_class=HTMLResponse)
def index():
    return (settings.HERE / "index.html").read_text()


@app.get("/realization-check.html")
def realization_report():
    path = settings.HERE / "test-results/realization-check.html"
    if not path.is_file():
        raise HTTPException(503, "Realization report is being prepared.")
    return FileResponse(path, media_type="text/html")


@app.get('/speaker-study/')
@app.get('/speaker-study.html')
def speaker_identity_report():
    path=settings.HERE/'speaker-study/index.html'
    if not path.is_file():
        raise HTTPException(503,'Speaker identity study is being prepared.')
    return FileResponse(path,media_type='text/html')


@app.get('/speaker-study/{filename}')
def speaker_identity_data(filename:str):
    # No source references, private embedding cache or arbitrary paths.
    if filename not in {'design.json','summary.json','results.json','reference-controls.json'}:
        raise HTTPException(404)
    path=settings.HERE/'speaker-study'/filename
    if not path.is_file():raise HTTPException(404)
    return FileResponse(path,media_type='application/json')


@app.get('/full-ft-speaker-study/')
def full_ft_identity_report():
    path=settings.HERE/'full-ft-speaker-study/index.html'
    if not path.is_file():
        raise HTTPException(503,'Full-FT identity/quality study is being prepared.')
    return FileResponse(path,media_type='text/html')


@app.get('/catalogue-steering-study/')
def catalogue_steering_report():
    path = settings.HERE / 'catalogue-steering-study/index.html'
    if not path.is_file():
        raise HTTPException(503, 'Catalog speaker-steering report is being prepared.')
    return FileResponse(path, media_type='text/html')


@app.get('/vc-performance/')
def vc_performance_report():
    path=settings.HERE/'vc-performance/index.html'
    if not path.is_file():raise HTTPException(503,'Voice conversion benchmark is being prepared.')
    return FileResponse(path,media_type='text/html')


@app.get('/acting-performance/')
def acting_performance_report():
    path=settings.HERE/'acting-performance.html'
    if not path.is_file():raise HTTPException(503,'Acting speed report is being prepared.')
    return FileResponse(path,media_type='text/html')


@app.get('/vc-performance/results.json')
def vc_performance_data():
    path=settings.HERE/'vc-performance/results.json'
    if not path.is_file():raise HTTPException(404)
    return FileResponse(path,media_type='application/json')


@app.get('/catalogue-steering-study/{filename}')
def catalogue_steering_data(filename: str):
    if filename not in {'design.json', 'results.json', 'summary.json'}:
        raise HTTPException(404)
    path = settings.HERE / 'catalogue-steering-study' / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type='application/json')


@app.get('/full-ft-speaker-study/{filename}')
def full_ft_identity_data(filename:str):
    if filename not in {'design.json','summary-pilot.json','summary-validation.json',
                        'results-pilot.json','results-validation.json','reference-controls.json','selection.json','validation-design.json','protocol-amendment.json',
                        'diagnostic-design.json','summary-diagnostic.json','results-diagnostic.json',
                        'followup-design.json','summary-followup.json','results-followup.json','measurement-sanity.json','backend-check.json'}:
        raise HTTPException(404)
    path=settings.HERE/'full-ft-speaker-study'/filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path,media_type='application/json')


@app.get('/mira-activation-study/')
def mira_activation_report():
    path = settings.HERE / 'mira-activation-study/index.html'
    if not path.is_file():
        raise HTTPException(503, 'Mira activation study is being prepared.')
    return FileResponse(path, media_type='text/html')


@app.get('/mira-activation-study/{filename}')
def mira_activation_data(filename: str):
    # Explicit public research artifacts only: never uploads, caches or arbitrary paths.
    allowed = {'design.json', 'summary.json', 'results.json', 'conclusions.json',
               'refine-plan.json', 'validation-plan.json', 'layer-contrast-plan.json',
               'balanced-diagnostic-plan.json', 'instrumentation-check.json',
               'measurement-amendment.json', 'reference-controls.json'}
    if filename == 'vectors.npz':
        path = settings.HERE / 'mira-activation-study/vectors/selected.npz'
        media_type = 'application/octet-stream'
    elif filename in allowed:
        path = settings.HERE / 'mira-activation-study' / filename
        media_type = 'application/json'
    else:
        raise HTTPException(404)
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type=media_type)


@app.get("/performance/")
def performance_report():
    path = settings.HERE / "performance/index.html"
    if not path.is_file():
        raise HTTPException(503, "Performance report is being prepared.")
    return FileResponse(path, media_type="text/html")


@app.get("/performance/{filename}")
def performance_data(filename: str):
    if filename not in {"baseline.json", "optimizations.json", "compile.json", "sweep.json", "validation.json", "latency.json", "long.json"}:
        raise HTTPException(404)
    path = settings.HERE / "performance" / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="application/json")


@app.get("/study/")
def emotion_report():
    path = settings.HERE / "study/index.html"
    if not path.is_file():
        raise HTTPException(503, "Emotion study is being prepared.")
    return FileResponse(path, media_type="text/html")


@app.get("/search/")
def search_report():
    path = settings.HERE / "search/index.html"
    if not path.is_file():
        raise HTTPException(503, "Search report is being prepared.")
    return FileResponse(path, media_type="text/html")


@app.get("/search/{filename}")
def search_data(filename: str):
    if filename not in {"design.json", "results.json", "validation.json", "emotion-check.json"}:
        raise HTTPException(404)
    path = settings.HERE / "search" / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="application/json")


@app.get("/search/audio/{filename}")
def search_audio(filename: str):
    if not re.fullmatch(r"(?:smoke|measure|quality|audit)-[a-z0-9_-]+\.mp3", filename):
        raise HTTPException(404)
    path = settings.HERE / "search/audio" / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/study/{filename}")
def study_data(filename: str):
    if filename not in {"design.json", "summary.json", "results.json", "scorer-environment.json"}:
        raise HTTPException(404)
    path = settings.HERE / "study" / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="application/json")


@app.get("/study/audio/{filename}")
def study_audio(filename: str):
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+\.mp3", filename):
        raise HTTPException(404)
    path = settings.HERE / "study/audio" / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/legacy.css")
def legacy_css():
    # Actual original demo styles, so changes there remain reflected here.
    original = (settings.PARENT / "index.html").read_text()
    return Response(re.search(r"<style>(.*?)</style>", original, re.S)[1], media_type="text/css")


@app.get("/api/state")
def state(request: Request):
    from acting_models import MODELS
    release=getattr(STATE['engine'],'release',MODELS[settings.DEFAULT_MODEL])
    return {"ready": STATE["phase"] == "ready", "phase": STATE["phase"], "error": STATE["error"],
            "cooperative_disconnect_cancellation": True,
            "continuous_tts": True,
            "model": release.get('repo',settings.REPO), "stage":release.get('label',STATE['phase']), "revision":release.get('revision',settings.REVISION),
            "model_id":getattr(STATE['engine'],'model_id',settings.DEFAULT_MODEL),
            "model_default":settings.DEFAULT_MODEL,"batch_size_default":release.get('batch_size',1),
            "batch_size_max":release.get('max_batch_size',4),"search_n_max":release.get('search_max_n',4),
            "checkpoint":getattr(STATE['engine'],'checkpoint_info',None),
            "checkpoint_ram":__import__('checkpoint_ram').RAM.describe(),
            "parameters": STATE["engine"].parameters if STATE["engine"] else None,
            "director": settings.LUNA_MODEL, "director_ready": bool(settings.luna_key()),
            "guidance_default": settings.DEFAULT_GUIDANCE,
            "temperature_default": settings.DEFAULT_TEMPERATURE,
            "format_default": settings.DEFAULT_FORMAT, "adapters": {"repo": settings.DPO_REPO, "revision": settings.DPO_REVISION, "default": None, "ref15_repo":settings.REF15_REPO,"enabled":False}, "ttl_seconds": settings.TTL,
            "reference_retention": "until_removed",
            "reference_seconds_max":settings.REFERENCE_SECONDS,
            "voice_default":settings.DEFAULT_VOICE,"speaker_default":SpeakerOptions().model_dump(),
            "reference_policy_default":"fixed",
            "speaker_steering_default":False,"speaker_steering_available":False,
            "dynamic_reference_default":False,
            "generation_default":settings.DEFAULT_GENERATION,"n_default":settings.DEFAULT_N,"streaming_default":settings.DEFAULT_STREAMING,"search_n_default":4,"search_seconds_default":30,
            "voice_conversion":vc_manager.describe(),
            "ranking_default":RankingOptions().model_dump(),
            "ears_loaded":sorted(getattr(getattr(STATE['scorers'],'ears',None),'models',{})),
            "search_chunk_frames_default":settings.DEFAULT_CHUNK_FRAMES,"chunk_frames_default":settings.DEFAULT_CHUNK_FRAMES,
            "audio_format": "mp3", "audio_channels": 1, "audio_bitrate_kbps": 96,
            "alignment_default": True,
            "alignment_available": bool(STATE["finisher"] and STATE["finisher"].available),
            "job_phase": STATE.get("job_phase"),
            "job_seconds": round(time.monotonic() - STATE["job_started"]) if STATE.get("job_started") else None,
            "busy": STATE["busy"], "references": reference_list(request.state.sid)}


def reference_list(owner):
    return [{"id": x.id, "slot": x.slot, "name": x.label,
             "url": f"/api/audio/{x.id}", "expires_at": None} for x in store.references(owner)]


def normalize(raw, output):
    with tempfile.NamedTemporaryFile(suffix=".upload", dir=store.root) as tmp:
        tmp.write(raw)
        tmp.flush()
        result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", tmp.name,
            "-vn", "-af", "silenceremove=start_periods=1:start_duration=0.02:start_threshold=-45dB",
            "-t", str(settings.REFERENCE_SECONDS), "-ac", "1", "-ar", "48000",
            "-c:a", "pcm_s16le", str(output)], capture_output=True, timeout=30)
        if result.returncode:
            raise ValueError("The audio file could not be decoded.")
    wave, sr = sf.read(output, dtype="float32")
    if len(wave) < sr * 0.24 or not np.isfinite(wave).all() or np.max(np.abs(wave)) < 0.001:
        raise ValueError("Upload audible speech lasting at least 0.24 seconds.")
    peak = float(np.max(np.abs(wave)))
    if peak > 0.95:
        wave *= 0.95 / peak
        sf.write(output, wave, sr, subtype="PCM_16")
    return len(wave) / sr


def encode_mp3(source, output):
    """Only the transport is lossy; alignment and judging use original PCM."""
    if isinstance(source, Path):
        input_args, data = ["-i", str(source)], None
    else:
        input_args = ["-f", "f32le", "-ar", "48000", "-ac", "1", "-i", "pipe:0"]
        data = np.asarray(source, dtype="<f4").reshape(-1).tobytes()
    result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *input_args,
        "-map_metadata", "-1", "-vn", "-ac", "1", "-ar", "48000", "-c:a", "libmp3lame",
        "-b:a", "96k", "-write_xing", "1", str(output)], input=data, capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError("Audio could not be encoded as mono MP3.")


@app.post("/api/reference")
async def upload_reference(request: Request, slot: Literal["A", "B"] = "A", file: UploadFile = File(...)):
    raw = bytearray()
    try:
        while chunk := await file.read(1024 * 1024):
            raw.extend(chunk)
            if len(raw) > settings.MAX_UPLOAD:
                raise HTTPException(413, "Maximum 20 MB per reference audio.")
    finally:
        await file.close()
    if not raw:
        raise HTTPException(400, "Empty audio file.")
    existing = [x for x in store.references(request.state.sid) if x.slot == slot]
    if any(x.active for x in existing):
        raise HTTPException(409, "This reference is in use.")
    item = store.new(request.state.sid, Path(file.filename or "reference").name[:100], "pending", slot)
    try:
        seconds = await asyncio.to_thread(normalize, raw, item.path)
        await asyncio.to_thread(encode_mp3, item.path, item.playback)
    except Exception as exc:
        store.remove(item)
        raise HTTPException(400, str(exc)) from exc
    existing = [x for x in store.references(request.state.sid) if x.slot == slot]
    if any(x.active for x in existing):
        store.remove(item)
        raise HTTPException(409, "This reference is in use.")
    for old in existing:
        if STATE["engine"]:
            STATE["engine"].forget_reference(old.path)
        if STATE['scorers'] and hasattr(STATE['scorers'], 'speakers'):
            STATE['scorers'].speakers.forget(old.path)
        store.remove(old)
    item.kind = "reference"
    store.persist_references()
    return {"id": item.id, "seconds": round(seconds, 2), "references": reference_list(request.state.sid)}


@app.delete("/api/reference/{id_}")
async def delete_reference(id_: str, request: Request):
    try:
        item = store.get(id_, request.state.sid, touch=False)
        if item.kind != "reference":
            raise KeyError()
        store.remove(item)
        if STATE["engine"]:
            STATE["engine"].forget_reference(item.path)
        if STATE['scorers'] and hasattr(STATE['scorers'], 'speakers'):
            STATE['scorers'].speakers.forget(item.path)
    except KeyError:
        raise HTTPException(404, "Reference not found.")
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return {"references": reference_list(request.state.sid)}


@app.get("/api/audio/{id_}")
def audio(id_: str, request: Request):
    try:
        item = store.get(id_, request.state.sid)
    except KeyError:
        raise HTTPException(404, "Audio expired or not found.")
    return FileResponse(item.playback, media_type="audio/mpeg", filename=f"humaneness-{item.kind}.mp3")


class Turn(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    model: Literal['large_acting','small_acting','small_alpha','large_alpha','small_rl','large_rl'] = settings.DEFAULT_MODEL
    batch_size: int = Field(default=1,ge=1,le=8)
    dynamic_reference: bool = settings.DEFAULT_DYNAMIC_REFERENCE
    format: Literal["structured", "caption", "tags", "transcript"] = settings.DEFAULT_FORMAT
    language: Literal["en", "de", "auto"] = "auto"
    input_mode: Literal["chat", "perform", "manual"] = "chat"
    generation: Literal["single", "best", "all", "search_winner", "search_beam"] = settings.DEFAULT_GENERATION
    search_n: int = Field(default=4, ge=1, le=16)
    reward_mode: Literal['clap','adaptive_auto','adaptive_manual'] = 'clap'
    ranking: RankingOptions = Field(default_factory=RankingOptions)
    reward_plan: RewardPlan | None = None
    reward_style_share: float = Field(default=.5,ge=0,le=1,allow_inf_nan=False)
    burst: BurstOptions = Field(default_factory=BurstOptions)
    search_chunk_frames: Literal[10, 20] = settings.DEFAULT_CHUNK_FRAMES
    search_seconds: float = Field(default=30, ge=0.8, le=60)
    n: int = Field(default=settings.DEFAULT_N, ge=1, le=12)
    guidance: float = Field(default=settings.DEFAULT_GUIDANCE, ge=1, le=5)
    align: bool = True
    temperature: float = Field(default=settings.DEFAULT_TEMPERATURE, ge=0, le=1.5)
    streaming: bool = settings.DEFAULT_STREAMING
    stream_transport: Literal['mp3', 'pcm_f32'] = 'mp3'
    live_only: bool = False
    cancel_on_disconnect: bool = False
    voice_conversion: bool = settings.DEFAULT_VOICE_CONVERSION
    vc_precision: Literal['bf16','fp32'] = 'bf16'
    speaker_steering: bool = settings.DEFAULT_SPEAKER_STEERING
    identity_cfg: bool = True  # Versioned opt-out for historical study reproduction.
    chunk_frames: Literal[2, 5, 10, 20] = settings.DEFAULT_CHUNK_FRAMES
    first_chunk_frames: Literal[2, 5, 10, 20] | None = None
    seed: int = Field(default=777, ge=0, le=2147483647)
    seconds: float | None = Field(default=None, ge=1, le=60)
    reference_id: str | None = Field(default=None, max_length=40)
    voice_mode: Literal['none', 'profile', 'custom'] = 'profile'
    voice_profile: str | None = Field(default=settings.DEFAULT_VOICE, max_length=80)
    speaker: SpeakerOptions = Field(default_factory=SpeakerOptions)
    adapters: AdapterSelection = Field(default_factory=lambda: AdapterSelection(id=settings.DEFAULT_ADAPTER))
    history: list[dict] = Field(default_factory=list, max_length=32)

    @model_validator(mode='before')
    @classmethod
    def legacy_custom_reference(cls, data):
        if isinstance(data,dict) and data.get('voice_mode')=='none' and 'voice_conversion' not in data:
            data={**data,'voice_conversion':False}
        if isinstance(data, dict) and data.get('reference_id') and 'voice_mode' not in data:
            return {**data, 'voice_mode': 'custom'}
        return data

    @model_validator(mode='after')
    def validate_reward(self):
        if self.live_only and (self.generation!='single' or not self.streaming or self.stream_transport!='pcm_f32'):
            raise ValueError('live_only requires a single streaming PCM take')
        from acting_models import MODELS
        release=MODELS[self.model]
        if self.batch_size>release['max_batch_size']:
            raise ValueError(f"{release['label']}: batch size limit is {release['max_batch_size']}")
        if self.generation.startswith('search_') and self.search_n>release['search_max_n']:
            raise ValueError(f"{release['label']}: live search limit is {release['search_max_n']}")
        if self.adapters.active():
            raise ValueError('Legacy DPO/Ref15 adapters are incompatible with the Acting checkpoints')
        if self.speaker_steering:
            raise ValueError('Legacy speaker steering vectors are incompatible with the Acting checkpoints')
        if self.ranking.mode!='legacy' and (self.reward_mode!='clap' or self.burst.mode!='off'):
            raise ValueError('Adaptive/burst rewards require legacy ranking; Ears ranking has no hidden extra weights')
        if self.reward_mode=='adaptive_manual' and self.reward_plan is None:
            raise ValueError('Manual adaptive reward requires 1–4 dimension terms')
        if self.voice_mode == 'profile' and self.voice_profile not in dynamic_references.profiles():
            raise ValueError('Unknown reference voice')
        if self.voice_mode in ('none', 'profile') and self.reference_id:
            raise ValueError('This voice mode cannot also use an uploaded reference')
        if self.adapters.ref15 and self.adapters.ref15_weight>0 and self.voice_mode=='none':
            raise ValueError('Ref15 requires a catalog voice or uploaded reference audio')
        if self.voice_conversion and self.voice_mode=='none':
            raise ValueError('Voice conversion requires a catalog voice or uploaded reference audio')
        return self


def language_for(text, selected, fallback="en"):
    if selected != "auto":
        return selected
    if re.search(r"[äöüß]|\b(ich|nicht|eine|einen|und|habe|mein|mir|dass|bitte|das|ist|heute)\b", text, re.I):
        return "de"
    return "de" if fallback == "de" else "en"


@app.post("/api/turn")
async def turn(body: Turn, request: Request):
    if STATE["phase"] != "ready":
        raise HTTPException(503, STATE["error"] or "Models are loading.")
    if STATE["busy"]:
        raise HTTPException(409, "Generation in progress. Please wait.")
    try:
        validate_selection(body.adapters)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    reference = None
    reference_id = body.reference_id
    if body.voice_mode == 'custom' and not reference_id:
        owned = store.references(request.state.sid)
        if not owned:
            raise HTTPException(422, 'Upload a custom reference or select another voice.')
        reference_id = max(owned, key=lambda x: x.touched).id
    if reference_id:
        try:
            reference = store.get(reference_id, request.state.sid)
            if reference.kind != "reference":
                raise KeyError()
        except KeyError:
            raise HTTPException(404, "Reference expired. Please upload it again.")
    STATE["busy"] = True
    STATE["job_phase"] = "Acting directions" if body.input_mode != "manual" else "Preparing prompt"
    STATE["job_started"] = time.monotonic()
    if reference:
        reference.active += 1
    queue = asyncio.Queue()
    cancellation = DisconnectCancellation(body.cancel_on_disconnect)
    loop = asyncio.get_running_loop()
    owner = request.state.sid

    async def work():
        nonlocal reference
        started = time.monotonic()
        conversion=None
        steering_report = {'requested': body.speaker_steering, 'enabled': False, 'active': False,
                           'experimental': True, 'reason': 'not_prepared'}
        try:
            if getattr(STATE['engine'],'model_id',None) and STATE['engine'].model_id!=body.model:
                await queue.put({'event':'status','phase':'Acting-Switching model'})
                await asyncio.to_thread(switch_acting_model,body.model)
            # Best-of-N / All N are a real parallel candidate batch, not N
            # serial calls with the single-stream UI's batch-1 default.
            effective_batch=candidate_batch_size(body)
            if hasattr(STATE['engine'],'batch_size'):
                STATE['engine'].batch_size=effective_batch
            await queue.put({'event':'model','model_id':body.model,
                'checkpoint':getattr(STATE['engine'],'checkpoint_info',None),'batch_size':effective_batch,
                'requested_batch_size':body.batch_size,'parallel_candidates':body.generation in ('best','all')})
            await clear_speaker_steering(STATE['engine'], 'request_start')
            await queue.put({"event": "status", "phase": "Acting directions" if body.input_mode != "manual" else "Preparing prompt"})
            if body.input_mode == "manual":
                directed = parse_literal(body.message, body.format)
                llm_ms = 0
            else:
                try:
                    directed, llm_ms = await asyncio.wait_for(director.direct(
                        body.message, body.format, body.language, body.input_mode, body.history,
                        reference_enabled=body.voice_mode != 'none'), timeout=35)
                except asyncio.TimeoutError as exc:
                    raise RuntimeError("Luna did not respond within 35 seconds. Request cancelled; retry or use a manual prompt.") from exc
            language = language_for(directed["text"], body.language, directed.get("language", "en"))
            spec = build(directed, body.format, language, body.seconds, body.message if body.input_mode == "manual" else None,
                         reference_identity=body.voice_mode != 'none' and body.identity_cfg)
            reference_report = {'mode': 'custom' if reference else 'none', 'name': reference.label if reference else None}
            if body.voice_mode == 'profile':
                STATE['job_phase'] = 'Selecting a reference clip'
                await queue.put({'event': 'status', 'phase': 'Selecting a matching VoiceCLAP reference clip' if body.dynamic_reference else 'Using the fixed reference clip of this speaker'})
                tick = time.monotonic()
                options={'fixed':False} if body.dynamic_reference else {}
                path, reference_report = await asyncio.to_thread(dynamic_references.choose,
                    STATE['scorers'].retriever, spec, body.voice_profile,**options)
                reference = store.new(owner, reference_report['name'] + (' · VoiceCLAP reference' if body.dynamic_reference else ' · fixed reference clip'), 'profile_reference')
                reference.active += 1
                # Already normalized and measured: preserve its exact samples.
                await asyncio.to_thread(shutil.copyfile, path, reference.path)
                await asyncio.to_thread(encode_mp3, reference.path, reference.playback)
                reference_report.update(retrieval_ms=round((time.monotonic()-tick)*1000), url=f'/api/audio/{reference.id}')
            elif reference:
                reference_report['url'] = f'/api/audio/{reference.id}'
            if reference is None and settings.STAGE == settings.FULL_FT_STAGE:
                await queue.put({'event':'status','phase':'Note: this checkpoint was trained with reference audio; reference-free generation is not validated.'})
            await queue.put({'event': 'reference', **reference_report})
            await queue.put({"event": "direction", **spec, "llm_ms": llm_ms, "reference": reference.label if reference else None})
            if body.voice_conversion:
                STATE['job_phase']='Preparing voice converter and target speaker'
                await queue.put({'event':'status','phase':'Loading Humaneness Converter and preparing the selected reference'})
                if hasattr(vc_manager,'prepare_optimized'):
                    conversion=await asyncio.to_thread(vc_manager.prepare_optimized,reference.path,body.seed,
                        profile=body.voice_mode=='profile',precision=body.vc_precision)
                else:
                    prepare=vc_manager.prepare_profile if body.voice_mode=='profile' and hasattr(vc_manager,'prepare_profile') else vc_manager.prepare
                    conversion=await asyncio.to_thread(prepare,reference.path,body.seed)
                await queue.put({'event':'voice_conversion',**conversion.report})
            reward_plan=None
            adaptive_scorer=None
            identity_reward=None
            speaker_request={'active':False,'method':body.speaker.method,'weight':body.speaker.weight,
                             'reason':'No reference speaker' if not reference else 'Disabled'}
            speaker_preparation_ms=0
            if body.ranking.mode=='legacy' and reference and body.speaker.method!='off' and body.speaker.weight>0:
                STATE['job_phase']='Preparing speaker similarity'
                await queue.put({'event':'status','phase':'Preparing reference speaker embedding'})
                # Measure identity against the same actual fixed clip used by TTS.
                anchor=reference.path
                tick=time.monotonic()
                try:
                    identity_reward=await asyncio.to_thread(SpeakerReward,STATE['scorers'].speakers,body.speaker,anchor,
                        body.search_chunk_frames*.08 if body.generation.startswith('search_') else None,
                        'fixed profile anchor' if body.voice_mode=='profile' else 'custom upload')
                except ReferenceTooShort as exc:
                    speaker_request['reason']=str(exc)
                speaker_preparation_ms=round((time.monotonic()-tick)*1000)
                if identity_reward:
                    speaker_request={'active':True,**identity_reward.describe()}
            if body.ranking.mode=='legacy' and identity_reward is None and body.speaker.method!='off':
                if body.speaker.weight==0:
                    speaker_request['reason']='Speaker weight 0%: prompt style only'
                identity_reward=UnavailableSpeakerReward(body.speaker,speaker_request['reason'])
            await queue.put({'event':'speaker_request',**speaker_request})
            reward_planning_ms=reward_loading_ms=0
            burst_request=None
            if body.ranking.mode!='legacy':
                from search_scoring import SearchScorer
                STATE['job_phase']='Preparing Humaneness Ears ranking'
                await queue.put({'event':'status','phase':'Humaneness Ears '+body.ranking.model+' ranking preparation'})
                tick=time.monotonic()
                adaptive_scorer=await asyncio.to_thread(SearchScorer,STATE['scorers'],spec,
                    ranking=body.ranking)
                reward_loading_ms=round((time.monotonic()-tick)*1000)
                await queue.put({'event':'ranking','options':body.ranking.model_dump()})
            elif body.reward_mode!='clap':
                from search_scoring import SearchScorer, style_query
                if body.reward_mode=='adaptive_auto':
                    STATE['job_phase']='Selecting reward dimensions'
                    await queue.put({'event':'status','phase':'Luna is selecting reward dimensions'})
                    tick=time.monotonic()
                    try:
                        reward_plan=await asyncio.wait_for(director.reward_plan(
                            'Audible acting direction only: '+(style_query(spec) or 'Natural conversational delivery; no specified emotion.')),timeout=35)
                    except asyncio.TimeoutError as exc:
                        raise RuntimeError('Luna reward selection timed out. Use manual dimensions.') from exc
                    reward_planning_ms=round((time.monotonic()-tick)*1000)
                else:reward_plan=body.reward_plan
                await queue.put({'event':'reward_plan','plan':reward_plan.model_dump(),
                    'style_share':body.reward_style_share,'mode':body.reward_mode})
                STATE['job_phase']='Preparing reward models'
                await queue.put({'event':'status','phase':'Loading reward models (downloads on first use)'})
                tick=time.monotonic()
                adaptive_scorer=await asyncio.to_thread(SearchScorer,STATE['scorers'],spec,reward_plan,
                    body.reward_style_share,body.burst if body.burst.mode!='off' else None,identity_reward)
                reward_loading_ms=round((time.monotonic()-tick)*1000)
            elif body.burst.mode!='off':
                from search_scoring import SearchScorer
                STATE['job_phase']='Preparing burst models'
                await queue.put({'event':'status','phase':'Preparing Burst Locator v2 (first use loads models)'})
                tick=time.monotonic()
                adaptive_scorer=await asyncio.to_thread(SearchScorer,STATE['scorers'],spec,None,
                    body.reward_style_share,body.burst,identity_reward)
                reward_loading_ms=round((time.monotonic()-tick)*1000)
            elif identity_reward is not None:
                from search_scoring import SearchScorer
                adaptive_scorer=await asyncio.to_thread(SearchScorer,STATE['scorers'],spec,None,
                    body.reward_style_share,None,identity_reward)
            if adaptive_scorer is not None and adaptive_scorer.burst_reward is not None:
                burst_request=adaptive_scorer.burst_reward.describe()
                await queue.put({'event':'burst_request',**burst_request})
                if not burst_request['active'] and body.reward_mode=='clap' and identity_reward is None:
                    # A no-target turn must retain the legacy full-utterance
                    # ranker when no adaptive dimensions were requested.
                    adaptive_scorer=None
            adapter_report = {'selection': body.adapters.model_dump(), 'active': [], 'switch_ms': 0}
            if hasattr(STATE['engine'], 'configure_adapters'):
                STATE['job_phase'] = 'Preparing LoRAs'
                await queue.put({'event': 'status', 'phase': 'Preparing selected LoRAs'})
                adapter_report = await asyncio.to_thread(STATE['engine'].configure_adapters, body.adapters)
            elif body.adapters.active():
                raise RuntimeError('LoRA engine is unavailable')
            await queue.put({'event': 'adapters', **adapter_report})
            steering_report = await prepare_speaker_steering(STATE['engine'], body, reference)
            await queue.put({'event': 'speaker_steering', **steering_report})
            STATE["job_phase"] = "Generating audio"
            cancellation.check()
            await queue.put({"event": "status", "phase": "Generating audio"})
            n = 1 if body.generation == "single" else body.n
            def progress(done, total):
                cancellation.check()
                loop.call_soon_threadsafe(queue.put_nowait, {"event": "progress", "done": done, "total": total})
            generation_started = time.monotonic()
            # Stack-only watchdog for unexpectedly stalled generation. Never
            # include locals, request text, audio or credentials in diagnostics.
            import faulthandler
            if log.isEnabledFor(logging.DEBUG):
                faulthandler.dump_traceback_later(60,repeat=True)
            search_performance = None
            live_cutoff=None
            live_detail=None
            live_event='search_chunk' if body.generation.startswith('search_') else 'audio_chunk'
            converted_chunks=[]
            if body.generation.startswith("search_"):
                from beam_search import ChunkSearch
                from search_scoring import SearchScorer
                scorer = adaptive_scorer or await asyncio.to_thread(SearchScorer, STATE["scorers"], spec)
                from alignment import LiveCutoff
                live_cutoff=LiveCutoff(spec,body.align)
                live_detail={}
                def committed(wave, detail):
                    cancellation.check()
                    live_detail.update(detail)
                    rows=detail.get('retained_rows',[0]);scores=detail.get('scores',[{}])
                    edited=live_cutoff.feed(wave,scores[rows[0]] if scores else {})
                    if len(edited):
                        live_cutoff.report.setdefault('first_playable_seconds',time.monotonic()-generation_started)
                        loop.call_soon_threadsafe(queue.put_nowait, {"event": "search_chunk", "_wave": edited,
                        "step": detail["step"], "committed_until_seconds": detail["committed_until_seconds"],
                        "seconds": len(edited)/48000, "retained_rows": detail["retained_rows"],
                        'playback_until_seconds':live_cutoff.emitted/48000,'live_alignment':dict(live_cutoff.report)})
                    return live_cutoff.done
                wave, search_performance = await asyncio.to_thread(ChunkSearch(STATE["engine"], scorer).generate,
                    spec, body.search_n, 2 if body.generation == "search_beam" else 1,
                    body.search_chunk_frames, body.search_seconds, body.guidance, body.temperature,
                    body.seed, reference.path if reference else None, committed,
                    **({'conversion':conversion} if conversion else {}))
                waves = [wave]
            else:
                live_kwargs={}
                if body.generation == 'single' and body.streaming:
                    from alignment import LiveCutoff
                    from search_scoring import SearchScorer, prefix_errors
                    import torch
                    import torchaudio
                    live_cutoff=LiveCutoff(spec,body.align)
                    live_detail={'step':0,'retained_rows':[0]}
                    pcm_chunks=[]
                    vc_stream=conversion.new_stream() if conversion else None
                    timestamp_reader=SearchScorer.__new__(SearchScorer) if body.align else None
                    if timestamp_reader is not None:timestamp_reader.asr=STATE['scorers'].asr
                    def single_chunk(chunk):
                        cancellation.check()
                        if conversion:
                            chunk=conversion.chunk(chunk,vc_stream)
                            converted_chunks.append(chunk)
                            if not len(chunk):return False
                        pcm_chunks.append(chunk)
                        live_detail['step']+=1
                        cumulative=np.concatenate(pcm_chunks)
                        seconds=len(cumulative)/48000
                        score={}
                        if body.align:
                            audio16=torchaudio.functional.resample(torch.from_numpy(cumulative),48000,16000)
                            texts,stamps=timestamp_reader.transcribe_batch([audio16])
                            stable=' '.join(w['word'] for w in stamps[0] if w['end']<=seconds-.12)
                            score={'timestamps':stamps[0],'stable_hyp':stable,
                                   **prefix_errors(spec['text'],stable)}
                        edited=live_cutoff.feed(chunk,score)
                        if len(edited):
                            live_cutoff.report.setdefault('first_playable_seconds',time.monotonic()-generation_started)
                            loop.call_soon_threadsafe(queue.put_nowait, {'event':live_event,'_wave':edited,
                                'step':live_detail['step'],'seconds':len(edited)/48000,
                                'committed_until_seconds':seconds,'playback_until_seconds':live_cutoff.emitted/48000,
                                'live_alignment':dict(live_cutoff.report)})
                        return live_cutoff.done
                    live_kwargs['on_audio']=single_chunk
                waves = await asyncio.to_thread(STATE["engine"].generate, spec, n, body.guidance,
                    body.seed, reference.path if reference else None, body.temperature, progress,
                    **({"streaming": True, "chunk_frames": body.chunk_frames,"first_chunk_frames":body.first_chunk_frames} if body.streaming else {}),**live_kwargs)
                if conversion:
                    if body.generation=='single' and body.streaming:
                        tail=await asyncio.to_thread(conversion.chunk,np.zeros(0,np.float32),vc_stream,True)
                        converted_chunks.append(tail)
                        # Feed the held converted seam through the same ASR/fade
                        # path without converting it a second time.
                        if len(tail):
                            pcm_chunks.append(tail)
                            edited=live_cutoff.feed(tail,{})
                            if len(edited):
                                await queue.put({'event':'audio_chunk','_wave':edited,'step':live_detail['step']+1,
                                    'seconds':len(edited)/48000,'committed_until_seconds':live_cutoff.total/48000,
                                    'playback_until_seconds':live_cutoff.emitted/48000,'live_alignment':dict(live_cutoff.report)})
                        waves=[np.concatenate(converted_chunks)]
                    else:
                        STATE['job_phase']='Converting takes with Humaneness Converter'
                        await queue.put({'event':'status','phase':'Converting all takes to the reference speaker before scoring'})
                        waves=await asyncio.to_thread(conversion.full,waves,body.streaming,body.chunk_frames)
            generation_ms = round((time.monotonic() - generation_started) * 1000)
            cancellation.check()
            faulthandler.cancel_dump_traceback_later()
            if reference:
                context=getattr(STATE['engine'],'reference_info',{}).get(str(reference.path))
                if context:
                    reference_report['tts_context']=dict(context)
                    await queue.put({'event':'reference_context',**context})
            STATE["job_phase"] = "Speech boundaries & fades"
            await queue.put({"event": "status", "phase": "Aligning speech boundaries and applying fades"})
            alignment_started = time.monotonic()
            waves, alignments = await asyncio.to_thread(STATE["finisher"].finish, waves, spec, body.align)
            if live_cutoff is not None:
                tail=live_cutoff.finish(waves[0])
                if len(tail):
                    live_cutoff.report.setdefault('first_playable_seconds',time.monotonic()-generation_started)
                    await queue.put({'event':live_event,'_wave':tail,'step':live_detail.get('step',0)+1,
                        'seconds':len(tail)/48000,'committed_until_seconds':len(waves[0])/48000,
                        'playback_until_seconds':live_cutoff.emitted/48000,
                        'retained_rows':live_detail.get('retained_rows',[0]),'live_alignment':dict(live_cutoff.report)})
                if search_performance is not None:
                    search_performance['live_alignment']=live_cutoff.report
                    search_performance['first_raw_commit_seconds']=search_performance.get('first_commit_seconds')
                    search_performance['first_commit_seconds']=live_cutoff.report.get('first_playable_seconds')
            alignment_ms = round((time.monotonic() - alignment_started) * 1000)
            if body.live_only:
                # Robby already received the whole single take. There is no
                # candidate ranking to perform and no download artifact needed.
                await queue.put({'event':'result','candidates':[],'winner':0,'live_only':True,
                    'generation_ms':generation_ms,'alignment_ms':alignment_ms,'scoring_ms':0,
                    'speaker_preparation_ms':speaker_preparation_ms,
                    'performance':getattr(STATE['engine'],'last_performance',None),
                    'live_stream':dict(live_cutoff.report) if live_cutoff else None,
                    'total_ms':round((time.monotonic()-started)*1000)})
                return
            STATE["job_phase"] = "Scoring takes"
            await queue.put({"event": "status", "phase": "Scoring takes"})
            scoring_started = time.monotonic()
            if search_performance is not None or adaptive_scorer is not None:
                active_scorer=adaptive_scorer or scorer
                scores = await asyncio.to_thread(active_scorer.score, waves, spec["text"], [True]*len(waves))
                ranks={index:rank for rank,index in enumerate(sorted(range(len(scores)),
                    key=lambda i:(not scores[i].get('ranking_available',True),-scores[i]['reward'])))}
                for i, score in enumerate(scores):
                    score["wer"] = score["prefix_wer"]
                    score["rank"] = ranks[i]
            else:
                scores = await asyncio.to_thread(STATE["scorers"].score, waves, spec)
            scoring_ms = round((time.monotonic() - scoring_started) * 1000)
            candidates = []
            for i, (wave, score) in enumerate(zip(waves, scores)):
                item = store.new(owner, f"Take {i+1}", "take")
                try:
                    await asyncio.to_thread(encode_mp3, wave, item.playback)
                except Exception:
                    store.remove(item)
                    raise
                candidates.append({**score, "index": i, "url": f"/api/audio/{item.id}", "alignment": alignments[i]})
            best = min(candidates, key=lambda c: c["rank"])
            await queue.put({"event": "result", "candidates": candidates, "winner": best["index"],
                "generation": body.generation, "guidance": body.guidance, "seed": body.seed,
                "llm_ms": llm_ms, "generation_ms": generation_ms, "alignment_ms": alignment_ms, "scoring_ms": scoring_ms,
                "performance": getattr(STATE["engine"], "last_performance", None) if search_performance is None else None,
                "search_performance": search_performance,
                'live_stream':dict(live_cutoff.report) if live_cutoff else None,
                'voice_conversion':dict(conversion.report) if conversion else {'enabled':False},
                'checkpoint':getattr(STATE['engine'],'checkpoint_info',None),
                'ranking':body.ranking.model_dump(),
                'reward_mode':body.reward_mode,'reward_plan':reward_plan.model_dump() if reward_plan else None,
                'reward_style_share':body.reward_style_share,'reward_planning_ms':reward_planning_ms,
                'reward_loading_ms':reward_loading_ms,
                'burst_request':burst_request,
                'adapters': adapter_report,
                'speaker_steering': steering_report,
                'reference': reference_report,
                'speaker_request':speaker_request,'speaker_preparation_ms':speaker_preparation_ms,
                "total_ms": round((time.monotonic() - started) * 1000),
                "cfg_effective": body.guidance != 1 and spec["prompt"] != spec["unconditional"]})
        except TurnCancelled:
            log.info('Disconnected opt-in turn stopped at a cooperative audio boundary')
        except Exception as exc:
            log.exception("Turn failed")
            await queue.put({"event": "error", "message": str(exc) or f"{type(exc).__name__}: Generierung fehlgeschlagen."})
        finally:
            try:
                await clear_speaker_steering(STATE['engine'], 'request_complete')
            except Exception:
                log.exception('Request-scoped speaker steering cleanup failed')
            if conversion is not None:
                try:
                    await asyncio.to_thread(conversion.close)
                except Exception:
                    log.exception('Voice conversion cleanup failed')
            import faulthandler
            faulthandler.cancel_dump_traceback_later()
            if reference:
                reference.active -= 1
                reference.touched = time.time()
                if reference.path.exists():
                    reference.path.touch()
            STATE["busy"] = False
            STATE["job_phase"] = None
            STATE["job_started"] = None
            await queue.put(None)

    task = asyncio.create_task(work())
    # Existing Playground requests still survive disconnects. Opt-in callers
    # stop cooperatively; model ownership remains held until the GPU unwinds.
    tasks.add(task)
    task.add_done_callback(tasks.discard)

    async def events():
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=15)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if item is None:
                break
            if item.get("event") in {"search_chunk", "audio_chunk"}:
                wave = item.pop("_wave")
                if body.stream_transport == 'pcm_f32':
                    # Opt-in server-to-server transport. Avoid per-chunk MP3
                    # processes, temporary files and an additional HTTP GET.
                    # The public Playground and completed takes remain MP3.
                    import base64
                    item.update(pcm_f32=base64.b64encode(np.asarray(wave,dtype='<f4').tobytes()).decode('ascii'),
                                sample_rate=48000,channels=1)
                    yield 'data: '+json.dumps(item,ensure_ascii=False)+'\n\n'
                    continue
                chunk = store.new(owner, f"Live chunk {item['step']}", "take")
                try:
                    await asyncio.to_thread(encode_mp3, wave, chunk.playback)
                    item["url"] = f"/api/audio/{chunk.id}"
                except Exception:
                    store.remove(chunk)
                    log.exception("Search chunk encoding failed")
                    item = {"event": "status", "phase": "Live MP3 encoding failed; wait for the completed take"}
            yield "data: " + json.dumps(item, ensure_ascii=False) + "\n\n"
    return StreamingResponse(cancellation.watch(events()), media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


tasks = set()


@app.post("/api/asr")
async def asr(request: Request):
    if not STATE["scorers"]:
        raise HTTPException(503, "Speech recognition is loading.")
    if STATE["busy"]:
        raise HTTPException(409, "Wait for the current generation.")
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > settings.MAX_UPLOAD:
            raise HTTPException(413, "Recording too large.")
    if not raw:
        raise HTTPException(400, "Empty recording.")
    text = await asyncio.to_thread(STATE["scorers"].asr.transcribe, bytes(raw))
    return {"text": text}


from live_batch_api import install as install_live_batch
install_live_batch(app, STATE, switch_acting_model)
