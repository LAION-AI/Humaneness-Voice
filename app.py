"""MOSS voice-acting web demo — FastAPI orchestrator.

One turn:
  user text -> Gemma (GPU LLM_GPU) picks the line + the GENERAL/SCRIPT direction
            -> select_reference_voice resolves to pre-tokenised codes in RAM
            -> MOSS 4.55B (GPU TTS_GPU) streams 48 kHz PCM as it is generated
            -> browser plays the first chunk while the rest is still decoding

Wire format (so the browser can start playing before the reply is finished):
    [1 byte tag][4 byte big-endian length][payload]
    tag 0 = UTF-8 JSON event, tag 1 = raw PCM int16 LE mono @ 48 kHz
"""
import asyncio, json, os, struct, threading, time

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import (HTMLResponse, JSONResponse, Response,
                               StreamingResponse)

import config
import levers
import lora_bank
import align_engine
import bestofn
import steer_engine
import cues
import sidon
import timed_script
from llm_agent import VOICE_TOOL, LLMAgent
from lora_bank import LoraBank
from tts_engine import TTSEngine
from voice_bank import VoiceBank
import personas
import retrieval
import voice_profiles
from voice_codes import CodeBook

HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="MOSS Voice Acting")
STATE = {"tts": None, "bank": None, "agent": None, "lora": None, "asr": None,
         "agents": {}, "codebook": None, "vc": None, "sim": None,
         "score": None, "profiles": {}, "error": None,
         # the second and third levers: the measured coefficient table and the
         # steering vectors.  Either may be absent, in which case the modes that
         # need it degrade to `adapter` and the payload says so.
         "wiki": None, "vectors": None}


def _frame(tag: int, payload: bytes) -> bytes:
    return struct.pack(">BI", tag, len(payload)) + payload


def _ev(obj) -> bytes:
    return _frame(0, json.dumps(obj).encode())


def _pcm(seg: np.ndarray) -> bytes:
    return _frame(1, np.clip(seg * 32767, -32768, 32767).astype("<i2").tobytes())


# Voice context per conversation.  The manual is explicit that chaining whole
# previous clips is harmful — measured similarity collapsed 0.777 -> 0.280 by the
# fourth clip, because drift compounds and a long prefix drags the previous
# emotion into the new line.  What works is the anchor plus a short tail, so this
# keeps only the last few seconds of each of the last few replies.
SESSIONS = {}


def _session(sid):
    if not sid:
        return None
    s = SESSIONS.get(sid)
    if s is None:
        if len(SESSIONS) > config.MAX_SESSIONS:      # crude cap, oldest first
            for k in list(SESSIONS)[:len(SESSIONS) // 2]:
                SESSIONS.pop(k, None)
        s = SESSIONS[sid] = {"tails": []}
    return s


def _handoff(q, loop, item, gone, timeout=60.0):
    """Hand one item to the streaming consumer, giving up if it has gone away.

    Without the timeout a dropped client leaves this thread parked on a queue
    forever while it still holds the generation lock, and every later request
    queues behind it.
    """
    try:
        asyncio.run_coroutine_threadsafe(q.put(item), loop).result(timeout=timeout)
        return True
    except Exception:
        gone.set()
        return False


# ------------------------------------------------------------------ startup
def _boot():
    try:
        tts = TTSEngine(device="cuda:0")     # CUDA_VISIBLE_DEVICES pins the card
        STATE["tts"] = tts
        bank = VoiceBank(tts.proc, tts.device, gpu_lock=tts.lock)
        STATE["bank"] = bank
        cat, desc = bank.catalog(), bank.descriptions()
        cb = CodeBook(cat, desc)
        # what the director may reach for is what is installed, nothing else
        import glob as _glob
        _broot = config.LORA_ROOTS.get("burst") or ""
        _have = [os.path.basename(d) for d in _glob.glob(os.path.join(_broot, "*"))
                 if os.path.isdir(d)]
        burst_cues = lora_bank.burst_catalog(_have)
        print(f"[bursts] {len(burst_cues)} burst adapters offered to the director",
              flush=True)
        STATE["codebook"] = cb
        # one agent per (brain, prompt style); they share the http client cost
        # only at construction, so building all four up front keeps switching free
        # one agent per (brain, prompt style, skills on/off).  The skills flag
        # changes only the system prompt, so building both is a string apiece
        # and lets the two be compared without a restart.
        import skills as _skills
        _skills.load()
        for be in ("local",) + tuple(config.HOSTED_MODELS):
            for st in ("prose", "codes"):
                for sk in (True, False):
                    try:
                        STATE["agents"][(be, st, sk)] = LLMAgent(
                            cat, descriptions=desc, backend=be, style=st,
                            codebook=cb, bursts=burst_cues, use_skills=sk)
                    except Exception as e:
                        print(f"[app] agent {be}/{st}/skills={sk} unavailable: {e}",
                              flush=True)
        STATE["agent"] = STATE["agents"].get(
            (config.DEFAULT_BRAIN, "codes", config.SKILLS_ON)) \
            or STATE["agents"][("local", "prose", True)]
        STATE["profiles"] = voice_profiles.discover()
        for _v, _p in STATE["profiles"].items():      # own matrix beats borrowing
            _p["has_conditions"] = _p["has_conditions"] or bank.has_matrix(_v)
        print(f"[profiles] {len(STATE['profiles'])} voice profiles available",
              flush=True)
        try:
            # after the TTS model, so torch has already pulled cuDNN and cuBLAS
            # into the process -- onnxruntime finds them there and takes the CUDA
            # provider.  Loaded before torch it silently falls back to CPU, which
            # is 20x slower and too slow to keep a stream fed.
            STATE["aligner"] = align_engine.make_aligner()
        except Exception as e:
            print(f"[align] unavailable: {e}", flush=True)
        try:
            # scored candidates for best-of-N.  Shares the CLAP tower with the
            # retriever rather than holding a second copy of the same weights.
            _j = bestofn.Judge(asr=STATE.get("asr"))
            STATE["judge"] = _j if _j.ok else None
        except Exception as e:
            print(f"[align] unavailable: {e}", flush=True)
        try:
            STATE["retriever"] = retrieval.Retriever(device=str(tts.device))
            if STATE.get("judge") is not None:
                STATE["judge"].attach_clap(STATE["retriever"])
        except Exception as e:
            print(f"[retrieval] unavailable: {e}", flush=True)
        if config.USE_LORA:
            lb = LoraBank(tts.model, tts.device)
            lb.discover(config.LORA_ROOTS)
            tts.lora = lb
            STATE["lora"] = lb
        # The other two levers.  Both are optional assets: with neither present this
        # server behaves exactly as it did before generation modes existed.
        STATE["wiki"] = levers.Wiki()
        if config.STEER_ENABLED:
            vp = steer_engine.VectorPack(device=str(tts.device))
            STATE["vectors"] = vp
            tts.vectors = vp
        tts.warmup()
        # speech recognition lives on the other card, next to the language model
        # Both of these load on the second card.  They must load one after the
        # other, not in parallel: transformers flips torch's global default dtype
        # while building a bfloat16 model, and a model being constructed in
        # another thread at that moment comes out bf16 too — which is how WavLM
        # ended up dying with "mixed dtype (CPU)".
        def _side_models():
            try:
                from asr_engine import ParakeetASR
                STATE["asr"] = ParakeetASR()
                if STATE.get("judge") is not None:
                    STATE["judge"].asr = STATE["asr"]
            except Exception as e:
                print(f"[asr] unavailable: {e}", flush=True)
            try:
                from vc_engine import VoiceConverter
                v = VoiceConverter()
                ref = os.path.join(config.REF_DIR, "reference",
                                   "reference_target.mp3")
                if v.set_target(ref):
                    STATE["vc"] = v
                    print("[vc] target set to the corpus anchor", flush=True)
            except Exception as e:
                import traceback
                print(f"[vc] unavailable: {e}", flush=True)
                traceback.print_exc()
            try:
                from score_engine import VoiceScorer
                STATE["score"] = VoiceScorer()
            except Exception as e:
                print(f"[score] unavailable: {e}", flush=True)
            try:
                from sim_engine import SpeakerSim
                STATE["sim"] = SpeakerSim(os.path.join(
                    config.REF_DIR, "reference", "reference_target.mp3"))
            except Exception as e:
                print(f"[sim] unavailable: {e}", flush=True)
        threading.Thread(target=_side_models, daemon=True).start()
        bank.preload()                       # pre-tokenise the corpus in the background
        if STATE.get("lora"):
            lb = STATE["lora"]
            names = [n for k in config.PRELOAD_LORA_KINDS
                     for n in lb.repos if n.startswith(k + ":")]
            threading.Thread(target=lb.preload, args=(names,), daemon=True).start()
        print("[app] ready", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        STATE["error"] = str(e)


@app.on_event("startup")
async def startup():
    threading.Thread(target=_boot, daemon=True).start()


# -------------------------------------------------------------------- routes
@app.get("/", response_class=HTMLResponse)
def index():
    return open(os.path.join(HERE, "index.html")).read()


@app.get("/studio", response_class=HTMLResponse)
def studio():
    return open(os.path.join(HERE, "studio.html"), encoding="utf-8").read()


@app.get("/api/personas")
def api_personas():
    return {"personas": personas.listing(), "default": personas.DEFAULT}


@app.post("/api/listen")
async def listen(req: Request):
    """Transcribe and, in parallel, score what the voice sounds like."""
    raw = await req.body()
    if not raw:
        return JSONResponse({"error": "empty audio"}, status_code=400)
    asr, sc = STATE["asr"], STATE["score"]
    if asr is None:
        return JSONResponse({"error": "speech recognition not ready"},
                            status_code=503)
    loop = asyncio.get_running_loop()
    t0 = time.time()
    try:
        wav = await loop.run_in_executor(None, asr._to_wav16k, raw)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    # both read the same 16 kHz array; they sit on different models so the two
    # calls overlap instead of queueing
    tasks = [loop.run_in_executor(None, asr.transcribe, wav)]
    if sc is not None:
        tasks.append(loop.run_in_executor(None, sc.score, wav))
    done = await asyncio.gather(*tasks, return_exceptions=True)
    text = done[0] if not isinstance(done[0], Exception) else ""
    heard = None
    if len(done) > 1 and not isinstance(done[1], Exception):
        heard = done[1]
    return {"text": text, "heard": heard,
            "listen_ms": round((time.time() - t0) * 1000, 1)}


@app.get("/report", response_class=HTMLResponse)
def report():
    return open(os.path.join(HERE, "report.html")).read()


@app.get("/api/state")
async def state():
    agent = STATE["agent"]
    return {
        "ready": STATE["tts"] is not None and agent is not None,
        "error": STATE["error"],
        "bank": STATE["bank"].stats() if STATE["bank"] else None,
        "lora": STATE["lora"].stats() if STATE["lora"] else None,
        "asr": bool(STATE["asr"]), "vc": bool(STATE["vc"]),
        "sim": bool(STATE["sim"]), "score": bool(STATE["score"]),
        "tts_model": config.TTS_REPO.split("/")[-1],
        "brains": ["local"] + list(config.HOSTED_MODELS),
        "profiles": voice_profiles.listing(STATE["profiles"]),
        "default_profile": config.DEFAULT_PROFILE,
        "sidon": {"on": config.SIDON_ON, **(sidon.health() or {"ok": False})},
        "personas": personas.listing(), "speaker_lora": config.SPEAKER_LORA in (
            STATE["lora"].repos if STATE["lora"] else {}),
        "llm": await agent.health() if agent else False,
        "sr": STATE["tts"].sr if STATE["tts"] else None,
        # What the three levers can actually do on this box, so the UI never offers
        # a mode that is not wired up.
        "levers": {
            "mode": config.GEN_MODE,
            "modes": list(levers.ALL_MODE_WORDS),
            "strengths": list(levers.STRENGTHS),
            "agent_picks_mode": config.AGENT_PICKS_MODE,
            "steer": {
                "enabled": config.STEER_ENABLED,
                # steering needs a direction and the layers to put it at, and only the
                # vector pack has either
                "needs": ["steer_pack"],
                "available": bool(STATE["vectors"] and STATE["vectors"].available),
                "dimensions": len(STATE["vectors"].names) if STATE["vectors"] else 0,
                "error": STATE["vectors"].error if STATE["vectors"] else "not loaded",
                "alpha": config.STEER_ALPHA,
                "alpha_ceiling": config.STEER_ALPHA_CEILING,
            },
            "cfg": {
                "enabled": config.CFG_ENABLED,
                # guidance needs neither asset: `g` has a family default measured in the
                # CFG study, and the neutralised branch is built from the turn's own prompt
                "needs": [],
                "available": config.CFG_ENABLED,
                "cost_factor": config.CFG_COST_FACTOR,
                "streams": False,
                "g": config.CFG_G,
            },
            "wiki": {
                # the coefficient table is what `auto` reads.  Without it `auto` asks for no
                # lever; an explicitly requested one still runs, on its family default.
                "available": bool(STATE["wiki"] and STATE["wiki"].available),
                "attributes": len(STATE["wiki"].attributes) if STATE["wiki"] else 0,
                "error": STATE["wiki"].error if STATE["wiki"] else "not loaded",
                "gates": "auto; per-attribute operating points",
            },
            "delivery_lever": config.DELIVERY_LEVER,
        },
    }


@app.get("/api/adapters")
def adapters():
    """Every adapter the overlay can dial, grouped, with its default weight.

    A default of 0 does not mean "off": it means nothing is forced and the
    director decides.  Only the three quality axes are on by default.
    """
    lb = STATE.get("lora")
    if lb is None:
        return JSONResponse({"error": "adapters disabled"}, status_code=503)
    def group(kind):
        return sorted(n.split(":", 1)[1] for n in lb.repos if n.startswith(kind + ":"))
    out = []
    out.append({"kind": "sft3_quality", "title": "Quality axes",
                "note": "On by default at the trained value of 1.0.",
                "items": [{"name": n, "label": config.QUALITY_LABELS.get(n, n),
                           "default": config.QUALITY_LORAS.get(n, 0.0), "max": 2.0}
                          for n in sorted(config.QUALITY_LORAS)
                          if n in lb.repos]})
    out.append({"kind": "sft3_qdpo", "title": "Preference adapters",
                "note": f"Quality DPO is on at "
                        f"{config.QDPO_LORAS['sft3_qdpo:quality_dpo']}; "
                        f"burst+stop is off until somebody has listened to it. "
                        f"Neither has been heard.",
                "items": [{"name": n, "label": config.QDPO_LABELS.get(n, n),
                           "default": config.QDPO_LORAS.get(n, 0.0), "max": 2.0}
                          for n in config.QDPO_LORAS if n in lb.repos]})
    out.append({"kind": "sft3_voicenet", "title": "Delivery axes",
                "note": "The director picks up to two of these itself. A slider "
                        "forces one on regardless.",
                "items": [{"name": f"sft3_voicenet:{n}", "label": n,
                           "hint": config.SFT3_VN_ADAPTERS.get(n, ""),
                           "default": 0.0, "max": 2.0} for n in group("sft3_voicenet")]})
    out.append({"kind": "sft3_emotion", "title": "Emotions",
                "note": f"The retrieval picks one per turn at "
                        f"{config.SFT3_EMOTION_LAM}. A slider adds or replaces "
                        f"it.",
                "items": [{"name": f"sft3_emotion:{n}", "label": n.replace("_", " "),
                           "default": 0.0, "max": 2.0} for n in group("sft3_emotion")]})
    for _k, _t, _n in (
            ("burst_v2", "Burst adapters v2 — per class",
             "The 105-adapter release. Which arm serves a class is decided by "
             "that class's own recipe; these sliders force one regardless."),
            ("burst_v2_top1", "Burst adapters v2 — per class, top-1", ""),
            ("burst_grp", "Burst adapters v2 — per group", ""),
            ("burst_grp25", "Burst adapters v2 — per group, dose 25", ""),
            ("burst_abl", "Burst ablation arms (evidence, not for serving)", ""),
            ("burst_dose", "Burst dose arms (evidence, not for serving)", "")):
        _items = group(_k)
        if _items:
            out.append({"kind": _k, "title": _t, "note": _n,
                        "items": [{"name": f"{_k}:{n}", "label": n.replace("_", " "),
                                   "default": 0.0, "max": config.BURST_LAM_MAX}
                                  for n in _items]})
    out.append({"kind": "burst", "title": "Vocal bursts",
                "note": "Added automatically when the script contains one. With "
                        "skills on, at the weight measured for that class "
                        "(0.25-2.3); a class with no measured recipe falls back "
                        f"to {config.BURST_LAM} "
                        f"({config.BURST_LAM_INTENSE} standing alone).",
                "items": [{"name": f"burst:{n}", "label": n.replace("_", " "),
                           "default": 0.0, "max": config.BURST_LAM_MAX}
                          for n in group("burst")]})
    return {"groups": out,
            "always": [{"name": config.SFT3_DPO_LORA, "label": "Quality (DPO p2)",
                        "default": config.SFT3_DPO_LAM},
                       {"name": "sft3_voice:<profile>", "label": "Voice identity",
                        "default": config.PROFILE_LORA_LAM}]}


@app.get("/api/voices")
def voices():
    if not STATE["bank"]:
        return JSONResponse({"error": "not ready"}, status_code=503)
    return {"catalog": STATE["bank"].catalog(), "tool": VOICE_TOOL,
            "stats": STATE["bank"].stats()}


@app.post("/api/say")
async def say(req: Request):
    """Speak an exact text with an exact instruction — no language model in the
    loop.  This is the harness for measuring the voice model on its own."""
    import base64
    b = await req.json()
    tts, bank = STATE["tts"], STATE["bank"]
    if not tts:
        return JSONResponse({"error": "not ready"}, status_code=503)
    ref = []
    if b.get("anchor", True) and bank:
        # a specific speaker's own recording, so a harness can condition on the
        # same voice whose adapter it is testing; the default is the corpus
        # anchor, which is a different speaker entirely
        a = bank.anchor(b["anchor_path"]) if b.get("anchor_path") else bank.anchor()
        if a is not None:
            ref.append(a)
    if b.get("reference") and bank:
        hit = bank.select(b["reference"], language=b.get("ref_lang", "en"),
                          speed=float(b.get("speed", 1.0)))
        if hit:
            ref.append(hit["codes"])

    chunks, meta = [], {}
    def run():
        for kind, payload in tts.stream_pcm(
                text=b["text"], instruction=b.get("instruction", ""),
                language=b.get("language", "English"), ref_codes=ref or None,
                lora_specs=b.get("loras"), tokens=b.get("tokens"),
                seed=b.get("seed"), audio_temperature=b.get("audio_temperature"),
                max_new_tokens=b.get("max_new_tokens"),
                chunk_frames=b.get("chunk_frames"),
                stop_bias=b.get("stop_bias")):
            if kind == "pcm":
                chunks.append(payload)
            elif kind == "end":
                meta.update(payload)
    await asyncio.get_running_loop().run_in_executor(None, run)
    w = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    if b.get("align") and STATE.get("aligner") is not None:
        # offline the whole take is in hand, so one alignment pass settles both
        # edges exactly; no lookahead and no repeated passes
        w, rep = align_engine.trim(w, tts.sr, b.get("text", ""),
                                   b.get("plain") or b.get("text", ""),
                                   STATE["aligner"])
        meta["align"] = rep
    pcm = np.clip(w * 32767, -32768, 32767).astype("<i2").tobytes()
    return {"sr": tts.sr, "pcm": base64.b64encode(pcm).decode(), **meta}


@app.post("/api/cfg_sweep")
async def cfg_sweep(req: Request):
    """Render one line several times, changing only the guidance strength.

    The point is an A/B a person can actually judge: same script, same adapters,
    same seed, same reference clips, one number different.  Nothing is streamed —
    the takes are generated and returned together, and the browser lets you pick
    between them.  A guided take costs about 1.93x wall clock, so a four-value
    sweep is a couple of minutes.
    """
    import base64
    b = await req.json()
    tts, bank = STATE["tts"], STATE["bank"]
    if not tts:
        return JSONResponse({"error": "not ready"}, status_code=503)
    script = (b.get("script") or "").strip()
    if not script:
        return JSONResponse({"error": "no script"}, status_code=400)
    general = b.get("general") or ""
    language = b.get("language") or "English"
    values = b.get("guidance") or [1.0, 1.5, 2.0, 3.0]
    try:
        values = [float(v) for v in values][:8]
    except (TypeError, ValueError):
        return JSONResponse({"error": "bad guidance values"}, status_code=400)

    profs = STATE["profiles"] or {}
    prof = profs.get(b.get("profile") or config.DEFAULT_PROFILE)
    ref = []
    if bank:
        a = bank.anchor(prof["anchor"]) if prof else bank.anchor()
        if a is not None:
            ref.append(a)
    specs = [(n, float(l)) for n, l in (b.get("loras") or [])]
    seed = int(b.get("seed") or 1234)
    lc = "DE" if str(language).lower().startswith(("ger", "de")) else "EN"

    tagged, frames, plain = timed_script.render(script)
    if not plain:
        return JSONResponse({"error": "script renders empty"}, status_code=400)
    gl = timed_script.general_line(general, frames / config.FRAME_RATE, lc,
                                   b.get("reads_as"))
    instruction = f"GENERAL: {gl}\nSCRIPT:\n{tagged}"
    # the director builds this by dropping its own delivery clause; the client
    # passes it back so both branches differ in affect and in nothing else
    gen_unc = b.get("general_unc") or general
    unc_tagged = timed_script.neutralise(tagged)
    gl_unc = timed_script.general_line(gen_unc, frames / config.FRAME_RATE, lc, None)
    instruction_unc = f"GENERAL: {gl_unc}\nSCRIPT:\n{unc_tagged}"

    out = []
    for g in values:
        plan = levers.Plan()
        plan.requested = "adapter+cfg" if g > 1.0001 else "adapter"
        plan.mode = plan.requested
        plan.guidance = float(g)   # wants_cfg is derived from it
        chunks, meta = [], {}

        def run(_g=g, _plan=plan):
            for kind, payload in tts.stream_pcm(
                    text=tagged, instruction=instruction, language=language,
                    ref_codes=ref or None, lora_specs=specs, tokens=frames,
                    seed=seed, lever_plan=_plan,
                    instruction_unc=instruction_unc if _g > 1.0001 else None,
                    text_unc=unc_tagged if _g > 1.0001 else None):
                if kind == "pcm":
                    chunks.append(payload)
                elif kind == "end":
                    meta.update(payload)
        t0 = time.time()
        try:
            await asyncio.get_running_loop().run_in_executor(None, run)
        except Exception as e:
            out.append({"guidance": g, "error": str(e)[:200]})
            continue
        w = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
        arep = None
        if b.get("align", config.ALIGN_ON) is not False \
                and STATE.get("aligner") is not None:
            w, arep = align_engine.trim(w, tts.sr, tagged, plain, STATE["aligner"])
        raw = w
        used = False
        if b.get("sidon", config.SIDON_ON) is not False:
            w, used = await asyncio.get_running_loop().run_in_executor(
                None, lambda _w=w: sidon.enhance(_w, tts.sr))
        pcm = np.clip(w * 32767, -32768, 32767).astype("<i2").tobytes()
        row = {"guidance": g, "align": arep, "sidon": used,
               "pcm": base64.b64encode(pcm).decode(),
               "sec": round(len(w) / tts.sr, 3),
               "ms": round((time.time() - t0) * 1000, 1),
               "rtf": meta.get("rtf")}
        if used:
            # the take as generated, so the restoration can be judged by ear
            # rather than taken on trust
            r16 = np.clip(raw * 32767, -32768, 32767).astype("<i2").tobytes()
            row["pcm_raw"] = base64.b64encode(r16).decode()
        out.append(row)
    return {"sr": tts.sr, "takes": out, "tokens": frames,
            "script": tagged, "script_unc": unc_tagged,
            "instruction": instruction, "instruction_unc": instruction_unc}


@app.post("/api/say_batch")
async def say_batch(req: Request):
    """Several exact texts in one forward pass, one adapter set for all of them.

    The sweep harness lives on this: a condition is ten utterances, and running
    them together instead of one after another is where the time goes.
    """
    import base64
    b = await req.json()
    tts, bank = STATE["tts"], STATE["bank"]
    if not tts:
        return JSONResponse({"error": "not ready"}, status_code=503)
    ref = []
    if b.get("anchor", True) and bank:
        a = bank.anchor(b["anchor_path"]) if b.get("anchor_path") else bank.anchor()
        if a is not None:
            ref.append(a)
    items = []
    for it in b.get("items") or []:
        one = {"text": it["text"], "instruction": it.get("instruction", ""),
               "language": it.get("language", "English"),
               "tokens": int(it["tokens"]), "ref_codes": ref or None}
        # the neutralised half, when the caller wants guidance
        if it.get("instruction_unc"):
            one["instruction_unc"] = it["instruction_unc"]
            one["text_unc"] = it.get("text_unc") or it["text"]
        items.append(one)
    if not items:
        return JSONResponse({"error": "no items"}, status_code=400)
    t0 = time.time()
    waves = await asyncio.get_running_loop().run_in_executor(
        None, lambda: tts.generate_batch(
            items, lora_specs=b.get("loras"), seed=b.get("seed"),
            audio_temperature=b.get("audio_temperature"),
            guidance=b.get("guidance"),
            stop_bias=b.get("stop_bias")))
    out = []
    for w in waves:
        pcm = np.clip(np.asarray(w) * 32767, -32768, 32767).astype("<i2").tobytes()
        out.append(base64.b64encode(pcm).decode())
    return {"sr": tts.sr, "pcm": out, "n": len(out),
            "gpu_ms": round((time.time() - t0) * 1000, 1)}


@app.post("/api/asr")
async def asr(req: Request):
    """Transcribe a browser recording with Parakeet TDT 0.6B v3."""
    asr = STATE["asr"]
    if asr is None:
        return JSONResponse({"error": "speech recognition not ready"},
                            status_code=503)
    raw = await req.body()
    if not raw:
        return JSONResponse({"error": "empty audio"}, status_code=400)
    t0 = time.time()
    try:
        text = await asyncio.get_running_loop().run_in_executor(
            None, asr.transcribe, raw)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"text": text, "asr_ms": round((time.time() - t0) * 1000, 1)}


@app.post("/api/turn")
async def turn(req: Request):
    return await _turn(await req.json())


async def _turn(body):
    """The whole acting turn, as a stream of events and PCM.

    Split out of the endpoint so `/api/speak` can drive exactly the same
    pipeline instead of a copy of it — a second implementation of this would
    drift from the first within a week.
    """
    message = (body.get("message") or "").strip()
    history = body.get("history") or []
    language = body.get("language") or "English"
    if not message:
        return JSONResponse({"error": "empty message"}, status_code=400)
    if STATE["error"]:
        return JSONResponse({"error": STATE["error"]}, status_code=500)
    if not (STATE["tts"] and STATE["agent"]):
        return JSONResponse({"error": "model still loading"}, status_code=503)

    tts, bank = STATE["tts"], STATE["bank"]
    # the browser sends only the choice, never a key
    be = str(body.get("brain") or config.DEFAULT_BRAIN)
    if be != "local" and be not in config.HOSTED_MODELS:
        be = config.DEFAULT_BRAIN
    # Default is the language model's own prose: it writes the GENERAL delivery
    # line and the inline cues itself, instead of emitting codes that get
    # expanded procedurally.  Costs output tokens, buys direction that is written
    # for this moment rather than assembled from a table.
    st = "codes" if str(body.get("prompt_style")) == "codes" else "prose"
    use_skills = body.get("skills", config.SKILLS_ON) is not False
    agent = STATE["agents"].get((be, st, use_skills)) or STATE["agent"]
    persona = personas.brief_for(body.get("persona"), body.get("persona_custom"))
    persona_loras = personas.loras_for(body.get("persona"),
                                       body.get("persona_custom"))
    profs = STATE["profiles"] or {}
    prof = profs.get(str(body.get("profile") or config.DEFAULT_PROFILE)) \
        or profs.get(config.DEFAULT_PROFILE)
    heard = body.get("heard_summary")
    t_req = time.time()

    async def gen():
        # ---- 1. the brain -------------------------------------------------
        try:
            out, llm_ms, _ = await agent.turn(
                message, history, persona=persona, heard=heard,
                identity=(prof or {}).get("identity"),
                extra=body.get("prompt_extra"))
        except Exception as e:
            yield _ev({"type": "error", "where": "llm", "message": str(e)})
            return

        # ---- 2. reference retrieval (already in RAM, pre-tokenised) --------
        # the model decides which language it is speaking; the reference must be
        # drawn from that language and the voice model told the same thing
        spoken = out.get("language") or language
        t_sel = time.time()
        lang2 = "de" if str(spoken).lower().startswith(("ger", "de")) else "en"
        # The declared language is not reliable — a turn written wholly in German
        # came back declared as English, which then drew an English reference
        # clip for it.  The words themselves settle it.
        if lang2 == "en" and retrieval.looks_german(out.get("reply") or ""):
            lang2 = "de"
            spoken = "German"
        # Every bracket in English, whatever is being spoken.  The prompt has
        # asked for this since the corpus was described and the director writes
        # German cues on German turns anyway; measured, that costs a median word
        # error of 0.267 against 0.000 with the same words and English cues.
        # This runs BEFORE retrieval on purpose: the retriever falls back to the
        # named emotion when the cues are German, which is how a horror scene
        # ended up conditioned on "Jealousy_and_Envy".
        if lang2 == "de" and body.get("english_cues", config.ENGLISH_CUES) \
                is not False:
            async def _tr(items):
                msg = ("Rewrite each of these stage directions in English. Keep "
                       "the meaning and the intensity adverb exactly; keep them "
                       "the same length or shorter; return only a JSON array of "
                       "strings, same order, no other text.\n\n"
                       + json.dumps(items, ensure_ascii=False))
                # The local 12B took 15.6 s on this, which is a third of the
                # turn spent translating three brackets.  Use the fastest
                # hosted model when a key is present and keep the local one as
                # the fallback, since this must not become a hard dependency.
                key = config.luna_key()
                if key:
                    base, model = config.LUNA_BASE.rstrip("/"), \
                        config.HOSTED_MODELS.get("gemini-flash-lite",
                                                 "gemini-3.5-flash-lite")
                    extra = {"reasoning_effort": "none"}
                else:
                    base, model, extra = agent.base, agent.model, {}
                r = await agent.client.post(
                    f"{base}/v1/chat/completions",
                    json={"model": model,
                          "messages": [{"role": "user", "content": msg}],
                          "temperature": 0.2,
                          "response_format": {"type": "json_object"}, **extra},
                    headers={"Authorization": f"Bearer {key}"} if key else None)
                r.raise_for_status()
                # A local `import re` further down this handler makes `re` a
                # local name for the whole function, so this closure cannot see
                # the module-level one and raised "name 're' is not defined" —
                # which silently fell through to the lexicon and produced
                # half-translated cues like "(clearly amused, frei heraus)".
                import re as _re_cue
                txt = r.json()["choices"][0]["message"]["content"] or ""
                m = _re_cue.search(r"\[.*\]", txt, _re_cue.S)
                got = json.loads(m.group(0)) if m else \
                    json.loads(txt).get("directions")
                return [str(x) for x in (got or [])]
            try:
                t_cue = time.time()
                sc2, n_cue, _orig = await cues.englishise_async(
                    out.get("script"), _tr, force=True)
                gen2, n_gen = await cues.englishise_general_async(
                    out.get("general"), _tr, force=True)
                if n_cue or n_gen:
                    out["script"] = sc2
                    out["general"] = gen2
                    print(f"[cues] {n_cue} direction(s) and {n_gen} GENERAL "
                          f"clause(s) rewritten in English in "
                          f"{(time.time()-t_cue)*1000:.0f} ms", flush=True)
            except Exception as e:
                print(f"[cues] left as written: {type(e).__name__}: {e}",
                      flush=True)
        speed = config.SPEED_WORDS.get(out.get("speed") or "normal", 1.0)
        try:
            pv = prof["id"] if prof else None
            hit = bank.select(out.get("voice"), language=lang2, speed=speed,
                              voice=pv)
            # an optional second condition, concatenated after the first, for a
            # line that genuinely moves through two states
            v2 = out.get("voice2") or {}
            hit2 = (bank.select(v2, language=lang2, speed=speed, voice=pv)
                    if v2.get("mode") not in (None, "none", "") else None)
        except Exception as e:
            print("[app] voice select failed:", e, flush=True)
            hit = hit2 = None

        # Retrieval: the director's own prose decides the reference clip.
        # Everything it wrote about sound — the standing GENERAL description and
        # every bracketed cue — is concatenated, embedded with the
        # VoiceCLAP-commercial text tower and matched against the corpus.  This
        # replaces the clip the code path picked, because it is chosen from what
        # the director actually asked for rather than from a decoded label.
        retr = None
        want_emo = body.get("emotion_nuance", config.EMOTION_NUANCE_ON) is not False
        R = STATE.get("retriever")
        if R is not None and R.ok and body.get("retrieval", config.RETRIEVAL_ON) is not False:
            try:
                _gen, _cues = retrieval.split_direction(
                    out.get("general") or (prof or {}).get("identity"),
                    out.get("script"), out.get("text"))
                direction = " ".join(b for b in (_gen, _cues) if b)[:800]
                lvl = {"intense": "A", "contained": "B"}.get(
                    (out.get("voice") or {}).get("intensity"))
                # The emotion is read off the cues, but only when they are in
                # the language the text tower actually speaks.  Otherwise fall
                # back to the emotion the director named, which is always an
                # English label from a fixed vocabulary.
                _eq = _cues
                _named = (out.get("voice") or {}).get("emotion")
                if _named and (not _cues or retrieval.looks_german(_cues)):
                    _eq = f"a person speaking with {_named.replace('_', ' ').lower()} in their voice"
                retr = R.query(direction, voice=pv, lang=lang2, level=lvl,
                               emotion_nuance=want_emo, cues=_eq)
                for h in (retr or {}).get("hits", []):
                    got = bank.select_gid(h["gid"], speed)
                    if got is not None:
                        hit = got
                        retr["used"] = h
                        break
            except Exception as e:
                print("[app] retrieval failed:", e, flush=True)
                retr = None
        sel_ms = (time.time() - t_sel) * 1000
        # identity anchor first, then the delivery clip(s) in order
        ref_codes = []
        if config.USE_ANCHOR and out.get("voice", {}).get("mode") != "character":
            a = bank.anchor(prof["anchor"]) if prof else bank.anchor()
            if a is not None:
                ref_codes.append(a)
        # The 832 conditioned clips are renderings of one specific speaker, so
        # they only belong in the prompt when that is the speaker being asked
        # for.  For any other profile the delivery comes from the emotion and
        # VoiceNet adapters instead, and the reference stays purely its anchor.
        if hit and (prof is None or prof.get("has_conditions")
                    or bank.has_matrix(prof["id"])):
            ref_codes.append(hit["codes"])
        if hit2 and hit2["entry"]["id"] != (hit or {}).get("entry", {}).get("id"):
            ref_codes.append(hit2["codes"])
        # anchor (who) -> tails of the last turns (how they were just speaking)
        # -> this moment's condition clip (what to do now)
        sess = _session(body.get("session"))
        if sess and config.USE_TAIL_CONTEXT and ref_codes:
            ref_codes = ref_codes[:1] + sess["tails"] + ref_codes[1:]
        ref_codes = ref_codes or None
        chosen = None
        if hit:
            e = hit["entry"]
            chosen = {k: e.get(k) for k in
                      ("id", "block", "emotion", "intensity", "containment",
                       "dimension", "level", "character", "edge_case", "language")}
            chosen["frames"] = int(hit["codes"].shape[0])
            chosen["anchored"] = config.USE_ANCHOR
            chosen["speed"] = hit.get("speed", 1.0)
            if hit2:
                e2 = hit2["entry"]
                chosen["then"] = (e2.get("emotion") or e2.get("character")
                                  or e2.get("dimension") or e2.get("edge_case"))

        # ---- 2b. which adapters this turn gets (doses from the manual) -----
        # "Pure" mode: the v2 base model, the retrieved reference clips, and the
        # voice's own character adapter — nothing else.  Every expressive adapter
        # (emotion, VoiceNet, burst, aesthetics, base style) is left off, so the
        # performance comes from the reference audio and the prompt alone, the
        # way the base model was meant to be driven.
        # Pure is the default: base model + this voice's own reference clips +
        # its character adapter.  Every profile now has its own 842-condition
        # matrix, so the delivery can come from real recordings of that speaker
        # rather than from expressive adapters stacked on a borrowed voice.
        # Pure: the tuned base model plus this voice's own reference clips, and
        # nothing else.  The character adapter is a separate switch because it
        # was trained against the untuned weights.
        pure = body.get("pure_mode", False) is True
        want_char = body.get("char_lora", True) is not False
        specs = []
        lb = STATE.get("lora")
        # perceptual-quality adapters: on by default, each with its own slider
        q_spec = []
        if lb:
            ql = body.get("quality_lams") or {}
            for nm, dflt in config.QUALITY_LORAS.items():
                lam = ql.get(nm, ql.get(nm.split(":")[-1], dflt))
                try:
                    lam = float(lam)
                except (TypeError, ValueError):
                    lam = dflt
                if lam > 0.001 and nm in lb.repos:
                    q_spec.append((nm, lam))
        # preference-tuned adapters: each has its own checkbox and slider, and a
        # weight of 0 means the adapter is simply not loaded
        qd_spec = []
        if lb:
            ql = body.get("qdpo_lams") or {}
            for nm, dflt in config.QDPO_LORAS.items():
                lam = ql.get(nm, ql.get(nm.split(":")[-1], dflt))
                try:
                    lam = float(lam)
                except (TypeError, ValueError):
                    lam = dflt
                if lam > 0.001 and nm in lb.repos:
                    qd_spec.append((nm, lam))
        vn_spec = []
        if lb and body.get("delivery_loras", True) is not False:
            seen_vn = set()
            for it in (out.get("style") or []):
                if not isinstance(it, dict):
                    continue
                a = it.get("adapter")
                if a not in config.SFT3_VN_ADAPTERS or a in seen_vn:
                    continue
                try:
                    lam = float(it.get("strength") or 0)
                except (TypeError, ValueError):
                    continue
                # the set is a pilot and unevaluated; keep it inside the dial
                lam = max(0.0, min(lam, max(config.SFT3_VN_LEVELS)))
                nm = f"sft3_voicenet:{a}"
                if lam > 0.001 and nm in lb.repos:
                    seen_vn.add(a)
                    vn_spec.append((nm, lam))
                if len(vn_spec) >= config.SFT3_VN_MAX:
                    break
        # ------------------------------------------------------------------
        # Measured interference.  A quality adapter that is known to cancel the
        # delivery axis now in play is scaled down rather than left to fight it:
        # ESTH and S_RANT at equal strength produce neither effect (+0.464 alone
        # against -0.012 together).  See config.QUALITY_CONFLICTS.
        if q_spec and vn_spec:
            active_vn = {nm.split(":", 1)[-1] for nm, _ in vn_spec}
            adjusted = []
            for nm, lam in q_spec:
                rule = getattr(config, "QUALITY_CONFLICTS", {}).get(nm, {})
                factor = min([rule[a] for a in active_vn if a in rule], default=1.0)
                if factor < 1.0:
                    print(f"[lora] {nm} {lam:g} -> {lam * factor:g}: measured to "
                          f"conflict with {sorted(active_vn & set(rule))}", flush=True)
                if lam * factor > 0.001:
                    adjusted.append((nm, lam * factor))
            q_spec = adjusted

        dpo_spec = []
        if lb and body.get("dpo_lora", True) is not False \
                and config.SFT3_DPO_LORA in lb.repos and config.SFT3_DPO_LAM > 0.001:
            dpo_spec = [(config.SFT3_DPO_LORA, config.SFT3_DPO_LAM)]
        emo_spec = []
        if (lb and want_emo and retr and retr.get("emotion")
                and f"sft3_emotion:{retr['emotion']}" in lb.repos):
            # 1.5 is the published operating point: emotion 0.408 -> 0.471 with
            # genuineness and burst blend rising alongside it, and median word
            # error still 0.000.  Off when the emotion nuances are switched off,
            # which leaves the base checkpoint and the retrieved clip alone.
            emo_spec = [(f"sft3_emotion:{retr['emotion']}", config.SFT3_EMOTION_LAM)]
        if lb and pure:
            if want_char:
                lam = body.get("profile_lora_lam")
                lam = config.PURE_PROFILE_LAM if lam is None else float(lam)
                if prof and prof["lora"] in lb.repos and lam > 0.001:
                    specs = [(prof["lora"], lam)]
            specs = dpo_spec + q_spec + qd_spec + specs + vn_spec + emo_spec
        elif lb:
            try:
                mix = out.get("blend")
                if mix:
                    # code mode gave us the whole mixture: merge every emotion in
                    # it, weighted by its level, plus the voice qualities
                    specs = lora_bank.plan_blend(
                        mix, out.get("script"),
                        have_emotion=set(lb.names("emotion")),
                        have_voicenet=set(lb.names("voicenet")),
                        have_burst=lb.names("burst"))
                else:
                    specs = lora_bank.plan(
                        out.get("voice"), out.get("script"),
                        have_emotion=set(lb.names("emotion")),
                        have_character=set(lb.names("character")),
                        have_burst=lb.names("burst"),
                        have_voicenet=set(lb.names("voicenet")),
                        style=None)
            except Exception as e:
                print("[app] lora plan failed:", e, flush=True)
            # The speaker adapter goes on top of everything else: it is what makes
            # the voice the same person from turn to turn, so it is not one of the
            # per-moment choices the director gets to make.
            # Both of these are dials on the voice itself rather than choices
            # about this moment, so they are prepended, and a dose of 0 simply
            # means the adapter is not merged at all.
            def _dial(flag, dose, name, default_lam):
                if flag is False or name not in lb.repos:
                    return None
                lam = default_lam if dose is None else float(dose)
                return (name, lam) if lam > 0.001 else None

            # The base style is a floor, not a ceiling: if the director asked for
            # one of these dimensions itself — WARM low for something cold, or
            # S_CASU low for something formal — that choice wins, because it was
            # made for this moment and the default was not.
            chosen_dims = {n.rsplit("__", 1)[0] for n, _ in specs
                           if n.startswith("voicenet:")}
            base_style = [] if body.get("base_style") is False else [
                (n, l) for n, l in config.BASE_STYLE_LORAS
                if n in lb.repos and n.rsplit("__", 1)[0] not in chosen_dims]
            fixed = base_style + [(n, l) for n, l in persona_loras
                                  if n in lb.repos and
                                  n not in {x for x, _ in base_style}]
            have_profile = False
            if prof and prof["lora"] in lb.repos:
                lam = body.get("profile_lora_lam")
                lam = config.PROFILE_LORA_LAM if lam is None else float(lam)
                if lam > 0.001:
                    fixed = [(prof["lora"], lam)] + fixed
                    have_profile = True
            head = fixed + [x for x in (
                _dial(body.get("speaker_lora", not have_profile)
                      and not have_profile,
                      body.get("speaker_lora_lam"),
                      config.SPEAKER_LORA, config.SPEAKER_LORA_LAM),
                _dial(body.get("aesth_lora", True), body.get("aesth_lora_lam"),
                      config.AESTH_LORA, config.AESTH_LORA_LAM),
            ) if x]
            # a fixed or dialled adapter wins over the same one picked as a
            # per-moment style, so it keeps its intended dose
            picked = {n for n, _ in head}
            specs = head + [s for s in specs if s[0] not in picked]
            if dpo_spec:
                specs = dpo_spec + [x for x in specs if x[0] != config.SFT3_DPO_LORA]
            if qd_spec:
                picked_qd = {n for n, _ in qd_spec}
                specs = [x for x in specs if x[0] not in picked_qd] + qd_spec
            if q_spec:
                picked_q = {n for n, _ in q_spec}
                specs = [x for x in specs if x[0] not in picked_q] + q_spec
            if vn_spec:
                picked_vn = {n for n, _ in vn_spec}
                specs = [x for x in specs if x[0] not in picked_vn] + vn_spec
            if emo_spec:
                # the v3 emotion adapters were trained against the untuned v2
                # weights; on sft3 the matching set is the one trained with it
                specs = [s for s in specs if not s[0].startswith("emotion:")] + emo_spec

        # A burst named inside a delivery direction never becomes a sound: the
        # bracket is an instruction, no timing is budgeted for it and no adapter
        # is pulled.  Observed on a real turn whose "(a raw, tearing scream, ...)"
        # produced no scream at all.  Repaired here, once, so the guard, the
        # engine and the transcript in the UI all see the same script.
        if use_skills:
            try:
                import skills as _sk0
                _s0 = _sk0.load()
                if _s0 is not None and _s0.ok and lb is not None:
                    _fixed, _added = _s0.repair_script(
                        out.get("script") or "", lb.names("burst"))
                    if _added:
                        out["script"] = _fixed
                        print(f"[skills] burst named inside a direction, given its "
                              f"own bracket: {_added}", flush=True)
            except Exception as e:
                print(f"[skills] script repair skipped: {e}", flush=True)

        # Per-class burst weights.  The flat 0.25 / 0.5 was a conservative guess
        # made when seven adapters existed, and it was capped by a genuineness
        # gate that has since been dropped on purpose; the measured optimum is
        # per class and runs 0.25 to 2.3, so for most classes the shipped dose
        # sat well under it.  The wiki is the source of truth where it has a
        # measured recipe; the flat default stays the fallback for a class it has
        # never measured.  Only applied with skills on, so the two are comparable.
        if use_skills:
            try:
                import skills as _sk
                _s = _sk.load()
                if _s is not None and _s.ok:
                    # which adapter set the automatic resolution draws from.
                    # "recipe" asks each class's own page; anything else pins one
                    # set for the whole turn so the two can be compared.
                    _sel = str(body.get("burst_set") or config.BURST_SET)
                    def _reroot(n):
                        cls = n.split(":", 1)[-1]
                        if _sel == "recipe":
                            cand = _s.adapter_for(cls)
                        else:
                            cand = f"{config.BURST_SET_ROOT.get(_sel, 'burst')}:{cls}"
                        return cand if cand in lb.repos else n
                    try:
                        _cap = float(body.get("burst_lam_max")
                                     if body.get("burst_lam_max") is not None
                                     else config.BURST_LAM_MAX)
                    except (TypeError, ValueError):
                        _cap = config.BURST_LAM_MAX

                    def _burst_lam(name, flat):
                        # `weight_for` returns the flat default unchanged when the
                        # class has no measured recipe, so the cap only ever bites
                        # on a measured weight.
                        return min(_s.weight_for(name.split(":", 1)[-1], flat), _cap)
                    _isb = lambda n: n.split(":", 1)[0].startswith("burst")
                    # One adapter per burst the script actually tags.  The
                    # planner returns a single best match over the whole reply,
                    # chosen by longest string — so a line with a scream and an
                    # exasperated sigh loaded only the sigh, because its name is
                    # the longer one.  Read the tags instead, in order.
                    tagged = []
                    for it in timed_script.parse(out.get("script") or ""):
                        if it[0] == "burst":
                            lab = str(it[1]).strip().lower().replace(" ", "_")
                            if lab and lab not in tagged:
                                tagged.append(lab)
                    tagged = tagged[:config.BURST_MAX_ADAPTERS]
                    # SOLO CEILING.  A single burst adapter may go to
                    # BURST_LAM_MAX_SOLO (1.5); two may not.  Both halves are
                    # measured: two at 1.5 destroyed the line in 5 of 5 seeds
                    # (median word error 0.82) while one at 1.5 came back clean
                    # every time, and study `vb_opt` found the 1.5 ceiling adds
                    # +0.0275 to the candidate mean but only +0.0050 to the
                    # DELIVERED take -- the extra burst was generated and then
                    # not selected.  So it is gated on a burst-aware ranker
                    # existing, which is BON_BURST_WEIGHT > 0 with best-of-N on.
                    # An explicit burst_lam_max in the request always wins.
                    # Gated on a burst-aware ranker EXISTING, which is not the
                    # same as best-of-N being the server default: the UI turns
                    # best-of-N on per request, and `config.BON_ON` is only the
                    # default.  What actually matters is whether this turn will
                    # be ranked by something that can see the burst — N > 1, a
                    # non-zero weight, and a detector able to populate the term.
                    # `bon_n` is computed further down this handler, so read
                    # the request directly rather than closing over a name that
                    # is not bound yet — referencing it here threw, the whole
                    # burst block fell into its `except`, and every burst
                    # adapter silently dropped to the flat 0.25 default.
                    try:
                        _n_req = int(body.get("best_of")
                                     if body.get("best_of") is not None
                                     else (config.BON_N if config.BON_ON else 1))
                    except (TypeError, ValueError):
                        _n_req = 1
                    if (len(tagged) == 1
                            and body.get("burst_lam_max") is None
                            and _n_req > 1
                            and config.BON_BURST_WEIGHT > 0
                            and config.BON_BURST_READY
                            and config.BURST_LAM_MAX_SOLO > _cap):
                        print(f"[skills] one burst ({tagged[0]}): ceiling "
                              f"{_cap:g} -> {config.BURST_LAM_MAX_SOLO:g}",
                              flush=True)
                        _cap = config.BURST_LAM_MAX_SOLO
                    specs = [(n, l) for n, l in specs if not n.startswith("burst")]
                    # Fit the whole set inside the budget by SCALING, not by
                    # dropping.  Bursts carry the drama of a line — a scream that
                    # is quietly not merged is a worse outcome than a scream at
                    # two thirds weight — and dropping also silently changed
                    # which classes a reply could use depending on their order.
                    # Measured: two adapters at 1.5 each destroyed the line in
                    # 5 of 5 seeds (median word error 0.82); the same script with
                    # one at 1.5, or two at 0.5, came back clean every time.  It
                    # is the SUM that breaks a line here, which is the opposite
                    # of what the single-adapter ladder found on a bare model.
                    _want = []
                    for lab in tagged:
                        _want.append((lab, _burst_lam(f"burst:{lab}",
                                                      config.BURST_LAM)))
                    _sum = sum(w for _, w in _want)
                    _scale = 1.0
                    if _sum > config.BURST_LAM_BUDGET and _sum > 0:
                        _scale = config.BURST_LAM_BUDGET / _sum
                        print(f"[skills] burst budget {config.BURST_LAM_BUDGET:g}: "
                              f"{len(_want)} adapters want {_sum:g}, scaling by "
                              f"{_scale:.2f}", flush=True)
                    _spent = 0.0
                    for lab, _w0 in _want:
                        cand = _reroot(f"burst:{lab}")
                        _w = round(_w0 * _scale, 3)
                        if _w < 0.05:
                            print(f"[skills] {lab} scaled below audibility "
                                  f"({_w:g}), not merged", flush=True)
                            continue
                        if cand in lb.repos:
                            _spent += _w
                            # weight by the CLASS, not the adapter name: the
                            # ablation arms are filed as
                            # `ablation_d2_matched__scream`, whose tail is not
                            # the class, so weighting by the name silently fell
                            # back to the flat default.
                            specs.append((cand, _w))

                    # Cap only.  This used to re-run `_burst_lam`, which looks
                    # the weight up from the recipe again — so it overwrote the
                    # budget scaling for any adapter whose name tail IS the
                    # class.  `burst_v2:fearful_gasp` went back to its recipe
                    # 1.5 while `burst_abl:ablation_d2_matched__scream` kept its
                    # scaled 0.923, because that tail is not a class name and
                    # the lookup fell through to the value passed in.  The two
                    # then summed to 2.42 against a budget of 2.0.
                    specs = [((n, min(l, _cap)) if _isb(n) else (n, l))
                             for n, l in specs]
                    # Last word on the budget, whatever put a burst adapter in
                    # the list — a slider, an override, a second code path.
                    _bs = [(n, l) for n, l in specs if _isb(n)]
                    _tot = sum(l for _, l in _bs)
                    if _tot > config.BURST_LAM_BUDGET and _tot > 0:
                        _f = config.BURST_LAM_BUDGET / _tot
                        print(f"[skills] burst total {_tot:.3g} over budget "
                              f"{config.BURST_LAM_BUDGET:g}, scaling by {_f:.2f}",
                              flush=True)
                        specs = [((n, round(l * _f, 3)) if _isb(n) else (n, l))
                                 for n, l in specs]
            except Exception as e:
                print(f"[skills] burst weights unchanged: {e}", flush=True)

        # The overlay's sliders: a full adapter name mapped to a weight.  0 means
        # "leave it to the director", which is not the same as forcing it off —
        # the director's own picks stay unless a slider names them.
        ov = body.get("adapter_overrides") or {}
        if lb and isinstance(ov, dict) and ov:
            forced = []
            for nm, lam in ov.items():
                try:
                    lam = float(lam)
                except (TypeError, ValueError):
                    continue
                if lam > 0.001 and nm in lb.repos:
                    forced.append((nm, lam))
            if forced:
                names = {n for n, _ in forced}
                specs = [x for x in specs if x[0] not in names] + forced

        # ---- 2c. which of the three levers pushes it ----------------------
        # `voice` and `style` above chose WHAT to push.  This chooses whether the
        # adapter does it alone (today's behaviour), whether a steering vector is
        # added to the hidden state, or whether guidance runs the model twice per
        # frame.  levers.py owns the policy and every rule in it is a measurement.
        pf = out.get("perform") or {}
        # Which attribute is being pushed.  The director may name one explicitly;
        # otherwise it is whatever it already chose — the emotion first, because a
        # delivery adapter is already doing its own job, and the delivery axis only
        # if no emotion is in play.
        vsel = out.get("voice") or {}
        style_first = next((it.get("adapter") for it in (out.get("style") or [])
                            if isinstance(it, dict) and it.get("adapter")), None)
        attr_name = (pf.get("dimension")
                     or (retr or {}).get("emotion")
                     or (vsel.get("emotion") if vsel.get("mode") == "emotion" else None)
                     or style_first)
        # The adapter in this turn's plan that carries that attribute, if any: a
        # lever-only mode drops exactly that one and leaves the rest of the stack
        # (DPO, profile, quality) alone, which is what `lora_w = 0` meant in the study.
        _prefix = {"emo": "sft3_emotion:", "vn": "sft3_voicenet:",
                   "qual": "sft3_quality:"}.get(levers.family(attr_name) or "")
        attr_adapter = None
        if _prefix:
            want = _prefix + str(attr_name)
            attr_adapter = next((n for n, _ in specs if n == want), None)
        req_mode = body.get("gen_mode")
        if not req_mode:
            req_mode = pf.get("mode") if config.AGENT_PICKS_MODE else "auto"
            if (req_mode or "auto") == "auto" and config.GEN_MODE != "auto":
                req_mode = config.GEN_MODE      # the server pins it
        plan = levers.plan(
            req_mode, attr_name, pf.get("strength"),
            wiki=STATE["wiki"],
            pack=STATE["vectors"],
            active_delivery_adapters={n.split(":", 1)[-1] for n, _ in specs
                                      if n.startswith("sft3_voicenet:")},
            attribute_adapter=attr_adapter,
            cfg_available=bool(out.get("general_unc")))
        # an explicit strength from the UI slider overrides the measured operating
        # point, and says so, so the payload never claims a number it did not use
        gv = body.get("guidance")
        if gv is not None and plan is not None and plan.wants_cfg:
            try:
                gv = float(gv)
            except (TypeError, ValueError):
                gv = None
            if gv and abs(gv - float(plan.guidance or 1.0)) > 1e-6:
                plan.note(f"guidance set to {gv:g} by the request, overriding the "
                          f"{plan.operating_point or 'measured'} value "
                          f"{float(plan.guidance or 1.0):g}")
                plan.guidance = gv
                plan.operating_point = "request"
        if plan.drop_adapter:
            specs = [x for x in specs if x[0] != plan.drop_adapter]
        print(plan.log_line(), flush=True)

        yield _ev({"type": "llm", "levers": plan.payload(),
                   "reply": out["reply"], "general": out["general"],
                   # the neutralised half, so the browser can ask for a sweep on
                   # exactly this line without the director running again
                   "general_unc": out.get("general_unc"),
                   "script": out["script"], "voice": out.get("voice"),
                   "chosen": chosen, "llm_ms": round(llm_ms, 1),
                   "select_ms": round(sel_ms, 1), "language": out.get("language"),
                   "speed": out.get("speed"), "style": out.get("style"),
                   "brain": be, "prompt_style": st, "codes": out.get("codes"),
                   "profile": prof["id"] if prof else None,
                   "pure": pure, "char_lora": want_char, "skills": use_skills,
                   "retrieval": ({"direction": retr.get("direction"),
                                  "emotion": retr.get("emotion"),
                                  "emotions": retr.get("emotions", [])[:3],
                                  "used": {k: retr["used"].get(k) for k in
                                           ("gid", "emotion", "dim", "level",
                                            "lang", "score", "genuineness",
                                            "blend")}
                                  if retr.get("used") else None}
                                 if retr else None),
                   "emotion_nuance": want_emo,
                   "ref_clips": len(ref_codes or []),
                   # only the corpus voice has the 832 conditioned clips, so for
                   # any other profile the reference is its anchor alone
                   "ref_conditioned": bool(hit and (prof is None
                                                    or prof.get("has_conditions")
                                                    or bank.has_matrix(prof["id"]))),
                   "loras": [{"name": n, "lam": l} for n, l in specs],
                   "cached": bool(hit and hit["entry"]["id"] in bank.codes)})

        # ---- 3. stream the performance ------------------------------------
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        loop = asyncio.get_running_loop()
        want_vc = bool(body.get("voice_conversion")) and STATE["vc"] is not None
        vc = STATE["vc"] if want_vc else None
        if vc is not None and prof:
            # otherwise every profile is converted onto the corpus anchor, which
            # is exactly how every voice ended up sounding like Velvet Sage
            try:
                await loop.run_in_executor(None, vc.use_target, prof["anchor"])
            except Exception as e:
                print("[vc] target switch failed:", e, flush=True)
        vc_st = vc.new_stream(tts.sr or 48000) if vc is not None else None
        take = []          # kept only to score speaker similarity at the end

        # End-trimming: fade in at the first scripted word, cut after the last.
        # The guard holds a lookahead so there is room to cut before the filler
        # would be audible; with the aligner missing or the box unticked it stays
        # None and this path is exactly what it was.
        guard = None
        if body.get("align", config.ALIGN_ON) is not False \
                and STATE.get("aligner") is not None and STATE["aligner"].ok \
                and config.TIMED_SCRIPT:
            try:
                _tg, _fr, _pl = timed_script.render(out["script"], speed=speed)
                if _pl:
                    guard = align_engine.StreamGuard(
                        STATE["aligner"], tts.sr, _tg, _pl, enabled=True,
                        expect_s=_fr / config.FRAME_RATE)
            except Exception as e:
                print(f"[align] guard not built: {e}", flush=True)

        # ---- best-of-N -----------------------------------------------------
        # Generate the whole turn N times in one batched pass, score every
        # candidate and stream only the winner.  Nothing is streamed while this
        # runs -- the point is to choose, and choosing needs all of them.
        bon_n = body.get("best_of")
        try:
            bon_n = int(bon_n) if bon_n is not None else (
                config.BON_N if config.BON_ON else 1)
        except (TypeError, ValueError):
            bon_n = 1
        bon = None
        bon_error = None
        # Restoration needs the whole take, so a turn that will be restored
        # cannot be streamed: it is generated offline, restored, and then sent
        # down the same chunked protocol.  That makes an ordinary chat turn take
        # the best-of-N path with N = 1.
        want_sidon = body.get("sidon", config.SIDON_ON) is not False
        if want_sidon and not sidon.up():
            want_sidon = False
        # Guidance cannot stream either, so a single take that asks for it has
        # to take this path too.  Without this, `best_of: 1` with
        # `best_of_guidance: 3.0` streamed unguided and reported nothing — the
        # setting was accepted and silently ignored.
        try:
            _gv_req = float(body.get("best_of_guidance")
                            if body.get("best_of_guidance") is not None
                            else (config.BON_GUIDANCE if bon_n > 1 else 1.0))
        except (TypeError, ValueError):
            _gv_req = 1.0
        if (bon_n > 1 or want_sidon or _gv_req > 1.0001) \
                and STATE.get("judge") is not None and config.TIMED_SCRIPT:
            try:
                _tg, _fr, _pl = timed_script.render(out["script"], speed=speed)
                lc = "DE" if str(spoken).lower().startswith(("ger", "de")) else "EN"
                _gl = timed_script.general_line(
                    out["general"], _fr / config.FRAME_RATE, lc,
                    (retr or {}).get("emotion"))
                item = {"text": _tg, "tokens": _fr, "language": spoken,
                        "instruction": f"GENERAL: {_gl}\nSCRIPT:\n{_tg}",
                        "ref_codes": ref_codes}
                # Guidance is not a restoration setting: a plain turn that is
                # only here to be restored must not silently acquire a 1.93x
                # cost it never asked for.  `_gv_req` carries what was asked.
                gv = _gv_req
                if gv > 1.0001:
                    # The director is asked for a neutralised delivery line and
                    # does not always return one.  Guidance used to fall back to
                    # 1.0 when it was missing, silently — a best-of-N run that
                    # reported g=3.0 in the UI and had actually generated
                    # unguided.  The script neutralisation alone already gives
                    # the two branches something to differ by, which is what
                    # /api/cfg_sweep has always done, so fall back to the same
                    # GENERAL rather than to no guidance at all.
                    _gu = timed_script.general_line(
                        out.get("general_unc") or out.get("general") or "",
                        _fr / config.FRAME_RATE, lc, None)
                    _tu = timed_script.neutralise(_tg)
                    if _tu.strip() and _tu != _tg:
                        item["instruction_unc"] = f"GENERAL: {_gu}\nSCRIPT:\n{_tu}"
                        item["text_unc"] = _tu
                    else:
                        print("[bestofn] nothing to neutralise; guidance off",
                              flush=True)
                        gv = 1.0
                t_bon = time.time()

                _seed0 = int(body.get("seed") or 1234)

                def _gen(items, g, seed=None):
                    return tts.generate_batch(items, lora_specs=specs,
                                              seed=_seed0 if seed is None else seed,
                                              guidance=g, seed_per_item=True)

                def _gen_chunked():
                    """Generate N candidates, in as few passes as fit in memory.

                    Guidance doubles the batch -- two branches per candidate --
                    so eight candidates are sixteen sequences, and that OOMs on a
                    24 GB card often enough to matter.  It used to fail the whole
                    block and fall back to streaming, which produced exactly one
                    take and no explanation.  Halve and retry instead: a slower
                    best-of-N is still best-of-N.
                    """
                    import torch as _t
                    size = min(bon_n, config.BON_BATCH_CFG if gv > 1.0001
                               else config.BON_BATCH)
                    while size >= 1:
                        out, ok = [], True
                        try:
                            for off in range(0, bon_n, size):
                                k = min(size, bon_n - off)
                                # A different seed per chunk.  `torch.manual_seed`
                                # is called once per batch, so every chunk of
                                # identical items started from the same RNG state
                                # and produced the same takes again: best-of-8 at
                                # guidance, which chunks 4 + 4, was really
                                # best-of-4 with each candidate listed twice.
                                out += _gen([dict(item) for _ in range(k)], gv,
                                            seed=_seed0 + off)
                        except _t.cuda.OutOfMemoryError:
                            ok = False
                            _t.cuda.empty_cache()
                            print(f"[bestofn] out of memory at batch {size}, "
                                  f"halving", flush=True)
                            size //= 2
                        if ok:
                            return out
                    raise RuntimeError("not enough memory for a single candidate")

                waves = await loop.run_in_executor(None, _gen_chunked)
                # Restore BEFORE ranking, so the reward is computed on the audio
                # that will actually be heard rather than on a version of it
                # nobody gets.  The originals travel alongside for comparison.
                raw_waves = list(waves)
                sidon_used = False
                if want_sidon:
                    waves, sidon_used = await loop.run_in_executor(
                        None, lambda: sidon.enhance_many(raw_waves, tts.sr))
                # Trim EVERY candidate, not only the one that gets played.
                # The players in the UI were showing untrimmed audio while the
                # stream carried a trimmed take — a reply whose player said
                # 18.4 s next to a stream of 16.5 s, which reads as "end
                # trimming is not working" when it is working and the listener
                # is simply hearing the wrong copy.  Trimming before the judge
                # also means the reward ranks what will actually be heard.
                bon_aligns = [None] * len(waves)
                if STATE.get("aligner") is not None \
                        and body.get("align", config.ALIGN_ON) is not False:
                    def _trim_all(ws, raws):
                        outs, raws2, reps = [], [], []
                        for k, w in enumerate(ws):
                            try:
                                t, r = align_engine.trim(w, tts.sr, _tg, _pl,
                                                         STATE["aligner"])
                            except Exception as e:
                                print(f"[align] candidate {k} skipped: {e}",
                                      flush=True)
                                t, r = w, None
                            outs.append(t)
                            reps.append(r)
                            # cut the original at the same point: restoration
                            # preserves length to about 20 ms, so the same
                            # sample index is the same moment, and the two
                            # players should not differ in length for a reason
                            # that has nothing to do with restoration
                            rw = raws[k] if k < len(raws) else None
                            raws2.append(rw if rw is None else rw[:len(t)])
                        return outs, raws2, reps
                    waves, raw_waves, bon_aligns = await loop.run_in_executor(
                        None, lambda: _trim_all(waves, raw_waves))
                cands = STATE["judge"].score(waves, tts.sr, _pl,
                                             general=out["general"],
                                             script=out["script"],
                                             tagged=_tg)
                bestofn.rank(cands)
                best = min(range(len(cands)), key=lambda i: cands[i]["rank"])
                bon = {"n": len(cands), "guidance": gv,
                       "ms": round((time.time() - t_bon) * 1000, 1),
                       "chosen": best, "sidon": sidon_used,
                       # `burst` and `n_burst` belong here too: without them
                       # the term ranked correctly and was invisible in the UI
                       # and in every log, which reads exactly like a term that
                       # is not working.
                       "candidates": [{k: c.get(k) for k in
                                       ("reward", "rank", "gate", "wer", "extra_w",
                                        "genuineness", "blend", "clap", "sec",
                                        "burst", "n_burst")}
                                      for c in cands]}
                print(f"[bestofn] {len(cands)} candidates, g={gv:g}, "
                      f"{bon['ms']:.0f} ms, best reward "
                      f"{cands[best]['reward']:.3f} (wer {cands[best]['wer']:.3f})",
                      flush=True)
                bon["wave"] = waves[best]
                # Every candidate travels with the ranking so they can be
                # compared by ear.  The reward is a model scoring another
                # model's output, and the only way to find out whether rank 0
                # is really the best take is to listen to the others.
                if body.get("best_of_audio", True) is not False:
                    import base64 as _b64
                    order = sorted(range(len(cands)),
                                   key=lambda i: cands[i]["rank"])
                    for pos, i in enumerate(order):
                        w = waves[i]
                        if w is None or not len(w):
                            continue
                        pcm16 = np.clip(np.asarray(w, np.float32) * 32767,
                                        -32768, 32767).astype("<i2").tobytes()
                        bon["candidates"][i]["pcm"] = _b64.b64encode(pcm16).decode()
                        bon["candidates"][i]["index"] = i
                        bon["candidates"][i]["align"] = bon_aligns[i] \
                            if i < len(bon_aligns) else None
                        if sidon_used and i < len(raw_waves) \
                                and raw_waves[i] is not None:
                            r16 = np.clip(np.asarray(raw_waves[i], np.float32)
                                          * 32767, -32768,
                                          32767).astype("<i2").tobytes()
                            bon["candidates"][i]["pcm_raw"] = \
                                _b64.b64encode(r16).decode()
            except Exception as e:
                import traceback
                print(f"[bestofn] failed, falling back to streaming: "
                      f"{type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                bon = None
                # Say so.  Falling back silently produces exactly one player and
                # looks like an ordinary turn, which is how this went unnoticed.
                bon_error = f"{type(e).__name__}: {str(e)[:160]}"

        # A winning candidate is emitted here and the streaming loop skipped:
        # it is already generated, so re-generating it would be a second take and
        # a different one.  The chunking mirrors the streaming protocol so the
        # client cannot tell the difference apart from when the audio starts.
        if bon is not None and bon.get("wave") is not None \
                and len(bon["wave"]):
            w = np.asarray(bon["wave"], np.float32)
            # already trimmed above, with every other candidate; trimming a
            # second time would only re-run the aligner on its own output
            bon["align"] = (bon.get("candidates") or [{}])[
                bon.get("chosen", 0)].get("align") \
                if bon.get("candidates") else None
            payload = {k: v for k, v in bon.items() if k != "wave"}
            payload["sr"] = tts.sr
            yield _ev({"type": "best_of", **payload})
            first = True
            step = int(tts.sr * 0.2)
            for off in range(0, len(w), step):
                chunk = w[off:off + step]
                if first:
                    first = False
                    yield _ev({"type": "start", "sr": tts.sr,
                               "ttfa_server_ms": round(
                                   (time.time() - t_req) * 1000, 1),
                               "best_of": bon["n"]})
                take.append(chunk)
                yield _pcm(chunk)
            dur = len(w) / float(tts.sr)
            yield _ev({"type": "end", "audio_sec": round(dur, 3),
                       "gpu_total_ms": bon["ms"], "best_of": bon["n"],
                       "loras": [{"name": n, "lam": l} for n, l in specs],
                       "align": bon.get("align"),
                       "rtf": round((bon["ms"] / 1000.0) / dur, 3) if dur else None})
            return

        # If the client goes away mid-reply — tab closed, connection dropped — the
        # producer thread would otherwise block forever handing the next chunk to
        # a queue nobody drains, and it holds the generation lock while it does,
        # which wedges every later request.  The flag lets it notice and unwind.
        gone = threading.Event()

        def produce():
            try:
                from llm_agent import LLMAgent as _A
                for kind, payload in tts.stream_utterance(
                        script=out["script"], general=out["general"],
                        strip=_A._strip_tags,
                        language=spoken, ref_codes=ref_codes, lora_specs=specs,
                        speed=speed, lever_plan=plan,
                        general_unc=(out.get("general_unc")
                                     if plan.wants_cfg else None),
                        reads_as=((retr or {}).get("emotion")
                                  or (out.get("voice") or {}).get("emotion")),
                        max_new_tokens=body.get("max_new_tokens"),
                        chunk_frames=body.get("chunk_frames"),
                        seed=body.get("seed"),
                        audio_temperature=body.get("audio_temperature"),
                        stop_bias=body.get("stop_bias")):
                    if gone.is_set():
                        break
                    if not _handoff(q, loop, (kind, payload), gone):
                        break
            except Exception as e:
                import traceback; traceback.print_exc()
                _handoff(q, loop, ("error", {"message": str(e)}), gone)
            finally:
                _handoff(q, loop, (None, None), gone)

        threading.Thread(target=produce, daemon=True).start()

        try:
          while True:
            kind, payload = await q.get()
            if kind is None:
                break
            if kind == "pcm":
                if vc is not None:
                    payload = await loop.run_in_executor(
                        None, vc.convert_chunk, payload, vc_st)
                    if payload is None or not payload.size:
                        continue
                if guard is not None:
                    payload = await loop.run_in_executor(None, guard.feed, payload)
                    if payload is None or not payload.size:
                        continue
                take.append(payload)
                yield _pcm(payload)
            elif kind == "start":
                payload = dict(payload)
                # wall-clock from request arrival to the first audio leaving the server
                payload["ttfa_server_ms"] = round((time.time() - t_req) * 1000, 1)
                payload["type"] = "start"
                yield _ev(payload)
            elif kind == "end":
                if bon_error:
                    payload = dict(payload)
                    payload["best_of_error"] = bon_error
                if guard is not None:
                    tail = guard.flush()
                    if tail is not None and tail.size:
                        take.append(tail)
                        yield _pcm(tail)
                    payload = dict(payload)
                    payload["align"] = guard.report
                    if guard.report.get("applied"):
                        r = guard.report
                        print(f"[align] lead={r.get('lead_s')} cut={r.get('cut_s')} "
                              f"removed={r.get('removed_s')}s "
                              f"checks={r.get('checks')}", flush=True)

                if vc is not None:                 # emit the held-back seam
                    tailp = vc.flush(vc_st)
                    if tailp.size:
                        yield _pcm(tailp)
                payload = dict(payload)
                payload["type"] = "end"
                payload["server_total_ms"] = round((time.time() - t_req) * 1000, 1)
                payload["llm_ms"] = round(llm_ms, 1)
                payload["speaker_lora"] = next(
                    (l for n, l in specs if n == config.SPEAKER_LORA), 0.0)
                payload["aesth_lora"] = next(
                    (l for n, l in specs if n == config.AESTH_LORA), 0.0)
                payload["vc"] = vc is not None
                tail = payload.pop("tail_codes", None)
                if sess is not None and tail is not None and len(tail):
                    sess["tails"].append(tail)
                    del sess["tails"][:-config.TAIL_TURNS]
                payload["tail_turns"] = len(sess["tails"]) if sess else 0
                if STATE["sim"] and take:
                    t_s = time.time()
                    w = np.concatenate(take)
                    payload["speaker_sim"] = await loop.run_in_executor(
                        None, STATE["sim"].score, w, tts.sr,
                        (prof or {}).get("anchor"))
                    payload["sim_ms"] = round((time.time() - t_s) * 1000, 1)
                yield _ev(payload)
            elif kind == "error":
                yield _ev({"type": "error", "where": "tts", **payload})
        finally:
            # the client may have disconnected; tell the producer to stop so it
            # releases the generation lock instead of parking on the queue
            gone.set()
            try:
                while True:
                    q.get_nowait()
            except Exception:
                pass

    return StreamingResponse(gen(), media_type="application/octet-stream",
                             headers={"Cache-Control": "no-store",
                                      "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.APP_PORT)


def _to_mp3(pcm_i16: bytes, sr: int, bitrate="192k") -> bytes:
    """Little-endian int16 mono to MP3, through ffmpeg.

    ffmpeg is already a dependency of this box and encoding in-process would
    mean another wheel; this is one pipe and about 40 ms for a fifteen-second
    take.
    """
    import subprocess
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
         "-codec:a", "libmp3lame", "-b:a", bitrate, "-f", "mp3", "pipe:1"],
        input=pcm_i16, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg: {p.stderr.decode()[:200]}")
    return p.stdout


@app.post("/api/speak")
async def speak(req: Request):
    """Text in, MP3 out — the whole demo behind one request.

    Send what you want said. The language model writes the performance, the
    reference recording is retrieved, the adapters are chosen, `best_of`
    candidates are generated and ranked, and the winner comes back as an MP3.

    Everything the web UI can set is optional here and defaults to what the UI
    ships with, so an empty body beyond `text` gives the demo's own settings.
    """
    import base64 as _b64
    body = await req.json()
    text = (body.get("text") or body.get("message") or "").strip()
    if not text:
        return JSONResponse({"error": "no text"}, status_code=400)

    # the shipped defaults, each overridable by name
    call = dict(body)
    call["message"] = text
    call.pop("text", None)
    call.setdefault("best_of", config.SPEAK_BEST_OF)
    call.setdefault("best_of_guidance", config.SPEAK_GUIDANCE)
    call["best_of_audio"] = bool(body.get("best_of_audio"))  # only the winner unless asked
    call.setdefault("sidon", config.SIDON_ON)
    want = str(body.get("format") or "mp3").lower()
    if want not in ("mp3", "wav", "json"):
        return JSONResponse({"error": "format must be mp3, wav or json"},
                            status_code=400)

    t0 = time.time()
    resp = await _turn(call)
    if isinstance(resp, JSONResponse):
        return resp
    pcm, meta, cands = [], {}, []
    async for chunk in resp.body_iterator:
        i = 0
        while i < len(chunk):
            kind = chunk[i]
            n = struct.unpack(">I", chunk[i + 1:i + 5])[0]
            payload = chunk[i + 5:i + 5 + n]
            i += 5 + n
            if kind == 1:
                pcm.append(payload)
                continue
            try:
                ev = json.loads(payload)
            except Exception:
                continue
            if ev.get("type") == "error":
                return JSONResponse({"error": ev.get("message")},
                                    status_code=500)
            if ev.get("type") == "llm":
                meta.update({k: ev.get(k) for k in
                             ("reply", "general", "script", "language", "voice",
                              "chosen", "brain", "llm_ms", "profile")})
            elif ev.get("type") == "best_of":
                meta["best_of"] = ev.get("n")
                meta["guidance"] = ev.get("guidance")
                meta["sidon"] = ev.get("sidon")
                cands = ev.get("candidates") or []
            elif ev.get("type") == "end":
                meta.update({k: ev.get(k) for k in
                             ("audio_sec", "rtf", "loras", "align")})
    raw = b"".join(pcm)
    if not raw:
        return JSONResponse({"error": "no audio produced", "meta": meta},
                            status_code=500)
    sr = STATE["tts"].sr
    meta["total_ms"] = round((time.time() - t0) * 1000, 1)
    if cands:
        meta["candidates"] = [{k: c.get(k) for k in
                               ("rank", "reward", "gate", "wer", "extra_w",
                                "genuineness", "blend", "clap", "burst",
                                "n_burst", "sec")} for c in cands]
        # With best_of_audio the candidates carry their own audio too, encoded
        # the same way the winner is.  Without this a caller who wanted to hear
        # all ten got ten scores and one file, which is the least useful half.
        if body.get("best_of_audio") and want in ("json", "mp3", "wav"):
            # json is how a caller asks for ALL the candidates, and they want
            # files rather than raw samples; `candidate_format` overrides.
            enc = str(body.get("candidate_format")
                      or ("mp3" if want == "json" else want)).lower()
            for i, c in enumerate(cands):
                raw_c = c.get("pcm")
                if not raw_c:
                    continue
                pcm_c = _b64.b64decode(raw_c)
                if enc == "mp3":
                    data_c = await asyncio.get_running_loop().run_in_executor(
                        None, lambda p=pcm_c: _to_mp3(
                            p, sr, str(body.get("bitrate") or "192k")))
                elif enc == "wav":
                    data_c = (b"RIFF" + struct.pack("<I", 36 + len(pcm_c)) +
                              b"WAVEfmt " +
                              struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
                              + b"data" + struct.pack("<I", len(pcm_c)) + pcm_c)
                else:
                    data_c = pcm_c
                meta["candidates"][i]["audio"] = _b64.b64encode(data_c).decode()
                meta["candidates"][i]["audio_format"] = enc

    if want == "json":
        return {"sr": sr, "pcm": _b64.b64encode(raw).decode(), **meta}
    if want == "wav":
        data = (b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVEfmt " +
                struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) +
                b"data" + struct.pack("<I", len(raw)) + raw)
        media = "audio/wav"
    else:
        data = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _to_mp3(raw, sr, str(body.get("bitrate") or "192k")))
        media = "audio/mpeg"
    # the whole performance travels in the headers, so a curl user who only
    # wanted audio still gets to see what was done to produce it
    head = {"x-audio-sec": str(meta.get("audio_sec")),
            "x-language": str(meta.get("language")),
            "x-best-of": str(meta.get("best_of")),
            "x-guidance": str(meta.get("guidance")),
            "x-total-ms": str(meta.get("total_ms")),
            "x-script": json.dumps(meta.get("script") or "")[:3800],
            "x-reply": json.dumps(meta.get("reply") or "")[:1800]}
    return Response(content=data, media_type=media, headers=head)
