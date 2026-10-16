"""Controlled CFG x temperature pilot, resumable generation and scoring.

All generated takes are retained: no best-of-N filtering of study outcomes.
The design and selection rule are saved before the first quality evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import soundfile as sf

import settings
from prompting import build

ROOT = settings.HERE / "study"
CFG = [1., 1.5, 2., 3.]
TEMPERATURES = [0., .5, 1., 1.2]
SEEDS = [777, 2027]
EMOTIONS = {
    "Anger": "Intense contained anger at a betrayal. Tight jaw, clipped attacks, hard consonants, pressure under every word; a believable confrontation, not a neutral read or a scream.",
    "Fear": "Strong intimate fear on seeing an unexpected person. Breath held, trembling voice, fragile starts and urgent restrained panic; stay intelligible, do not scream.",
    "Sadness": "Deep genuine sadness and grief on a reunion. Heavy slow breath, a fragile falling voice, a small catch in the throat; sound lived-in rather than theatrically recited.",
    "Elation": "Overflowing genuine joy at an unexpected reunion. Bright ringing voice, smiling energetic attacks and spontaneous rising excitement, with clear words; not flat cheerful narration.",
    "Affection": "Deep warm affection in a private reunion. Close gentle voice, soft smiling breath, vulnerable warmth and tender unhurried phrasing; sincere and natural, not detached narration.",
}
TEXT = {
    "main": "You came back. I thought I would never see you again.",
    "german": "Du bist wieder da. Ich dachte, ich würde dich nie wiedersehen.",
    "reference_b": "You came back. I thought I would never see you again.",
    "short_cues": "You came back. I thought I would never see you again.",
    "legacy": "You came back. I thought I would never see you again.",
}
CUES = {"Anger": "barely contained anger, tight jaw, clipped attacks", "Fear": "fearful, breath held, trembling, urgent",
    "Sadness": "deeply sad, fragile voice, falling cadence", "Elation": "elated, bright smiling voice, spontaneous excitement",
    "Affection": "deeply affectionate, tender warmth, intimate"}


def write_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def spec_for(emotion, fmt, stage):
    text, direction = TEXT[stage], EMOTIONS[emotion]
    first, rest = text.split(". ", 1)
    script = f"({direction}) {first}. [0.35 seconds pause] {rest}"
    if stage == "short_cues":
        # Follow the model card's terse, sentence-level structured recipe.
        script = f"({CUES[emotion]}) [1.4 seconds duration] {first}.\n[0.35 seconds pause]\n({CUES[emotion]}) [3.3 seconds duration] {rest}"
    return build({"text": text, "general": direction, "caption": "Voice and performance: " + direction,
                  "script": script}, fmt, "de" if stage == "german" else "en", seconds=5.5)


def create_design():
    ROOT.mkdir(exist_ok=True)
    (ROOT / "audio").mkdir(exist_ok=True)
    (ROOT / "raw").mkdir(exist_ok=True)
    path = ROOT / "design.json"
    if path.exists():
        return json.loads(path.read_text())
    configurations = [(g, t) for g in CFG for t in TEMPERATURES]
    random.Random(9403).shuffle(configurations)
    rows = []
    for g, t in configurations:
        for seed in SEEDS:
            for emotion in EMOTIONS:
                for fmt in ["caption", "structured"]:
                    rows.append({"id": f"main-g{g:g}-t{t:g}-s{seed}-{emotion}-{fmt}",
                        "stage": "main", "reference": "A", "guidance": g, "temperature": t,
                        "seed": seed, "emotion": emotion, "format": fmt, "spec": spec_for(emotion, fmt, "main")})
    design = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": settings.REPO, "revision": settings.REVISION, "stage": settings.STAGE,
        "cfg_grid": CFG, "temperature_grid": TEMPERATURES, "seeds": SEEDS,
        "emotions": EMOTIONS, "texts": TEXT, "main_samples": len(rows), "rows": rows,
        "codec_dtype": "bf16 weights/compute; quantizer fp32", "tts_dtype": "bf16",
        "stop_temperature": "0 when audio T=0, otherwise fixed 1; stop logits never CFG-guided",
        "sampling": {"audio_top_k": 50, "audio_top_p": .95, "repetition_penalty": 1.},
        "generation": "Natural stop/continue, no forced minimum length; 5.5-second hint plus 60-frame safety budget. Batch size 4. No Luna, no LoRA, all takes retained.",
        "selection_rule": "Within each prompt format: eligible if mean RAW conventional WER <= 0.10 and mean processed genuineness >= same-format CFG=1.5,T=1 baseline minus 0.25. Maximize mean target-emotion rank percentile; ties within 0.01 prefer higher direction CLAP. If no eligible configuration, explicitly report no qualifying optimum and present Pareto trade-offs.",
        "uncertainty": "Paired bootstrap over 10 emotion-format blocks, averaging the two seeds within each block. Only five emotions and one English utterance: intervals are descriptive, not population guarantees. T=0 seeds are duplicates, not independent replications.",
        "holdout_plan": "After the main grid: baseline plus up to three shortlisted settings, same five emotions/formats/two seeds, separately German utterance with reference A and English utterance with reference B. This is a selected-setting generalization check, NOT a full independent grid search.",
        "metrics": {"primary_wer": "Unicode-aware Levenshtein word error rate on raw PCM using Parakeet v3; errors may exceed 100% with insertions.",
            "post_wer": "Same ASR/WER after production forced alignment and fades; secondary, to expose tail trimming effects.",
            "naturalness": "VoiceCLAP-commercial Genuineness full head 0-6 on processed PCM; proxy, not listener MOS.",
            "emotion": "Empathic-Insight-Voice-Small 40 heads, documented ReLU and EmoWhisper-AnS-Small-v0.1 encoder; target rank percentile (40-rank)/39 and raw target score, neither calibrated probability.",
            "direction": "Centered VoiceCLAP-commercial cosine against exactly the requested acting direction; separate correlated proxy.",
            "blend": "Existing 0-10 vocal-burst blend head as diagnostic only; these prompts contain no explicit bursts."}}
    write_json(path, design)
    return design


def reference_path(label):
    if label == "A":
        return settings.HERE / "performance/benchmark-reference.wav"
    path = ROOT / "reference-b.wav"
    if not path.exists():
        source = sorted(Path("/mnt/nvme/moss-15-v2-assets/refs3/audio/emolia_c1682").glob("*__en.*.mp3"))[0]
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
            "-af", "silenceremove=start_periods=1:start_duration=0.02:start_threshold=-45dB",
            "-t", "2.88", "-ac", "1", "-ar", "48000", str(path)], check=True)
    return path


def generate(stage="main"):
    from engine import SmallEngine
    import torch
    design = create_design()
    requested_stage = stage
    rows = [r for r in design["rows"] if r["stage"] == stage or stage == "all"]
    engine = SmallEngine()
    # Explicit device selection ensures inference RNG/graph capture concerns
    # GPU 0 even when the independent scorer process uses GPU 1.
    torch.cuda.set_device(engine.device)
    plan = {}
    for row in rows:
        plan.setdefault((row["stage"], row["guidance"], row["temperature"], row["seed"], row["reference"]), []).append(row)
    for (group_stage, g, t, seed, reference), group in plan.items():
        if all((ROOT / "raw" / (r["id"] + ".json")).exists() for r in group):
            continue
        if (group_stage == "legacy") != (not settings.FAST_INFERENCE):
            raise RuntimeError("Use SMALL_FAST_INFERENCE=0 SMALL_COMPILE_GLOBAL=0 SMALL_BATCH_SIZE=4 only for --stage legacy; other stages require optimized inference.")
        started = time.monotonic()
        try:
            waves = engine.generate_many([r["spec"] for r in group], g, seed,
                reference_path(reference), t, batch_size=4)
            for i, (row, wave, codes) in enumerate(zip(group, waves, engine.last_codes)):
                sf.write(ROOT / "raw" / (row["id"] + ".wav"), wave, 48000, subtype="FLOAT")
                record = {**row, "raw_seconds": len(wave) / 48000,
                    "raw_pcm_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
                    "codes_sha256": hashlib.sha256(codes.numpy().tobytes()).hexdigest(),
                    "frames": len(codes), "hit_frame_limit": len(codes) >= min(settings.MAX_FRAMES+60, row["spec"]["frames"]+60),
                    "effective_batch_seed": seed + (i // 4) * 4,
                    "batch_row": i % 4,
                    "performance": engine.last_performance}
                write_json(ROOT / "raw" / (row["id"] + ".json"), record)
        except Exception as exc:
            # Record a failed condition; never silently exclude it as a good
            # generation or relabel a partial run complete.
            write_json(ROOT / f"generation-error-{group_stage}-g{g}-t{t}-s{seed}.json",
                {"error": str(exc), "ids": [r["id"] for r in group]})
            raise
        print(json.dumps({"stage": group_stage, "cfg": g, "temperature": t, "seed": seed,
                          "samples": len(group), "seconds": time.monotonic()-started}), flush=True)
    write_json(ROOT / f"generation-complete-{requested_stage}.json", {"stage": requested_stage, "samples": len(rows)})


def score(stage="main", follow=False):
    import torch
    import torchaudio
    import config
    from huggingface_hub import snapshot_download
    from engine import Scorers
    from alignment import AudioFinisher
    from metrics import word_errors
    from score_engine import VoiceScorer
    from small_app import encode_mp3

    class EmotionOnly(VoiceScorer):
        def __init__(self):
            self.device = settings.SCORE_DEVICE
            self._load_emotion()

        @torch.inference_mode()
        def evaluate(self, wave):
            embed = self._whisper_embed(wave)
            values = {name: float(head(embed).squeeze()) for name, head in self.emo.items()}
            attrs = {name: float(head(embed).squeeze()) for name, head in self.emo_attr.items()}
            return values, attrs

    design = create_design()
    torch.cuda.set_device(settings.SCORE_DEVICE)
    # Match the public emotion model's explicit encoder contract, rather than
    # silently reusing the older server's generic BUD-E Whisper backbone.
    config.WHISPER_DIR = snapshot_download("mkrausio/EmoWhisper-AnS-Small-v0.1",
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt"])
    scorers, finisher = Scorers(), AudioFinisher()
    emotion = EmotionOnly()
    metadata = {"encoder_path": config.WHISPER_DIR, "activation": "relu",
        "emotion_heads": len(emotion.emo), "asr": config.ASR_MODEL,
        "torch": torch.__version__, "scorer_device": torch.cuda.get_device_name(1),
        "genuineness": "full", "processed_audio_scored_before_mp3": True}
    write_json(ROOT / "scorer-environment.json", metadata)
    while True:
        design = create_design()
        rows = [r for r in design["rows"] if r["stage"] == stage or stage == "all"]
        remaining, completed = 0, 0
        for row in rows:
            output = ROOT / (row["id"] + ".json")
            if output.exists():
                completed += 1
                continue
            raw_path = ROOT / "raw" / (row["id"] + ".wav")
            raw_meta = raw_path.with_suffix(".json")
            if not raw_meta.exists():
                remaining += 1
                continue
            raw, sr = sf.read(raw_path, dtype="float32")
            raw16 = torchaudio.functional.resample(torch.from_numpy(raw), sr, 16000).numpy()
            raw_hyp = scorers.asr.transcribe(raw16)
            edited, alignments = finisher.finish([raw], row["spec"], True)
            wave = edited[0]
            judged = scorers.score([wave], row["spec"])[0]
            # Study WER uses anti-aliased resampling, consistently for raw and
            # processed recordings. The demo's old scorer uses box averaging.
            processed16 = torchaudio.functional.resample(torch.from_numpy(wave), 48000, 16000).numpy()
            post_hyp = scorers.asr.transcribe(processed16)
            emotions, attrs = emotion.evaluate(processed16)
            ordered = sorted(emotions, key=lambda name: -emotions[name])
            rank = ordered.index(row["emotion"]) + 1
            filename = row["id"] + ".mp3"
            encode_mp3(wave, ROOT / "audio" / filename)
            encode_mp3(raw, ROOT / "audio" / (row["id"] + "-raw.mp3"))
            result = {**json.loads(raw_meta.read_text()),
                "raw_hypothesis": raw_hyp, "raw_word_errors": word_errors(row["spec"]["text"], raw_hyp),
                "processed_hypothesis": post_hyp, "processed_word_errors": word_errors(row["spec"]["text"], post_hyp),
                "genuineness": judged["genuineness"], "blend": judged["blend"], "clap": judged["clap"],
                "emotion_rank": rank, "emotion_percentile": (40-rank)/39,
                "target_emotion_score": emotions[row["emotion"]], "emotions_all": emotions, "attributes": attrs,
                "alignment": alignments[0], "audio": "audio/" + filename,
                "raw_audio": "audio/" + row["id"] + "-raw.mp3"}
            assert all(np.isfinite(x) for x in [judged["genuineness"], judged["clap"], *emotions.values()])
            write_json(output, result)
            completed += 1
            print(json.dumps({"completed": completed, "total": len(rows), "id": row["id"],
                "raw_wer": result["raw_word_errors"]["wer"], "post_wer": result["processed_word_errors"]["wer"],
                "genuineness": result["genuineness"], "emotion_rank": rank}), flush=True)
        if not remaining:
            break
        if not follow:
            raise RuntimeError(f"{remaining} generations not available yet")
        time.sleep(3)
    write_json(ROOT / f"scoring-complete-{stage}.json", {"stage": stage, "samples": len(rows)})


def add_holdout(configurations):
    design = create_design()
    design["holdout_added_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    design["holdout_selected_configurations"] = configurations
    for stage in ["german", "reference_b"]:
        for g, t in configurations:
            for seed in SEEDS:
                for emotion in EMOTIONS:
                    for fmt in ["caption", "structured"]:
                        id_ = f"{stage}-g{g:g}-t{t:g}-s{seed}-{emotion}-{fmt}"
                        if any(r["id"] == id_ for r in design["rows"]):
                            continue
                        design["rows"].append({"id": id_, "stage": stage, "reference": "B" if stage == "reference_b" else "A",
                            "guidance": g, "temperature": t, "seed": seed, "emotion": emotion, "format": fmt,
                            "spec": spec_for(emotion, fmt, stage)})
    write_json(ROOT / "design.json", design)


def add_diagnostics():
    design = create_design()
    design["diagnostics_added_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for stage, configs, formats in [("short_cues", [(g,t) for g in CFG for t in TEMPERATURES], ["structured"]),
                                    ("legacy", [(1.5,1.)], ["caption","structured"])]:
        for g,t in configs:
            for seed in SEEDS:
                for emotion in EMOTIONS:
                    for fmt in formats:
                        id_ = f"{stage}-g{g:g}-t{t:g}-s{seed}-{emotion}-{fmt}"
                        if any(r["id"] == id_ for r in design["rows"]):
                            continue
                        design["rows"].append({"id":id_,"stage":stage,"reference":"A","guidance":g,"temperature":t,
                            "seed":seed,"emotion":emotion,"format":fmt,"spec":spec_for(emotion,fmt,stage)})
    design["diagnostic_plan"] = "Added after observing main-grid failure: 160 takes with terse per-sentence cues and explicit sentence durations, same full global direction/text/reference/seeds. Exploratory recipe rescue, not an untouched holdout; multiple recipe details change. Also 20 original-backend baseline takes (CFG1.5,T1) for numerical-optimization quality diagnostics, not a powered equivalence trial."
    write_json(ROOT / "design.json", design)


def audit_references():
    design=create_design()
    source_a=next(Path("/mnt/nvme/moss-15-v2-assets/refs3/audio/emolia_c1699").rglob("*.mp3"))
    source_b=sorted(Path("/mnt/nvme/moss-15-v2-assets/refs3/audio/emolia_c1682").glob("*__en.*.mp3"))[0]
    design['reference_provenance']={
        'A':{'source_file':source_a.name,'source_sha256':hashlib.sha256(source_a.read_bytes()).hexdigest(),
             'prepared_wav_sha256':hashlib.sha256(reference_path('A').read_bytes()).hexdigest(),
             'note':'Positive-valence-labelled corpus speech. Not a neutral-speaker control.'},
        'B':{'source_file':source_b.name,'source_sha256':hashlib.sha256(source_b.read_bytes()).hexdigest(),
             'prepared_wav_sha256':hashlib.sha256(reference_path('B').read_bytes()).hexdigest(),
             'note':'Hiss-labelled corpus take containing speech. Possible burst/timbre contamination; this is a reference-sensitivity stress check, not a neutral clean-reference replication.'},
        'preparation':'Leading-silence removal at -45 dB, first 2.88 s, mono 48 kHz. Neither clip contains the target utterance. No reference audio is served publicly.'}
    write_json(ROOT/'design.json',design)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["design", "generate", "score", "holdout", "diagnostics", "audit"])
    parser.add_argument("--stage", default="main", choices=["main", "german", "reference_b", "short_cues", "legacy", "all"])
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--configs", default="1.5:1,1:0.5,1.5:0.5,2:0.5")
    args = parser.parse_args()
    if args.phase == "design":
        print(create_design()["main_samples"])
    elif args.phase == "generate":
        generate(args.stage)
    elif args.phase == "score":
        score(args.stage, args.follow)
    elif args.phase == "diagnostics":
        add_diagnostics()
    elif args.phase == "audit":
        audit_references()
    else:
        add_holdout([tuple(map(float, c.split(":"))) for c in args.configs.split(",")])
