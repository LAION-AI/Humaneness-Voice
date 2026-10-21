"""Combine immutable pilot runs into one readable English technical report."""
import html
import json
from pathlib import Path

from .train import write_report


def main():
    root = Path(__file__).resolve().parent
    runs = []
    for folder,title in (("results","V1: uncentered architecture winner"),
                         ("results_centered","V1: centered-target control"),
                         ("results_v2_raw","V2: MLP, class-balanced activity"),
                         ("results_v2_centered","V2: MLP, centered + class-balanced"),
                         ("results_v2_gru","V2: GRU, centered + class-balanced")):
        file = root/folder/"results.json"
        if file.exists():
            report = json.loads(file.read_text())
            data_folder = "pilot" if folder in ("results","results_centered") else "pilot_v2"
            metadata_file = root/data_folder/"manifest.json"
            if metadata_file.exists():
                metadata = json.loads(metadata_file.read_text())
                report["dataset_limitations"] = ["Only 32 synthetic identities and 4 held-out test identities; no real-microphone robustness claim.",*metadata.get("limitations",[])]
                report["source_metadata"] = {k:metadata.get(k) for k in ("dataset_repo","dataset_revision","source_license","attribution","teacher_repo","teacher_revision","pilot_revision")}
            write_report(report,root/folder/"report.html")
            runs.append((folder,title,report))
    ood = []
    for name,title in (("ood_v1_v2.json","Frozen V1 uncentered → new emotions"),
                       ("ood_centered_v2.json","Frozen V1 centered → new emotions")):
        if (root/name).exists(): ood.append((title,json.loads((root/name).read_text())))
    def row(name,metrics,val=None):
        per = metrics.get("per_speaker",{})
        macro = sum(v["closed_set_accuracy"] for v in per.values())/max(len(per),1)
        return "<tr>"+"".join(f"<td>{html.escape(str(x))}</td>" for x in (
            name, f"{val*100:.1f}" if val is not None else "frozen",
            f"{metrics['eer']*100:.1f}",f"{metrics['positive_cosine']:.3f}",
            f"{macro*100:.1f}",f"{metrics['false_accept_rate']*100:.1f}",
            f"{metrics['false_reject_rate']*100:.1f}"))+"</tr>"
    rows = "".join(row(title,r["test"],r["selected"]["validation"]["eer"]) for _,title,r in runs)
    rows += "".join(row(title,r["test"]) for title,r in ood)
    oracle_rows = "".join(row(title+" — long clean ECAPA",r["long_clean_teacher_oracle"]["test"],
                        r["long_clean_teacher_oracle"]["validation"]["eer"])
                        for _,title,r in runs if r.get("long_clean_teacher_oracle",{}).get("available"))
    details = "".join(f"<li><a href='{folder}/report.html'>{html.escape(title)}: full architecture table and protocol</a></li>" for folder,title,_ in runs)
    context_table = ""
    if (root/"context_ablation_test.json").exists():
        cases = json.loads((root/"context_ablation_test.json").read_text())
        context_rows = "".join(f"<tr><td>{r['context_ms']}</td><td>{r['validation']['eer']*100:.1f}</td><td>{r['test']['eer']*100:.1f}</td><td>{r['stable_context_oracle'].get('eer',0)*100:.1f}</td></tr>" for r in cases)
        context_table = "<h2>Context-length diagnostic</h2><table><tr><th>Feature context ms</th><th>Validation EER %</th><th>Test EER %</th><th>Oracle stable-context EER %</th></tr>"+context_rows+"</table><p>These are the predefined V1 MLP-192 context ablations, not a test-set architecture selection. The selected model remains the validation winner. Stable-context diagnostics retain only contexts with at least90% of frames from the target speaker, using ground-truth labels; this barely changes error. Longer context alone does not solve the pilot's speaker-generalization failure.</p>"
    artifact = root/"artifacts"/"selection.json"
    selection = json.loads(artifact.read_text()) if artifact.exists() else {}
    scaling = ""
    manifest_path = root/"pilot_v2"/"manifest.json"
    if manifest_path.exists():
        records = json.loads(manifest_path.read_text())["conversations"]
        duration = sum(r.get("duration_s",0) for r in records)
        wall = sum(r.get("feature_seconds",0) for r in records)
        rtf = wall/max(duration,1)
        scaling = f"<p>Measured V2 feature calls processed {duration:.0f} seconds of audio in {wall:.1f} seconds: wall RTF {rtf:.3f}. Linear extrapolation to 100,000 × 24-second conversations is {100000*24*rtf/3600:.1f} hours just for feature calls, excluding waveform synthesis, downloads and teacher work. This includes local HTTP/serialization and is not isolated on-device ASR timing. Full-frame float16 caching would occupy about 61.4 GB; 500,000 independent 25-frame contexts alone would occupy 25.6 GB before metadata/caches.</p>"
    (root/"report.html").write_text("""<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width, initial-scale=1'><title>TYGER V2 — measured speaker-identity pilot</title><style>body{max-width:1150px;margin:3rem auto;padding:0 1rem;background:#faf8f3;color:#26333c;font:17px/1.6 system-ui}table{border-collapse:collapse;font-size:14px;width:100%}td,th{padding:.6rem;border:1px solid #ccc;text-align:left}pre{white-space:pre-wrap;background:#eee;padding:1rem}.notice{background:#fff0d7;border-left:5px solid #b67628;padding:1rem}.scroll{overflow:auto}a{color:#205e87}</style><nav><a href='/'>← Back to Robby</a></nav><h1>TYGER V2: audio synchronization and speaker identity are separate problems</h1>
<p class=notice><strong>Result: keep the experimental student identity gate off by default.</strong> The lightweight predictors are very fast, and the centered-target control improves speaker discrimination, but this small held-out pilot does not establish reliable multi-user filtering. The original TYGER turn-taking GRU remains unchanged. A direct ECAPA teacher is a separate baseline, not proof that the student works.</p>
<h2>Actual held-out results</h2><p>Fifteen actual CPU training runs were completed: eleven initial architecture/size/context/loss comparisons, one centered-target control, and three broader-emotion class-balanced runs. The linked per-run reports include all eleven initial comparisons. All identity splits are disjoint. Architecture selection and operating thresholds use validation identities only. EER below is evaluated on unseen test identities; lower is better. The operating point targets approximately 5% validation impostor false accepts and is then frozen for testing. Adjacent windows are correlated. Macro accuracy averages speakers equally.</p><div class=scroll><table><tr><th>Run</th><th>Validation EER %</th><th>Test EER % ↓</th><th>Teacher cosine ↑</th><th>Macro identity % ↑</th><th>False accept % ↓</th><th>False reject % ↓</th></tr>"""+rows+"</table></div><h2>Long clean teacher controls</h2><p>These controls use full clean source utterances and separate long enrollment clips, unlike the short noisy student windows. They quantify teacher/voice-profile consistency, not equal-context speed or accuracy.</p><div class=scroll><table><tr><th>Corpus</th><th>Validation EER %</th><th>Test EER %</th><th>Teacher cosine</th><th>Macro identity %</th><th>False accept %</th><th>False reject %</th></tr>"+oracle_rows+"""</table></div>
"""+context_table+"""<h2>Method and limitations</h2><p>Frozen Nemotron ASR 3.5 layer 15 produces 1024-dimensional frames every 80 ms. An MLP, small GRU, or causal temporal CNN predicts normalized 192D SpeechBrain ECAPA embeddings, plus a three-way speech/silence/ambiguous-overlap head. Every 320 ms supplies four new frames. Models span 148k–470k parameters; contexts are 320, 640, 960, and 2000 ms. Predictor training and latency measurements use CPU only; the TTS GPU is untouched.</p>
<p>The pilot has 32 synthetic identities: 24 train, 4 validation, 4 test, 64 procedural 24-second conversations and 4,800 decision hops per corpus. Two clean long enrollment utterances per voice are excluded from conversation source clips. V1's bounded extraction found only fatigue/exhaustion and low-background-noise conditions, so it cannot establish emotional robustness. V2 samples anger, amusement, elation, sadness and background-noise-conditioned speech through bounded remote tar ranges; its probe utterances are new while identity splits and enrollment targets remain fixed. V2's 383 source clips, including enrollment sources, are German-heavy (338 German, 45 English), so this is not balanced English-emotion validation. Room noise, reverb, telephone filtering, interruptions, overlaps and synchronized synthetic playback leakage are procedural, not recordings of real users.</p>
<p>Standard ECAPA embeddings have a shared mean component. Average cosine can therefore look deceptively high even for a constant-vector predictor. The centered control fits a mean and residual radius from training identities only, regresses residual directions, and reconstructs standard ECAPA outputs. Its lower absolute cosine can coexist with better verification. Normalized MSE is a rescaled cosine loss; the loss ablation changes regression/activity weighting, not independent geometry. Class balancing addresses rare overlap labels but does not guarantee overlap rejection.</p>
<p>Temporal accumulation analyses use ground-truth speaker resets and are optimistic; they did not rescue V1 verification. Unknown rejection must be read alongside known acceptance: rejecting almost everyone is not success. Word-alignment occupancy may label untranscribed vocal bursts as silence. Four held-out voices cannot establish performance for real microphones, ages, accents, or unrelated background speakers. Enrollment embeddings are approximate cues, never secure authentication.</p>
<h2>Scaling decision</h2><p>Do not begin a 100,000-conversation/500,000-window run merely because the predictor is fast. V1 measured roughly 2,300–5,200 training windows/s on CPU, around 2–4 minutes per 500,000-window epoch before feature I/O. Actual frozen-ASR feature extraction dominates end-to-end corpus cost; multiply measured feature throughput, not student throughput, when budgeting. Caching every frame of 100,000 24-second conversations would exceed a 40 GB working budget, so a larger run must selectively store supervised windows or use an iterable/procedural feature pipeline. A broader intermediate pilot and stronger validation are the next justified steps.</p>
"""+scaling+"<h2>Reproducibility and files</h2><ul>"+details+"</ul><p>Run <code>python -m unittest speaker_v2.test_predictor -v</code>; use <code>python -m speaker_v2.train --help</code> and the data preparation README for exact commands and source metadata. Frozen checkpoint out-of-domain tests use <code>python -m speaker_v2.evaluate</code> without threshold adaptation. No private microphone recordings or enrolled user voiceprints are included.</p><details><summary>Current integration selection</summary><pre>"+html.escape(json.dumps(selection,indent=2))+"</pre></details></html>")
    print(root/"report.html")


if __name__ == "__main__": main()
