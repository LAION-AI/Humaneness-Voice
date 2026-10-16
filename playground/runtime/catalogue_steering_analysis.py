"""Prospective per-reference layer-16 steering protocol and CPU-only analysis.

Ten catalogue labels are ten fixed recording profiles, not ten independently
verified human identities. ECAPA cosine is not a percentage or probability.
"""
from __future__ import annotations

import base64
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path

import numpy as np

import mira_activation_analysis as mira

POINT = "layer_16"
ALPHA = .75
QUALITY_GATES = {"content_enjoyment_delta_min":-.30, "production_quality_delta_min":-.30,
    "mos_delta_min":-.20, "wer_delta_max":.03, "wer_mean_max":.15,
    "bad_wer_rate_delta_max":.10, "ecapa_delta_min":.01}
VALIDATION_TEXTS = {
    "en": ["I will leave the envelope at reception. You can collect it whenever you arrive.",
           "We finally have a date for the celebration. I hope you can join us that evening.",
           "Please stop interrupting me. I need you to listen before making another promise.",
           "I kept the cup you gave me. It reminds me of the mornings we spent together.",
           "Something moved near the stairs. Stay here while I check what is happening."],
    "de": ["Ich lasse den Umschlag am Empfang. Du kannst ihn abholen, sobald du ankommst.",
           "Der Termin für die Feier steht endlich fest. Hoffentlich kannst du an diesem Abend dabei sein.",
           "Bitte unterbrich mich nicht ständig. Hör erst zu, bevor du wieder etwas versprichst.",
           "Ich habe deine Tasse behalten. Sie erinnert mich an unsere gemeinsamen Morgen.",
           "Neben der Treppe hat sich etwas bewegt. Bleib hier, während ich nachsehe."]}


def design(manifest, collection_n=100, reuse_mira=True):
    """No clock-dependent fields: the caller freezes this before generation."""
    if collection_n not in (50,100): raise ValueError("Collection must be 50 or 100 per voice")
    candidates = manifest["candidates"] if isinstance(manifest,dict) else manifest
    if len(candidates)!=10 or len({v["voice"] for v in candidates})!=10:
        raise ValueError("Expected ten distinct catalogue profiles")
    voices, collection, validation = [], [], []
    for index, item in enumerate(candidates):
        voice = item["voice"]
        if len(item["sha256"])!=64: raise ValueError("Reference SHA256 must be pinned")
        reuse = reuse_mira and voice=="emolia_c1699"
        voices.append({"voice":voice,"name":item["name"],"reference_sha256":item["sha256"],
            "reference_seconds":item["seconds"],"reference_file":item["wav"],
            "reuse_existing_mira":reuse,"collection_n":100 if reuse else collection_n})
        if reuse: continue
        for block in mira.blocks("collection",collection_n):
            collection.append({**block,"voice":voice,"voice_name":item["name"],
                "phase":"catalogue_collection","condition":"baseline",
                "id":f'catalogue_collection-{voice}-{block["block"]}',
                "seed":11000000+index*100000+block["block"]*1009,
                "reference_sha256":item["sha256"]})
        for i, emotion in enumerate(mira.EMOTIONS):
            for j, language in enumerate(("en","de")):
                text = VALIDATION_TEXTS[language][i]
                block = 2*i+j
                prompt = 'CAPTION: '+mira.IDENTITY+' '+mira.EMOTIONS[emotion]+'\nTRANSCRIPT: '+json.dumps(text,ensure_ascii=False)
                for condition in ("baseline","steered"):
                    validation.append({"voice":voice,"voice_name":item["name"],"phase":"catalogue_validation",
                        "condition":condition,"id":f'catalogue_validation-{voice}-{condition}-{block}',
                        "block":block,"seed":21000000+index*100000+block*1009,
                        "emotion":emotion,"language":language,"text":text,"prompt":prompt,
                        "point":POINT,"alpha":0. if condition=="baseline" else ALPHA,
                        "reference_sha256":item["sha256"]})
    return {"version":1,"voices":voices,"collection_n":collection_n,"point":POINT,"alpha":ALPHA,
        "collection_rows":collection,"validation_rows":validation,
        "sampling":{"cfg":2.,"temperature":1.,"duration":"automatic","adapters":[],"fixed_reference":True},
        "selection_rule":"Strict >.50 high vs <=.50 low if both pools contain >=2 samples per each of five emotions; else strict >.25 vs <=.25 with same coverage. Balance ten emotion/language cells only when every high and low cell contains >=2 samples; otherwise equal-emotion weighting with disclosed language confounding. If neither threshold qualifies, top/bottom two samples in each emotion/language cell ranked by ECAPA, with deterministic ID ties, disjoint pools and explicit exploratory rank-fallback metadata. If a cell has <4 valid samples, no vector is fabricated.",
        "capture":"Generated acoustic-frame prediction states only; last conditional prefill state included iff it predicts frame0. Exclude unused prompt/reference positions, EOS and padded/finished tails. Each sample mean gets equal weight within balancing cells.",
        "validation":"Ten fresh mixed-task matched baseline/steered pairs per new voice. Mira reuses the independently validated original vector and 20+20 held-out takes unchanged. No quality retries, fades, alignment trimming or best-of-N selection.",
        "quality_gates":QUALITY_GATES,
        "publication_policy":"All valid derived vectors remain experimentally selectable, including quality-failed vectors with prominent warnings. The user's default-on request is not a claim that every voice improved. Statistical eligibility/recommendations are separate from actual activation policy. Hard bypass for unsupported custom/no-reference, adapter, reference-hash or base-checkpoint mismatches is implemented by the runtime owner.",
        "limitations":["One fixed reference recording per profile; a per-reference vector is not a general identity embedding.",
            "Ten validation pairs are a small proxy pilot, not ten repetitions per emotion or human listening evidence.",
            "Batch tensor rows share RNG; matched grouping and actual batch seeds must be recorded.",
            "Threshold/rank selection can select phonetic, linguistic, content or generic activation directions.",
            "Raw alpha .75 is transferred from Mira; no per-voice alpha optimization is claimed.",
            "50 samples give five per language/emotion cell and fragile threshold coverage; 100 gives ten and is preferred.",
            "No ECAPA-to-steering MLP is trained or validated here."]}


def _valid(row):
    return not row.get("failed") and row.get("status")!="failed" and mira._metric(row,"ecapa") is not None


def paired_design(manifest, repeats=10):
    """Pure-emotion reference/no-reference experiment, separate from quality filtering.

    Each repeat batches different conditions 4/4/2. Thus each identical
    condition appears in ten distinct actually seeded batches, not merely ten
    tensor rows from three seeds. The same schedule is used in both arms.
    """
    if repeats!=10: raise ValueError("Prospective paired study uses ten actual seed repetitions")
    candidates = manifest["candidates"] if isinstance(manifest,dict) else manifest
    voices = [{"voice":v["voice"],"name":v["name"],"reference_sha256":v["sha256"],
               "reference_file":v["wav"],"reference_seconds":v["seconds"]} for v in candidates]
    with_rows, without_rows = [],[]
    for repeat in range(repeats):
        for i,emotion in enumerate(mira.EMOTIONS):
            for j,language in enumerate(("en","de")):
                offset = 2*i+j
                text = mira.TEXTS["collection"][language][i]
                prompt = 'CAPTION: '+mira.EMOTIONS[emotion]+'\nTRANSCRIPT: '+json.dumps(text,ensure_ascii=False)
                seed = 31000000+repeat*1009+(offset//4)*100000
                shared = {"phase":"paired_collection","block":repeat*10+offset,"repeat":repeat,
                    "condition_id":emotion+'-'+language,"emotion":emotion,"language":language,"text":text,
                    "prompt":prompt,"reference_identity":False,"seed":seed,"actual_batch_seed":seed,
                    "batch_condition_order":[e+'-'+l for e in mira.EMOTIONS for l in ("en","de")][(offset//4)*4:(offset//4)*4+4]}
                without_rows.append({**shared,"id":f'paired-without-{emotion}-{language}-{repeat}',
                    "voice":"none","condition":"reference_absent","reference_sha256":None})
                for v in voices:
                    with_rows.append({**shared,"id":f'paired-with-{v["voice"]}-{emotion}-{language}-{repeat}',
                        "voice":v["voice"],"condition":"reference_present","reference_sha256":v["reference_sha256"]})
    return {"version":1,"voices":voices,"with_reference_rows":with_rows,"without_reference_rows":without_rows,
        "repeats_per_identical_condition":10,"conditions":10,"capture_points":[POINT,"semantic_global","semantic_local"],
        "caption_rule":"Exact same pure-emotion CAPTION and transcript in both arms; no identity/reference-speaker/timbre/gender/age descriptors. reference_identity=False.",
        "batching_rule":"For each repeat, ten different condition_id values in stable 4/4/2 batches. Ten different actual batch seeds per identical condition; same batch seed, condition order and companions in reference-present and common reference-absent arm.",
        "contrast_rule":"No high-quality or ECAPA outcome filtering. For each fixed text/emotion/language condition, average paired with-minus-without per-sample generated-position means across ten seeds; then equally average ten conditions. Save arm means, contrasts, pair IDs and prediction counts.",
        "interpretation":"This is the total reference-conditioning effect on generated trajectories, not an isolated speaker/identity direction. It can contain reference language/recording effects, different generated tokens, duration and prosody. Common no-reference audio may be reused because prompt/batch seed/grouping are identical, never because of a chosen speaker.",
        "limits":"Raw no-reference support is itself tested, not presumed reliable. Missing/failed pairs prevent a silently reweighted contrast; no quality retries or substitution of a threshold-selected vector."}


def paired_reference_vector(with_rows, without_rows, arrays):
    """Unfiltered equal-condition paired contrast; return vectors plus metadata.

    arrays maps ID -> {layer_16, semantic_global, semantic_local} CPU means.
    No ECAPA, quality, emotion-recognition or WER score enters this estimator.
    """
    def pair_key(row):
        repeat = row.get("repeat", row.get("seed"))
        if repeat is None: raise ValueError("Pair needs repeat or seed")
        return (row["emotion"],row["language"],row["text"],repeat)
    def mapping(rows):
        result = {}
        for row in rows:
            key = pair_key(row)
            if key in result: raise ValueError("Duplicate reference pair")
            result[key] = row
        return result
    with_map,without_map = mapping(with_rows),mapping(without_rows)
    if set(with_map)!=set(without_map): raise ValueError("Reference arms have unmatched pairs")
    dims = {POINT:1024,"semantic_global":1024,"semantic_local":2560}
    groups = defaultdict(list)
    pair_ids,counts = [],[]
    for key,row in sorted(with_map.items()):
        control = without_map[key]
        for field in ("prompt","seed"):
            if row.get(field)!=control.get(field): raise ValueError("Reference arms differ in "+field)
        if "batch_seed" in row and row.get("batch_seed")!=control.get("batch_seed"):
            raise ValueError("Reference arms differ in actual batch seed")
        if row.get("batch_ids") and control.get("batch_ids"):
            if [x.rsplit('-',1)[-1] for x in row['batch_ids']]!=[x.rsplit('-',1)[-1] for x in control['batch_ids']]:
                raise ValueError("Reference arms differ in batch companion order")
        if row.get("reference_identity") is not False or control.get("reference_identity") is not False:
            raise ValueError("Reference identity prompting must be explicitly disabled")
        for candidate in (row,control):
            if candidate.get("failed") or candidate.get("status")=="failed": raise ValueError("Failed reference pair retained; no contrast silently derived")
            if candidate["id"] not in arrays: raise ValueError("Missing activation capture")
            if candidate.get("generated_frames",1)<=0: raise ValueError("Empty generated-state capture")
        diffs = {}
        for point,dim in dims.items():
            a,b = np.asarray(arrays[row["id"]][point],dtype=np.float64),np.asarray(arrays[control["id"]][point],dtype=np.float64)
            if a.shape!=(dim,) or b.shape!=(dim,) or not(np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                raise ValueError("Invalid paired capture at "+point)
            diffs[point] = {"with":a,"without":b,"difference":a-b}
        groups[(row["emotion"],row["language"],row["text"])].append(diffs)
        pair_ids.append({"with":row["id"],"without":control["id"],"repeat":row.get("repeat"),"seed":row.get("seed")})
        counts.append({"with":row.get("generated_frames"),"without":control.get("generated_frames")})
    expected = {(e,l) for e in mira.EMOTIONS for l in ("en","de")}
    if {(e,l) for e,l,_ in groups}!=expected: raise ValueError("Paired design lacks emotion/language coverage")
    vectors,arm_means = {},{}
    for point in dims:
        arms = {arm:np.mean([np.mean([d[point][arm] for d in values],axis=0)
                            for values in groups.values()],axis=0).astype(np.float32)
                for arm in ("with","without","difference")}
        vectors[point] = arms["difference"]
        arm_means[point] = {"with":arms["with"],"without":arms["without"]}
    return {"vectors":vectors,"arm_means":arm_means,"metadata":{
        "method":"unfiltered_paired_reference_effect","quality_selection":False,"paired_n":len(pair_ids),
        "condition_count":len(groups),"repeats_per_condition":{e+'/'+l:len(values) for (e,l,_),values in groups.items()},
        "weighting":"equal paired sample within identical text/emotion/language condition, equal conditions",
        "pair_ids":pair_ids,"generated_frame_counts":counts,
        "points":{p:{"dimension":dims[p],"difference_l2":float(np.linalg.norm(v)),
                     "difference_rms":float(np.sqrt(np.mean(v*v)))} for p,v in vectors.items()},
        "interpretation":"Total conditioning effect, including changed generated token/duration trajectories; not a pure speaker vector. No similarity-based selection."}}


def select_vector(rows, arrays, point=POINT):
    """Return (FP32 vector, JSON metadata), never silently lower a threshold.

    Arrays map sample IDs to either a [1024] mean or a dict with layer_16 mean.
    The primary Mira vector must be reused exactly, not recomputed by this API.
    """
    values = {}
    for row in rows:
        if not _valid(row) or row["id"] not in arrays: continue
        source = arrays[row["id"]]
        vector = np.asarray(source[point] if isinstance(source,dict) else source,dtype=np.float32)
        if vector.shape!=(1024,) or not np.all(np.isfinite(vector)):
            raise ValueError("Capture must be a finite 1024D layer-16 sample mean")
        values[row["id"]] = vector
    valid = [r for r in rows if r["id"] in values]
    expected_emotions = set(mira.EMOTIONS)
    expected_cells = {(e,l) for e in expected_emotions for l in ("en","de")}
    pools = []
    high = low = None
    threshold = None
    method = "threshold_high_minus_low"
    for t in (.50,.25):
        h = [r for r in valid if mira._metric(r,"ecapa")>t]
        l = [r for r in valid if mira._metric(r,"ecapa")<=t]
        hc,lc = Counter(r["emotion"] for r in h),Counter(r["emotion"] for r in l)
        eligible = len(h)>=10 and len(l)>=10 and all(hc[e]>=2 and lc[e]>=2 for e in expected_emotions)
        pools.append({"threshold":t,"high_n":len(h),"low_n":len(l),"eligible":eligible,
            "high_emotion_counts":dict(hc),"low_emotion_counts":dict(lc)})
        if high is None and eligible: high,low,threshold=h,l,t
    if high is None:
        method = "exploratory_within_cell_rank_contrast"
        high,low = [],[]
        for emotion,language in sorted(expected_cells):
            cell = sorted([r for r in valid if (r["emotion"],r["language"])==(emotion,language)],
                          key=lambda r:(mira._metric(r,"ecapa"),r["id"]))
            if len(cell)<4: raise ValueError(f"Cannot derive rank fallback: fewer than four valid samples in {emotion}/{language}")
            low.extend(cell[:2]);high.extend(cell[-2:])
    hcells = Counter((r["emotion"],r["language"]) for r in high)
    lcells = Counter((r["emotion"],r["language"]) for r in low)
    joint = all(hcells[c]>=2 and lcells[c]>=2 for c in expected_cells)
    def balanced(pool):
        groups = defaultdict(list)
        for row in pool:
            key = (row["emotion"],row["language"]) if joint else row["emotion"]
            groups[key].append(values[row["id"]])
        return np.mean([np.mean(v,axis=0,dtype=np.float64) for _,v in sorted(groups.items())],axis=0).astype(np.float32)
    high_mean,low_mean = balanced(high),balanced(low)
    vector = high_mean-low_mean
    if set(r["id"] for r in high)&set(r["id"] for r in low): raise ValueError("Contrast pools overlap")
    metadata = {"point":point,"alpha":ALPHA,"method":method,"threshold":threshold,"strict_threshold":threshold is not None,
        "eligible_threshold":threshold is not None,"exploratory_rank_fallback":threshold is None,
        "valid_n":len(valid),"excluded_n":len(rows)-len(valid),"threshold_pools":pools,
        "weighting":"equal sample per emotion/language cell, equal ten cells" if joint else "equal sample per emotion, equal five emotions",
        "joint_language_balanced":joint,"language_confounded":not joint,
        "high_ids":[r["id"] for r in high],"low_ids":[r["id"] for r in low],
        "high_language_counts":dict(Counter(r["language"] for r in high)),
        "low_language_counts":dict(Counter(r["language"] for r in low)),
        "high_cell_counts":{f'{e}/{l}':n for (e,l),n in sorted(hcells.items())},
        "low_cell_counts":{f'{e}/{l}':n for (e,l),n in sorted(lcells.items())},
        "high_mean_l2":float(np.linalg.norm(high_mean)),"low_mean_l2":float(np.linalg.norm(low_mean)),
        "vector_l2":float(np.linalg.norm(vector)),"vector_rms":float(np.sqrt(np.mean(vector*vector))),
        "vector_sha256":hashlib.sha256(vector.tobytes()).hexdigest(),
        "interpretation":"Candidate per-reference activation direction, not an isolated speaker vector or calibrated voice-match percentage."}
    return vector,metadata


def assessment(rows, baseline="baseline", condition=None):
    """One voice, lightweight statistical audit, not runtime activation policy.

    Known negative effects may remain selectable at the user's explicit request.
    Unknown effects never count as evidence of improvement.
    """
    voices = set(r.get("voice","unknown") for r in rows)
    if len(voices)!=1: raise ValueError("Assess one voice at a time")
    targets = set(r["condition"] for r in rows if r["condition"]!=baseline)
    if condition is None:
        if len(targets)!=1: raise ValueError("Specify one target condition")
        condition = next(iter(targets))
    controls = {r["block"]:r for r in rows if r["condition"]==baseline}
    takes = [r for r in rows if r["condition"]==condition]
    means,matched = {},{}
    reasons = []
    metrics = ("ecapa","content_enjoyment","production_quality","mos","wer")
    if not takes: reasons.append("no_steered_takes")
    failures = sum(bool(r.get("failed") or r.get("status")=="failed") for r in takes)
    if failures: reasons.append("generation_failure")
    if any(c.get("failed") or c.get("status")=="failed" for c in controls.values()): reasons.append("control_failure")
    for metric in metrics:
        measured = [mira._metric(r,metric) for r in takes if not r.get("failed") and r.get("status")!="failed"]
        valid = [v for v in measured if v is not None]
        means[metric] = float(np.mean(valid)) if valid else None
        diffs = []
        for r in takes:
            c = controls.get(r["block"])
            if c is None or r.get("failed") or c.get("failed") or r.get("status")=="failed" or c.get("status")=="failed": continue
            a,b = mira._metric(r,metric),mira._metric(c,metric)
            if a is not None and b is not None: diffs.append(a-b)
        matched[metric] = mira.paired_interval(diffs)
        if len(diffs)!=len(takes) or not takes: reasons.append("incomplete_matched_"+metric)
    delta = lambda metric: matched[metric]["mean"] if matched[metric] else None
    for metric,limit in (("content_enjoyment",-.30),("production_quality",-.30),("mos",-.20)):
        if delta(metric) is not None and delta(metric)<limit: reasons.append(metric+"_regression")
    if delta("wer") is not None and delta("wer")>.03: reasons.append("wer_regression")
    if means["wer"] is not None and means["wer"]>.15: reasons.append("absolute_wer_limit")
    bad = lambda group: sum(bool(r.get("failed") or r.get("status")=="failed" or (mira._metric(r,"wer") is not None and mira._metric(r,"wer")>.25)) for r in group)/len(group) if group else None
    control_takes = [controls[r["block"]] for r in takes if r["block"] in controls]
    bad_delta = bad(takes)-bad(control_takes) if takes and control_takes else None
    if bad_delta is not None and bad_delta>.10: reasons.append("catastrophic_wer_regression")
    if delta("ecapa") is None or delta("ecapa")<.01: reasons.append("insufficient_speaker_gain")
    per_language = {}
    for language in ("en","de"):
        language_rows = [r for r in rows if r.get("language")==language]
        if not language_rows: continue
        diagnostics = mira.language_diagnostics(language_rows,baseline=baseline)
        cell = next((c for c in diagnostics if c["condition"]==condition),None)
        if cell: per_language[language]=cell
    safe = not(set(reasons)-{"insufficient_speaker_gain"})
    enabled = not reasons
    confirmed = bool(enabled and matched["ecapa"]["low"]>0)
    return {"voice":next(iter(voices)),"condition":condition,"baseline":baseline,"n":len(takes),"failures":failures,
        "safe":safe,"eligible":enabled,"default_enabled":enabled,"recommended_enabled":enabled,"confirmed_gain":confirmed,
        "experimental":not confirmed,"gate_reasons":sorted(set(reasons)),"means":means,"matched":matched,
        "bad_wer_rate_delta":bad_delta,"per_language":per_language,
        "warning":"Small single-reference proxy pilot; gain remains unconfirmed." if enabled and not confirmed
            else "Audit did not clear safeguards; experimental activation can remain available with a prominent quality warning." if not enabled
            else "Positive held-out proxy interval on this reference, not universal identity proof."}


def mlp_recommendations():
    return {"train_now":False,"recommendation":"First establish a useful per-reference vector bank and honest held-out gains. Ten voices are insufficient to substantiate an ECAPA-to-delta MLP.",
        "target_bias":"Derived deltas are selected using ECAPA against one conditioning clip. The same embedding as MLP input risks learning this score-selection bias and recording/language/prosody leakage instead of identity.",
        "data":"Use multiple independent reference recordings per speaker, multiple text/emotion/language batches, repeated vector estimates and no-effect/negative controls. Store reference/session provenance and vector estimation uncertainty.",
        "splits":"Hold out whole speakers and recording sessions, not random clips. Neither threshold-pool selection nor validation audio may leak into targets or hyperparameter tuning.",
        "baseline":"Compare nearest-neighbor retrieval, regularized linear/ridge mapping and low-rank basis coefficients before a small MLP. Estimate and separately report the grand mean generic reference-presence effect before treating residual directions as speaker-specific. A 1024D raw output from ten examples is severely underdetermined.",
        "evaluation":"Fresh paired generations on unseen speakers/references, reference plus independently verified anchor embeddings, CE/PQ/MOS/WER, language and emotion stratification, batch-aware intervals and a blind listening panel.",
        "deployment":"Predict bounded coefficients over validated directions with confidence/abstention, reference/model hash gating and a no-steering fallback. A regression fit alone is not evidence of improved voice cloning."}


def reference_effect_diagnostics(vector_sets):
    """CPU-only shared-direction diagnostic; no centered intervention is tested.

    Input voice -> point/vector mapping, or paired_reference_vector() outputs.
    Return explicit arrays separately from JSON metadata suitable for reports.
    """
    source={voice:value.get('vectors',value) for voice,value in vector_sets.items()}
    if len(source)<2:raise ValueError('Need at least two voices for a shared-reference-effect diagnostic')
    points=set.intersection(*(set(value) for value in source.values()))
    generic,centered,metadata={},defaultdict(dict),{}
    voices=sorted(source)
    for point in sorted(points):
        arrays=[np.asarray(source[v][point],dtype=np.float64) for v in voices]
        if any(a.ndim!=1 or a.shape!=arrays[0].shape or not np.all(np.isfinite(a)) for a in arrays):raise ValueError('Incompatible reference effect vectors')
        matrix=np.stack(arrays)
        mean=matrix.mean(axis=0)
        norms=np.linalg.norm(matrix,axis=1)
        units=matrix/np.maximum(norms[:,None],1e-12)
        cosines=units@units.T
        mean_norm=float(np.linalg.norm(mean))
        generic[point]=mean.astype(np.float32)
        per_voice={}
        for i,voice in enumerate(voices):
            residual=matrix[i]-mean
            centered[voice][point]=residual.astype(np.float32)
            per_voice[voice]={'raw_l2':float(norms[i]),'residual_l2':float(np.linalg.norm(residual)),
                'cosine_to_generic':float(matrix[i]@mean/max(norms[i]*mean_norm,1e-12)),
                'generic_projection_coefficient':float(matrix[i]@mean/max(mean_norm*mean_norm,1e-12))}
        off_diagonal=cosines[~np.eye(len(voices),dtype=bool)]
        metadata[point]={'dimension':int(matrix.shape[1]),'generic_l2':mean_norm,
            'generic_rms':float(np.sqrt(np.mean(mean*mean))), 'voice_order':voices,
            'cross_voice_cosine_matrix':cosines.tolist(),'mean_cross_voice_cosine':float(off_diagonal.mean()),
            'per_voice':per_voice}
    return {'generic_vectors':generic,'centered_vectors':dict(centered),'metadata':{
        'points':metadata,'voice_count':len(voices),
        'interpretation':'Grand mean is a candidate generic reference-presence/history effect, not a proven nuisance-only component. Centering may remove shared legitimate voice features. Centered vectors are CPU diagnostics only and have not been generation-tested or installed as defaults.'}}


def render_catalogue_report(path, rows, voice_reports, bank, plan=None, audio_root=None,
                            evidence=None, narrative=None, downloads=None, reference_audio=None):
    """English report with per-voice matching and evaluation-only embedded audio.

    voice_reports is a list of assessment() dictionaries, or voice->dict/list.
    bank is voice->vector metadata (a `voices` wrapper is accepted). All raw
    collection results/audio stay on disk and in the downloadable ledger;
    only actual non-collection evaluation takes are embedded in this HTML.
    """
    path=Path(path)
    audio_root=Path(audio_root) if audio_root is not None else path.parent
    esc=lambda value:html.escape(str(value),quote=True)
    fmt=lambda value,n=3:'—' if value is None else f'{value:.{n}f}'
    interval=lambda value:'—' if not value else f'{value["mean"]:+.3f} [{value["low"]:+.3f}, {value["high"]:+.3f}]'
    if isinstance(voice_reports,dict):
        reports=[]
        for voice,value in voice_reports.items():
            for item in (value if isinstance(value,list) else [value]):
                reports.append({'voice':voice,**item})
    else: reports=list(voice_reports or [])
    entries=bank.get('voices',bank) if isinstance(bank,dict) else {e['voice']:e for e in bank}
    if isinstance(entries,list): entries={e['voice']:e for e in entries}
    evaluations=[r for r in rows if 'collection' not in r.get('phase','').lower()]
    collection_count=len(rows)-len(evaluations)
    failures=sum(bool(r.get('failed') or r.get('status')=='failed') for r in rows)
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Catalogue voice steering and paired reference effects</title><style>*{box-sizing:border-box}html{max-width:100%}body{font:16px system-ui;width:100%;max-width:1350px;margin:auto;padding:clamp(12px,3vw,20px);background:#f5f5f6;color:#20232a}h1,h2,h3,p,summary{overflow-wrap:anywhere}p{line-height:1.5}.scroll{max-width:100%;min-width:0;overflow:auto}table{width:100%;min-width:1050px;border-collapse:collapse;font-size:14px}td,th{padding:8px;text-align:left;border-bottom:1px solid #ccc;white-space:nowrap}td:first-child,td:last-child{white-space:normal;overflow-wrap:anywhere;max-width:300px;min-width:130px}.card,article{min-width:0;padding:16px;margin:12px 0;border:1px solid #ddd;border-radius:10px;background:white}article[hidden]{display:none}audio{width:100%;max-width:600px;min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere}.filters{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(230px,100%),1fr));gap:10px}.filters label,select{min-width:0}select{display:block;width:100%;max-width:100%;padding:8px}.warning{color:#8c351d}summary{cursor:pointer}</style>',
        '<h1>Catalogue voice steering: quality contrast versus reference effect</h1>',
        '<p>Full-FT base model, no DPO/LoRAs, CFG 2.0 and temperature 1.0. Ten fixed catalogue recording profiles. Two distinct estimators are kept separate: <strong>quality contrast</strong> compares high- versus low-ECAPA takes while both use a reference; <strong>reference-effect contrast</strong> compares the same pure-emotion prompt and actual batch seed with and without reference audio, without quality filtering.</p>',
        f'<p>{len(rows)} recorded takes: {collection_count} collection/control takes retained in the machine-readable ledger, {len(evaluations)} evaluation takes embedded below, {failures} generation failures retained. Collection audio is not duplicated inline in this report.</p>',
        '<section class="card"><h2>Experimental UI policy — on is not a quality guarantee</h2><p>The user requested default-on access to every valid catalogue vector. Vectors with failed quality screens remain tryable with explicit warnings; this report does not claim that all voices improved. Statistical gate eligibility is separate from actual activation policy. The runtime must bypass custom/no-reference, active-LoRA, wrong-reference-hash or wrong-base-checkpoint cases. The fixed quality-vector dose is layer 16, alpha 0.75; no hidden automatic dose optimization is implied.</p></section>']
    if downloads:
        from urllib.parse import urlsplit
        links=[]
        for label,href in downloads.items():
            href=str(href).strip()
            if urlsplit(href).scheme not in ('','http','https'):raise ValueError('Unsafe download URL')
            links.append(f'<a href="{esc(href)}" download>{esc(label)}</a>')
        parts.append('<p>Downloads: '+' · '.join(links)+'</p>')
    if narrative:
        for paragraph in ([narrative] if isinstance(narrative,str) else narrative):parts.append('<p>'+esc(paragraph)+'</p>')
    confirmed=[]
    with_reference,without_reference=[],[]
    for report in reports:
        reference_free=report.get('baseline')=='no-reference-baseline' or str(report.get('condition','')).startswith('reference-without-')
        (without_reference if reference_free else with_reference).append(report)
    def audit_table(group,title):
        parts.append('<h2>'+esc(title)+'</h2><div class="scroll" tabindex="0"><table><tr><th>Voice / condition versus control</th><th>N / failures</th><th>ECAPA</th><th>Paired Δ ECAPA [95% CI]</th><th>Δ CE</th><th>Δ PQ / MOS</th><th>WER</th><th>Δ WER</th><th>Evidence / warning</th></tr>')
        for report in group:
            matched=report.get('matched',{})
            mean=report.get('means',{})
            confirmed_gain=report.get('confirmed_gain',report.get('heldout_positive_interval',False))
            status='Positive held-out proxy interval' if confirmed_gain else 'Positive mean only; unconfirmed' if report.get('eligible') else 'Did not clear statistical safeguards'
            if confirmed_gain:confirmed.append(report.get('voice'))
            status+='; '+', '.join(report.get('gate_reasons',[])) if report.get('gate_reasons') else ''
            d=lambda key:matched[key]['mean'] if matched.get(key) else None
            control=report.get('baseline','no-reference-baseline' if str(report.get('condition','')).startswith('reference-without-') else 'baseline')
            parts.append(f'<tr><td>{esc(report.get("voice","unknown"))} / {esc(report.get("condition","unknown"))}<br>versus {esc(control)}</td><td>{report.get("n",0)} / {report.get("failures",0)}</td><td>{fmt(mean.get("ecapa"))}</td><td>{interval(matched.get("ecapa"))}</td><td>{fmt(d("content_enjoyment"))}</td><td>{fmt(d("production_quality"))} / {fmt(d("mos"))}</td><td>{fmt(100*mean["wer"],1) if mean.get("wer") is not None else "—"}%</td><td>{fmt(d("wer"))}</td><td>{esc(status)}</td></tr>')
        parts.append('</table></div>')
    audit_table(with_reference,'With-reference reinforcement: matched per-voice audit')
    if without_reference:audit_table(without_reference,'Without-reference replacement: versus the no-reference control')
    parts.append('<p>Each comparison uses its explicitly named control from the same voice, phase, text block and seeded batch schedule; controls are never overwritten across voices. Reference-free injection is compared against a no-reference baseline, not quietly against reference-conditioned synthesis. Intervals are exploratory, not corrected for all candidate comparisons. WER deltas are absolute fractions. ECAPA cosine is not a calibrated percentage of identity. CE/PQ/MOS are automated proxies, not human listening scores.</p>')
    if not reports:parts.append('<p class="warning">No per-voice validation audit is available yet. No improvement claim is supported.</p>')
    parts.append('<h2>Vector provenance and selection</h2><div class="scroll"><table><tr><th>Voice</th><th>Method / threshold</th><th>High / low samples</th><th>Balancing</th><th>Known warning</th></tr>')
    for voice,item in entries.items():
        meta=item.get('selection',item.get('metadata',item))
        fallback_method='prior validated Mira vector' if voice=='emolia_c1699' and item.get('status')=='ready' else 'not yet derived'
        parts.append(f'<tr><td>{esc(voice)}</td><td>{esc(meta.get("method",fallback_method))} / {esc(meta.get("threshold","not supplied"))}</td><td>{len(meta.get("high_ids",[]))} / {len(meta.get("low_ids",[]))}</td><td>{esc(meta.get("weighting","see provenance"))}</td><td>{esc(item.get("warning", "Language imbalance / limited coverage" if meta.get("language_confounded") else "Exploratory rank fallback" if meta.get("exploratory_rank_fallback") else "Reference-specific, not a general identity model"))}</td></tr>')
    parts.append('</table></div><p>Mira keeps its original independently validated quality vector unchanged where specified in the bank. The new pure-caption collection is a separate dataset for reference-effect estimation. Other quality vectors may use strict cosine &gt;0.50 or &gt;0.25 with both pools covering every emotion; deterministic top/bottom rank fallbacks are explicitly exploratory. Joint language/emotion balancing is used only with sufficient cell coverage. Means may still encode phonetics, reference language, generic speaker-prompt presence, recording style and duration.</p>')
    parts.append('<h2>Why the paired reference experiment differs</h2><p>Each of ten identical text/emotion/language conditions is repeated with ten distinct actual seeded batches. Every repeat contains different conditions in a stable 4/4/2 schedule; both arms use exactly the same literal caption/transcript, batch companions, order and seed. Captions contain no identity, gender, age or timbre description. Common no-reference outputs are reusable across profiles because neither the prompt nor seed depends on voice. They are a shared control, not ten independent control populations.</p><p>No ECAPA or quality filter enters the reference-effect vector. Average paired generated-position differences within each condition, then equally average the ten conditions. The last conditional prefill position is counted only if it predicts the first emitted frame; unused prompt/reference positions, EOS and finished/padded tails are excluded. Different output tokens, generated durations and prosody remain part of the total reference effect. It is not a pure speaker vector.</p><p>semantic_global is the post-normalization 1024D state; semantic_local is its bias-free 2560D acoustic projection. Raw mean differences at these sites are approximately linearly related, with BF16 rounding potentially changing autoregressive trajectories. Their results do not represent independent identity mechanisms.</p>')
    parts.append('<details><summary>Per-language and whole-batch diagnostics</summary><pre>'+esc(json.dumps(reports,ensure_ascii=False,indent=2))+'</pre></details>')
    if plan:parts.append('<details><summary>Frozen protocol</summary><pre>'+esc(json.dumps(plan,ensure_ascii=False,indent=2))+'</pre></details>')
    for key,value in (evidence or {}).items():parts.append('<details><summary>'+esc(key)+'</summary><pre>'+esc(json.dumps(value,ensure_ascii=False,indent=2))+'</pre></details>')
    parts.append('<h2>ECAPA-to-steering MLP: proposal only, not trained</h2><pre>'+esc(json.dumps(mlp_recommendations(),indent=2))+'</pre>')
    reference_count=0
    if reference_audio:
        parts.append('<section class="card"><h2>Fixed reference previews</h2><p>These are conditioning recordings, not validation outputs. Corpus names are profile labels, not independently verified identity ground truth.</p>')
        for label,filename in reference_audio.items():
            candidate=Path(filename)
            if not candidate.is_absolute():candidate=audio_root/candidate
            if candidate.is_file():
                parts.append('<h3>'+esc(label)+'</h3><audio controls preload="none" src="data:audio/mpeg;base64,'+base64.b64encode(candidate.read_bytes()).decode('ascii')+'"></audio>')
                reference_count+=1
        parts.append('</section>')
    parts.append('<h2>Listen to every recorded evaluation take</h2><p>All evaluation takes below, including bad outputs and failures. Collection takes are retained as separate files and ledger entries, not embedded here.</p><div class="filters">')
    filter_keys=('voice','phase','condition','emotion','language','block')
    for key in filter_keys:
        options=sorted(set(str(r.get(key,'unknown')) for r in evaluations))
        parts.append('<label>'+esc(key.title())+f'<select data-filter="{key}"><option value="">All</option>'+''.join(f'<option value="{esc(v)}">{esc(v)}</option>' for v in options)+'</select></label>')
    parts.append('</div><p id="count"></p>')
    embedded=0
    for row in evaluations:
        attrs=' '.join(f'data-{k}="{esc(row.get(k,"unknown"))}"' for k in filter_keys)
        parts.append(f'<article {attrs}><h3>{esc(row["id"])}</h3><p>{esc(row.get("text",""))}</p><p>'+' · '.join(esc(m)+': '+fmt(mira._metric(row,m)) for m in ('ecapa','content_enjoyment','production_quality','mos','wer'))+'</p>')
        filename=row.get('audio_path')
        candidate=Path(filename) if filename else None
        if candidate and not candidate.is_absolute():candidate=audio_root/candidate
        if candidate and candidate.is_file():
            parts.append('<audio controls preload="none" src="data:audio/mpeg;base64,'+base64.b64encode(candidate.read_bytes()).decode('ascii')+'"></audio>')
            embedded+=1
        else:parts.append('<p class="warning">No playable audio: '+esc(row.get('error','missing artifact'))+'</p>')
        if row.get('failed') or row.get('status')=='failed':parts.append('<p class="warning">Failure retained: '+esc(row.get('error','unknown'))+'</p>')
        parts.append('<details><summary>Prompt and intervention</summary><pre>'+esc(row.get('prompt',''))+'\n'+esc(json.dumps(row.get('intervention',{}),indent=2))+'</pre></details></article>')
    parts.append('<script>const selects=[...document.querySelectorAll("select[data-filter]")],takes=[...document.querySelectorAll("article")];function filter(){let n=0;for(const t of takes){t.hidden=selects.some(s=>s.value&&t.dataset[s.dataset.filter]!==s.value);if(!t.hidden)n++;}document.querySelector("#count").textContent=n+" of "+takes.length+" evaluation takes shown";}selects.forEach(s=>s.addEventListener("change",filter));filter();</script></html>')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(parts),encoding='utf-8')
    return {'recorded_takes':len(rows),'collection_takes_not_embedded':collection_count,
        'evaluation_takes':len(evaluations),'audio_embedded':embedded,'reference_audio_embedded':reference_count,
        'confirmed_proxy_gain_voices':sorted(set(confirmed))}
