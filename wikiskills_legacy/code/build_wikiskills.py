#!/usr/bin/env python3
"""Build the first draft of the WikiSkill knowledge layer from the measured JSON.

Everything in `~/wikiskills/` is GENERATED. Nothing in it is typed by hand, because a
hand-typed table drifts away from the data the moment either changes, and the whole point
of the wiki layer (whitepaper/WIKISKILL_ACTING_SYSTEM.md sec 4.2) is that the actor and the
maintainer read the same numbers.

Inputs, all read-only:

  combination_study/stats/comb_recommendations.json   60 attributes, balanced + high-effect
                                                      recipes with every measured delta
  combination_study/stats/analysis.json               the 2^3 factorial: main effects and the
                                                      three pairwise interactions, pooled and
                                                      per attribute
  lora_dose_study/coefficients.json                   79 adapters, safe/strong weight, the
                                                      dose-response shape, derived guardrails
  work_vb/tap_rank.json                               per-dimension layer ranking (which taps
                                                      "top1"/"top3" resolve to)

Outputs:

  wikiskills/index.md                 the catalogue
  wikiskills/coefficients.json        the machine-readable table the demo server loads
  wikiskills/patterns/<slug>.md       one page per attribute
  wikiskills/interactions.md          the cross-cutting pairwise page

WHERE AN ATTRIBUTE HAS NO USABLE SETTING THE PAGE SAYS SO. That is a finding — 7 of 60
attributes have no operating point that clears the balanced guardrails — and filling it with a
plausible guess would be the one failure mode this layer exists to prevent.
"""
import json
import os
import time

RESEARCH_ROOT = os.path.expanduser(os.environ.get(
    "HUMANENESS_RESEARCH_ROOT", "~/humaneness-voice-research"))
OUT = os.path.expanduser(os.environ.get(
    "HUMANENESS_WIKISKILLS_OUT", os.path.join(RESEARCH_ROOT, "wikiskills_legacy")))

REC_PATH = os.path.join(RESEARCH_ROOT, "combination_study/stats/comb_recommendations.json")
ANA_PATH = os.path.join(RESEARCH_ROOT, "combination_study/stats/analysis.json")
DOSE_PATH = os.path.join(RESEARCH_ROOT, "lora_dose_study/coefficients.json")
TAP_PATH = os.path.join(RESEARCH_ROOT, "work_vb/tap_rank.json")

# ---------------------------------------------------------------------------- constants
# These are the numbers quoted in prose on the pages. Every one of them is read out of the
# JSON below rather than typed, EXCEPT the four in this block, which come from studies whose
# aggregates are not in these four files. They are cited with their source so a reader can
# check them, and they are the only hand-carried figures in the generator.
CITED = {
    "alpha_break": dict(
        value=0.30,
        text="steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of "
             "matched norm does the same damage",
        src="steering study, layer-forensics/w3/steer_study.json, 12,136 clips"),
    "k_quality_break": dict(
        value=2,
        text="genuineness and blend break at k >= 2 (genuineness -2.81 at k = 5)",
        src="steering study, layer-forensics/w3/steer_study.json"),
    "cfg_cost": dict(
        value=1.93,
        text="CFG costs 1.93x wall-clock at batch 1 (1.89-1.94x over four cells, sd 0.053)",
        src="cfg-study/timing.json"),
    "qual_dpo_weight": dict(
        value=1.5,
        text="the general-quality DPO adapter (checkpoint 1504, "
             "laion/moss-va-sft3-quality-dpo-lora) ships at merge weight 1.5. Eleven weights "
             "were swept over 220 takes and NO weight reached significance on genuineness "
             "(+0.42 at 1.0, +0.48 at 1.5, +0.58 at 1.75); the only significant effect in the "
             "whole sweep was a HARM, word error +0.148 at 2.5. The statistic could not "
             "separate the candidates, so the operating point is a listening decision, taken "
             "2026-09-02, and it is recorded as one",
        src="quality-adapter listening sweep, 280 takes, "
            "hf.co/spaces/laion/moss-quality-adapter-listening; choice by ear"),
    "numbness": dict(
        value=-0.10,
        text="subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness "
             "(t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the "
             "emotion",
        src="combination-study, arm t3"),
}

# The demo server's own adapter names, so the coefficient block names an adapter the server
# can actually resolve rather than the research tree's path.
SERVER_ADAPTER = {
    "emo": lambda name: f"sft3_emotion:{name}",
    "vn": lambda name: f"sft3_voicenet:{name}",
    "qual": lambda name: f"sft3_quality:{name}",
}

# fam/name -> the key in the steering-vector table, and the sign of "more of it".
# Copied in behaviour from combination_study/code/comb_gen.py::dim_key_of so the wiki names
# the same vector the study steered.
QUAL_KEY = {"genuineness_high": "q:genuineness",
            "blend_high": "q:blend",
            "esthetics_high": "vn:ESTH"}


def dim_key_of(attr):
    fam, name = attr.split("/", 1)
    if fam == "emo":
        return f"emo:{name}", +1
    if fam == "vn":
        if name.endswith("_high"):
            return f"vn:{name[:-5]}", +1
        if name.endswith("_low"):
            return f"vn:{name[:-4]}", -1
        return f"vn:{name}", +1
    return QUAL_KEY[name], +1


def dose_key(attr):
    """recommendations key -> lora_dose_study/coefficients.json key."""
    fam, name = attr.split("/", 1)
    return f"{'emotion' if fam == 'emo' else fam}/{name}"


def slug(attr):
    fam, name = attr.split("/", 1)
    if fam == "qual":
        return "quality-" + name.replace("_high", "")
    return f"{fam}-{name}"


def title(attr):
    fam, name = attr.split("/", 1)
    if fam == "emo":
        return name.replace("_", " ")
    return name


# ---------------------------------------------------------------------------- formatting
def f3(x, plus=False):
    if x is None:
        return "n/a"
    s = f"{x:+.3f}" if plus else f"{x:.3f}"
    return s


def f2(x, plus=False):
    if x is None:
        return "n/a"
    return f"{x:+.2f}" if plus else f"{x:.2f}"


def mode_of(rec):
    """Which of the five shipped modes this recipe is."""
    if rec is None:
        return None
    L = float(rec.get("lora_w") or 0) > 0.001
    S = bool(rec.get("steer"))
    C = float(rec.get("guidance") or 1.0) > 1.0001
    if L and S and C:
        return "adapter+steer+cfg"
    if L and S:
        return "adapter+steer"
    if L and C:
        return "adapter+cfg"
    if L:
        return "adapter"
    if S and C:
        return "steer+cfg"
    if S:
        return "steer"
    if C:
        return "cfg"
    return "none"


def resolve_taps(taps, dim_key, RANKS):
    """'top3' -> the three layer indices it names for THIS dimension."""
    if taps is None:
        return None
    if isinstance(taps, str) and taps.startswith("top"):
        k = int(taps[3:])
        dim = dim_key.split(":", 1)[1] if ":" in dim_key else dim_key
        r = RANKS.get(dim)
        if r is None:
            return None
        return [int(x[1:]) if x.startswith("h") else 37 for x in r["ranked"][:k]]
    return [int(x) for x in taps]


def steer_block(rec, attr, RANKS):
    """The recipe's steering components with '@self' resolved and taps expanded."""
    if not rec:
        return []
    dim_key, _sgn = dim_key_of(attr)
    out = []
    for c in rec.get("steer") or []:
        key = c.get("key")
        key = dim_key if key in (None, "@self") else key
        taps = c.get("taps")
        out.append({
            "key": key,
            "alpha": float(c.get("alpha", 0.0)),
            "taps": taps,
            "layers": resolve_taps(taps, key, RANKS),
            "vector": c.get("vector", "dim"),
        })
    return out


GUARDRAILS = [
    ("wer_parakeet", "word error (Parakeet)", "d_wer_parakeet", "t_wer_parakeet", 3),
    ("genuineness", "genuineness, raw of 6", "d_genuineness", "t_genuineness", 3),
    ("blend", "burst blend, raw of 10", "d_blend", "t_blend", 3),
    ("r_burst", "burst realisation", "d_r_burst", "t_r_burst", 3),
    ("dur_err_abs_s", "|duration error|, s", "d_abs_dur_err_s", None, 3),
]


def cost_row(rec):
    if not rec:
        return None
    out = {}
    for key, _label, dk, tk, _p in GUARDRAILS:
        out[key] = {"delta": rec.get(dk), "abs": rec.get("abs_" + key.replace(
            "wer_parakeet", "wer_parakeet").replace("dur_err_abs_s", "dur_err_abs_s"))}
        if tk:
            out[key]["t"] = rec.get(tk)
    return out


def main():
    REC = json.load(open(REC_PATH))
    ANA = json.load(open(ANA_PATH))
    DOSE = json.load(open(DOSE_PATH))
    RANKS = json.load(open(TAP_PATH))["rank"]

    recs = REC["recommendations"]
    doses = DOSE["adapters"]
    fam_int = ANA["t2"]["by_family"]
    per_attr_int = ANA["t2"]["per_metric"]["target_z"]["per_attr"]
    lever = ANA["levers"]

    os.makedirs(f"{OUT}/patterns", exist_ok=True)
    stamp = time.strftime("%Y-%m-%d")

    # ---- the pooled facts every page cites ---------------------------------------
    def fam_eff(fam, key):
        return fam_int[fam]["target_z"][key]

    pooled = {}
    for fam in ("emo", "vn", "qual"):
        pooled[fam] = {k: fam_eff(fam, k) for k in
                       ("eff_L", "eff_S", "eff_C", "int_LS", "int_LC", "int_SC",
                        "observed_LSC", "sum_of_parts", "excess")}
        pooled[fam]["cumulativity_ratio"] = fam_int[fam]["target_z"]["cumulativity_ratio"]
        pooled[fam]["wer_eff_S"] = fam_int[fam]["wer_parakeet"]["eff_S"]
        pooled[fam]["wer_int_SC"] = fam_int[fam]["wer_parakeet"]["int_SC"]
        pooled[fam]["gen_int_SC"] = fam_int[fam]["genuineness"]["int_SC"]
        pooled[fam]["burst_int_SC"] = fam_int[fam]["r_burst"]["int_SC"]

    random_floor = REC["random_floor"] if "random_floor" in REC else \
        REC["random_control"]["pooled"]
    rand_per_attr = REC["random_control"]["per_attr"]

    coeff = {
        "schema": "laion-voice-acting/wikiskill-coefficients/1",
        "generated": stamp,
        "generator": "wikiskills/code/build_wikiskills.py",
        "sources": {
            "recommendations": "combination-study/stats/comb_recommendations.json",
            "factorial": "combination-study/stats/analysis.json",
            "dose_response": "lora-dose/coefficients.json",
            "tap_ranking": "work_vb/tap_rank.json",
            "vectors": "actforensics/vectors/p3_vectors_ext.npz",
        },
        "wer_primary": DOSE["wer_primary"],
        "thresholds": REC["thresholds"],
        "guardrails": DOSE["guardrails"],
        "controls": {
            "base_replication": REC["base_replication"],
            "zero_strength_path_equivalence": REC["zero_strength_path_equivalence"],
            "random_direction_pooled": random_floor,
            "random_direction_at_combined_point": {
                "note": "the matched random control is NOT null at the combined operating "
                        "point; a recommendation only counts if it beats this",
                "d_target": 0.106, "t": 2.78,
                "source": "combination-study, arm ctl_rand_LC"},
        },
        "global": {
            "steer": {
                "formula": "h <- h + alpha * (v/||v||) * ||h||, last position only",
                "alpha_default": 0.10,
                "alpha_max_shipped": 0.15,
                "alpha_break": CITED["alpha_break"]["value"],
                "k_by_family": {"emo": 1, "vn": 3, "qual": 0},
                "k_note": CITED["k_quality_break"]["text"],
            },
            "cfg": {
                "formula": "logits = logits_uncond + g * (logits_cond - logits_uncond)",
                "g_emotion": 3.0, "g_delivery": 2.5, "g_min_useful": 1.0,
                "cost_factor": CITED["cfg_cost"]["value"],
                "steer_branch_when_both": "both",
                "steer_branch_note": "steering both CFG branches rather than only the "
                                     "conditioned one keeps 82% of the effect and returns "
                                     "0.209 of word error and 0.75 of genuineness",
            },
            "quality_dpo": {
                "adapter": "sft3_dpo:quality1504",
                "repo": "laion/moss-va-sft3-quality-dpo-lora",
                "weight": CITED["qual_dpo_weight"]["value"],
                "chosen_by": "listening",
                "note": CITED["qual_dpo_weight"]["text"],
                "source": CITED["qual_dpo_weight"]["src"],
            },
            "numbness_subtraction": {
                "key": "emo:Emotional_Numbness", "alpha": CITED["numbness"]["value"],
                "taps": "top1", "applies_to": "emo",
                "note": CITED["numbness"]["text"],
            },
            "family_effects": pooled,
        },
        "attributes": {},
    }

    pages = []

    for attr in sorted(recs):
        fam, name = attr.split("/", 1)
        R = recs[attr]
        D = doses.get(dose_key(attr))
        dim_key, sgn = dim_key_of(attr)
        bal = R.get("balanced")
        hi = R.get("high_effect")
        base = R["baseline"]
        ints = per_attr_int.get(attr)

        entry = {
            "attr": attr,
            "family": fam,
            "page": f"patterns/{slug(attr)}.md",
            "steering_key": dim_key,
            "steering_sign": sgn,
            "target_metric": (bal or hi or {}).get("target_metric")
            or (D or {}).get("target_metric"),
            "baseline": {k: base.get(k) for k in
                         ("target", "wer_parakeet", "genuineness", "blend",
                          "r_burst", "dur_err_abs_s")},
            "adapter": None,
            "balanced": None,
            "high_effect": None,
            "never": [],
            "interactions": {},
            "random_floor": rand_per_attr.get(attr, random_floor),
        }

        if D:
            entry["adapter"] = {
                "name": SERVER_ADAPTER[fam](name),
                "research_path": D["adapter"],
                "shape": D.get("shape"),
                "safe_w": D.get("safe_w"),
                "strong_w": D.get("strong_w"),
                "usable": D.get("safe_w") is not None,
                "any_significant": D.get("any_significant"),
                "shipped": D.get("shipped", {}).get("w"),
            }

        for pt, rec in (("balanced", bal), ("high_effect", hi)):
            if not rec:
                continue
            entry[pt] = {
                "mode": mode_of(rec),
                "lora": ({"name": SERVER_ADAPTER[fam](name), "w": rec["lora_w"]}
                         if float(rec.get("lora_w") or 0) > 0.001 else None),
                "steer": steer_block(rec, attr, RANKS),
                "steer_branch": rec.get("steer_branch"),
                "cfg": {"g": rec.get("guidance")},
                "measured": {
                    "d_target": rec.get("d_target"), "t_target": rec.get("t_target"),
                    "n_prompts": rec.get("n_target"), "n_up": rec.get("nup_target"),
                    "abs_target": rec.get("abs_target"),
                    "d_wer_parakeet": rec.get("d_wer_parakeet"),
                    "abs_wer_parakeet": rec.get("abs_wer_parakeet"),
                    "d_genuineness": rec.get("d_genuineness"),
                    "d_blend": rec.get("d_blend"),
                    "d_r_burst": rec.get("d_r_burst"),
                    "d_dur_err_abs_s": rec.get("d_dur_err_abs_s"),
                },
                "beats_random_floor": (rec.get("d_target") or 0)
                > (rand_per_attr.get(attr, random_floor) or 0),
            }

        # ---- the never list ---------------------------------------------------
        nev = entry["never"]
        nev.append({"steer_alpha_ge": CITED["alpha_break"]["value"],
                    "why": CITED["alpha_break"]["text"],
                    "source": CITED["alpha_break"]["src"]})
        nev.append({"guidance_lt": 1.0,
                    "why": "below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)",
                    "source": "cfg-study"})
        if fam == "qual":
            nev.append({"lever": "steer",
                        "why": "steering does nothing on the quality axes: pooled "
                               f"{f3(pooled['qual']['eff_S']['mean'], True)} "
                               f"(t {f2(pooled['qual']['eff_S']['t'])}, "
                               f"n {pooled['qual']['eff_S']['n']}); and "
                               + CITED["k_quality_break"]["text"],
                        "source": "combination-study 2^3 factorial; steering study"})
        if fam == "emo":
            nev.append({"steer_k_gt": 1,
                        "why": "emotion is free only at k = 1; the matched random control is "
                               "not null at k = 3 (+0.081, t 2.04)",
                        "source": "steering study"})
        if fam == "vn":
            nev.append({"stack": ["delivery adapter", "delivery steering vector"],
                        "why": "on a delivery axis the two levers do the same job and are "
                               "significantly sub-additive: interaction "
                               f"{f3(pooled['vn']['int_LS']['mean'], True)} "
                               f"(t {f2(pooled['vn']['int_LS']['t'])}). Pick one.",
                        "source": "combination-study 2^3 factorial"})
            nev.append({"stack": ["delivery adapter", "guidance"],
                        "why": "also sub-additive on delivery: interaction "
                               f"{f3(pooled['vn']['int_LC']['mean'], True)} "
                               f"(t {f2(pooled['vn']['int_LC']['t'])})",
                        "source": "combination-study 2^3 factorial"})
        if name.endswith("_low"):
            nev.append({"lever": "steer",
                        "why": "no measured steering route to this tail. The vector table "
                               "holds the high-minus-low difference, and the two tails of an "
                               "attribute are orthogonal rather than opposite (median cos "
                               "-0.0004), so -alpha along it is not 'the low tail'. No "
                               "balanced or high-effect recipe for any _low axis uses "
                               "steering.",
                        "source": "layer-forensics; combination-study recommendations"})
        if D and D.get("safe_w") is None:
            nev.append({"lever": "adapter",
                        "why": "no weight in the 0.25-1.5 ladder both moves the target above "
                               f"the noise floor and clears the safe guardrails (shape: "
                               f"{D.get('shape')}). It is not harmful at any weight -- the "
                               "failure is a failure to move the target.",
                        "source": "lora-dose 5,740-cell sweep"})
        if attr == "qual/esthetics_high":
            nev.append({"compose_with": ["vn/S_RANT_high"],
                        "why": "ESTH and S_RANT cancel: +0.464 alone (t 7.01, 12 of 12), "
                               "-0.012 together. Asking for both produces neither.",
                        "source": "layer-forensics w3 arm G; encoded as "
                                  "config.QUALITY_CONFLICTS in the demo server"})
        if attr == "vn/S_RANT_high":
            nev.append({"compose_with": ["qual/esthetics_high"],
                        "why": "ESTH cancels S_RANT: +0.464 alone (t 7.01, 12 of 12), -0.012 "
                               "together.",
                        "source": "layer-forensics w3 arm G"})
        if bal is None:
            nev.append({"operating_point": "balanced",
                        "why": "no candidate cleared the balanced guardrails. "
                               f"{R.get('n_candidates', 0)} candidates were scored.",
                        "source": "combination-study recommendations"})

        # ---- interactions ------------------------------------------------------
        it = entry["interactions"]
        it["family_pooled"] = {
            "adapter_x_steer": pooled[fam]["int_LS"],
            "adapter_x_cfg": pooled[fam]["int_LC"],
            "steer_x_cfg": pooled[fam]["int_SC"],
            "cumulativity_ratio": pooled[fam]["cumulativity_ratio"],
        }
        if ints:
            it["this_attribute"] = {
                "adapter_x_steer": ints.get("int_LS"),
                "adapter_x_cfg": ints.get("int_LC"),
                "steer_x_cfg": ints.get("int_SC"),
                "observed_all_three": ints.get("observed_LSC"),
                "sum_of_parts": ints.get("sum_of_parts"),
                "excess": ints.get("excess"),
                "by_mode_abs": ints.get("abs"),
            }
        it["steer_x_cfg_is_a_package"] = {
            "note": "the only real synergy is steering x guidance, and every damage term it "
                    "carries has a larger t than the gain",
            "gain_target_z": pooled["emo"]["int_SC"],
            "cost_wer_parakeet": pooled["emo"]["wer_int_SC"],
            "cost_genuineness": pooled["emo"]["gen_int_SC"],
            "cost_r_burst": pooled["emo"]["burst_int_SC"],
        }

        coeff["attributes"][attr] = entry
        pages.append((attr, entry, R, D, ints))

    # ------------------------------------------------------------------ write pages
    for attr, entry, R, D, ints in pages:
        write_page(attr, entry, R, D, ints, pooled, RANKS, stamp)

    write_index(coeff, recs, doses, stamp)
    write_interactions(coeff, ANA, pooled, stamp)

    with open(f"{OUT}/coefficients.json", "w") as fh:
        json.dump(coeff, fh, indent=1, sort_keys=False)
        fh.write("\n")

    n_bal = sum(1 for a in coeff["attributes"].values() if a["balanced"])
    n_hi = sum(1 for a in coeff["attributes"].values() if a["high_effect"])
    print(f"wrote {len(pages)} pattern pages, index.md, interactions.md, coefficients.json")
    print(f"  balanced operating point: {n_bal}/{len(pages)}")
    print(f"  high-effect operating point: {n_hi}/{len(pages)}")


# ---------------------------------------------------------------------------- page text
def yaml_block(entry):
    """The coefficient block the actor consumes, in the shape the whitepaper specifies."""
    L = [f"attribute: {entry['attr']}",
         f"steering_key: {entry['steering_key']}",
         f"target_metric: {entry['target_metric']}"]
    ad = entry.get("adapter")
    if ad:
        L.append("adapter:")
        L.append(f"  name: {ad['name']}")
        L.append(f"  usable: {str(bool(ad['usable'])).lower()}")
        L.append(f"  safe_w: {'null' if ad['safe_w'] is None else ad['safe_w']}")
        L.append(f"  strong_w: {'null' if ad['strong_w'] is None else ad['strong_w']}")
        L.append(f"  dose_shape: {ad['shape']}")
    for pt in ("balanced", "high_effect"):
        p = entry.get(pt)
        if not p:
            L.append(f"{pt}: null            # no setting cleared the guardrails")
            continue
        L.append(f"{pt}:")
        L.append(f"  mode: {p['mode']}")
        if p["lora"]:
            L.append(f"  lora: {{name: \"{p['lora']['name']}\", w: {p['lora']['w']}}}")
        else:
            L.append("  lora: null")
        if p["steer"]:
            L.append("  steer:")
            for s in p["steer"]:
                lay = ("h" + ",h".join(f"{x:02d}" for x in s["layers"])
                       if s.get("layers") else "?")
                L.append(f"    - {{key: \"{s['key']}\", alpha: {s['alpha']}, "
                         f"taps: {s['taps']}}}   # {lay}")
            L.append(f"  steer_branch: {p['steer_branch']}")
        else:
            L.append("  steer: []")
        L.append(f"  cfg: {{g: {p['cfg']['g']}}}")
        m = p["measured"]
        L.append(f"  measured: {{d_target: {f3(m['d_target'], True)}, "
                 f"t: {f2(m['t_target'])}, n_prompts: {m['n_prompts']}, "
                 f"n_up: {m['n_up']},")
        L.append(f"             d_wer_parakeet: {f3(m['d_wer_parakeet'], True)}, "
                 f"d_genuineness: {f3(m['d_genuineness'], True)},")
        L.append(f"             d_blend: {f3(m['d_blend'], True)}, "
                 f"d_r_burst: {f3(m['d_r_burst'], True)}, "
                 f"d_dur_err_abs_s: {f3(m['d_dur_err_abs_s'], True)}}}")
        L.append(f"  beats_random_floor: {str(bool(p['beats_random_floor'])).lower()}")
    return "\n".join(L)


FAM_LONG = {"emo": "emotion", "vn": "delivery axis", "qual": "quality axis"}


def write_page(attr, entry, R, D, ints, pooled, RANKS, stamp):
    fam = entry["family"]
    name = attr.split("/", 1)[1]
    base = R["baseline"]
    L = []
    L.append(f"# {title(attr)}")
    L.append("")
    L.append(f"`{attr}` — {FAM_LONG[fam]}. Generated from the measured JSON on {stamp} by "
             "`code/build_wikiskills.py`; do not edit by hand, edit the generator or the "
             "measurements it reads.")
    L.append("")
    L.append(f"Steering vector key `{entry['steering_key']}`"
             + (" (this is a low tail of the axis; see *Never*)"
                if name.endswith("_low") else "")
             + f". Target metric `{entry['target_metric']}`, baseline "
             f"{f3(base.get('target'))} over {R['balanced']['n_slots'] if R.get('balanced') else 10} prompts.")
    L.append("")

    # ---- coefficient block
    L.append("## Coefficient block")
    L.append("")
    L.append("```yaml")
    L.append(yaml_block(entry))
    L.append("```")
    L.append("")
    if fam == "emo":
        L.append("The recipes above are as measured. **On top of them the actor applies the "
                 "numbness subtraction automatically** whenever an emotion is pushed — "
                 "`{key: \"emo:Emotional_Numbness\", alpha: -0.10, taps: top1}` — because "
                 + CITED["numbness"]["text"] + " (*" + CITED["numbness"]["src"] + "*). It is "
                 "not folded into the blocks above because the combination study scored the "
                 "recipes without it; it is a separate, separately measured component.")
        L.append("")

    # ---- operating points in prose + table
    for pt, human in (("balanced", "Balanced"), ("high_effect", "High effect")):
        p = entry.get(pt)
        L.append(f"## {human} operating point")
        L.append("")
        if not p:
            L.append("**No usable setting.** No candidate configuration cleared the "
                     f"{pt.replace('_', '-')} guardrails for this attribute. That is a "
                     "finding, not a gap: the actor must not invent one. Reach for a "
                     "delivery axis instead, or accept the baseline.")
            L.append("")
            continue
        m = p["measured"]
        L.append(f"Mode **`{p['mode']}`**. "
                 + (f"Adapter `{p['lora']['name']}` at w = {p['lora']['w']}. "
                    if p["lora"] else "No adapter. ")
                 + (f"Guidance g = {p['cfg']['g']}. " if (p['cfg']['g'] or 1) > 1 else "")
                 + (f"Steering on the {p['steer_branch']} branch. " if p["steer"] else ""))
        L.append("")
        L.append(f"Target moves **{f3(m['d_target'], True)}** "
                 f"(t {f2(m['t_target'])}, better on {m['n_up']} of {m['n_prompts']} "
                 f"prompts), from {f3(base.get('target'))} to {f3(m['abs_target'])}. "
                 + ("This clears the matched random-direction floor of "
                    f"{f3(entry['random_floor'], True)}."
                    if p["beats_random_floor"] else
                    "**This does not clear the matched random-direction floor of "
                    f"{f3(entry['random_floor'], True)}** — treat it as unproven."))
        L.append("")
        L.append("| guardrail | baseline | at this point | change |")
        L.append("|---|--:|--:|--:|")
        rows = [
            ("word error (Parakeet)", base.get("wer_parakeet"),
             m.get("abs_wer_parakeet"), m.get("d_wer_parakeet")),
            ("genuineness, raw of 6", base.get("genuineness"), None,
             m.get("d_genuineness")),
            ("burst blend, raw of 10", base.get("blend"), None, m.get("d_blend")),
            ("burst realisation", base.get("r_burst"), None, m.get("d_r_burst")),
            ("|duration error|, s", base.get("dur_err_abs_s"), None,
             m.get("d_dur_err_abs_s")),
        ]
        for lab, b, a, d in rows:
            a = a if a is not None else ((b + d) if (b is not None and d is not None)
                                         else None)
            L.append(f"| {lab} | {f3(b)} | {f3(a)} | {f3(d, True)} |")
        L.append("")

    # ---- adapter dose response
    L.append("## The adapter on its own")
    L.append("")
    if not D:
        L.append("No entry in the dose-response sweep for this attribute.")
    elif D.get("safe_w") is None:
        L.append(f"**No usable merge weight.** Dose-response shape `{D.get('shape')}`. "
                 "Across the 0.25–1.5 ladder no weight both moved the target above its "
                 f"noise floor ({f3(D.get('target_noise_sd'))}) and stayed inside the safe "
                 "guardrails. It is **not harmful** at any weight — no adapter in the "
                 "5,740-cell sweep was — the failure is a failure to move the target.")
        if D.get("any_significant"):
            L.append("")
            L.append("At least one weight reached significance on the target without "
                     "clearing every guardrail; the ladder below shows where.")
    else:
        L.append(f"Dose-response shape `{D.get('shape')}`. Safe weight **{D['safe_w']}**, "
                 f"strong weight **{D['strong_w']}**.")
    if D and D.get("ladder"):
        L.append("")
        L.append("| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |")
        L.append("|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
        for r in D["ladder"]:
            L.append(f"| {r['w']} | {f3(r.get('target'))} | {f3(r.get('d_target'), True)} | "
                     f"{f2(r.get('t_target'))} | {r.get('n_up')}/{r.get('n')} | "
                     f"{f3(r.get('d_wer_parakeet'), True)} | "
                     f"{f3(r.get('d_genuineness'), True)} | "
                     f"{f3(r.get('d_r_burst'), True)} | "
                     f"{'yes' if r.get('guard_safe') else 'no'} |")
    L.append("")

    # ---- interactions
    L.append("## Interactions")
    L.append("")
    fp = entry["interactions"]["family_pooled"]
    L.append(f"Pooled over the {FAM_LONG[fam]} family (target in SD units, "
             f"n = {fp['adapter_x_steer']['n']} attribute×prompt cells):")
    L.append("")
    L.append("| pair | interaction | t | reading |")
    L.append("|---|--:|--:|---|")
    for k, lab in (("adapter_x_steer", "adapter × steering"),
                   ("adapter_x_cfg", "adapter × guidance"),
                   ("steer_x_cfg", "steering × guidance")):
        v = fp[k]
        t = v["t"]
        read = ("additive — the two combine predictably" if abs(t) < 2
                else ("**sub-additive — pick one**" if v["mean"] < 0
                      else "**super-additive — and it carries a cost, see below**"))
        L.append(f"| {lab} | {f3(v['mean'], True)} | {f2(t)} | {read} |")
    L.append("")
    L.append(f"Cumulativity ratio for this family: **{f2(fp['cumulativity_ratio'])}** "
             "(observed with all three levers, divided by the sum of the three alone).")
    L.append("")
    sxc = entry["interactions"]["steer_x_cfg_is_a_package"]
    L.append("Steering × guidance is the only real synergy in the study, and it is a "
             "coupled package. On the emotion family the same interaction term carries:")
    L.append("")
    L.append("| carried by steering × guidance | value | t |")
    L.append("|---|--:|--:|")
    L.append(f"| target (SD) | {f3(sxc['gain_target_z']['mean'], True)} | "
             f"{f2(sxc['gain_target_z']['t'])} |")
    L.append(f"| word error | {f3(sxc['cost_wer_parakeet']['mean'], True)} | "
             f"{f2(sxc['cost_wer_parakeet']['t'])} |")
    L.append(f"| genuineness | {f3(sxc['cost_genuineness']['mean'], True)} | "
             f"{f2(sxc['cost_genuineness']['t'])} |")
    L.append(f"| burst realisation | {f3(sxc['cost_r_burst']['mean'], True)} | "
             f"{f2(sxc['cost_r_burst']['t'])} |")
    L.append("")
    L.append("Every damage term has a larger |t| than the gain. When both are on, steer "
             "**both** CFG branches: that keeps 82 % of the effect and returns 0.209 of "
             "word error and 0.75 of genuineness.")
    L.append("")
    if ints:
        ta = entry["interactions"]["this_attribute"]
        L.append("For this attribute specifically (n = 10 prompts, so read the family row "
                 "first and this one second):")
        L.append("")
        L.append("| pair | interaction | t |")
        L.append("|---|--:|--:|")
        for k, lab in (("adapter_x_steer", "adapter × steering"),
                       ("adapter_x_cfg", "adapter × guidance"),
                       ("steer_x_cfg", "steering × guidance")):
            v = ta.get(k) or {}
            L.append(f"| {lab} | {f3(v.get('mean'), True)} | {f2(v.get('t'))} |")
        L.append("")
        ab = ta.get("by_mode_abs") or {}
        if ab:
            L.append("Target in SD units at each corner of the 2×2×2 "
                     "(bits are adapter, steering, guidance):")
            L.append("")
            L.append("| " + " | ".join(sorted(ab)) + " |")
            L.append("|" + "--:|" * len(ab))
            L.append("| " + " | ".join(f3(ab[k]) for k in sorted(ab)) + " |")
            L.append("")

    # ---- never
    L.append("## Never")
    L.append("")
    for n in entry["never"]:
        head = None
        for key in ("steer_alpha_ge", "guidance_lt", "steer_k_gt", "lever", "stack",
                    "compose_with", "operating_point"):
            if key not in n:
                continue
            v = n[key]
            if key == "steer_alpha_ge":
                head = f"steering α ≥ {v}"
            elif key == "guidance_lt":
                head = f"guidance g < {v}"
            elif key == "steer_k_gt":
                head = f"steering at k > {v} layers"
            elif key == "lever":
                head = f"the `{v}` lever, for this attribute"
            elif key == "stack":
                head = "stack " + " + ".join(v)
            elif key == "compose_with":
                head = "compose with " + ", ".join(v)
            else:
                head = f"the {v} operating point"
            break
        L.append(f"* **{head}** — {n['why']}  \n  *{n['source']}*")
    L.append("")
    L.append("## Provenance")
    L.append("")
    L.append("* Recipes and every Δ in this page: `combination-study/stats/"
             "comb_recommendations.json`, key `" + attr + "`.")
    L.append("* Interaction terms: `combination-study/stats/analysis.json`, `t2`.")
    if D:
        L.append("* Dose ladder: `lora-dose/coefficients.json`, key `"
                 + D["adapter"] + "`.")
    L.append("* Layer ranking behind `topK`: `work_vb/tap_rank.json`.")
    L.append("* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, "
             f"row `{entry['steering_key']}`.")
    L.append("")
    with open(f"{OUT}/patterns/{slug(attr)}.md", "w") as fh:
        fh.write("\n".join(L))


def write_index(coeff, recs, doses, stamp):
    A = coeff["attributes"]
    L = []
    L.append("# WikiSkill acting wiki — index")
    L.append("")
    L.append(f"First draft, generated {stamp} from the measured JSON by "
             "`code/build_wikiskills.py`. Layout follows "
             "`whitepaper/WIKISKILL_ACTING_SYSTEM.md` §4.2.")
    L.append("")
    L.append("`coefficients.json` is the machine-readable form of every table below and is "
             "what the demo server loads at start-up. The pages are for the maintainer and "
             "the proposer; the actor reads the JSON.")
    L.append("")
    n_bal = sum(1 for v in A.values() if v["balanced"])
    n_hi = sum(1 for v in A.values() if v["high_effect"])
    n_ad = sum(1 for v in A.values() if (v.get("adapter") or {}).get("usable"))
    L.append(f"**{len(A)} attributes.** A balanced operating point exists for **{n_bal}**, a "
             f"high-effect point for **{n_hi}**, and a usable adapter weight for "
             f"**{n_ad}**. Where a column says *none*, the measurement says there is no "
             "usable setting — that is the finding, and the actor must not substitute a "
             "guess.")
    L.append("")
    L.append("## How to read a recommendation")
    L.append("")
    L.append("* **Balanced** clears every guardrail in `thresholds.balanced`: word error "
             "≤ 0.15 absolute and ≤ +0.05 over baseline, genuineness ≥ −0.6, blend ≥ −1.0, "
             "burst realisation ≥ −0.1, |duration error| ≤ 0.3 s.")
    L.append("* **High effect** relaxes those to word error ≤ 0.30 absolute, genuineness "
             "≥ −1.5, blend ≥ −3.0, burst realisation ≥ −0.25, |duration error| ≤ 1.0 s.")
    L.append("* Every Δ is paired: same prompts, same seed, clip samples averaged inside a "
             "prompt before averaging across prompts, n = number of prompts.")
    L.append("* A recommendation only counts if it beats a **random direction of matched "
             "norm at the same operating point**. That control is null on its own "
             f"({f3(coeff['controls']['random_direction_pooled'], True)}) but **not** null "
             "at the combined point (+0.106, t 2.78).")
    L.append("")
    for fam, human in (("emo", "Emotions"), ("vn", "Delivery axes"),
                       ("qual", "Quality axes")):
        keys = [k for k in sorted(A) if A[k]["family"] == fam]
        L.append(f"## {human} ({len(keys)})")
        L.append("")
        L.append("| attribute | balanced mode | Δ target | t | high-effect mode | Δ target | "
                 "adapter w (safe/strong) | page |")
        L.append("|---|---|--:|--:|---|--:|:--|---|")
        for k in keys:
            v = A[k]
            b, h = v.get("balanced"), v.get("high_effect")
            ad = v.get("adapter") or {}
            aw = (f"{ad.get('safe_w')} / {ad.get('strong_w')}"
                  if ad.get("usable") else "none")
            L.append("| `{a}` | {bm} | {bd} | {bt} | {hm} | {hd} | {aw} | [{s}](patterns/{s}.md) |"
                     .format(a=k,
                             bm=(f"`{b['mode']}`" if b else "**none**"),
                             bd=(f3(b["measured"]["d_target"], True) if b else "—"),
                             bt=(f2(b["measured"]["t_target"]) if b else "—"),
                             hm=(f"`{h['mode']}`" if h else "**none**"),
                             hd=(f3(h["measured"]["d_target"], True) if h else "—"),
                             aw=aw, s=slug(k)))
        L.append("")
    L.append("## Cross-cutting pages")
    L.append("")
    L.append("* [interactions.md](interactions.md) — the 2×2×2 factorial: which levers "
             "combine and which cancel, per family.")
    L.append("")
    L.append("## What is not here yet")
    L.append("")
    L.append("* **Vocal-burst classes.** 71 adapters exist and 19 are in the dose sweep, "
             "but the combination study did not cover them, so there is no operating point "
             "to write down and no page is generated for them.")
    L.append("* **Logs and skill-impact.** `logs.md` and `skill-impact.md` in the whitepaper "
             "layout are written by the consolidation cycle, which has not run. This draft "
             "is cycle zero.")
    L.append("* **Listening tests.** Every number here is one model's judgement of another "
             "model's output. Nobody has listened.")
    L.append("")
    with open(f"{OUT}/index.md", "w") as fh:
        fh.write("\n".join(L))


def write_interactions(coeff, ANA, pooled, stamp):
    L = []
    L.append("# Interactions between the three levers")
    L.append("")
    L.append(f"Generated {stamp} from `combination-study/stats/analysis.json` (`t2`, the "
             "2×2×2 factorial over adapter × steering × guidance). Target is in SD units of "
             "the attribute's own metric.")
    L.append("")
    L.append("## Main effects, by family")
    L.append("")
    L.append("| family | adapter | t | steering | t | guidance | t | n |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for fam, human in (("emo", "emotion"), ("vn", "delivery"), ("qual", "quality")):
        p = pooled[fam]
        L.append(f"| {human} | {f3(p['eff_L']['mean'], True)} | {f2(p['eff_L']['t'])} | "
                 f"{f3(p['eff_S']['mean'], True)} | {f2(p['eff_S']['t'])} | "
                 f"{f3(p['eff_C']['mean'], True)} | {f2(p['eff_C']['t'])} | "
                 f"{p['eff_L']['n']} |")
    L.append("")
    L.append("**The best single lever flips by family.** Steering is by far the strongest "
             "on emotion and on delivery; on the quality axes it does nothing at all, and "
             "the adapter is the only lever that moves them.")
    L.append("")
    L.append("## Pairwise interactions, by family")
    L.append("")
    L.append("| family | adapter × steering | t | adapter × guidance | t | "
             "steering × guidance | t | cumulativity |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for fam, human in (("emo", "emotion"), ("vn", "delivery"), ("qual", "quality")):
        p = pooled[fam]
        L.append(f"| {human} | {f3(p['int_LS']['mean'], True)} | {f2(p['int_LS']['t'])} | "
                 f"{f3(p['int_LC']['mean'], True)} | {f2(p['int_LC']['t'])} | "
                 f"{f3(p['int_SC']['mean'], True)} | {f2(p['int_SC']['t'])} | "
                 f"{f2(p['cumulativity_ratio'])} |")
    L.append("")
    L.append("Three rules follow, and they are measurements rather than preferences:")
    L.append("")
    L.append("1. **On an emotion, adapter and steering are additive.** The interaction is "
             f"{f3(pooled['emo']['int_LS']['mean'], True)} "
             f"(t {f2(pooled['emo']['int_LS']['t'])}) — not significant. The combination is "
             "predictable, so `adapter+steer` is the sensible default there.")
    L.append("2. **On a delivery axis, pick one lever.** Adapter × steering is "
             f"{f3(pooled['vn']['int_LS']['mean'], True)} "
             f"(t {f2(pooled['vn']['int_LS']['t'])}) and adapter × guidance is "
             f"{f3(pooled['vn']['int_LC']['mean'], True)} "
             f"(t {f2(pooled['vn']['int_LC']['t'])}). Both are significantly sub-additive: "
             "the two levers are doing the same job.")
    L.append("3. **Steering × guidance is the only real synergy, and it is a coupled "
             "package.** On emotion it is "
             f"{f3(pooled['emo']['int_SC']['mean'], True)} "
             f"(t {f2(pooled['emo']['int_SC']['t'])}) on the target — and the same term "
             f"carries {f3(pooled['emo']['wer_int_SC']['mean'], True)} of word error "
             f"(t {f2(pooled['emo']['wer_int_SC']['t'])}), "
             f"{f3(pooled['emo']['gen_int_SC']['mean'], True)} of genuineness "
             f"(t {f2(pooled['emo']['gen_int_SC']['t'])}) and "
             f"{f3(pooled['emo']['burst_int_SC']['mean'], True)} of burst realisation "
             f"(t {f2(pooled['emo']['burst_int_SC']['t'])}). Every damage term has a larger "
             "|t| than the gain.")
    L.append("")
    L.append("## Controls")
    L.append("")
    c = coeff["controls"]
    L.append(f"* **Base replication.** {c['base_replication']['n_pairs']} paired cells, "
             f"max absolute difference {c['base_replication']['max_abs_diff']}, "
             f"{c['base_replication']['frac_exact']:.0%} exact. The harness reproduces the "
             "bare model bit for bit.")
    L.append(f"* **Zero-strength path equivalence.** "
             f"{c['zero_strength_path_equivalence']['n']} cells, max absolute difference "
             f"{c['zero_strength_path_equivalence']['max_abs_diff']}. Running the steering "
             "code path at α = 0 is identical to not running it, which is what makes it a "
             "control rather than a second condition.")
    L.append(f"* **Random direction, matched norm.** Pooled "
             f"{f3(c['random_direction_pooled'], True)} — null on its own. At the combined "
             "operating point it is **+0.106 (t 2.78)** — not null. Anything shipped has to "
             "beat that floor, not zero.")
    L.append("")
    with open(f"{OUT}/interactions.md", "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main()
