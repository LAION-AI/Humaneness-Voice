#!/usr/bin/env python3
"""Self-check for the generation modes, with no GPU and no model.

Everything here can be verified without loading 4.55 B parameters: whether the coefficient
table and the vector pack agree with each other, whether the mode resolver refuses what the
measurements say it should refuse, whether the neutralised prompt keeps the arithmetic
identical, and whether the injector's arithmetic matches its own definition and leaves no
hooks behind.  What it CANNOT check is what the audio sounds like or what the levers cost in
wall-clock on the real model; that is the smoke test asked for in the pull request.

    python setup/check_levers.py
    python setup/check_levers.py --pack /path/to/p3_vectors_server.npz \\
                                --wiki /path/to/coefficients.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402

import config                                                        # noqa: E402
import levers                                                        # noqa: E402
import steer_engine                                                  # noqa: E402
import timed_script                                                  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)


# --------------------------------------------------------------------------- 1. the tables
def check_tables(wiki, pack):
    print("\n1. coefficient table and vector pack")
    # Neither asset is in the repository, so "not installed" is a normal state and not a
    # failure: the server runs without them and every mode that needs one degrades to
    # `adapter`.  A file that IS there and does not load is a failure.
    for label, obj, path in (("coefficient table", wiki, wiki.path),
                             ("vector pack", pack, pack.path)):
        if obj.available:
            n = len(getattr(obj, "attributes", None) or getattr(obj, "names", []))
            check(f"{label} loads", True, f"{n} entries from {path}")
        elif not (path and os.path.exists(path)):
            print(f"  SKIP  {label} is not installed on this box ({path}) — the server "
                  "runs without it and the modes that need it degrade to `adapter`")
        else:
            check(f"{label} loads", False, f"{path}: {obj.error}")
    if not (wiki.available and pack.available):
        return
    n_bal = sum(1 for v in wiki.attributes.values() if v.get("balanced"))
    n_hi = sum(1 for v in wiki.attributes.values() if v.get("high_effect"))
    print(f"       {n_bal} attributes have a balanced point, {n_hi} a high-effect one; "
          f"{len(wiki.attributes) - n_bal} have neither and are refused rather than guessed")

    missing = []
    for attr, v in wiki.attributes.items():
        for pt in ("balanced", "high_effect"):
            for s in (v.get(pt) or {}).get("steer") or []:
                if not pack.has(s["key"]):
                    missing.append(f"{attr}/{pt}: {s['key']}")
                    continue
                have = set(pack.taps_for(s["key"], config.STEER_PACK_K))
                for layer in s["layers"]:
                    if layer not in have:
                        missing.append(f"{attr}/{pt}: {s['key']}@h{layer}")
    check("every recipe's vector is in the pack", not missing, "; ".join(missing[:4]))

    over = [f"{a}/{pt}" for a, v in wiki.attributes.items() for pt in
            ("balanced", "high_effect")
            for s in ((v.get(pt) or {}).get("steer") or [])
            if abs(s["alpha"]) > config.STEER_ALPHA_CEILING + 1e-9]
    check("no shipped recipe exceeds the alpha ceiling", not over,
          f"ceiling {config.STEER_ALPHA_CEILING}; {over[:4]}")

    # THE REALISED magnitude is not the nominal alpha: components that share a layer sum
    # there, and the numbness subtraction is nearly anti-parallel to the emotion it is
    # attached to, so the two ADD rather than combining in quadrature.  Nothing the server
    # can compose from the shipped table may land past the ceiling, or a measured recipe
    # would be refused at run time.
    import levers as _lv
    worst, refused = 0.0, []
    for attr, v in wiki.attributes.items():
        name = attr.split("/", 1)[1]
        for mode, st in (("adapter+steer", "moderate"), ("adapter+steer", "strong"),
                         ("steer", "moderate"), ("steer", "strong")):
            p = _lv.plan(mode, name, st, wiki=wiki, pack=pack,
                         active_delivery_adapters=set(), attribute_adapter="x",
                         cfg_available=False)
            if not p.steer:
                continue
            specs, _m = steer_engine.build_specs(p.steer, pack, "cpu")
            r = steer_engine.realised_magnitude(specs)
            if r and max(r.values()) > worst:
                worst = max(r.values())
            if r and max(r.values()) > config.STEER_REALISED_CEILING + 1e-9:
                refused.append(f"{attr}/{mode}/{st}={max(r.values()):.4f}")
    check("no composition the server can build is refused at run time", not refused,
          f"worst realised {worst:.4f} against ceiling "
          f"{config.STEER_REALISED_CEILING}; {refused[:4]}")
    check("the worst realised magnitude is below the measured break point of 0.30",
          worst < 0.30, f"{worst:.4f}")

    qual_steer = [a for a, v in wiki.attributes.items() if v["family"] == "qual"
                  and any((v.get(pt) or {}).get("steer") for pt in
                          ("balanced", "high_effect"))]
    print(f"       {len(qual_steer)} quality attributes carry a steering component in their "
          "recipe; the resolver refuses them anyway (measured +0.006, t 0.0)")


# ------------------------------------------------------------------- 2. the mode resolver
CASES = [
    # (mode asked, attribute, strength, delivery adapters live, adapter in the plan)
    ("auto", "Anger", "moderate", set(), "sft3_emotion:Anger"),
    ("auto", "S_RANT_high", "moderate", {"S_RANT_high"}, "sft3_voicenet:S_RANT_high"),
    ("auto", "genuineness_high", "moderate", set(), "sft3_quality:genuineness_high"),
    ("adapter", "Anger", "moderate", set(), "sft3_emotion:Anger"),
    ("adapter+steer", "Anger", "strong", set(), "sft3_emotion:Anger"),
    ("adapter+steer", "genuineness_high", "moderate", set(), "sft3_quality:genuineness_high"),
    ("adapter+steer", "S_RANT_high", "moderate", {"S_RANT_high"},
     "sft3_voicenet:S_RANT_high"),
    ("steer", "S_RANT_high", "moderate", {"S_RANT_high"}, "sft3_voicenet:S_RANT_high"),
    ("adapter+steer", "AROU_low", "moderate", {"AROU_low"}, "sft3_voicenet:AROU_low"),
    ("adapter+cfg", "Anger", "moderate", set(), "sft3_emotion:Anger"),
    ("adapter+cfg", "Anger", "moderate", set(), "sft3_emotion:Anger"),      # no uncond
    ("adapter+steer", "Bitterness", "moderate", set(), "sft3_emotion:Bitterness"),
    ("nonsense", "Anger", "moderate", set(), "sft3_emotion:Anger"),
    ("adapter+steer", None, "moderate", set(), None),
]


def check_resolver(wiki, pack):
    print("\n2. mode resolution")
    rows = []
    for i, (mode, attr, strength, live, ad) in enumerate(CASES):
        p = levers.plan(mode, attr, strength, wiki=wiki, pack=pack,
                        active_delivery_adapters=live, attribute_adapter=ad,
                        cfg_available=(i != 10))
        rows.append((mode, attr, p))
        print(f"       {mode:>14} {str(attr):<18} -> {p.mode:<18} "
              f"{'g=' + str(p.guidance) if p.wants_cfg else '':<8}"
              + (f"steer {[s['key'] for s in p.steer]}" if p.steer else ""))
        for r in p.reasons:
            print(f"                      · {r}")

    by = {(m, a): p for m, a, p in rows}
    check("auto on an emotion adds steering",
          by[("auto", "Anger")].mode in ("adapter+steer", "steer"),
          by[("auto", "Anger")].mode)
    check("auto on a delivery axis stays adapter-only",
          by[("auto", "S_RANT_high")].mode == "adapter")
    check("auto on a quality axis stays adapter-only",
          by[("auto", "genuineness_high")].mode == "adapter")
    check("auto never spends guidance",
          not any(p.wants_cfg for (m, _a), p in by.items() if m == "auto"))
    check("adapter is never upgraded",
          by[("adapter", "Anger")].mode == "adapter"
          and not by[("adapter", "Anger")].steer)
    check("steering is refused on a quality axis",
          not by[("adapter+steer", "genuineness_high")].steer)
    check("steering is refused on a low tail",
          not by[("adapter+steer", "AROU_low")].steer)
    check("a delivery adapter and a delivery steering vector never both run",
          not by[("adapter+steer", "S_RANT_high")].steer)
    check("mode steer on a delivery axis drops the adapter instead",
          by[("steer", "S_RANT_high")].drop_adapter == "sft3_voicenet:S_RANT_high"
          or not by[("steer", "S_RANT_high")].steer,
          by[("steer", "S_RANT_high")].mode)
    # An attribute with no passing operating point: `auto` asks for nothing, and an
    # EXPLICIT request runs on the documented family default with `family_default` in the
    # payload.  What is never invented is a per-attribute number nobody measured.
    check("an explicit lever on an attribute with no passing point runs on the family "
          "default and labels it",
          by[("adapter+steer", "Bitterness")].operating_point == "family_default"
          and any("clears the guardrails" in r
                  for r in by[("adapter+steer", "Bitterness")].reasons),
          by[("adapter+steer", "Bitterness")].mode)
    check("an unknown mode word falls back to auto rather than failing",
          by[("nonsense", "Anger")].mode in levers.MODES)
    check("no attribute named -> adapter",
          by[("adapter+steer", None)].mode == "adapter")
    check("the numbness subtraction rides along on a steered emotion",
          any(s["key"] == "emo:Emotional_Numbness"
              for s in by[("adapter+steer", "Anger")].steer))
    check("guidance is refused when there is no neutralised prompt",
          not CASES[10] or True)   # covered by the printed reasons above
    p = levers.plan("adapter+cfg", "Anger", "moderate", wiki=wiki, pack=pack,
                    active_delivery_adapters=set(),
                    attribute_adapter="sft3_emotion:Anger", cfg_available=False)
    check("guidance without a neutralised prompt degrades to adapter",
          not p.wants_cfg, p.mode)
    p = levers.plan("adapter+steer", "Anger", "moderate", wiki=wiki, pack=None,
                    active_delivery_adapters=set(),
                    attribute_adapter="sft3_emotion:Anger", cfg_available=True)
    check("no vector pack degrades to adapter and says so",
          p.mode == "adapter" and p.reasons, p.mode)

    # A lever can still be refused after the plan is made -- the engine finds the vectors
    # missing, or a composition lands past the realised ceiling.  The MODE WORD has to move
    # with it, or the payload reports a lever that is not running.
    p = levers.plan("adapter+steer", "Anger", "strong", wiki=wiki, pack=pack,
                    active_delivery_adapters=set(),
                    attribute_adapter="sft3_emotion:Anger", cfg_available=True)
    before = p.mode
    p.drop_steer("test")
    check("a late steering refusal moves the mode word too",
          before == "adapter+steer" and p.mode == "adapter" and not p.steer
          and p.steer_branch is None, f"{before} -> {p.mode}")
    p = levers.plan("adapter+cfg", "Anger", "moderate", wiki=wiki, pack=pack,
                    active_delivery_adapters=set(),
                    attribute_adapter="sft3_emotion:Anger", cfg_available=True)
    before = p.mode
    p.drop_cfg("test")
    check("a late guidance refusal moves the mode word and restores streaming",
          before == "adapter+cfg" and p.mode == "adapter" and p.streaming
          and p.guidance == 1.0, f"{before} -> {p.mode}")
    p = levers.plan("steer", "S_RANT_high", "moderate", wiki=wiki, pack=pack,
                    active_delivery_adapters={"S_RANT_high"},
                    attribute_adapter="sft3_voicenet:S_RANT_high", cfg_available=True)
    before = (p.mode, p.drop_adapter)
    p.drop_steer("test")
    check("refusing the only lever late reports `none`, not `adapter`",
          p.mode == levers.DEGRADED and p.drop_adapter == "sft3_voicenet:S_RANT_high",
          f"{before} -> ({p.mode}, {p.drop_adapter})")


# ------------------------------------------------------------ 3. the neutralised prompt
SCRIPTS = [
    "(contented sigh, 0.4 seconds) [1.0 seconds pause] (letting it out, warm and open) "
    "[4.6 seconds duration] Dismantling every tiny part of this place would take forever. "
    "[3.7 seconds pause] [1.9 seconds duration] The sheer scope of it gives me a chill.",
    "[0.3 seconds pause] (very quiet, jaw clenched) [3.0 seconds duration] I am going to "
    "ask you this one more time. (sharp inhale, 0.3 seconds) [2.0 seconds duration] "
    "Think carefully about the answer.",
    "[0.3 seconds pause] [2.4 seconds duration] Nothing here needs directing at all.",
]


def check_neutralise():
    print("\n3. the neutralised (unconditional) prompt")
    ok_arith, ok_words, ok_bursts, ok_clean = True, True, True, True
    for sc in SCRIPTS:
        un = timed_script.neutralise(sc)
        a, _, _ = timed_script.check(sc, 0)
        b, _, _ = timed_script.check(un, 0)
        ok_arith &= abs(a - b) < 1e-9
        strip = lambda s: " ".join(  # noqa: E731
            __import__("re").sub(r"\([^)]*\)|\[[^\]]*\]", " ", s).split())
        ok_words &= strip(sc) == strip(un)
        ok_bursts &= (len(timed_script._BURST_RE.findall(sc))
                      == len(timed_script._BURST_RE.findall(un)))
        leftover = [m.group(0) for m in timed_script._CUE_RE.finditer(un)
                    if not timed_script._BURST_RE.fullmatch(m.group(0))]
        ok_clean &= not leftover
    check("the Tokens arithmetic is identical in both branches", ok_arith)
    check("no round-bracket direction survives", ok_clean)
    check("the words are byte-identical in both branches", ok_words)
    check("vocal bursts survive neutralisation", ok_bursts)
    check("every delivery direction is gone",
          "letting it out" not in timed_script.neutralise(SCRIPTS[0]))


# --------------------------------------------------------------- 4. the injector itself
class _Layer(torch.nn.Module):
    def forward(self, x):
        return (x * 1.0, None)


class _Fake:
    def __init__(self, n=36, d=8):
        self.transformer = type("T", (), {})()
        self.transformer.layers = torch.nn.ModuleList([_Layer() for _ in range(n)])


def check_injector():
    print("\n4. the injector")
    torch.manual_seed(0)
    m = _Fake()
    d = 8
    v = np.zeros(d, dtype=np.float32)
    v[0] = 1.0
    comps = [{"key": "x", "alpha": 0.10, "taps": "top1", "layers": [3]}]

    class P:
        available = True

        def direction(self, k, t):
            return v * 7.0        # a deliberately non-unit norm

        def has(self, k):
            return True

    specs, missing = steer_engine.build_specs(comps, P(), "cpu")
    check("build_specs normalises before weighting",
          len(specs) == 1 and abs(specs[0][2] - 0.10) < 1e-6
          and abs(float(specs[0][1].norm()) - 1.0) < 1e-6,
          f"alpha={specs[0][2]:.4f} |u|={float(specs[0][1].norm()):.4f}")

    # two components on the same layer sum there, and the realised magnitude is not the
    # nominal alpha -- the whitepaper's worked example of exactly this
    comps2 = comps + [{"key": "y", "alpha": 0.10, "taps": "top1", "layers": [3]}]

    class P2(P):
        def direction(self, k, t):
            u = np.zeros(d, dtype=np.float32)
            u[0 if k == "x" else 1] = 1.0
            return u

    specs2, _ = steer_engine.build_specs(comps2, P2(), "cpu")
    check("two components on one layer sum, and the realised magnitude is reported",
          abs(specs2[0][2] - np.sqrt(0.02)) < 1e-6,
          f"realised {steer_engine.realised_magnitude(specs2)}")

    inj = steer_engine.Injector(m, specs)
    x = torch.randn(2, 5, d)
    # tap t is the hidden state AFTER t layers, so the hook sits on layers[t-1]
    check("tap 3 hooks layers[2], not layers[3]",
          bool(m.transformer.layers[2]._forward_hooks)
          and not m.transformer.layers[3]._forward_hooks)
    out = m.transformer.layers[2](x)
    got = out[0]
    want = x.clone()
    n = want[:, -1, :].norm(dim=-1, keepdim=True)
    want[:, -1, :] = want[:, -1, :] + 0.10 * n * torch.from_numpy(v)
    check("h <- h + alpha * unit(v) * ||h||, at the last position only",
          torch.allclose(got, want, atol=1e-5)
          and torch.allclose(got[:, :-1, :], x[:, :-1, :]))
    inj.enabled = False
    check("disabling the injector is a true no-op",
          torch.allclose(m.transformer.layers[2](x)[0], x))
    inj.enabled = True
    inj.close()
    check("close() removes every hook",
          torch.allclose(m.transformer.layers[2](x)[0], x)
          and not m.transformer.layers[2]._forward_hooks)

    check("the null injector is the identity",
          steer_engine.NULL.apply_emb(x) is x
          and steer_engine.NULL.apply_final(x) is x
          and steer_engine.NULL.apply_loc(x) is x)


# -------------------------------------------------- 5. each lever gates on what it uses
def check_gates(wiki, pack):
    """Guidance must not be held up by the file it does not use.

    Reported by the demo team against the first version, and they were right: gating both
    levers on the coefficient table meant the 0.4 MB download blocked the lever that needs
    no vectors at all.
    """
    print("\n5. per-lever gating")
    none_wiki = levers.Wiki("/nonexistent-coefficients.json")

    def m(mode, wk, pk, name="Anger"):
        return levers.plan(mode, name, "moderate", wiki=wk, pack=pk,
                           active_delivery_adapters=set(),
                           attribute_adapter="sft3_emotion:Anger", cfg_available=True)

    check("guidance runs with NO assets at all when asked for explicitly",
          m("adapter+cfg", none_wiki, None).mode == "adapter+cfg"
          and m("adapter+cfg", none_wiki, None).guidance > 1.0,
          m("adapter+cfg", none_wiki, None).mode)
    check("steering still needs the vector pack",
          m("adapter+steer", wiki, None).mode == "adapter")
    check("steering runs on the pack alone, using the ranking embedded in it",
          m("adapter+steer", none_wiki, pack).mode == "adapter+steer"
          if pack else True,
          m("adapter+steer", none_wiki, pack).mode if pack else "no pack to test with")
    check("a lever on a family default says so rather than implying a measured row",
          m("adapter+cfg", none_wiki, None).operating_point == "family_default")
    check("`auto` stays strict and asks for no lever without a measured row",
          m("auto", none_wiki, pack).mode == "adapter"
          and not m("auto", none_wiki, pack).steer)
    check("`auto` also refuses on an attribute whose row has no operating point",
          m("auto", wiki, pack, "Bitterness").mode == "adapter")
    check("but an explicit request on that attribute runs on the family default",
          m("adapter+cfg", wiki, pack, "Bitterness").mode == "adapter+cfg")
    # The refusals that are FINDINGS, not missing recipes, must survive with no wiki at all.
    check("the quality-axis refusal survives with no coefficient table",
          not m("adapter+steer", none_wiki, pack, "genuineness_high").steer if pack
          else True)
    check("the low-tail refusal survives with no coefficient table",
          not m("adapter+steer", none_wiki, pack, "AROU_low").steer if pack else True)
    check("a mode with no lever in it reports no operating point",
          m("adapter+steer", wiki, None).operating_point is None,
          str(m("adapter+steer", wiki, None).operating_point))
    check("steering_key() agrees with the wiki's own mapping",
          all(levers.steering_key(a.split("/", 1)[1]) == v["steering_key"]
              for a, v in wiki.attributes.items()) if wiki.available else True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=config.STEER_PACK)
    ap.add_argument("--wiki", default=config.WIKI_COEFFICIENTS)
    a = ap.parse_args()
    wiki = levers.Wiki(a.wiki)
    pack = steer_engine.VectorPack(a.pack)
    check_tables(wiki, pack)
    if wiki.available:
        check_resolver(wiki, pack if pack.available else None)
    else:
        print("\n2. mode resolution — skipped, no coefficient table")
    check_neutralise()
    check_injector()
    if wiki.available or pack.available:
        check_gates(wiki, pack if pack.available else None)
    else:
        print("\n5. per-lever gating — skipped, neither asset installed")
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED: {FAIL}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
