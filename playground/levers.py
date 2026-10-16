"""Which levers this turn gets, and at what strength.

There are exactly three ways this stack can push a performance towards an attribute:

  **adapter**  a rank-16 LoRA at a merge weight.  What this server already does.
  **steer**    a difference-of-means direction added to the hidden state (`steer_engine.py`).
  **cfg**      classifier-free guidance on the delivery condition (`tts_engine.py`).

This module owns the policy: given what the director asked for and what the wiki says has
been measured for that attribute, it decides which levers actually run.  It decides nothing
by taste — every rule below is a measurement, and the one that produced it is named.

THE SHAPE OF THE EVIDENCE, in one table.  Main effects on the target, in SD units, from the
2x2x2 factorial (`combination-study/stats/analysis.json`, `t2.by_family`):

    family      adapter        steering       guidance
    emotion     +0.077 (2.8)   +0.384 (9.4)   +0.050 (1.8)
    delivery    +0.377 (7.1)   +0.614 (9.6)   +0.026 (0.4)
    quality     +0.399 (6.0)   +0.006 (0.0)   +0.062 (0.9)

Three rules fall straight out of it, and they are why `auto` resolves the way it does:

  * **Emotion: adapter and steering are additive** (interaction +0.038, t 1.36 — not
    significant), so `adapter+steer` is predictable and is the default there.
  * **Delivery: pick one.**  adapter x steering is -0.164 (t -3.7) and adapter x guidance is
    -0.125 (t -3.5); both are significantly sub-additive because the two levers are doing the
    same job.  The incumbent adapter wins by default (`config.DELIVERY_LEVER`).
  * **Quality: adapter only.**  Steering does nothing at all there (+0.006, t 0.0), and the
    quality directions break at k >= 2 layers.

And two rules about combining:

  * **Steering x guidance is the only real synergy, and it is a coupled package.**  +0.277
    (t 7.5) on the emotion target — while the same term carries +0.078 word error (t 11.8),
    -0.862 genuineness (t -21.5) and -0.070 burst realisation (t -6.7).  Every damage term
    has a larger |t| than the gain.  When both are on, both branches are steered, which keeps
    82 % of the effect and returns 0.209 of word error and 0.75 of genuineness.
  * **Subtracting `Emotional_Numbness` at -0.10 is free.**  It returns +0.60 of genuineness
    (t 9.64, on 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion,
    so it rides along automatically on any emotion turn that has the steering machinery
    running.

WHERE THE WIKI SAYS THERE IS NOTHING, THIS MODULE REFUSES.  Seven of the sixty attributes
have no configuration that clears the balanced guardrails.  For those the levers are not
offered at a guessed setting; the turn runs on the adapter and the payload says why.  That is
the whole point of the knowledge layer: an absent recommendation is a finding.
"""
import json
import os

import config

# ------------------------------------------------------------------ the five shipped modes
MODES = ("adapter", "adapter+steer", "adapter+cfg", "steer", "cfg")
# Not offered to the director: what a lever-only mode becomes if the engine
# refuses the lever after the attribute's adapter has already been dropped.
DEGRADED = "none"
# `auto` is not a mode, it is a request to resolve one from the family.
ALL_MODE_WORDS = ("auto",) + MODES

# What the director's three strength words mean.  Doses are never a free number the language
# model picks: docs/ADAPTERS.md §4.4 — "a hallucinated number cannot get through".
STRENGTHS = ("gentle", "moderate", "strong")


def family(name):
    """A name the director already knows -> which of the three families it belongs to."""
    if not name:
        return None
    if name in config.SFT3_VN_ADAPTERS:
        return "vn"
    if name in config.QUALITY_AXES:
        return "qual"
    if name in config.EMOTION_NAMES:
        return "emo"
    return None


def attr_key(name):
    fam = family(name)
    return f"{fam}/{name}" if fam else None


class Wiki:
    """The generated coefficient table, loaded once at start-up.

    Written by `wikiskills/code/build_wikiskills.py` in the research log from the measured
    JSON; see docs/LEVERS.md for where to get it.  Absent, every lever beyond the adapter is
    refused and says so — the server never invents a setting.
    """

    def __init__(self, path=None):
        self.path = path or config.WIKI_COEFFICIENTS
        self.available = False
        self.error = None
        self.data = {}
        self.attributes = {}
        try:
            with open(self.path) as fh:
                self.data = json.load(fh)
            self.attributes = self.data.get("attributes") or {}
            self.available = bool(self.attributes)
            g = self.data.get("global", {})
            print(f"[levers] wiki coefficients: {len(self.attributes)} attributes from "
                  f"{self.path} (schema {self.data.get('schema')})", flush=True)
            if g.get("cfg", {}).get("cost_factor"):
                print(f"[levers] guidance costs {g['cfg']['cost_factor']}x wall clock; "
                      "it is never a default", flush=True)
        except Exception as e:                      # noqa: BLE001 - never fail start-up
            self.error = f"{type(e).__name__}: {e}"
            print(f"[levers] no wiki coefficients ({self.error}); `auto` will not reach "
                  "for a lever, and an explicitly requested one runs on its measured "
                  "family default rather than this attribute's own row", flush=True)

    def get(self, name):
        k = attr_key(name)
        return self.attributes.get(k) if k else None

    @property
    def g(self):
        return self.data.get("global", {})


class Plan:
    """What actually runs, and why — the object the payload and the log line are built from."""

    def __init__(self):
        self.requested = "auto"
        self.mode = "adapter"
        self.attribute = None
        self.family = None
        self.strength = "moderate"
        self.operating_point = None       # balanced | high_effect | default
        self.steer = []                   # [{key, alpha, taps, layers}]
        self.steer_branch = None          # cond | both
        self.guidance = 1.0               # 1.0 == off; g = 1 cancels the uncond term exactly
        self.drop_adapter = None          # adapter name to remove for a lever-only mode
        self.reasons = []                 # every downgrade, in order
        self.realised_alpha = {}          # layer -> the alpha actually applied there
        self.streaming = True
        self.has_adapter = False          # is the attribute's own adapter in the plan

    # -- helpers ------------------------------------------------------------
    @property
    def wants_steer(self):
        return bool(self.steer)

    @property
    def wants_cfg(self):
        return float(self.guidance or 1.0) > 1.0001

    def note(self, msg):
        self.reasons.append(msg)

    # -- late downgrades ----------------------------------------------------
    # The engine can still refuse a lever after the plan is made -- the vectors turn out not
    # to be loaded, a composition lands past the realised ceiling, the neutralised prompt
    # came out empty.  When that happens the MODE WORD has to move too: a payload that says
    # `adapter+steer` while nothing is being steered is exactly the dial that reads a value
    # while the thing it names is switched off (docs/LEARNINGS.md).
    def _resolve_mode(self):
        if self.steer and self.wants_cfg:
            self.mode = "adapter+steer+cfg" if self.has_adapter else "steer+cfg"
        elif self.steer:
            self.mode = "adapter+steer" if self.has_adapter else "steer"
        elif self.wants_cfg:
            self.mode = "adapter+cfg" if self.has_adapter else "cfg"
        else:
            self.operating_point = None   # no lever ran, so no point was used
            # `drop_adapter` is a record of what the app already did to the adapter list,
            # not an instruction that can be taken back: by the time a lever is refused
            # inside the engine the adapter has been gone for some time.  So a lever-only
            # mode whose lever is then refused is NOT `adapter` -- it is `none`, and saying
            # so is the difference between a traceable take and a mystery.
            self.mode = "none" if self.drop_adapter else "adapter"

    def drop_steer(self, reason):
        if self.steer:
            self.steer = []
            self.note(reason)
        self.realised_alpha = {}
        if not self.steer:
            self.steer_branch = None
        self._resolve_mode()

    def drop_cfg(self, reason):
        if self.wants_cfg:
            self.guidance = 1.0
            self.streaming = True
            self.note(reason)
        if self.steer:
            self.steer_branch = "cond"
        self._resolve_mode()

    def payload(self):
        """Everything applied, in the response and in the log, so a bad take is traceable."""
        return {
            "mode": self.mode,
            "mode_requested": self.requested,
            "attribute": self.attribute,
            "family": self.family,
            "strength": self.strength,
            "operating_point": self.operating_point,
            "steer": [{"key": s["key"], "alpha": s["alpha"], "taps": s["taps"],
                       "layers": s["layers"]} for s in self.steer],
            "steer_branch": self.steer_branch,
            "realised_alpha": self.realised_alpha,
            "guidance": round(float(self.guidance), 3),
            "cost_factor": (self.g_cost if self.wants_cfg else 1.0),
            "streaming": self.streaming,
            "dropped_adapter": self.drop_adapter,
            "reasons": list(self.reasons),
        }

    g_cost = config.CFG_COST_FACTOR

    def log_line(self):
        bits = [f"mode={self.mode}"]
        if self.requested != self.mode:
            bits.append(f"(asked {self.requested})")
        if self.attribute:
            bits.append(f"attr={self.attribute}")
        if self.operating_point:
            bits.append(f"point={self.operating_point}")
        for s in self.steer:
            bits.append(f"steer {s['key']} a={s['alpha']:+.2f} @h{','.join(str(x) for x in s['layers'])}")
        if self.realised_alpha:
            bits.append("realised=" + ",".join(f"h{k}:{v}" for k, v in
                                               sorted(self.realised_alpha.items())))
        if self.wants_cfg:
            bits.append(f"g={self.guidance} branch={self.steer_branch or 'n/a'} "
                        f"cost={self.g_cost}x non-streaming")
        if self.drop_adapter:
            bits.append(f"-{self.drop_adapter}")
        return "[levers] " + "  ".join(bits) + (
            ("  | " + "; ".join(self.reasons)) if self.reasons else "")


def steering_key(name, fam=None):
    """A name the director knows -> the key in the steering-vector table.

    The wiki carries this per attribute, but the mapping is mechanical and the resolver has
    to work with no wiki at all, so it is computed here too.  Kept identical in behaviour to
    `dim_key_of` in the combination study, which is what chose the vectors that were scored.
    """
    fam = fam or family(name)
    if fam == "emo":
        return f"emo:{name}"
    if fam == "vn":
        if str(name).endswith("_high"):
            return f"vn:{name[:-5]}"
        if str(name).endswith("_low"):
            return f"vn:{name[:-4]}"
        return f"vn:{name}"
    if fam == "qual":
        return {"genuineness_high": "q:genuineness", "blend_high": "q:blend",
                "esthetics_high": "vn:ESTH"}.get(name)
    return None


def _steer_components(rec, wiki, name, fam, alpha_scale=1.0):
    """The wiki recipe's steering block, scaled, with the numbness subtraction attached."""
    out = []
    for s in (rec or {}).get("steer") or []:
        if not s.get("layers"):
            continue
        out.append({"key": s["key"],
                    "alpha": round(float(s["alpha"]) * alpha_scale, 4),
                    "taps": s.get("taps"),
                    "layers": list(s["layers"])})
    return out


def _default_steer(entry, name, fam, alpha):
    """A steering component for an attribute with no *steering* recipe to read.

    Two ways to get here, and both are an explicit request rather than `auto`: the wiki has
    an operating point for this attribute but it does not use steering, or there is no wiki
    row at all.  Neither number is invented -- alpha = 0.10 at the attribute's own top layer
    is the measured free setting from the steering study, and k comes from the same study's
    per-family finding.  The layers come from the ranking embedded in the vector pack, so
    this path needs the pack and nothing else.
    """
    key = (entry or {}).get("steering_key") or steering_key(name, fam)
    k = config.STEER_K.get(fam, 1)
    return {"key": key, "alpha": alpha, "taps": f"top{k}", "layers": None, "_k": k}


def _numbness(wiki, pack):
    n = (wiki.g.get("numbness_subtraction") or {}) if wiki.available else {}
    key = n.get("key", "emo:Emotional_Numbness")
    alpha = float(n.get("alpha", -0.10))
    k = 1
    layers = pack.taps_for(key, k) if pack is not None else []
    if not layers:
        return None
    return {"key": key, "alpha": alpha, "taps": f"top{k}", "layers": layers}


def plan(requested, name, strength, *, wiki, pack, active_delivery_adapters,
         attribute_adapter, cfg_available):
    """Resolve one turn's levers.

    `requested`               a word from ALL_MODE_WORDS
    `name`                    the attribute the director named (emotion, delivery adapter or
                              quality axis), or None
    `strength`                gentle | measured | strong
    `active_delivery_adapters`  {"S_RANT_high", ...} already planned for this turn
    `attribute_adapter`       the adapter name carrying `name`, if one is in the plan
    `cfg_available`           whether a neutralised instruction could be built for this turn
    """
    p = Plan()
    p.requested = str(requested or "auto")
    p.strength = strength if strength in STRENGTHS else "measured"
    fam = family(name)
    p.attribute = attr_key(name)
    p.family = fam

    if p.requested not in ALL_MODE_WORDS:
        p.note(f"unknown mode {p.requested!r}; using auto")
        p.requested = "auto"

    # ---- global kill switches -------------------------------------------
    want_steer = "steer" in p.requested or p.requested == "auto"
    want_cfg = "cfg" in p.requested or p.requested == "auto"
    if not config.STEER_ENABLED and want_steer:
        if p.requested != "auto":
            p.note("steering is switched off on this server (MOSS_STEER=0)")
        want_steer = False
    if not config.CFG_ENABLED and want_cfg:
        if p.requested != "auto":
            p.note("guidance is switched off on this server (MOSS_CFG=0)")
        want_cfg = False

    # `auto` never spends guidance: it costs CFG_COST_FACTOR and the first two levers reach
    # the band on their own for the great majority of attributes.
    if p.requested == "auto":
        want_cfg = False

    if not name:
        p.mode = "adapter"
        if p.requested not in ("auto", "adapter"):
            p.note("no attribute named this turn, so there is nothing for a lever to push")
        return p

    have_wiki = bool(wiki is not None and wiki.available)
    entry = wiki.get(name) if have_wiki else None

    # ---- pick the operating point ---------------------------------------
    if p.strength == "strong":
        point, rec = "high_effect", (entry or {}).get("high_effect")
        if rec is None:
            point, rec = "balanced", (entry or {}).get("balanced")
    else:
        point, rec = "balanced", (entry or {}).get("balanced")
        if rec is None:
            point, rec = "high_effect", (entry or {}).get("high_effect")

    # ---- EACH LEVER IS GATED ON WHAT IT ACTUALLY USES --------------------
    # Guidance needs no steering vectors and no per-attribute row: `g` has a family
    # default measured in the CFG study (3.0 for emotion, 2.5 for delivery, at word error
    # <= 0.20).  Steering needs a direction and the layers to put it at, which only the
    # vector pack has.  Gating them together meant the 0.4 MB file was holding up the lever
    # that does not need the 5.3 MB one -- reported by the demo team, and they were right.
    #
    # `auto` is the exception and stays strict.  It is a claim about what was MEASURED for
    # this attribute, so with no row to read it makes no claim and asks for no lever.  An
    # explicitly requested mode is an operator or a director overriding that, and it runs on
    # the documented family default with `operating_point: "family_default"` in the payload
    # and a reason line saying so.  A default from the study is not a guess; what is refused
    # is inventing a per-attribute number that nobody measured.
    explicit = p.requested != "auto"
    if rec is None:
        why = ("no wiki entry for this attribute" if have_wiki else
               "wiki coefficients unavailable "
               f"({getattr(wiki, 'error', 'not loaded')})") if entry is None else \
            "no measured operating point for this attribute clears the guardrails"
        if not explicit:
            p.mode = "adapter"
            p.note(why + "; auto makes no claim without one")
            return p
        p.note(why + "; running the explicitly requested lever on its documented family "
                     "default instead, which is a measured setting but not this "
                     "attribute's own")
        point = "family_default"
    p.operating_point = point if (rec or explicit) else None

    # ---- family policy ---------------------------------------------------
    if p.requested == "auto":
        if fam == "emo":
            want_steer = True
        else:
            # delivery: the two levers are sub-additive, and the adapter is the incumbent.
            # quality: steering does nothing at all.
            want_steer = False
            if fam == "vn":
                p.note("auto: delivery axis, adapter and steering are sub-additive "
                       "(-0.164, t -3.7) — one lever only")
            elif fam == "qual":
                p.note("auto: quality axis, steering does not move it (+0.006, t 0.0)")

    if want_steer and fam == "qual":
        want_steer = False
        p.note("steering refused on a quality axis: measured +0.006 (t 0.0), and the "
               "quality directions break at k >= 2 layers")
    if want_steer and fam == "vn" and str(name).endswith("_low"):
        want_steer = False
        p.note("steering refused on a low tail: the vector table holds the high-minus-low "
               "difference and the two tails are orthogonal (median cos -0.0004), so there "
               "is no measured route to this one")

    # ---- delivery: adapter and steering do the same job ------------------
    drop_attr_adapter = p.requested in ("steer", "cfg")
    if want_steer and fam == "vn" and active_delivery_adapters and not drop_attr_adapter:
        if config.DELIVERY_LEVER == "adapter":
            want_steer = False
            p.note(f"delivery adapter {sorted(active_delivery_adapters)} already in play; "
                   "not stacking a delivery steering vector on it (interaction -0.164, "
                   "t -3.7). config.DELIVERY_LEVER picks the winner.")
        else:
            drop_attr_adapter = True
            p.note("config.DELIVERY_LEVER=steer: dropping the delivery adapter rather than "
                   "stacking it with the steering vector")

    # ---- build the steering components -----------------------------------
    if want_steer:
        if pack is None or not pack.available:
            want_steer = False
            p.note("no steering vectors on this box; see docs/LEVERS.md for the pack "
                   + (f"({pack.error})" if pack is not None and pack.error else ""))
    if want_steer:
        alpha_scale = config.STRENGTH_ALPHA_SCALE.get(p.strength, 1.0)
        comps = _steer_components(rec, wiki, name, fam, alpha_scale)
        if not comps:
            # the wiki's point for this attribute does not use steering; fall back to the
            # attribute's own top-k layers at the family default alpha
            d = _default_steer(entry, name, fam, config.STEER_ALPHA * alpha_scale)
            layers = pack.taps_for(d["key"], d.pop("_k")) if d["key"] else []
            if layers:
                d["layers"] = layers
                comps = [d]
                p.note(("no steering recipe to read" if rec is None else
                        f"the {point} recipe for this attribute does not use steering")
                       + f"; applying the measured family default alpha {d['alpha']:+.2f} "
                         f"at its own top-{len(layers)} layers")
        comps = [c for c in comps if pack.has(c["key"])]
        if fam == "emo" and comps and config.NUMBNESS_SUBTRACTION != "off":
            nb = _numbness(wiki, pack)
            if nb and not any(c["key"] == nb["key"] for c in comps):
                comps.append(nb)
        # HARD CEILING.  Nominal alpha, before summation; the realised magnitude per layer is
        # checked separately once the specs are built.
        for c in comps:
            if abs(c["alpha"]) > config.STEER_ALPHA_CEILING + 1e-9:
                c["alpha"] = (config.STEER_ALPHA_CEILING
                              * (1 if c["alpha"] > 0 else -1))
                p.note(f"clamped {c['key']} to alpha {c['alpha']:+.2f} "
                       f"(ceiling {config.STEER_ALPHA_CEILING}, half the measured "
                       "break point of 0.3)")
        if not comps:
            want_steer = False
            p.note("the steering vectors for this attribute are not in the pack")
        else:
            p.steer = comps

    # ---- guidance ---------------------------------------------------------
    if want_cfg:
        if not cfg_available:
            want_cfg = False
            p.note("guidance needs a neutralised instruction and this turn has none "
                   "(the director wrote no separable delivery line)")
    if want_cfg:
        g = float((rec or {}).get("cfg", {}).get("g") or 0) or config.CFG_G.get(fam, 2.5)
        if g <= 1.0:
            g = config.CFG_G.get(fam, 2.5)
            p.note((f"the {point} recipe does not use guidance" if rec is not None else
                    "no guidance recipe to read")
                   + f"; applying the measured family default g = {g}")
        g = max(config.CFG_G_MIN, min(g, config.CFG_G_MAX))
        p.guidance = g
        p.streaming = False
        p.note(f"guidance costs {config.CFG_COST_FACTOR}x wall clock; at the measured "
               "realtime factor of 0.764 that is ~1.47, past the streaming budget of 1.0, "
               "so this take is rendered whole and then played")

    # ---- resolve the mode word -------------------------------------------
    p.has_adapter = bool(attribute_adapter) and not drop_attr_adapter
    if drop_attr_adapter and attribute_adapter:
        p.drop_adapter = attribute_adapter
    if p.steer:
        p.steer_branch = config.CFG_STEER_BRANCH if p.wants_cfg else "cond"
        if p.wants_cfg:
            p.note("both levers on: steering BOTH guidance branches, which keeps 82 % of "
                   "the effect and returns 0.209 of word error and 0.75 of genuineness")
    p._resolve_mode()
    if p.mode == "none":
        # Refused at PLAN time, before anything was removed from the adapter list, so the
        # adapter simply stays and this is an ordinary `adapter` turn.
        p.drop_adapter = None
        p.has_adapter = bool(attribute_adapter)
        p._resolve_mode()
    return p
