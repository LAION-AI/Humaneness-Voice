#!/usr/bin/env python3
"""The measured knowledge layer, rendered into something the director can read.

`wikiskills/` is a generated corpus: sixty attribute pages, a machine-readable
`coefficients.json` the lever planner already reads, and `VOCAL_BURSTS.md`, which
is the part that changes what the director should write.  This module turns the
burst tables into two things the server can act on — a per-class merge weight and
a prompt block naming the sounds that actually exist.

Why it matters more than a documentation update: the instructions this server was
shipping contradict the measurements in two places.

  * It told the director to place bursts **mid-sentence**.  Measured, mid-clause
    placement is worse on 15 of 15 classes (-0.07..-0.12 hit rate, miss rate
    +0.31..+0.37, t 8-10).  One inversion: `clears_throat` goes 0.250 -> 0.483.
  * It offered every burst that has an adapter.  Nineteen classes never realise
    at any dose under either prompt form, and seventeen more sit below the 0.15
    bar.  Every mouth class and every whistle class in the bank is on one of
    those lists.  Offering them produces a silent take and no warning.

And one thing it did not know: the useful merge weight is **per class** and runs
0.25 to 2.3, against the flat 0.25 / 0.5 shipped here.

The tables are parsed rather than transcribed, for the same reason the wiki is
generated rather than typed: a copy stops agreeing with its source the moment
either moves.
"""
import os
import re

import config

_NUM = re.compile(r"^-?[0-9]+(?:[.,][0-9]+)?$")


def _f(x):
    x = str(x).strip().replace("**", "").replace(",", ".")
    return float(x) if _NUM.match(x.replace(",", ".")) else None


def _rows(md, header_must):
    """Every pipe-table row under a header containing all of `header_must`."""
    out, cols = [], None
    for line in md.splitlines():
        if not line.strip().startswith("|"):
            cols = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        low = [c.lower() for c in cells]
        if cols is None:
            if all(any(m in c for c in low) for m in header_must):
                cols = cells
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        out.append(dict(zip(cols, cells)))
    return out


def _cls(cell):
    """`[`vb/chuckle`](patterns/vb-chuckle.md)` -> `chuckle`."""
    m = re.search(r"`([a-z]+)/([a-z0-9_]+)`", cell)
    if m:
        return m.group(2)
    m = re.search(r"`([a-z0-9_]+)`", cell)
    return m.group(1) if m else None


class Skills:
    def __init__(self, root=None):
        self.root = root or config.SKILLS_DIR
        self.ok = False
        self.recipes = {}          # class -> {w, form, hit, n, source}
        self.never = set()         # never realises at any dose
        self.weak = {}             # class -> hit rate, below the shipping bar
        p = os.path.join(self.root, "VOCAL_BURSTS.md")
        if not os.path.exists(p):
            print(f"[skills] no wikiskills at {self.root}", flush=True)
            return
        md = open(p, encoding="utf-8").read()
        self._parse_main(md)
        self._parse_merge(md)
        self._parse_lists(md)
        self._parse_arms()
        self.ok = bool(self.recipes)
        print(f"[skills] {len(self.recipes)} burst recipes, "
              f"{len(self.never)} classes that never realise, "
              f"{len(self.weak)} below the bar", flush=True)

    # ------------------------------------------------------------------ parse
    def _parse_main(self, md):
        for r in _rows(md, ["class", "weight", "hit"]):
            c = _cls(r.get("class", ""))
            w = _f(r.get("weight"))
            hit = _f(next((v for k, v in r.items() if "hit" in k.lower()), None))
            if not c or w is None:
                continue
            self.recipes[c] = {"w": w, "form": r.get("prompt form", "").strip(),
                               "hit": hit, "n": _f(r.get("N")) or 1,
                               "source": "§51/52"}

    def _parse_merge(self, md):
        """The §64 addendum: some recipes are superseded, most are not."""
        for r in _rows(md, ["klasse", "steht"]):
            c = _cls(r.get("Klasse", "") or r.get("klasse", ""))
            if not c:
                continue
            steht = (r.get("steht") or "").replace("*", "").strip().lower()
            newcell = next((v for k, v in r.items() if k.startswith("neu:")), "")
            parts = [p.strip() for p in newcell.split("/")]
            w = next((_f(p) for p in parts if _f(p) is not None), None)
            hit = _f(parts[-1]) if parts else None
            if steht.startswith("neu") and w is not None:
                cur = self.recipes.get(c, {})
                self.recipes[c] = {"w": w, "form": cur.get("form", ""),
                                   "hit": hit if hit is not None else cur.get("hit"),
                                   "n": _f(r.get("N")) or cur.get("n", 1),
                                   "source": "§64"}
            elif c not in self.recipes and w is not None:
                self.recipes[c] = {"w": w, "form": "", "hit": hit,
                                   "n": _f(r.get("N")) or 1, "source": "§64"}

    def _parse_lists(self, md):
        m = re.search(r"\*\*Never realised, at any dose:\*\*\s*\n\n?(.+?)\n\n", md, re.S)
        if m:
            self.never = set(re.findall(r"`([a-z0-9_]+)`", m.group(1)))
        m = re.search(r"\*\*Below the bar[^\n]*\*\*\s*\n\n?(.+?)\n\n", md, re.S)
        if m:
            for cls, val in re.findall(r"`([a-z0-9_]+)`\s*\(([0-9.]+)\)", m.group(1)):
                self.weak[cls] = float(val)

    # Which adapter set each recipe names.  The `bester Adapter` row of a class
    # page says whether the newly measured arm superseded the shipped one, and
    # that decision is per class -- 12 of the merge table's rows moved, the rest
    # stayed.  Serving one set for everything would override that judgement in
    # one direction or the other, so the arm is read per class.
    ARMS = {"bulk_mix_full": "burst_v2", "bulk_mix_top1": "burst_v2_top1",
            "grp_mix_full": "burst_grp", "grp_mix25": "burst_grp25",
            "shipped": "burst"}
    # Two classes name an ablation arm as their best.  Those adapters are filed
    # per arm rather than per class -- `ablation_d2_matched__scream` -- so they
    # need the name as well as the root, which `root_for` alone cannot express.
    ABL_ARMS = ("d2_matched", "d1_matched", "d2_full", "mix_full", "mix_matched",
                "real")

    def _parse_arms(self):
        import glob as _g
        pat = os.path.join(self.root, "patterns", "vb-*.md")
        for f in _g.glob(pat):
            cls = os.path.basename(f)[3:-3]
            try:
                txt = open(f, encoding="utf-8").read()
            except Exception:
                continue
            m = re.search(r"\|\s*bester Adapter\s*\|([^|]*)\|", txt)
            if not m:
                continue
            cell = m.group(1)
            new = "neu" in cell.lower()
            arm = None
            for a in self.ARMS:
                if f"`{a}`" in cell:
                    arm = a
                    break
            abl = None
            if arm is None and new:
                for a in self.ABL_ARMS:
                    if f"`{a}`" in cell:
                        abl = a
                        break
            if cls in self.recipes:
                self.recipes[cls]["arm"] = (self.ARMS.get(arm, "burst")
                                            if (new and arm) else
                                            ("burst_abl" if abl else "burst"))
                if abl:
                    self.recipes[cls]["adapter"] = f"ablation_{abl}__{cls}"

    def root_for(self, burst_class, default="burst"):
        """The adapter set this class's own recipe names."""
        r = self.recipes.get(str(burst_class or "").lower())
        return (r or {}).get("arm") or default

    def adapter_for(self, burst_class):
        """`root:name` for this class, when the recipe names a specific one."""
        c = str(burst_class or "").lower()
        r = self.recipes.get(c) or {}
        root = r.get("arm") or "burst"
        return f"{root}:{r.get('adapter') or c}"

    # ------------------------------------------------------------------ use
    def weight_for(self, burst_class, default=None):
        """The measured merge weight for this class, or the flat default."""
        r = self.recipes.get(str(burst_class or "").lower())
        return r["w"] if r else default

    def offerable(self, available):
        """Of the adapters on disk, the ones actually worth offering.

        Ordered by measured hit rate, because the director should reach for the
        reliable sounds first and the tail of the list is where the misses live.
        """
        av = {a.lower() for a in available}
        got = [(c, r) for c, r in self.recipes.items()
               if c in av and c not in self.never
               and (r.get("hit") or 0) >= config.SKILLS_MIN_HIT]
        got.sort(key=lambda kv: -(kv[1].get("hit") or 0))
        return got

    def prompt_block(self, available):
        """What the director is told about bursts, from the measurements."""
        got = self.offerable(available)
        if not got:
            return ""
        strong = [c for c, r in got if (r.get("hit") or 0) >= 0.40]
        rest = [c for c, r in got if (r.get("hit") or 0) < 0.40]
        gone = sorted(self.never | set(self.weak))
        L = ["\n\nVOCAL BURSTS — measured, not guessed. Every sound below has a "
             "trained adapter AND a measured hit rate; naming one in a "
             "round-bracket cue pulls its adapter in at the weight that was "
             "measured best for that class.",
             "",
             "USE THEM CONSTANTLY. Real speech is full of them and a reply "
             "without a breath, a laugh or a sigh sounds read rather than "
             "spoken. One in most replies, two or three when the moment is "
             "emotional.",
             "",
             "MOST RELIABLE — reach for these first (they land 40-75% of the time):",
             "  " + ", ".join(strong)]
        if rest:
            L += ["", "ALSO AVAILABLE, less reliable (15-40%):", "  " + ", ".join(rest)]
        if gone:
            L += ["",
                  "NEVER ASK FOR THESE. They do not exist in this voice: measured "
                  "across every dose and both prompt forms, they produce nothing "
                  "or something else. Asking yields a silent gap, not a sound. "
                  "Every mouth sound and every whistle is in this group:",
                  "  " + ", ".join(gone)]
        L += ["",
              "HOW TO WRITE THEM — each of these was measured on this model:",
              "  * WRITE EVERY CUE IN ENGLISH, even when the spoken line is "
              "German. This is not a style preference: it is how the training "
              "data is written. German corpus lines read "
              "\"Das zerreisst einen einfach, weisst du? (relief sigh)\" — "
              "German words, English cue. A German cue is out of distribution "
              "and behaves unpredictably. The words you speak stay in the "
              "user's language; only the brackets are English.",
              "  * Put the burst BETWEEN sentences, not inside one. Mid-clause "
              "placement is worse on 15 of 15 classes tested (hit rate -0.07 to "
              "-0.12, miss rate +0.31 to +0.37). The single exception is "
              "clears_throat, which is better mid-sentence.",
              "  * Name the CAUSE of the sound in your GENERAL line — what makes "
              "the character breathe in, laugh, sigh. Worth +0.026 hit rate on "
              "its own, and it composes with the next one.",
              "  * A burst that matters gets a longer stated duration. Worth "
              "+0.022, and together with the cause sentence +0.044.",
              "  * Write the sound, never the action that makes it. "
              "\"(chuckle)\" works; \"(he chuckles)\" degrades to silence "
              "(-0.08 to -0.11 hit rate, misses +0.12).",
              "  * Do not substitute a neighbour. Asking for a tired groan to "
              "get a frustrated one is measured as a harm, not a fallback.",
              "  * Never open or close the line on a burst; words must follow."]
        return "\n".join(L)


    def repair_script(self, script, available):
        """Give a burst named inside a delivery direction its own bracket.

        The failure this fixes, observed in a real turn: the director wrote
        `(a raw, tearing scream, completely overwhelmed by sudden shock)` and no
        scream was produced.  A round bracket without a number is a *direction* —
        an instruction about how to speak the next sentence — so the label inside
        it is prose, no `(scream, N.N seconds)` tag reaches the model, no time is
        budgeted for the sound, and no adapter is pulled.  `scream` had a recipe
        the whole time.

        Telling the director not to do this is already in the prompt and it did
        it anyway, so the server repairs it: the direction is KEPT, and a bare
        burst bracket for the label is inserted after it.  Additive, so a
        direction that also carries real delivery information does not lose it.

        The match is deliberately narrow — whole word, and only labels that have
        a measured recipe above the bar.  That is what stops
        `(spitting the words out)` from becoming a spit: `spitting` never
        realises, so it is not offerable, so it is not repaired.
        """
        import re as _re
        offer = {c for c, _ in self.offerable(available)}
        if not offer:
            return script, []
        # A label the director already wrote as its own bracket needs no repair,
        # and adding a second one puts the sound in twice.  Seen live: a
        # direction mentioning a scream sat two words from the director's own
        # `(scream)`, and both became bursts.
        import re as _re0
        # This has to see a burst however it is written.  It used to match only
        # brackets with no comma and no digit, which caught "(scream)" and
        # missed "(scream, 0.7 seconds)" -- and the prompt now asks the director
        # for exactly the second form.  So from the day the length was
        # introduced, a director who wrote "(scream, 0.7 seconds)" and then
        # mentioned the scream in the next direction got a SECOND bare
        # "(scream)" inserted, and the line opened on two screams.
        already = set()
        for _m in _re0.finditer(r"\(([^)]*)\)", str(script or "")):
            _lab = _re0.split(r",\s*[0-9]", _m.group(1))[0].strip().lower()
            if _lab and not _re0.search(r"[0-9]", _lab):
                already.add(_lab.replace(" ", "_"))
        # longest first, so "fearful gasp" wins over "gasp"
        labels = sorted(offer, key=len, reverse=True)
        out, added, pos = [], [], 0
        for m in _re.finditer(r"\(([^)]*)\)", str(script or "")):
            body = m.group(1)
            if _re.search(r"[0-9]", body):          # already a burst
                continue
            # A bracket the renderer already reads as a burst needs no repair.
            # Without this, "(short chuckle)" -- the length syntax the director
            # is now allowed to use -- had a second bare "(chuckle)" inserted
            # after it, and the line got two chuckles in a row.
            try:
                import timed_script as _ts
                if _ts._is_burst_label(body):
                    continue
            except Exception:
                pass
            low = body.lower()
            for lab in labels:
                spaced = lab.replace("_", " ")
                if lab in already or spaced.replace(" ", "_") in already:
                    continue
                if _re.search(rf"\b{_re.escape(spaced)}\b", low) and spaced != low.strip():
                    out.append((m.end(), spaced))
                    added.append(spaced)
                    break
        if not out:
            return script, []
        s2, last = [], 0
        for at, lab in out:
            s2.append(script[last:at])
            s2.append(f" ({lab})")
            last = at
        s2.append(script[last:])
        return "".join(s2), added


_CACHE = {}


def load():
    if "s" not in _CACHE:
        try:
            _CACHE["s"] = Skills()
        except Exception as e:
            print(f"[skills] unavailable: {e}", flush=True)
            _CACHE["s"] = None
    return _CACHE["s"]
