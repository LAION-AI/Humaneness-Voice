"""LoRA adapters, applied at runtime instead of merged.

The obvious approach is `weight += (B @ A) * alpha/r` before generating and the
inverse afterwards.  That does not survive contact with this demo: undoing the
merge accumulates bf16 error over hundreds of turns, keeping a pristine copy of
the base weights costs ~9 GB of VRAM on a card that is already 20/24 GB full, and
every turn would pay for hundreds of `B@A` products twice — directly on
time-to-first-audio.

So the factors stay factored.  A forward hook on each target Linear adds
`lam * (alpha/r) * (x @ Aᵀ) @ Bᵀ` to its output.  Swapping adapters is a pointer
change, stacking two adapters is a sum, and nothing about the base model is ever
written to.  These adapters are rank 32 against hidden 2560, so the extra work is
a couple of percent per token.

Doses come from the manual's measurements (see PLAN_LORA.md), never from the
language model:

    emotion alone        0.5   (band 0.35-0.75)
    burst mid-sentence   0.5
    burst isolated       0.75
    burst + emotion      burst 1.0, emotion at most half the burst dose
    character            0.75
"""
import glob, json, os, re, threading

import torch
from safetensors.torch import load_file

import config

# manual-derived doses
LAM_EMOTION = 0.5
LAM_EMOTION_UNDER_BURST = 0.5     # capped again against the burst dose at apply time
# Fallback burst doses, for a class the wiki has no measured recipe for and for
# runs with MOSS_SKILLS=0.  A measured class is re-dosed per class in app.py at
# the weight VOCAL_BURSTS.md records for it (0.25-2.3); see config.BURST_LAM.
# The "too much drifts the whole line towards the burst" reasoning that used to
# stand here was the genuineness gate talking, and that gate has been retired.
LAM_BURST_INLINE = config.BURST_LAM
LAM_BURST_SOLO = config.BURST_LAM_INTENSE
LAM_CHARACTER = 0.75
LAM_EXPLICIT = 1.0
LAM_SPORTS = 0.75

_PREFIXES = ("base_model.model.", "base_model.")


class LoraBank:
    """Holds adapters in CPU RAM, pushes the active ones to the GPU."""

    def __init__(self, model, device, max_gpu_adapters=8, max_cpu_adapters=None):
        self.model = model
        self.device = device
        self.max_gpu = max_gpu_adapters
        # The host cache was unbounded, and there are 65 GB of adapters on disk
        # (~33 GB as fp16).  A long session that exercises many dimensions and
        # bursts grows it until the kernel kills the process — which is exactly
        # what happened.  LRU, same as the GPU side.
        self.max_cpu = max_cpu_adapters or config.MAX_CPU_ADAPTERS
        self.cpu = {}          # name -> {module_path: (A, B, scale)}  on CPU
        self.cpu_order = []    # LRU
        self.gpu = {}          # name -> same, on device
        self.gpu_order = []    # LRU
        self.repos = {}        # name -> directory
        self._hooks = {}       # module_path -> handle
        self._active = {}      # module_path -> list[(A, B, gain)]
        self.lock = threading.Lock()
        self._modules = None

    # ------------------------------------------------------------- discovery
    def discover(self, roots):
        """roots: {kind: directory}.  Each adapter is a subdir with a safetensors."""
        n = 0
        for kind, root in roots.items():
            if not root or not os.path.isdir(root):
                continue
            for f in glob.glob(os.path.join(root, "*", "adapter_model.safetensors")) + \
                     glob.glob(os.path.join(root, "*", "*", "adapter_model.safetensors")):
                rel = os.path.relpath(os.path.dirname(f), root)
                name = f"{kind}:{rel.split(os.sep)[0]}"
                if name not in self.repos:
                    self.repos[name] = f
                    n += 1
        print(f"[lora] discovered {n} adapters across {len(roots)} sets", flush=True)
        return n

    def names(self, kind=None):
        return sorted(k.split(":", 1)[1] for k in self.repos
                      if kind is None or k.startswith(kind + ":"))

    # ------------------------------------------------------------- loading
    def _module_map(self):
        if self._modules is None:
            self._modules = dict(self.model.named_modules())
        return self._modules

    def load(self, name):
        """Parse one adapter into {module_path: (A, B, scale)} and keep it in RAM."""
        with self.lock:
            if name in self.cpu:
                if name in self.cpu_order:
                    self.cpu_order.remove(name)
                self.cpu_order.append(name)
                return self.cpu[name]
        path = self.repos.get(name)
        if not path:
            return None
        cfg_p = os.path.join(os.path.dirname(path), "adapter_config.json")
        scale = 1.0
        if os.path.exists(cfg_p):
            cfg = json.load(open(cfg_p))
            r, alpha = cfg.get("r"), cfg.get("lora_alpha")
            if r:
                scale = float(alpha or r) / float(r)
        sd = load_file(path)
        mods = self._module_map()
        out = {}
        for k in sd:
            if ".lora_A." not in k:
                continue
            bk = k.replace(".lora_A.", ".lora_B.")
            if bk not in sd:
                continue
            mp = k.split(".lora_A.")[0]
            for pre in _PREFIXES:
                if mp.startswith(pre):
                    mp = mp[len(pre):]
                    break
            if mp not in mods:
                continue
            out[mp] = (sd[k].to(torch.float16), sd[bk].to(torch.float16), scale)
        with self.lock:
            self.cpu[name] = out
            self.cpu_order.append(name)
            while len(self.cpu_order) > self.max_cpu:
                drop = self.cpu_order.pop(0)
                if drop not in self.gpu:          # still resident on the GPU: keep
                    self.cpu.pop(drop, None)
                else:
                    self.cpu_order.append(drop)
                    break
        print(f"[lora] loaded {name}: {len(out)} modules, scale {scale:g} "
              f"({len(self.cpu)} in RAM)", flush=True)
        return out

    def preload(self, names):
        for n in names:
            try:
                self.load(n)
            except Exception as e:
                print(f"[lora] preload failed {n}: {e}", flush=True)

    def _to_gpu(self, name):
        if name in self.gpu:
            self.gpu_order.remove(name); self.gpu_order.append(name)
            return self.gpu[name]
        w = self.load(name)
        if not w:
            return None
        g = {mp: (A.to(self.device), B.to(self.device), s) for mp, (A, B, s) in w.items()}
        self.gpu[name] = g
        self.gpu_order.append(name)
        while len(self.gpu_order) > self.max_gpu:
            drop = self.gpu_order.pop(0)
            self.gpu.pop(drop, None)
        return g

    # ------------------------------------------------------------- hooks
    def _ensure_hook(self, mp):
        if mp in self._hooks:
            return
        mod = self._module_map().get(mp)
        if mod is None:
            return
        state = self._active

        def hook(module, args, output, _mp=mp):
            terms = state.get(_mp)
            if not terms:
                return output
            x = args[0]
            delta = None
            for A, B, gain in terms:
                d = torch.nn.functional.linear(
                    torch.nn.functional.linear(x.to(A.dtype), A), B) * gain
                delta = d if delta is None else delta + d
            return output + delta.to(output.dtype)

        self._hooks[mp] = mod.register_forward_hook(hook, with_kwargs=False)

    # ------------------------------------------------------------- merging
    #
    # Hooks were the first design and they are elegant, but they cost 536 extra
    # kernel launches per token here: the streaming loop runs batch-1, one token
    # at a time, so launch overhead dwarfs the rank-32 arithmetic.  Measured
    # realtime factor went 0.737 -> 1.29, i.e. slower than playback, which breaks
    # streaming outright.  Folding the delta into the weights once per turn costs
    # a fixed sum up front and nothing per token.
    #
    # Undoing it is the catch.  Subtracting the recomputed delta is cheap but
    # rounds in bf16, so a pristine copy of every touched weight is kept in host
    # RAM (~9 GB of 110 GB free) and restored periodically to stop drift from
    # accumulating over a long session.
    RESYNC_EVERY = 25

    # Modules whose weight is shared with another module and therefore must not
    # be written to.  MossTTSLocal calls tie_weights() to make
    # `audio_lm_heads.N.weight` and `audio_embeddings.N.weight` the *same*
    # tensor, so folding a head delta into the weight silently rewrites the
    # audio embedding as well and corrupts the model — the DPO adapter's own
    # card measures both tensors moving by exactly 6.103515625e-05.  The twelve
    # head modules are therefore run as forward hooks instead, which is the same
    # arithmetic (h + scaling·B(A(x))) without touching any stored weight.
    #
    # Hooks were rejected for the model as a whole because 536 of them cost more
    # in kernel launches than the rank-32 arithmetic saves at batch 1.  Twelve is
    # not 536, and correctness is not optional.
    TIED = re.compile(r"(?:^|\.)audio_(?:lm_heads|embeddings)\.\d+$")

    def _tied_paths(self):
        """Every module whose weight tensor is SHARED with another module.

        The regex above is right for the twelve audio heads, but it is a list of
        names that has to be kept in step with the checkpoint by hand, and it
        misses the other tie the same model makes:

            self.text_lm_head.weight = self.transformer.embed_tokens.weight

        Nothing in this stack currently adapts `text_lm_head`, so that one is
        latent rather than live — but "latent" is what the audio-head tie was
        until an adapter set started targeting it, and a hand-maintained list
        cannot notice a new tie appearing in a future checkpoint.

        Asking the model is cheaper than remembering.  Two parameters are the
        same tensor exactly when they share storage, so group every weight by
        `data_ptr()` and take any group with more than one owner.  Computed once
        and cached; a few hundred pointer reads.
        """
        if getattr(self, "_tied_cache", None) is not None:
            return self._tied_cache
        by_ptr = {}
        for mp, m in self._module_map().items():
            w = getattr(m, "weight", None)
            if w is None or not hasattr(w, "data_ptr"):
                continue
            by_ptr.setdefault((w.data_ptr(), w.device), []).append(mp)
        tied = {mp for owners in by_ptr.values() if len(owners) > 1 for mp in owners}
        self._tied_cache = tied
        if tied:
            print(f"[lora] {len(tied)} modules share weights and will never be "
                  f"merged: {', '.join(sorted(tied)[:4])}"
                  f"{' …' if len(tied) > 4 else ''}", flush=True)
        return tied

    def _snapshot(self, paths):
        if not hasattr(self, "_pristine"):
            self._pristine = {}
            self._merges = 0
        mods = self._module_map()
        for mp in paths:
            if mp in self._pristine:
                continue
            m = mods.get(mp)
            if m is not None and hasattr(m, "weight"):
                self._pristine[mp] = m.weight.detach().to("cpu", copy=True)

    def _delta(self, A, B, gain):
        # W is [out, in]; A is [r, in], B is [out, r]
        return (B.float() @ A.float()).mul_(gain)

    def apply(self, specs):
        """Fold the requested adapters into the weights.  Returns what was used."""
        self.unapply()
        applied, touched = [], {}
        for name, lam in specs or []:
            if not name or lam <= 0:
                continue
            g = self._to_gpu(name)
            if not g:
                print(f"[lora] missing adapter {name}", flush=True)
                continue
            for mp, (A, B, s) in g.items():
                touched.setdefault(mp, []).append((A, B, s * float(lam)))
            applied.append((name, float(lam)))
        # tied weights cannot be merged; they are hooked instead
        # A module is hooked rather than merged if EITHER the name matches the
        # known-tied pattern OR the model itself reports the weight as shared.
        # The union is deliberate: the regex documents intent, the storage check
        # catches whatever the regex has not been told about yet.
        tied = self._tied_paths()
        hooked = {mp: t for mp, t in touched.items()
                  if self.TIED.search(mp) or mp in tied}
        touched = {mp: t for mp, t in touched.items() if mp not in hooked}
        # Belt and braces.  If a shared weight ever reaches the merge path the
        # damage is silent -- the model keeps generating, slightly wrong, and
        # nothing in the logs says so.  Fail loudly instead.
        assert not (set(touched) & tied), (
            "refusing to merge into shared weights: "
            f"{sorted(set(touched) & tied)}")
        if hooked:
            if not getattr(self, "_said_hooked", False):
                self._said_hooked = True
                print(f"[lora] {len(hooked)} tied modules run as hooks, not merged "
                      f"(shared weights: {sorted(hooked)[0]} …)", flush=True)
            for mp, terms in hooked.items():
                self._active[mp] = terms
                self._ensure_hook(mp)
        if not touched:
            self._merged = None
            return applied
        self._snapshot(touched.keys())
        mods = self._module_map()
        with torch.no_grad():
            for mp, terms in touched.items():
                m = mods.get(mp)
                if m is None or not hasattr(m, "weight"):
                    continue
                d = None
                for A, B, gain in terms:
                    dd = self._delta(A, B, gain)
                    d = dd if d is None else d + dd
                m.weight.add_(d.to(m.weight.dtype))
        self._merged = touched
        self._merges = getattr(self, "_merges", 0) + 1
        return applied

    def unapply(self):
        """Take the delta back out again.

        Normally by recomputing and subtracting it on the GPU — no PCIe traffic,
        same cost as the merge.  Every RESYNC_EVERY turns the pristine host copy
        is written back instead, so bf16 rounding cannot accumulate.
        """
        self._active.clear()          # drop any hooked (tied-weight) adapters
        touched = getattr(self, "_merged", None)
        if not touched:
            return
        mods = self._module_map()
        exact = (getattr(self, "_merges", 0) % self.RESYNC_EVERY) == 0
        pri = getattr(self, "_pristine", {})
        with torch.no_grad():
            for mp, terms in touched.items():
                m = mods.get(mp)
                if m is None or not hasattr(m, "weight"):
                    continue
                if exact and mp in pri:
                    m.weight.copy_(pri[mp].to(m.weight.device))
                    continue
                d = None
                for A, B, gain in terms:
                    dd = self._delta(A, B, gain)
                    d = dd if d is None else d + dd
                m.weight.sub_(d.to(m.weight.dtype))
        self._merged = None

    def clear(self):
        self.unapply()
        self._active.clear()

    def stats(self):
        return {"discovered": len(self.repos), "in_ram": len(self.cpu),
                "ram_cap": self.max_cpu,
                "on_gpu": len(self.gpu), "active_modules": len(self._active)}


# ------------------------------------------------------------------ policy
# Cue wording the model actually writes, mapped to the adapter families that
# exist.  The adapter names are terse ("smack_one_s_lips"), so matching only on
# them misses most natural direction; these are the words a director types.
# German too: the reply is written in the user's language, so the cues are as
# well, and an English-only table silently misses every German take.
_BURST_SYNONYMS_DE = {
    "chuckle": ("glucksen", "gluckst", "leises lachen", "schmunzelt"),
    "childlike_giggle": ("kichert", "kichern", "gekicher"),
    "nervous_giggle": ("nervoeses kichern", "verlegenes lachen"),
    "guffaw": ("lacht laut", "brüllt vor lachen"),
    "cackle": ("gackert", "boeses lachen"),
    "contented_sigh": ("zufriedener seufzer", "wohliger seufzer"),
    "relief_sigh": ("seufzer der erleichterung", "erleichtert"),
    "wistful_sigh": ("leiser seufzer", "wehmuetiger seufzer"),
    "exasperated_sigh": ("seufzer", "seufzt", "seufzend", "genervter seufzer",
                         "schwerer seufzer"),
    "sharp_inhale": ("scharfer atemzug", "zieht scharf die luft ein",
                     "scharf einatmend"),
    "surprised_gasp": ("keucht", "keuchen", "luft anhalten", "erschrocken"),
    "fearful_gasp": ("angstvolles keuchen", "erschrockenes keuchen"),
    "deep_breath": ("tiefer atemzug", "atmet tief", "holt tief luft"),
    "heavy_breathing": ("schweres atmen", "atmet schwer"),
    "panting": ("hechelt", "ausser atem"),
    "trembling_whimper": ("wimmern", "wimmert", "leises wimmern"),
    "quiet_sob": ("leises schluchzen", "unterdruecktes schluchzen"),
    "convulsive_sob": ("schluchzt", "schluchzen", "schluchzend"),
    "sobs": ("weint", "weinen", "unter traenen"),
    "mournful_wail": ("klagelaut", "jammert"),
    "scream": ("schreit", "schrei", "schreien"),
    "shriek": ("kreischt", "kreischen"),
    "growl": ("knurrt", "knurren", "grollt", "grollen"),
    "hiss": ("zischt", "zischen"),
    "snort": ("schnaubt", "schnauben", "veraechtliches schnauben", "snortet"),
    "effort_grunt": ("aechzt vor anstrengung", "stoehnt vor anstrengung"),
    "displeased_grunt": ("brummt", "grunzt", "missmutiges brummen"),
    "frustrated_groan": ("stoehnt", "stoehnen", "genervtes stoehnen"),
    "exhausted_groan": ("muedes stoehnen", "erschoepftes stoehnen"),
    "pain_moan": ("stoehnt vor schmerz", "schmerzlaut"),
    "resonant_hum": ("brummt zustimmend", "summt"),
    "soft_hum": ("leises summen", "nachdenkliches hmm", "hmm"),
    "humming": ("summt vor sich hin",),
    "yawn": ("gaehnt", "gaehnen"),
    "sniff": ("schnieft", "schnueffelt"),
    "cough": ("hustet", "husten"),
    "clears_throat": ("raeuspert sich", "raeuspern"),
    "lip_smack": ("schmatzt", "schmatzen"),
    "gulps": ("schluckt schwer", "schluckt"),
    "low_mumble": ("murmelt", "nuschelt", "vor sich hin"),
    "soft_whistle": ("pfeift leise",),
    "person_whistling_playfully": ("pfeift", "pfeifen"),
}

_BURST_SYNONYMS = {
    "chuckle": ("chuckle", "chuckling", "chuckles"),
    "snicker": ("snicker", "snickering", "smirk"),
    "guffaw": ("guffaw", "belly laugh", "roars with laughter", "howls"),
    "cackle": ("cackle", "cackling", "wicked laugh"),
    "breathy_giggle": ("breathy giggle", "breathless giggle"),
    "childlike_giggle": ("childlike giggle", "little giggle", "giggle", "giggling",
                         "giggles"),
    "nervous_giggle": ("nervous giggle", "nervous laugh", "awkward laugh"),
    "snorting_giggle": ("snorting giggle", "snort-laugh"),
    "contented_sigh": ("contented sigh", "happy sigh", "satisfied sigh"),
    "relief_sigh": ("sigh of relief", "relieved sigh", "breathes out in relief"),
    "wistful_sigh": ("wistful sigh", "longing sigh", "soft sigh"),
    "exasperated_sigh": ("exasperated sigh", "exasperated", "sighs hard",
                         "heavy sigh", "sigh", "sighs", "sighing"),
    "sharp_inhale": ("sharp inhale", "sharp breath in", "inhales sharply",
                     "hisses in a breath"),
    "surprised_gasp": ("gasp", "gasps", "gasping", "surprised gasp"),
    "fearful_gasp": ("fearful gasp", "frightened gasp", "terrified gasp"),
    "deep_breath": ("deep breath", "breathes in deeply", "steadying breath"),
    "heavy_breathing": ("heavy breathing", "breathing heavily"),
    "fast_breathing": ("fast breathing", "breathing fast", "hyperventilating"),
    "slow_breathing": ("slow breathing", "breathing slowly"),
    "panting": ("panting", "pants", "out of breath"),
    "trembling_whimper": ("whimper", "whimpers", "whimpering", "small whimper"),
    "quiet_sob": ("quiet sob", "stifled sob", "swallowed sob"),
    "convulsive_sob": ("sob", "sobs", "sobbing", "convulsive sob"),
    "sobs": ("crying", "cries", "in tears", "weeping"),
    "mournful_wail": ("wail", "wailing", "mournful cry"),
    "scream": ("scream", "screams", "screaming"),
    "shriek": ("shriek", "shrieks", "shrieking"),
    "growl": ("growl", "growls", "growling", "low growl"),
    "hiss": ("hiss", "hisses", "hissing"),
    "snort": ("snort", "snorts", "snorting", "derisive snort"),
    "effort_grunt": ("grunt of effort", "strains", "effort grunt"),
    "affirmative_grunt": ("affirmative grunt", "grunts in agreement", "mm-hm"),
    "displeased_grunt": ("displeased grunt", "disapproving grunt", "grunt"),
    "frustrated_groan": ("frustrated groan", "groan", "groans", "groaning"),
    "exhausted_groan": ("exhausted groan", "weary groan", "tired groan"),
    "pain_moan": ("moan of pain", "pained moan", "moans in pain"),
    "pleasure_moan": ("moan", "moans", "moaning", "moan of pleasure"),
    "resonant_hum": ("hum", "hums", "humming", "low hum", "resonant hum"),
    "soft_hum": ("soft hum", "quiet hum", "thoughtful hum", "knowing hum"),
    "humming": ("hums a tune", "humming to himself", "humming to herself"),
    "yawn": ("yawn", "yawns", "yawning"),
    "sniff": ("sniff", "sniffs", "wet sniff", "sniffle"),
    "cough": ("cough", "coughs"),
    "coughing": ("coughing fit", "coughing"),
    "clears_throat": ("clears throat", "clearing his throat",
                      "clearing her throat", "ahem"),
    "hiccup": ("hiccup", "hiccups"),
    "lip_smack": ("smacks lips", "lip smack", "smacking his lips"),
    "smack_one_s_lips": ("smacking lips", "licks lips"),
    "kissing_noises": ("kissing noise", "blows a kiss", "mwah"),
    "gulps": ("gulp", "gulps", "swallows hard"),
    "swallows": ("swallows", "swallowing"),
    "slurping_noises": ("slurp", "slurping"),
    "drinking_noises": ("drinking", "takes a drink", "sips"),
    "spitting": ("spits", "spitting"),
    "gurgling": ("gurgle", "gurgling"),
    "low_mumble": ("mumble", "mumbles", "mumbling", "under his breath"),
    "soft_whistle": ("soft whistle", "whistles softly", "low whistle"),
    "sharp_whistle": ("sharp whistle", "whistles sharply"),
    "wolf_whistle": ("wolf whistle", "appreciative whistle"),
    "person_whistling_playfully": ("whistling", "whistles a tune",
                                   "playful whistle"),
    "person_whistling_to_get_attention": ("whistles for attention",),
    "normal_breathing": ("breathing", "breathes"),
    "deep_breathing": ("deep breathing", "breathing deeply"),
}


# Noun forms and looser wordings the director actually writes.  Merged into the
# tables above rather than listed there, so the catalogue shown to the model
# stays one clean cue per adapter.
_EXTRA_EN = {'sharp_inhale': ('intake of breath', 'sharp intake', 'breath in'), 'trembling_whimper': ('small whimper', 'little whimper'), 'surprised_gasp': ('intake', 'sudden gasp'), 'deep_breath': ('breath', 'breathes'), 'soft_hum': ('hmm', 'mmh'), 'chuckle': ('laugh', 'laughs', 'laughing', 'little laugh', 'soft laugh')}
_EXTRA_DE = {'sharp_inhale': ('einatmen', 'scharfes einatmen', 'atemzug', 'luftholen'), 'trembling_whimper': ('wimmerer', 'gewimmer', 'winseln'), 'deep_breath': ('tiefes einatmen', 'atemholen'), 'surprised_gasp': ('keuchen', 'aufkeuchen', 'luftschnappen'), 'convulsive_sob': ('schluchzer', 'aufschluchzen'), 'chuckle': ('glucksen', 'schmunzeln'), 'childlike_giggle': ('kichern', 'gekicher'), 'snort': ('schnauben', 'schnauber'), 'frustrated_groan': ('stoehnen', 'gestoehn'), 'growl': ('knurren', 'grollen'), 'soft_hum': ('summen', 'hmm', 'brummen'), 'yawn': 'gaehnen', 'clears_throat': 'raeuspern', 'cough': 'husten', 'sniff': 'schniefen'}
for _t, _x in ((_BURST_SYNONYMS, _EXTRA_EN), (_BURST_SYNONYMS_DE, _EXTRA_DE)):
    for _k, _v in _x.items():
        _v = (_v,) if isinstance(_v, str) else tuple(_v)
        _t[_k] = tuple(_t.get(_k, ())) + _v


def burst_catalog(available):
    """The bursts that are actually installed, as director wording."""
    out = []
    for name in sorted(available):
        syn = _BURST_SYNONYMS.get(name)
        cue = syn[0] if syn else name.replace("_", " ")
        out.append((name, cue))
    return out


def detect_burst(script, available):
    """Find a burst cue in the SCRIPT and map it to an adapter we have.

    Prompt tags alone land a burst 23.6% of the time; with the matching adapter
    merged it is 71.9%, so it is worth resolving a cue even when the director did
    not name an adapter.  Longer synonyms win, so "fearful gasp" beats "gasp".
    """
    import re
    tags = " ".join(re.findall(r"\(([^)]*)\)", script or "")).lower()
    if not tags:
        return None
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        tags = tags.replace(a, b)
    avail = set(available)
    best, best_len = None, 0
    for name in avail:
        for syn in (_BURST_SYNONYMS.get(name, ())
                    + _BURST_SYNONYMS_DE.get(name, ())
                    + (name.replace("_", " "),)):
            if syn in tags and len(syn) > best_len:
                best, best_len = name, len(syn)
    return best


LAM_STYLE = 0.5           # voice-quality dimension stacked on top
_LEVEL_DIR = {"extremely_low": "low", "moderately_low": "low",
              "moderately_high": "high", "very_high": "high"}

# Each emotion is dosed from its own level, not from a shared budget: the total
# mass is allowed to exceed a single adapter's ceiling when the director really
# asks for two strong feelings at once.  Level 4 lands on 0.75, the top of the
# manual's band for one emotion; level 1 lands on 0.1875.
EMOTION_PER_LEVEL = 0.1875
EMOTION_MAX_EACH = 0.75
EMOTION_MAX_TOTAL = 2.0      # runaway guard only
MAX_EMOTIONS = 3
MAX_STYLE = 4


def vn_name(dim, direction):
    return f"voicenet:vn_{dim}__{direction}"


def plan_blend(mix, script, have_emotion, have_voicenet, have_burst):
    """A blend of emotions plus voice qualities, each at its own strength.

    `mix` is what the code language produced: [(kind, name, level 1-4), ...].
    Two feelings at once — embarrassment under amusement — needs both adapters
    merged, not one of them picked.  The emotion doses share a fixed budget
    weighted by their levels, so adding a second feeling dilutes rather than
    doubles the pull on the voice; the manual's ceiling for a single emotion is
    0.75 and stacking past it degrades the take.
    """
    specs = []
    emos = [(n, l) for k, n, l in mix if k == "emotion" and n in have_emotion]
    emos = emos[:MAX_EMOTIONS]
    if emos:
        running = 0.0
        for n, l in emos:
            lam = min(EMOTION_PER_LEVEL * l, EMOTION_MAX_EACH)
            if running + lam > EMOTION_MAX_TOTAL:
                lam = max(0.0, EMOTION_MAX_TOTAL - running)
            if lam >= 0.05:
                specs.append((f"emotion:{n}", round(lam, 3)))
                running += lam
    for k, n, l in mix:
        if k != "voicenet":
            continue
        name = vn_name(n, "high" if l >= 3 else "low")
        if name.split(":", 1)[1] in have_voicenet and \
                name not in [s[0] for s in specs]:
            specs.append((name, LAM_STYLE))
        if sum(1 for s in specs if s[0].startswith("voicenet:")) >= MAX_STYLE:
            break

    burst = detect_burst(script, have_burst)
    if burst:
        # The manual's measured recommendation for a burst sitting inside speech
        # is 0.5 — roughly a 50% landing rate per take while leaving ~90% of the
        # speech after it intact.  0.75 is the "knee" for a burst that carries a
        # whole line on its own, which is not what happens in a chat reply.
        solo = len((script or "").split()) < 10
        b_lam = LAM_BURST_SOLO if solo else LAM_BURST_INLINE
        specs.append((f"burst:{burst}", b_lam))
        # the manual: under a burst adapter the emotion sits at or below half its dose
        specs = [(n, min(l, b_lam / 2) if n.startswith("emotion:") else l)
                 for n, l in specs]
    return specs


def plan(voice, script, have_emotion, have_character, have_burst,
         have_voicenet=(), style=None):
    """Turn the agent's choice into a concrete adapter + dose list.

    The language model picks *which* condition; the doses are the manual's
    measured values, applied here so a hallucinated number cannot get through.
    """
    mode = (voice or {}).get("mode")
    specs = []
    burst = detect_burst(script, have_burst)

    if mode == "character":
        c = voice.get("character")
        if c and c in have_character:
            specs.append((f"character:{c}", LAM_CHARACTER))
    elif mode == "emotion":
        e = voice.get("emotion")
        if e and e in have_emotion:
            # under a burst adapter the emotion must sit at or below half its dose
            lam = LAM_EMOTION_UNDER_BURST if burst else LAM_EMOTION
            specs.append((f"emotion:{e}", lam))
    elif mode == "voicenet":
        d, lvl = voice.get("dimension"), voice.get("level")
        direction = _LEVEL_DIR.get(lvl)
        if d and direction:
            n = vn_name(d, direction)
            if n.split(":", 1)[1] in have_voicenet:
                specs.append((n, LAM_STYLE))
    elif mode == "sports":
        specs.append(("sports:r32_e2", LAM_SPORTS))

    # style adapters stack on top of whatever the mode picked — this is what lets
    # two takes of the same emotion sound like different performances
    for s in (style or [])[:2]:
        d, direction = (s or {}).get("dimension"), (s or {}).get("direction")
        if not d or direction not in ("high", "low"):
            continue
        n = vn_name(d, direction)
        if n.split(":", 1)[1] in have_voicenet and n not in [x[0] for x in specs]:
            specs.append((n, LAM_STYLE))

    if burst:
        # a burst that carries the whole line gets the higher "knee" dose;
        # one sitting inside speech gets the gentler one
        solo = len((script or "").split()) < 14
        b_lam = LAM_BURST_SOLO if solo else LAM_BURST_INLINE
        specs.append((f"burst:{burst}", b_lam))
        specs = [(n, min(l, b_lam / 2) if n.startswith("emotion:") else l)
                 for n, l in specs]
    return specs
