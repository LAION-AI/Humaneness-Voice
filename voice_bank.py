"""Reference-voice bank.

Holds the TTS-AGI/moss-voice-profile-references corpus in RAM as *pre-tokenised*
12-codebook MOSS codes, so a turn never pays for mp3 decode + codec encode: the
agent picks a condition, we hand the streaming model a tensor that is already
sitting in memory.

The corpus is a complete matrix over four query axes, which is exactly what the
language model gets to choose from (see llm_agent.VOICE_TOOL):

    emotion   x intensity(intense|moderate) x containment(free|contained)
    voicenet  x level(extremely_low..very_high)
    edge_case (screams, sobbing, laughter)
    character (dragon, murloc, sprightly-pixie, ...)
"""
import json, os, re, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor

import torch

import config


def _sh(cmd):
    return subprocess.run(cmd, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, check=False).returncode


class VoiceBank:
    def __init__(self, proc, device, variant=None, gpu_lock=None):
        self.proc = proc
        self.device = device
        # the generation lock, so background pre-tokenising never runs on the GPU
        # at the same time as a live turn — sharing the card pushed the measured
        # realtime factor from 0.74 to over 3
        self.gpu_lock = gpu_lock
        # "original" keeps the performance, "voice_converted" keeps the identity
        self.variant = variant or config.REF_VARIANT
        self.root = config.REF_DIR
        self.codes = {}                 # entry id -> LongTensor (frames, 12) on CPU
        self.entries = {}               # entry id -> dict
        self.index = {}                 # nested lookup tree from index.json
        self.schema = {}
        self._lock = threading.Lock()
        self._encode_lock = threading.Lock()
        self.ready = threading.Event()
        self.n_total = 0
        self._load_index()

    # ------------------------------------------------------------------ index
    def _load_index(self):
        # vc_sidon is not part of index.json — it ships as a separate best-take
        # selection, mapped gid -> path.  Upstream's own scoring puts it well
        # ahead of plain voice conversion: genuineness 0.617 vs 0.496 and burst
        # blend 3.00 vs 2.64, for 0.027 less speaker similarity.
        self.alt = {}
        ap = os.path.join(self.root, "index_vc_sidon.json")
        if self.variant == "vc_sidon" and os.path.exists(ap):
            self.alt = json.load(open(ap, encoding="utf-8"))
        ip = os.path.join(self.root, "index.json")
        sp = os.path.join(self.root, "schema.json")
        if os.path.exists(ip):
            self.index = json.load(open(ip, encoding="utf-8"))
            if os.path.exists(sp):
                self.schema = json.load(open(sp, encoding="utf-8"))
            self._flatten_from_index()
        else:
            # dataset still syncing -> fall back to parsing the filenames, which
            # carry the same axes (…__E__Anger__C__de.mp3).
            self._flatten_from_filenames()
        self._load_profile_indexes()
        self.anchor_codes = None
        self.speeds_ready = {1.0}
        self.n_total = len(self.entries)
        print(f"[voicebank] {self.n_total} reference conditions indexed "
              f"({'index.json' if self.index else 'filenames'})", flush=True)

    def _pick(self, leaf):
        """Path for this variant, falling back through the ones that exist."""
        if self.alt:
            hit = self.alt.get(leaf.get("id"))
            if hit:
                return hit
        return leaf.get(self.variant) or leaf.get("voice_converted") \
            or leaf.get("original")

    _GID_COND = {"A": ("intense", "free"), "B": ("moderate", "free"),
                 "C": ("intense", "contained"), "D": ("moderate", "contained")}

    def _load_profile_indexes(self):
        """Per-voice condition matrices, one file per profile.

        The corpus index.json only covers the voice the matrix was first
        rendered with.  Each other profile ships the same 842 conditions under
        its own gids, extracted separately, so a profile can be retrieved
        against its own recordings instead of borrowing another speaker's.
        """
        import glob as _g
        self.profiles = {}
        for f in _g.glob(os.path.join(self.root, "index_profile_*.json")):
            voice = os.path.basename(f)[len("index_profile_"):-len(".json")]
            try:
                gids = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            table = {}
            for gid, rel in gids.items():
                path = os.path.join(self.root, rel)
                if not os.path.exists(path):
                    continue
                parts = gid.split("|")
                if len(parts) < 4:
                    continue
                kind, lang = parts[1], parts[-1]
                e = {"id": gid, "path": path, "language": lang,
                     "profile_voice": voice}
                if kind == "E" and len(parts) >= 5:
                    inten, con = self._GID_COND.get(parts[3], ("intense", "free"))
                    e.update(block="emotion", emotion=parts[2],
                             intensity=inten, containment=con)
                elif kind == "V" and len(parts) >= 5:
                    e.update(block="voicenet", dimension=parts[2], level=parts[3])
                elif kind == "C":
                    e.update(block="character", character=parts[2])
                elif kind == "X":
                    e.update(block="edge_case", edge_case=parts[2])
                elif kind == "B":
                    e.update(block="burst", burst=parts[2])
                else:
                    e.update(block=parts[2] if len(parts) > 2 else kind)
                table[gid] = e
            # The corpus voice already has a matrix here, and it is the better
            # one: index.json points at the vc_sidon renderings, which upstream
            # scored well above the raw takes these per-profile indexes carry
            # (genuineness 0.617 vs 0.553).  Do not shadow it.
            if any(k.startswith(voice + "|") for k in self.entries):
                print(f"[voicebank] {voice}: corpus matrix already present, "
                      f"keeping it", flush=True)
                continue
            if table:
                self.profiles[voice] = table
                self.entries.update(table)      # so encode() can find them
        try:
            self._load_refs3()
        except Exception as e:
            print(f"[voicebank] refs3 unavailable: {e}", flush=True)
        if self.profiles:
            print("[voicebank] per-profile matrices: "
                  + ", ".join(f"{v}:{len(t)}" for v, t in
                              sorted(self.profiles.items())), flush=True)

    def _load_refs3(self):
        """The top three takes of every condition, per voice.

        The earlier extraction kept only the single highest-reward take per
        condition.  Three gives the retrieval something to choose between, and
        the corpus scores every take on itself, so the choice can be made on
        measured genuineness rather than on reward alone.  The best take keeps
        the plain gid; the runners-up are addressable as `gid#2` and `gid#3`.
        """
        import glob as _g
        root = getattr(config, "REF3_DIR", "")
        if not root or not os.path.isdir(root):
            return
        n = 0
        for f in _g.glob(os.path.join(root, "index_*.json")):
            voice = os.path.basename(f)[len("index_"):-len(".json")]
            try:
                gids = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            table = self.profiles.get(voice)
            for gid, rels in gids.items():
                base = self.entries.get(gid)
                for i, rel in enumerate(rels):
                    path = os.path.join(root, rel)
                    if not os.path.exists(path):
                        continue
                    eid = gid if i == 0 else f"{gid}#{i+1}"
                    if base is not None:
                        e = dict(base)
                        e["id"], e["path"] = eid, path
                    else:
                        parts = gid.split("|")
                        e = {"id": eid, "path": path, "profile_voice": voice,
                             "language": parts[-1] if len(parts) > 1 else "en"}
                    self.entries[eid] = e
                    if table is not None:
                        table[eid] = e
                    n += 1
        if n:
            print(f"[voicebank] refs3: {n} takes registered (top 3 per condition)",
                  flush=True)

    def has_matrix(self, voice):
        return bool(self.profiles.get(voice))

    def _add(self, eid, **kw):
        rel = kw.get("audio")
        if not rel:
            return
        path = os.path.join(self.root, rel)
        if not os.path.exists(path):
            return
        kw["id"] = eid
        kw["path"] = path
        self.entries[eid] = kw

    def _flatten_from_index(self):
        ix = self.index
        for emo, by_int in ix.get("emotion", {}).items():
            for inten, by_con in by_int.items():
                for con, by_lang in by_con.items():
                    for lang, leaf in by_lang.items():
                        self._add(leaf["id"], block="emotion", emotion=emo,
                                  intensity=inten, containment=con, language=lang,
                                  text=leaf.get("text", ""),
                                  duration_s=leaf.get("duration_s"),
                                  audio=self._pick(leaf))
        for dim, by_lvl in ix.get("voicenet_dimension", {}).items():
            for lvl, by_lang in by_lvl.items():
                for lang, leaf in by_lang.items():
                    self._add(leaf["id"], block="voicenet", dimension=dim, level=lvl,
                              language=lang, text=leaf.get("text", ""),
                              duration_s=leaf.get("duration_s"),
                              audio=self._pick(leaf))
        for ec, by_lang in ix.get("edge_case", {}).items():
            for lang, leaf in by_lang.items():
                self._add(leaf["id"], block="edge_case", edge_case=ec, language=lang,
                          text=leaf.get("text", ""), duration_s=leaf.get("duration_s"),
                          audio=self._pick(leaf))
        for ch, by_lang in ix.get("character", {}).items():
            for lang, leaf in by_lang.items():
                self._add(leaf["id"], block="character", character=ch, language=lang,
                          text=leaf.get("text", ""), duration_s=leaf.get("duration_s"),
                          audio=self._pick(leaf))
        for other, by_lang in ix.get("other", {}).items():
            for lang, leaf in by_lang.items():
                self._add(leaf["id"], block=other, language=lang,
                          text=leaf.get("text", ""), duration_s=leaf.get("duration_s"),
                          audio=self._pick(leaf))

    _COND = {"A": ("intense", "free"), "B": ("moderate", "free"),
             "C": ("intense", "contained"), "D": ("moderate", "contained")}

    def _flatten_from_filenames(self):
        d = os.path.join(self.root, "audio", self.variant)
        if not os.path.isdir(d):
            d = os.path.join(self.root, "audio", "original")
        if not os.path.isdir(d):
            return
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".mp3"):
                continue
            parts = fn[:-4].split("__")
            rel = os.path.relpath(os.path.join(d, fn), self.root)
            eid = "|".join(parts)
            kind, lang = (parts[1] if len(parts) > 1 else ""), parts[-1]
            if kind == "E" and len(parts) >= 5:
                inten, con = self._COND.get(parts[3], ("intense", "free"))
                self._add(eid, block="emotion", emotion=parts[2], intensity=inten,
                          containment=con, language=lang, audio=rel)
            elif kind == "V" and len(parts) >= 5:
                self._add(eid, block="voicenet", dimension=parts[2], level=parts[3],
                          language=lang, audio=rel)
            elif kind == "C":
                self._add(eid, block="character", character=parts[2], language=lang, audio=rel)
            elif kind == "X":
                self._add(eid, block="edge_case", edge_case=parts[2], language=lang, audio=rel)
            elif kind == "S":
                self._add(eid, block=parts[2], language=lang, audio=rel)

    # ------------------------------------------------------- audio -> codes
    def _wav_for(self, entry):
        """mp3 -> 48 kHz mono wav (cached).  Keeps the codec off the mp3 decoder."""
        os.makedirs(config.WAV_CACHE, exist_ok=True)
        # the id is shared by both variants of a condition, so it must be part of
        # the cache key or the two would silently collide
        out = os.path.join(config.WAV_CACHE, self.variant + "__" +
                           entry["id"].replace("|", "__").replace("/", "_") + ".wav")
        if not os.path.exists(out) or os.path.getsize(out) == 0:
            rc = _sh(["ffmpeg", "-nostdin", "-y", "-i", entry["path"],
                      "-ac", "1", "-ar", "48000", out])
            if rc != 0 or not os.path.exists(out):
                return None
        return out

    def _stretched_wav(self, entry, speed):
        """A tempo-shifted copy of a reference, pitch preserved.

        "say that faster" should swap the reference for a faster take of the same
        performance, not just ask the model to hurry.  audiostretchy's ratio is a
        duration factor, so it is the inverse of the speed we want.
        """
        base = self._wav_for(entry)
        if base is None:
            return None
        if abs(speed - 1.0) < 1e-6:
            return base
        out = base[:-4] + f"__x{speed:g}.wav"
        if not os.path.exists(out) or os.path.getsize(out) == 0:
            try:
                from audiostretchy.stretch import stretch_audio
                stretch_audio(base, out, ratio=1.0 / speed)
            except Exception as e:
                print(f"[voicebank] stretch {speed} failed: {e}", flush=True)
                return None
        return out if os.path.exists(out) else None

    def _code_path(self, eid, speed):
        import hashlib
        h = hashlib.sha1(f"{eid}|{speed:.3f}".encode()).hexdigest()
        d = os.path.join(config.CODE_CACHE, h[:2])
        return os.path.join(d, h[2:] + ".npy")

    def encode(self, eid, speed=1.0):
        """Return pre-tokenised codes for one condition at one tempo.

        Cached in RAM and on disk.  The disk half matters more than it looks:
        without it every restart re-tokenised the whole corpus, and any clip the
        retrieval reached for that the preload had not got to yet paid a full
        mp3 decode plus a GPU codec pass — queued behind the preload's own
        encodes, which is how one turn spent 60 s picking its reference.  The
        codes themselves are tiny, a few hundred integers per clip.
        """
        key = (eid, round(float(speed), 3))
        with self._lock:
            hit = self.codes.get(key)
        if hit is not None:
            return hit
        entry = self.entries.get(eid)
        if entry is None:
            return None
        cp = self._code_path(eid, round(float(speed), 3))
        if os.path.exists(cp):
            try:
                import numpy as _np
                t = torch.from_numpy(_np.load(cp)).to(torch.long)
                with self._lock:
                    self.codes[key] = t
                return t
            except Exception:
                pass
        wav = self._stretched_wav(entry, float(speed))
        if wav is None:
            return None
        try:
            # The codec is a single shared GPU module -> serialise access.
            # It must be inference_mode, not no_grad: generation runs under
            # inference_mode and leaves the codec's internal buffers as inference
            # tensors, which a later no_grad in-place update refuses to touch.
            import contextlib
            gate = self.gpu_lock or contextlib.nullcontext()
            with gate, self._encode_lock, torch.inference_mode():
                c = self.proc.encode_audios_from_path([wav], n_vq=12)[0]
                t = torch.as_tensor(c).to(torch.long).cpu()
            t = t.clone()          # leave inference-tensor land; this is kept in RAM
        except Exception as e:
            print(f"[voicebank] encode failed {eid}: {e}", flush=True)
            return None
        with self._lock:
            self.codes[key] = t
        try:
            import numpy as _np
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            _np.save(cp, t.numpy().astype("int16"))
        except Exception as e:
            print(f"[voicebank] code cache write failed {eid}: {e}", flush=True)
        return t

    def anchor(self, src=None):
        """A speaker's own recording, pre-tokenised and cached per path.

        Prepending it to the reference list gives the model the unchanging
        identity alongside the condition clip that carries the delivery, which is
        the manual's anchor idea applied to a single turn.
        """
        src = src or os.path.join(self.root, "reference", "reference_target.mp3")
        if not hasattr(self, "_anchors"):
            self._anchors = {}
        if src in self._anchors:
            return self._anchors[src]
        if not os.path.exists(src):
            return None
        e = {"id": "_anchor_" + os.path.basename(os.path.dirname(src)) + "_"
             + os.path.basename(src), "path": src}
        wav = self._wav_for(e)
        if wav is None:
            return None
        try:
            with self._encode_lock, torch.inference_mode():
                c = self.proc.encode_audios_from_path([wav], n_vq=12)[0]
                t = torch.as_tensor(c).to(torch.long).cpu()
            self._anchors[src] = t.clone()
        except Exception as ex:
            print(f"[voicebank] anchor encode failed: {ex}", flush=True)
            return None
        print(f"[voicebank] anchor pre-tokenised {tuple(self._anchors[src].shape)} "
              f"({os.path.basename(src)})", flush=True)
        return self._anchors[src]

    def preload(self, langs=None, workers=4):
        """Warm every condition into RAM in the background."""
        langs = langs or config.PRELOAD_LANGS
        # Only the corpus matrix is warmed eagerly.  With ten per-profile
        # matrices in the table that would be 8,400 clips times five tempi;
        # they are cheap to encode on demand (~0.1 s) and cached after first use.
        # The corpus voice, plus the profile actually being spoken.  Skipping
        # every profile matrix meant nothing the retrieval could reach for was
        # ever warm, so each turn paid a decode-and-encode for its own clip.
        want_voice = getattr(config, "DEFAULT_PROFILE", "")
        todo = [e for e in self.entries.values()
                if e.get("language") in langs
                and "#" not in str(e.get("id", ""))
                and (not e.get("profile_voice")
                     or e.get("profile_voice") == want_voice)]

        def run():
            t0 = time.time()
            # decode mp3 -> wav in parallel (pure CPU), then encode serially on GPU
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(self._wav_for, todo))
            for i, e in enumerate(todo):
                self.encode(e["id"])
                if (i + 1) % 100 == 0:
                    print(f"[voicebank] pre-tokenised {i+1}/{len(todo)}", flush=True)
            self.ready.set()
            print(f"[voicebank] {len(self.codes)} conditions pre-tokenised in "
                  f"{time.time()-t0:.1f}s", flush=True)
            # tempo variants come after, so normal-speed replies are available
            # within ~90 s and "say that faster" gets its takes a few minutes later
            for sp in config.SPEEDS:
                if sp == 1.0:
                    continue
                t1 = time.time()
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    list(ex.map(lambda e, s=sp: self._stretched_wav(e, s), todo))
                for e in todo:
                    self.encode(e["id"], sp)
                self.speeds_ready.add(sp)
                print(f"[voicebank] speed x{sp:g} ready ({time.time()-t1:.0f}s, "
                      f"{len(self.codes)} takes cached)", flush=True)
            print("[voicebank] all tempo variants ready", flush=True)

        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------- retrieval
    def select(self, sel, language="en", speed=1.0, voice=None):
        """Resolve the language model's tool call to one reference condition.

        `sel` mirrors VOICE_TOOL: {"mode": ..., plus the axis fields}.
        Falls back gracefully so a hallucinated value still yields a voice.
        """
        if not sel:
            return None
        mode = (sel.get("mode") or "none").lower()
        if mode in ("none", "default", ""):
            return None
        lang = "de" if str(language).lower().startswith("de") else "en"

        pool = self.profiles.get(voice) if voice else None
        pool = pool.values() if pool else self.entries.values()

        def find(**must):
            cands = []
            for e in pool:
                if e.get("language") != lang:
                    continue
                if all(str(e.get(k, "")).lower() == str(v).lower()
                       for k, v in must.items() if v):
                    cands.append(e)
            return cands

        cands = []
        if mode == "emotion":
            cands = find(block="emotion", emotion=sel.get("emotion"),
                         intensity=sel.get("intensity"),
                         containment=sel.get("containment"))
            if not cands:      # relax the modifiers before giving up on the emotion
                cands = find(block="emotion", emotion=sel.get("emotion"))
        elif mode == "voicenet":
            cands = find(block="voicenet", dimension=sel.get("dimension"),
                         level=sel.get("level"))
            if not cands:
                cands = find(block="voicenet", dimension=sel.get("dimension"))
        elif mode == "character":
            cands = find(block="character", character=sel.get("character"))
        elif mode == "edge_case":
            cands = find(block="edge_case", edge_case=sel.get("edge_case"))
        elif mode in ("sports", "explicitness"):
            cands = find(block=mode)
        if not cands:
            return None
        e = cands[0]
        codes = self.encode(e["id"], speed)
        if codes is None and speed != 1.0:       # tempo variant not ready yet
            codes = self.encode(e["id"], 1.0)
            speed = 1.0
        if codes is None:
            return None
        return {"entry": e, "codes": codes, "speed": speed}

    def select_gid(self, gid, speed=1.0):
        """One specific condition, named by the gid the retrieval index uses.

        The index and this bank key conditions identically, so a nearest
        neighbour comes back as something that can be encoded directly.
        """
        e = self.entries.get(gid)
        if e is None:
            return None
        codes = self.encode(gid, speed)
        if codes is None and speed != 1.0:
            codes = self.encode(gid, 1.0)
            speed = 1.0
        if codes is None:
            return None
        return {"entry": e, "codes": codes, "speed": speed}

    # --------------------------------------------------------------- catalog
    def catalog(self):
        """Legal values per axis — fed to the model so it cannot invent one."""
        out = {"emotion": set(), "voicenet_dimension": set(), "level": set(),
               "character": set(), "edge_case": set()}
        for e in self.entries.values():
            if e.get("emotion"):
                out["emotion"].add(e["emotion"])
            if e.get("dimension"):
                out["voicenet_dimension"].add(e["dimension"])
            if e.get("level"):
                out["level"].add(e["level"])
            if e.get("character"):
                out["character"].add(e["character"])
            if e.get("edge_case"):
                out["edge_case"].add(e["edge_case"])
        return {k: sorted(v) for k, v in out.items()}

    _GLOSS_RE = re.compile(r"\bin ([a-z][a-z \-]{2,40})\.", re.I)
    _ART_RE = re.compile(r"^(an|a|the)\s+", re.I)

    @classmethod
    def _article(cls, s):
        return cls._ART_RE.sub("", s.strip()).strip()

    def descriptions(self):
        """Human-readable gloss per condition, mined from the corpus itself.

        Every clip carries the conditioning prompt it was generated from, whose
        GENERAL line names the dimension in words ("A voice extremely low in
        voice age.").  The 57 VoiceNet codes are otherwise opaque, and a model
        that cannot tell ARSH from BRGT falls back on the same two or three
        conditions every turn.
        """
        if getattr(self, "_desc", None) is not None:
            return self._desc
        out = {"voicenet_dimension": {}, "character": {}, "edge_case": {}}
        mp = os.path.join(self.root, "metadata.jsonl")
        if os.path.exists(mp):
            for line in open(mp):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("language") != "en":
                    continue
                head = (r.get("prompt") or "").split("SCRIPT:")[0]
                head = head.replace("GENERAL:", "").strip().split(".")[0].strip()
                if not head:
                    continue
                blk = r.get("block")
                if blk == "voicenet" and r.get("dimension") not in out["voicenet_dimension"]:
                    m = self._GLOSS_RE.search(head + ".")
                    if m:
                        out["voicenet_dimension"][r["dimension"]] = m.group(1).strip().lower()
                elif blk == "character" and r.get("character") not in out["character"]:
                    out["character"][r["character"]] = self._article(head)
                elif blk == "edge_case" and r.get("edge_case") not in out["edge_case"]:
                    g = head.split(",")[0].replace(
                        "A voice at the extreme edge of", "").strip()
                    out["edge_case"][r["edge_case"]] = self._article(g)
        self._desc = out
        return out

    def stats(self):
        return {"conditions": self.n_total, "pre_tokenised": len(self.codes),
                "ready": self.ready.is_set(), "variant": self.variant,
                "speeds_ready": sorted(self.speeds_ready)}
