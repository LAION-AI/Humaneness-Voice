"""Session-owned audio: takes expire; custom references persist until removal."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
import uuid

import settings


@dataclass
class Audio:
    id: str
    owner: str
    path: Path
    label: str
    kind: str
    touched: float
    slot: str = ""
    active: int = 0

    @property
    def playback(self):
        return self.path.with_suffix(".mp3")

    @property
    def files(self):
        return {self.path, self.playback}


class AudioStore:
    def __init__(self, root=None, ttl=None):
        self.root = Path(root or settings.DATA)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.ttl = settings.TTL if ttl is None else ttl
        self.items = {}
        self.ledger = self.root / '.references.json'
        if self.ledger.exists():
            for row in json.loads(self.ledger.read_text()):
                if (len(row['id']) != 32 or any(c not in '0123456789abcdef' for c in row['id'])
                        or len(row['owner']) != 32):
                    continue
                path = self.root / (row['id'] + '.wav')
                if path.is_file() and path.with_suffix('.mp3').is_file():
                    self.items[row['id']] = Audio(row['id'], row['owner'], path,
                        row['label'], 'reference', row['touched'], row['slot'])
        known = {path for item in self.items.values() for path in item.files}
        cutoff = time.time() - self.ttl
        for path in self.audio_files():
            if path not in known and path.stat().st_mtime < cutoff:
                path.unlink()

    def persist_references(self):
        rows = [{'id': x.id, 'owner': x.owner, 'label': x.label, 'slot': x.slot, 'touched': x.touched}
                for x in self.items.values() if x.kind == 'reference']
        if not rows:
            self.ledger.unlink(missing_ok=True)
            return
        tmp = self.ledger.with_suffix('.tmp')
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as file:
            json.dump(rows, file)
        tmp.replace(self.ledger)

    def new(self, owner, label, kind, slot=""):
        id_ = uuid.uuid4().hex
        suffix = ".mp3" if kind == "take" else ".wav"
        item = Audio(id_, owner, self.root / f"{id_}{suffix}", label, kind, time.time(), slot)
        self.items[id_] = item
        return item

    def get(self, id_, owner, touch=True):
        item = self.items.get(id_)
        if item is None or item.owner != owner or (item.kind != 'reference' and not item.active and time.time() - item.touched >= self.ttl):
            raise KeyError("Audio expired or unavailable.")
        if touch:
            item.touched = time.time()
            for path in item.files:
                if path.exists():
                    path.touch()
        return item

    def references(self, owner):
        return [x for x in self.items.values() if x.owner == owner and x.kind == "reference"]

    def remove(self, item):
        if item.active:
            raise ValueError("Diese Referenz wird gerade verwendet.")
        for path in item.files:
            path.unlink(missing_ok=True)
        self.items.pop(item.id, None)
        if item.kind == 'reference':
            self.persist_references()

    def cleanup(self, now=None):
        now = time.time() if now is None else now
        expired = [x for x in self.items.values() if x.kind != 'reference' and not x.active and now - x.touched >= self.ttl]
        for x in expired:
            self.remove(x)
        # Includes orphan files from a restart, but never an active item.
        known = {path for item in self.items.values() for path in item.files}
        for path in self.audio_files():
            if path not in known and now - path.stat().st_mtime >= self.ttl:
                path.unlink()
        return expired

    def audio_files(self):
        yield from self.root.glob("*.wav")
        yield from self.root.glob("*.mp3")
