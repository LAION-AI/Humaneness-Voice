"""Session-only conversation ownership; cosine scores are not probabilities.

Keep acoustic admission independent of the language model and of TYGER's
turn-taking proposal. A background voice must not acquire ownership simply
because it made the loudest sound or said a stop word.
"""
import time
import numpy as np


class ConversationFocus:
    def __init__(self, threshold=.28, timeout=180., clock=time.monotonic):
        self.enabled = True
        self.threshold = threshold
        self.timeout = timeout
        self.clock = clock
        self.owner = None
        self.members = set()
        self.primary = None
        self.seen = -1e9

    def release(self):
        self.owner = None
        self.seen = -1e9

    def active(self, users, speaking=False):
        self.members.intersection_update(users)
        if self.primary not in users:self.primary=None
        if self.owner not in users or (not speaking and self.clock()-self.seen > self.timeout):
            self.release()
        elif speaking and self.owner is not None:
            self.seen = self.clock()
        return self.owner if self.enabled else None

    def accept(self, sid):
        if sid and sid != 'unknown':
            self.members.add(sid)
            if self.primary is None:self.primary=sid
            self.owner = sid
            self.seen = self.clock()

    def check(self, embedding, users, speaking=False, bot=None):
        owner = self.active(users, speaking)
        if not self.enabled:
            return {'accepted': True, 'reason': 'Conversation focus disabled', 'speaker_id': None}
        if owner is None:
            return {'accepted': not speaking, 'reason': 'No active voice anchor', 'speaker_id': None}
        if embedding is None:
            return {'accepted': False, 'reason': 'Not enough clean speech to confirm the conversation partner', 'speaker_id': owner}
        q = np.asarray(embedding, np.float32).reshape(-1)
        q = q / max(1e-8, float(np.linalg.norm(q)))
        scores = {sid: max((float(q @ v) for v in user['vectors']), default=-1.)
                  for sid, user in users.items()}
        allowed=(self.members or {owner}) & users.keys()
        target=max(allowed,key=lambda sid:scores.get(sid,-1.)) if allowed else owner
        # Keep the current person when two admitted voices are nearly tied.
        if target!=owner and scores.get(target,-1.)<scores.get(owner,-1.)+.06:target=owner
        own = scores.get(target, -1.)
        rival = max((score for sid, score in scores.items() if sid not in allowed), default=-1.)
        echo = float(q @ bot) if bot is not None else -1.
        accepted = own >= self.threshold and own >= rival + .06 and (echo < .40 or own >= echo + .06)
        return {'accepted': bool(accepted), 'speaker_id': target, 'similarity': round(own, 4),
                'rival_similarity': round(rival, 4), 'bot_similarity': round(echo, 4),
                'reason': 'Active conversation partner' if accepted else 'Background, echo or uncertain voice; keeping the current conversation'}

    def state(self, users, speaking=False):
        owner = self.active(users, speaking)
        return {'enabled': self.enabled, 'threshold': self.threshold, 'timeout_s': self.timeout,
                'speaker_id': owner, 'name': users.get(owner, {}).get('name'),'primary_id':self.primary,
                'participants':[{'id':sid,'name':users[sid]['name']} for sid in sorted(self.members) if sid in users]}
