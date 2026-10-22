"""Bounded, session-only avatar context. Browser gestures are NOT speaker identity."""
import math
import time
from avatar_actions import VARIANTS

NOTE = '''Robby has a playful, warm cartoon personality. The supplied avatar context
describes actual browser animations and mouse gestures, not physical sensations
or proof of who is touching the mouse. Use it if asked what you are doing: explain
your little carousel or silly faces naturally, as playful make-believe during a
quiet moment. You may be cute, grateful, friendly and amused. Do not claim genuine
physical touch, subjective suffering, loneliness, neediness or dependence on the
user. Do not guilt them for silence, request exclusive friendship, or promise a
human relationship. No emotional obligation, repeated coaxing or pressure.
Mouse strokes may be acknowledged warmly in your own words; keep it grounded in
this playful interaction, not claims about skin, nerves or bodily feelings.
Do not recite an AI disclaimer in every reply. If directly asked what you really
feel, be transparent about simulated expressions rather than inventing sensations.
The user may ask for quiet or decline touch; respect that across the conversation.
An avatar event is NOT a user utterance. For petted events generate one brief,
fresh, affectionate acknowledgment (at most 20 words), not a canned sentence.
For ask_pet events, optionally offer ONE brief, low-pressure invitation. Never
say they must do it or that you are sad if they don't. If history indicates quiet,
discomfort, refusal or a serious topic, return silence instead. You may also stay
silent on any avatar event by returning exactly {"silent": true}. Silence is valid
only for these autonomous events. No knowledge/tool calls on autonomous events.
Use the latest conversation language; the supplied language is a fallback.
For ordinary user questions answer the question, not an unsolicited pet request.'''

class AvatarPersonality:
    def __init__(self,clock=time.monotonic):
        self.clock=clock;self.started=clock();self.last_user=self.started
        self.last_asr=-math.inf;self.last_pet=-math.inf;self.last_invite=-math.inf
        self.last_event=-math.inf;self.invites=0;self.invited_quiet=False;self.events=[]

    def user(self):
        self.last_user=self.clock();self.invited_quiet=False

    def snapshot(self):
        now=self.clock()
        return {'recent_events':[dict(e,age_s=round(now-at,1)) for at,e in self.events if now-at<600][-6:]}

    def observe(self,message,busy=False,language='en'):
        if not isinstance(message,dict) or set(message)-{'event','variant','carousel','duration_ms','age_ms'}:raise ValueError('Avatar event fields')
        event=message.get('event');now=self.clock()
        if event not in ('idle_play','petted','ask_pet'):raise ValueError('Avatar event')
        age=message.get('age_ms',0)
        if type(age) is not int or not 0<=age<=600000 or (event!='idle_play' and age):raise ValueError('Avatar event age')
        if now-self.last_event<1:return None
        entry={'event':event}
        if event=='idle_play':
            if message.get('variant') not in VARIANTS or type(message.get('carousel')) is not bool:raise ValueError('Idle animation')
            if busy or now-self.last_asr<2:return None
            entry.update(variant=message['variant'],carousel=message['carousel'])
        if event=='petted':
            duration=message.get('duration_ms')
            if type(duration) is not int or not 450<=duration<=15000:raise ValueError('Stroke duration')
            entry['duration_ms']=duration
        if event=='idle_play' and any(e==entry and abs(at-(now-age/1000))<1 for at,e in self.events):return None
        self.last_event=now;self.events.append((now-age/1000,entry));self.events=self.events[-12:]
        if event=='idle_play':return None
        if busy or now-self.last_asr<2:return None
        if event=='petted':
            if now-self.last_pet<90:return None
            self.last_pet=now
        else:
            if self.invited_quiet or self.invites>=2 or now-self.started<180 or now-self.last_user<60 or now-self.last_invite<600:return None
            self.invited_quiet=True;self.invites+=1;self.last_invite=now
        return {'event':event,'language':language if language in ('de','en') else 'en',**self.snapshot()}
