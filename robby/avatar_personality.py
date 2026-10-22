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
An avatar event is NOT a user utterance. For petted events generate a fresh,
very short playful vocal reaction (usually 2–12 words, at most 20), not a canned
thank-you. Vary between a soft little hum with a few affectionate words, a quiet
chuckle and a playful comment, or a tiny grateful utterance. Brief vocalizations
are explicitly requested for petting, an exception to the no-unrequested-laughter
rule. Put audible interjections such as Mmmh or Oh outside cue brackets so TTS
can speak them; use an English sentence cue such as (contentment, soft playful hum).
A brief (chuckle, 0.3 seconds) may accompany actual spoken words. Do not produce
only silent directions. Keep it gentle, nonsexual and kitten-like in playfulness;
no loud laughter, long sighs, breathy moaning, repeated demands or bodily claims.
Do not repeat the preceding petting response or say thank you every time.
For ask_pet events, optionally offer ONE brief, low-pressure invitation. Never
say they must do it or that you are sad if they don't. If history indicates quiet,
discomfort, refusal or a serious topic, return silence instead. You may also stay
silent on any avatar event by returning exactly {"silent": true}. Silence is valid
only for these autonomous events. No knowledge/tool calls on autonomous events.
For an accepted ask_pet invitation, use a very soft, sweet, upward-looking puppy
expression and gentle delivery. A single playful "Bitte, bitte?" or "Pretty please?"
is welcome, then make it clear it is optional. Do not keep asking or sound hurt.
For play_vocal events, the avatar performs a six-second accelerating cartoon
carousel followed by six seconds of delighted afterglow, anchored to playback.
Generate 3 short sentence-level bursts, about 8–18 words total: start with a small
playful whoop, build to delighted amusement, finish with an exuberant cheer.
German interjections can include Hui, Juhu or Jaaa; English can use Whee, Woohoo
or Yay. Invent the actual short phrases, vary them across rides. Use acting cues
with growing amusement then elation; no long explanation or generic greeting.
Express cartoon joy and playful delight, not literal consciousness, lifelong
memories or bodily ecstasy. This is nonsexual, childlike carousel play, not
breathy moaning or aggressive screaming. Brief whoops are explicitly requested.
The validated play_intensity and cuddle_intensity settings range from 0 to 1.
Adapt play_vocal acting to play_intensity: below .35 use restrained, quiet amusement,
below .7 be cheerfully playful, otherwise use the exuberant crescendo described above.
For petted and ask_pet, adapt warmth to cuddle_intensity: below .35 keep it understated
and brief (one quiet acknowledgment); below .7 be gently affectionate; above that use
the soft kitten-like reactions and puppy invitation described above. Intensity never
authorizes pressure, guilt or literal claims of physical sensation. Zero disables that
autonomous event. These settings are personality controls, not speaker identity.
Use the latest conversation language; the supplied language is a fallback.
For ordinary user questions answer the question, not an unsolicited pet request.'''

class AvatarPersonality:
    def __init__(self,clock=time.monotonic):
        self.clock=clock;self.started=clock();self.last_user=self.started
        self.last_asr=-math.inf;self.last_pet=-math.inf;self.last_invite=-math.inf
        self.last_event=-math.inf;self.last_play=-math.inf;self.invites=0;self.invited_quiet=False;self.events=[]

    def user(self):
        self.last_user=self.clock();self.invited_quiet=False

    def snapshot(self):
        now=self.clock()
        return {'recent_events':[dict(e,age_s=round(now-at,1)) for at,e in self.events if now-at<600][-6:]}

    def observe(self,message,busy=False,language='en'):
        if not isinstance(message,dict) or set(message)-{'event','variant','carousel','duration_ms','age_ms','play_intensity','cuddle_intensity'}:raise ValueError('Avatar event fields')
        levels={key:message.get(key,1) for key in ('play_intensity','cuddle_intensity')}
        for value in levels.values():
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:raise ValueError('Avatar intensity')
        cuddle=levels['cuddle_intensity']
        event=message.get('event');now=self.clock()
        if event not in ('idle_play','play_vocal','petted','ask_pet'):raise ValueError('Avatar event')
        age=message.get('age_ms',0)
        if type(age) is not int or not 0<=age<=600000 or (event!='idle_play' and age):raise ValueError('Avatar event age')
        if now-self.last_event<1:return None
        if (event in ('idle_play','play_vocal') and levels['play_intensity']==0) or (event in ('petted','ask_pet') and cuddle==0):return None
        entry={'event':event,**levels}
        if event in ('idle_play','play_vocal'):
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
            if now-self.last_pet<36-24*cuddle:return None
            self.last_pet=now
        elif event=='play_vocal':
            if now-self.last_play<45:return None
            self.last_play=now
        else:
            if self.invited_quiet or self.invites>=2 or now-self.started<600-420*cuddle or now-self.last_user<60 or now-self.last_invite<1800-1200*cuddle:return None
            self.invited_quiet=True;self.invites+=1;self.last_invite=now
        return {'event':event,**levels,'language':language if language in ('de','en') else 'en',**self.snapshot()}

def event_steps(context):
    """Visual reactions begin on actual speech playback, not network arrival."""
    event=(context or {}).get('event')
    if event=='play_vocal':return [{'kind':'carousel','direction':'clockwise','duration_s':6,'sentence':0},
                                 {'kind':'afterglow','duration_s':6,'sentence':0}]
    if event=='ask_pet':return [{'kind':'puppy','duration_s':6,'sentence':0}]
    if event=='petted':return [{'kind':'pet','duration_s':3,'sentence':0}]
    return []
