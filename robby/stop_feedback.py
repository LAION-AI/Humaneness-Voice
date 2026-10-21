"""Optional polite feedback AFTER cancellation, never a prerequisite to stop.

No LLM call. Silence is the default and explicit wait/quiet requests always win.
Session-only task state; browser persists only the non-biometric preference.
"""
import asyncio
import re
from stop_commands import is_stop_command

PHRASES={
    'de':('Alles klar. Ich höre zu.', 'Entschuldige. Du bist dran.',
          'Okay, ich mache eine Pause.', 'Oh, sorry. Wie kann ich dir sonst helfen?'),
    'en':('Of course. I’m listening.', 'Sorry. Go ahead.',
          'Okay, I’ll pause here.', 'Oh, sorry. How else can I help?'),
}

def allows_ack(text):
    return is_stop_command(text) and not re.search(
        r'\b(?:warte|wait|pause|leise|still|ruhe|quiet|silence|hold)\b',text,re.I)

class StopFeedback:
    def __init__(self,conversation):
        self.c=conversation;self.enabled=False;self.task=None;self.index=0
    def cancel(self):
        if self.task and self.task is not asyncio.current_task():self.task.cancel()
        self.task=None
    def schedule(self,event):
        self.cancel()
        if not self.enabled or event.get('source')!='verified_stop_command' or not allows_ack(event.get('stop_text','')):return
        c=self.c;rid=c.playback.rid;speaker=dict(c.current_speaker or {})
        language=c.decisions.language if c.decisions.language in PHRASES else 'de'
        async def later():
            # Let the user keep the floor; never overlap the cancelled producer
            # or answer over continuing speech, pending turns or a new reply.
            await asyncio.sleep(1.2)
            ext=c.speaker_ext
            if (not self.enabled or c.ws.closed or c.speaking or c.halting or c.playback.rid!=rid
                or c.pending is not None or not ext or ext.closed
                or speaker.get('speaker_id') not in ext.memory.users
                or (c.current_speaker or {}).get('speaker_id')!=speaker.get('speaker_id')
                or ext.pending_turns or ext.live_speech_recent(c.hop_pos)):return
            phrase=PHRASES[language][self.index%4];self.index+=1
            c.session.reset();c._start_speaking()
            await c.ws.send_json({'type':'stop_feedback','text':phrase})
            await c.to_mouth(b'T',{'transcript':'','speaker':speaker,
                'interaction_reply':phrase,'interaction_language':language,'stop_feedback':True})
        self.task=asyncio.create_task(later())
