"""One shared encoder, serialized across native and owned-ASR worker threads.

This changes scheduling, not model math. Fatal CUDA contexts cannot be fixed
by reconnecting a microphone; expose that failure instead of reporting ready.
"""
import functools,threading

class AudioPriorityLock:
    """Reentrant; queued live encoder hops run before another ASR decode.

    Does not preempt an in-flight kernel or drop microphone packets. A plain
    RLock allowed repeated ASR jobs to reacquire ahead of the live hop thread.
    """
    def __init__(self):
        self.condition=threading.Condition();self.owner=None;self.depth=0;self.waiting=[];self.ticket=0
    def acquire(self,priority):
        ident=threading.get_ident()
        with self.condition:
            if self.owner==ident:self.depth+=1;return
            ticket=(priority,self.ticket);self.ticket+=1;self.waiting.append(ticket)
            while self.owner is not None or ticket!=min(self.waiting):self.condition.wait()
            self.waiting.remove(ticket);self.owner=ident;self.depth=1
    def release(self):
        with self.condition:
            if self.owner!=threading.get_ident():raise RuntimeError('Not the audio lock owner')
            self.depth-=1
            if not self.depth:self.owner=None;self.condition.notify_all()


def serialize_audio_gpu(engine):
    if getattr(engine,'robo_gpu_serialized',False):return
    lock=AudioPriorityLock()
    def guarded(method,priority=10):
        @functools.wraps(method)
        def call(*args,**kwargs):
            lock.acquire(priority)
            try:return method(*args,**kwargs)
            except RuntimeError as exc:
                message=str(exc).lower()
                if any(s in message for s in ('cuda error','illegal memory access','device-side assert','unable to find an engine')):
                    engine.robo_audio_failure='Audio GPU inference failed; the Robby worker needs a restart.'
                raise
            finally:lock.release()
        return call
    engine.process=guarded(engine.process,priority=0)
    engine.asr.encode=guarded(engine.asr.encode)
    engine.asr.transcribe=guarded(engine.asr.transcribe)
    engine.robo_gpu_serialized=True


def audio_failure(ears,thread_names=None):
    if ears is None:return None
    engine=ears.service.engine
    error=getattr(engine,'robo_audio_failure',None)
    if error:return error
    if getattr(engine,'robo_gpu_serialized',False):
        names=set(thread_names) if thread_names is not None else {t.name for t in threading.enumerate() if t.is_alive()}
        if not {'ears-gpu','ears-asr'}.issubset(names):return 'Audio GPU worker stopped; the Robby worker needs a restart.'
    return None
