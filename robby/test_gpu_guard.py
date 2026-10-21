import concurrent.futures,threading,time,unittest
from types import SimpleNamespace
from gpu_guard import serialize_audio_gpu,audio_failure,AudioPriorityLock

class GpuGuardTests(unittest.TestCase):
    def test_waiting_live_hop_precedes_second_asr_job(self):
        lock=AudioPriorityLock();lock.acquire(10);order=[]
        def job(name,priority):
            lock.acquire(priority)
            try:order.append(name)
            finally:lock.release()
        a=threading.Thread(target=job,args=('asr',10));b=threading.Thread(target=job,args=('hop',0))
        a.start();b.start()
        deadline=time.monotonic()+2
        while len(lock.waiting)<2 and time.monotonic()<deadline:time.sleep(.001)
        lock.release();a.join(2);b.join(2)
        self.assertEqual(order,['hop','asr'])
    def test_native_and_transcription_calls_do_not_overlap_across_threads(self):
        owners={};maximum=0
        def work():
            nonlocal maximum
            tid=threading.get_ident();owners[tid]=owners.get(tid,0)+1;maximum=max(maximum,len(owners))
            time.sleep(.002)
            owners[tid]-=1
            if not owners[tid]:del owners[tid]
            return 'ok'
        engine=SimpleNamespace(asr=SimpleNamespace(encode=work,transcribe=work))
        engine.process=lambda:engine.asr.encode()
        serialize_audio_gpu(engine);serialize_audio_gpu(engine)
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            jobs=[pool.submit(engine.process if i%2 else engine.asr.transcribe)for i in range(24)]
            self.assertTrue(all(f.result(timeout=2)=='ok' for f in jobs))
        self.assertEqual(maximum,1)
    def test_fatal_context_and_dead_worker_are_not_reported_ready(self):
        def bad():raise RuntimeError('CUDA error: an illegal memory access was encountered')
        engine=SimpleNamespace(process=lambda:None,asr=SimpleNamespace(encode=lambda:None,transcribe=bad))
        serialize_audio_gpu(engine);ears=SimpleNamespace(service=SimpleNamespace(engine=engine))
        self.assertIsNone(audio_failure(ears,['ears-gpu','ears-asr']))
        with self.assertRaises(RuntimeError):engine.asr.transcribe()
        self.assertIn('restart',audio_failure(ears,['ears-gpu','ears-asr']))
        del engine.robo_audio_failure
        self.assertIn('stopped',audio_failure(ears,['ears-asr']))

if __name__=='__main__':unittest.main()
