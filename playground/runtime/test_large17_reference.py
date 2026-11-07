"""Reference duration contract without loading model weights or allocating CUDA."""
import shutil,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch
from large17_engine import Large17Engine,ActingEngine


class ReferenceTests(unittest.TestCase):
    def engine(self):
        engine=Large17Engine.__new__(Large17Engine)
        engine.refs={};engine.reference_info={};engine.reference_seconds_cap=14.96
        self.calls=[]
        def encode(wave,sr,n_vq):
            seconds=wave.shape[-1]/sr;self.calls.append(seconds)
            return [torch.zeros((int(np.ceil(seconds*12.5)),12),dtype=torch.long)]
        engine.proc=SimpleNamespace(encode_audios_from_wav=encode)
        return engine

    def test_crop_before_encoding_and_cache_by_content(self):
        engine=self.engine()
        with tempfile.TemporaryDirectory() as tmp,patch('torch.cuda.is_available',return_value=False):
            first=Path(tmp)/'first.wav';second=Path(tmp)/'other-name.wav'
            sf.write(first,np.zeros(20*16000,np.float32),16000);shutil.copyfile(first,second)
            self.assertEqual(tuple(engine.reference(first).shape),(187,12))
            self.assertEqual(tuple(engine.reference(second).shape),(187,12))
            self.assertEqual(self.calls,[14.96]);self.assertTrue(engine.reference_info[str(first)]['truncated'])

    def test_short_reference_does_not_silently_pad(self):
        engine=self.engine()
        with tempfile.TemporaryDirectory() as tmp,patch('torch.cuda.is_available',return_value=False):
            path=Path(tmp)/'short.wav';sf.write(path,np.zeros(4*16000,np.float32),16000)
            with self.assertRaisesRegex(ValueError,'5.04'):engine.reference(path)

    def test_extended_warmup_obeys_prompt_limit_and_stops_after_first_chunk(self):
        engine=self.engine();calls=[]
        engine.generate_many=lambda specs,**kw:calls.append((specs,kw))
        with patch.object(ActingEngine,'warm_live_stream'):
            engine.warm_live_stream()
        specs,options=calls[0]
        self.assertLessEqual(len(specs[0]['text'].split()),120)
        self.assertTrue(options['on_audio'](np.ones(8,np.float32)))
        self.assertEqual(options['temperature'],.8)


if __name__=='__main__':unittest.main()
