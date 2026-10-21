import asyncio,io,unittest
from unittest.mock import AsyncMock,patch
import numpy as np
import soundfile as sf
from tts_bridge import decode_mp3

class AudioTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_mono_48k_without_process(self):
        wave=np.sin(np.arange(4800,dtype=np.float32)*.03)*.1
        raw=io.BytesIO();sf.write(raw,wave,48000,format='WAV',subtype='FLOAT')
        with patch('tts_bridge.decode_mp3_ffmpeg',new=AsyncMock()) as fallback:
            pcm=await decode_mp3(raw.getvalue())
        fallback.assert_not_awaited();np.testing.assert_array_equal(pcm,wave)
    async def test_other_rates_and_corrupt_files_keep_bounded_fallback(self):
        raw=io.BytesIO();sf.write(raw,np.zeros(2400,np.float32),24000,format='WAV')
        for data in (raw.getvalue(),b'not audio'):
            with patch('tts_bridge.decode_mp3_ffmpeg',new=AsyncMock(return_value=np.zeros(10,np.float32))) as fallback:
                self.assertEqual(len(await decode_mp3(data)),10);fallback.assert_awaited_once_with(data)
    async def test_cancellation_not_swallowed(self):
        with patch('asyncio.to_thread',new=AsyncMock(side_effect=asyncio.CancelledError)):
            with self.assertRaises(asyncio.CancelledError):await decode_mp3(b'')

if __name__=='__main__':unittest.main()
