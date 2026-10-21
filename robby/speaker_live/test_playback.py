import unittest
from .playback import SyntheticPlayback

class PlaybackTests(unittest.TestCase):
    def test_arrival_does_not_mean_played(self):
        d=SyntheticPlayback();d.add(1,24000,0)
        self.assertEqual(d.acknowledgments(0),{})
        self.assertEqual(d.pieces(0,320),[(0,320,0,-1)])
        self.assertEqual(d.pieces(640,320),[(0,320,1,0)])
        self.assertEqual(d.acknowledgments(8640),{1:12000})
        self.assertEqual(d.acknowledgments(16640),{1:24000})
    def test_chunks_gaps_and_position_offset(self):
        d=SyntheticPlayback();d.add(1,2400,0);d.add(1,2400,3000)
        self.assertEqual(d.pieces(2200,1600),[(0,40,1,2340),(40,1440,0,-1),(1440,1600,1,2400)])
    def test_cancel_removes_future_and_allows_new_reply(self):
        d=SyntheticPlayback();d.add(1,24000,0);d.cancel(1000)
        self.assertEqual(d.pieces(1200,320),[(0,320,0,-1)])
        d.add(1,2400,1000);self.assertEqual(d.sent[1],24000)
        d.add(2,2400,1000);self.assertEqual(d.pieces(1640,320),[(0,320,2,0)])

if __name__=='__main__':unittest.main()
