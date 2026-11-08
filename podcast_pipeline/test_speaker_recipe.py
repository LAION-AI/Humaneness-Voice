import json
import unittest
from pathlib import Path
from recipe import specification, HERE

class SpeakerRecipeTests(unittest.TestCase):
    def test_each_sentence_is_directed_without_changing_spoken_text(self):
        for lang in ('en','de'):
            original=json.loads((HERE/f'script-{lang}.json').read_text())
            current=json.loads((HERE/f'script-speaker-{lang}.json').read_text())
            for old,turn in zip(original['turns'],current['turns']):
                self.assertEqual(old['text'],turn['text'])
                self.assertEqual(' '.join(s['text'] for s in turn['sentences']),turn['text'])
                prompt=specification(turn,lang)['prompt']
                self.assertEqual(prompt.count('(Very genuine, very natural, very authentic;'),len(turn['sentences']))
                self.assertNotIn('pleasant',prompt.lower())

    def test_equal_weights_and_no_wer_gate(self):
        from speaker_scoring import speaker_reward
        self.assertAlmostEqual(speaker_reward(3,.8),.65)
        self.assertAlmostEqual(speaker_reward(6,0),speaker_reward(0,1))
        self.assertEqual(speaker_reward(100,-1),.5)
        with self.assertRaises(ValueError):speaker_reward(float('nan'),.5)

if __name__=='__main__':unittest.main()
