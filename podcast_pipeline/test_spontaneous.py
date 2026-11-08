import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from recipe import HERE,specification
class SpontaneousTests(unittest.TestCase):
    def test_exact_words_and_sparse_cues(self):
        for lang in ('en','de'):
            original=json.loads((HERE/f'script-{lang}.json').read_text())['turns']
            new=json.loads((HERE/f'script-spontaneous-{lang}.json').read_text())['turns']
            self.assertEqual(len(new),len(original))
            for a,b in zip(original,new):
                self.assertEqual(a['text'],b['text']);self.assertEqual(a['speaker'],b['speaker'])
                self.assertEqual(' '.join(c['text'] for c in b['delivery_chunks']),a['text'])
                self.assertLessEqual(len(b['delivery_chunks']),2)
                self.assertIn('Very authentic, very genuine',specification(b,lang)['prompt'])
                self.assertNotIn('sentences',b)
    def test_failed_archive_does_not_publish_complete_receipt(self):
        from archive import archive_batch
        with tempfile.TemporaryDirectory() as tmp:
            turn=dict(index=0,speaker='S1',text='Hello.',direction='warm')
            with patch('archive.sf.write',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    archive_batch(tmp,[('en',turn)],[None],[None],[{'reward':1}],{}, {},1,1)
            self.assertFalse(list(Path(tmp).rglob('result.json')))
if __name__=='__main__':unittest.main()
