import unittest
from performance import SLUGS,cue,plan


class PerformanceTests(unittest.TestCase):
    def test_all_forty_canonical_labels(self):
        self.assertEqual(len(SLUGS),40)
        for slug in SLUGS:self.assertEqual(cue(slug)['emotion'],slug)

    def test_no_emotion_inference_from_spoken_words(self):
        p=plan({'general':'calm','tagged':'[2.0 seconds duration] (angry, intense) I am not angry.'})
        self.assertEqual(p['sentences'][0]['emotion'],'anger')

    def test_negated_direction(self):self.assertNotEqual(cue('not angry, relieved')['emotion'],'anger')

    def test_timed_sentence_changes(self):
        p=plan({'tagged':'[0.3 seconds pause] [2.0 seconds duration] (sad) Oh no. [0.4 seconds pause] [1.6 seconds duration] (relieved) We are safe.'})
        a,b=p['sentences'];self.assertEqual(a['emotion'],'sadness');self.assertEqual(b['emotion'],'relief')
        self.assertAlmostEqual(a['start_s'],.3);self.assertAlmostEqual(a['end_s'],2.3)
        self.assertAlmostEqual(b['start_s'],2.7);self.assertAlmostEqual(b['end_s'],4.3)

    def test_burst_not_spoken_sentence(self):
        p=plan({'tagged':'(sad) [1.2 seconds duration] Oh no. (wistful sigh, 0.3 seconds)'})
        self.assertEqual(len(p['sentences']),1);self.assertEqual(p['sentences'][0]['emotion'],'sadness')

    def test_empty_and_mix(self):
        self.assertTrue(plan({})['sentences']);c=cue('strong sadness with relief')
        self.assertEqual(c['secondary'],'relief');self.assertGreater(c['intensity'],.8)

if __name__=='__main__':unittest.main()
