"""Scoring/mixing contracts: prevent a test-harness error from becoming a pass."""
import unittest
import numpy as np
from adversarial_audio import SR, soundscape, sha_audio, wilson, word_error_rate, zero_event_rate_bound, judge_text
from adversarial_scenarios import scenarios, validate_plan


class TestAdversarialProtocol(unittest.TestCase):
    def setUp(self):
        t=np.arange(SR*2)/SR
        self.target=(.12*np.sin(t*2*np.pi*191)).astype(np.float32)
        self.bg=(.08*np.sin(t*2*np.pi*237)).astype(np.float32)

    def test_actual_snr_and_peak_preserved(self):
        for snr in (20,12,6,0,-6):
            wave,meta,bed=soundscape(self.target,[self.bg],snr_db=snr)
            self.assertAlmostEqual(meta['measured_snr_db'],snr,places=5)
            self.assertLessEqual(abs(wave).max(),.94001)
            self.assertGreater(np.linalg.norm(bed[-16000:]),0,'Background must continue after the main speaker')

    def test_seed_and_signal_hashes(self):
        a,meta,_=soundscape(self.target,[self.bg],seed=42)
        b,_,_=soundscape(self.target,[self.bg],seed=42)
        c,_,_=soundscape(self.target,[self.bg],seed=43)
        self.assertEqual(sha_audio(a),sha_audio(b));self.assertNotEqual(sha_audio(a),sha_audio(c))
        self.assertFalse(meta['activity_is_word_alignment'])

    def test_same_gain_for_peak_limited_components(self):
        _,meta,_=soundscape(self.target*50,[self.bg],snr_db=0)
        self.assertLess(meta['common_peak_gain'],1)
        self.assertAlmostEqual(meta['measured_snr_db'],0,places=5)

    def test_no_speech_cannot_define_snr(self):
        with self.assertRaises(ValueError):soundscape(np.zeros(SR),[self.bg])

    def test_bounded_memory(self):
        with self.assertRaises(ValueError):soundscape(self.target,[self.bg],tail_s=121)

    def test_wer_counts_background_insertions(self):
        self.assertEqual(word_error_rate('Hallo Welt','Hallo Welt'),0)
        self.assertEqual(word_error_rate('Hallo Welt','Hallo Welt Wetter Sonne Regen'),1.5)
        self.assertEqual(word_error_rate('Stopp',''),1)

    def test_keywords_do_not_substitute_for_length(self):
        checks=judge_text('Fips am Bahnhof',{'min_words':25,'required':['Fips','bahnhof']})
        self.assertFalse(all(checks.values()))

    def test_another_cannot_repeat_identically(self):
        self.assertFalse(judge_text('Fips lacht.',{'different_from_previous':True},'Fips lacht!')['not_identical'])

    def test_small_samples_not_100_percent_confidence(self):
        low,high=wilson(4,4);self.assertLess(low,.6);self.assertAlmostEqual(high,1)
        self.assertIsNone(wilson(0,0))
        self.assertGreater(zero_event_rate_bound(60),100)
        with self.assertRaises(ValueError):wilson(5,4)

    def test_plan_complete_unique_phase_matrix(self):
        plan=validate_plan(scenarios())
        interrupt=next(s for s in plan if s['id']=='interruptions')
        for cue in ('christoph-stop','christoph-wait'):
            self.assertEqual({s['phase_ms'] for s in interrupt['steps'] if s.get('clip')==cue},{0,80,160,240})
        self.assertTrue(any(s.get('required') is False for p in plan for s in p['steps']))

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):validate_plan([scenarios()[0],scenarios()[0]])

    def test_weather_numbers_and_rounding_evidence(self):
        from browser_tool_evidence_check import numeric_mentions
        self.assertEqual(numeric_mentions('minus fünf Grad bis achtzehn Komma drei Grad'),[-5,18.3])
        self.assertEqual(numeric_mentions('Zwischen 7,4 und 18.2 Grad, Wind vierunddreißig.'),[7.4,18.2,34])

if __name__=='__main__':unittest.main()
