import unittest
from recovery_rules import known_voice_consensus

class RecoveryConsensusTests(unittest.TestCase):
    def test_same_known_winner_can_be_corroborated_without_lowering_teacher_floor(self):
        vote={'id':'known-a','similarity':.7,'margin':.2}
        self.assertTrue(known_voice_consensus('known-a',.58,.07,vote))
        self.assertFalse(known_voice_consensus('known-a',.54,.07,vote))
        self.assertFalse(known_voice_consensus('known-a',.58,.03,vote))
        self.assertFalse(known_voice_consensus('known-a',.58,.07,{**vote,'id':'other'}))
        self.assertFalse(known_voice_consensus('known-a',.58,.07,{**vote,'similarity':.59}))
        self.assertFalse(known_voice_consensus('known-a',.58,.07,vote,teacher_threshold=.65))

if __name__=='__main__':unittest.main()
