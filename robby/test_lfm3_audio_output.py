import unittest

from lfm3_audio_output import parse_audio_output


class AudioOutputTest(unittest.TestCase):
    def test_array_is_preserved(self):
        result = parse_audio_output('[{"category":"music","description":"A melody","relevance":1}]')
        self.assertEqual(result['observations'][0]['description'], 'A melody')
        self.assertEqual(result['observations'][0]['relevance'], 1)

    def test_plain_description_is_not_discarded_or_assigned_a_score(self):
        result = parse_audio_output('A voice and quiet music.')
        self.assertEqual(result['observations'][0]['description'], 'A voice and quiet music.')
        self.assertIsNone(result['observations'][0]['relevance'])

    def test_empty_and_unfinished_thinking_are_not_presented(self):
        self.assertIsNone(parse_audio_output(''))
        self.assertIsNone(parse_audio_output('<think>unfinished reasoning'))

    def test_empty_object_remains_empty(self):
        self.assertEqual(parse_audio_output('{"observations":[]}'), {'observations': []})


if __name__ == '__main__':
    unittest.main()
