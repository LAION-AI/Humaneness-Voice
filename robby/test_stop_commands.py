import unittest
from stop_commands import is_stop_command,is_brief_stop_command,has_stop_intent

class StopCommandTests(unittest.TestCase):
    def test_wait_commands_are_consumed_but_waiting_statements_are_not(self):
        for text in ('Warte bitte.','Bitte warte kurz.','Warte einen Moment.','Wait, please.','Hold on.'):
            self.assertTrue(is_stop_command(text),text);self.assertTrue(has_stop_intent(text),text)
        for text in ('Warte nicht auf mich.','Warte auf den Bus.','Wait for the train.','Ich warte noch.','Do not stop.'):
            self.assertFalse(has_stop_intent(text),text)
    def test_bounded_asr_homophone_not_arbitrary_person_mention(self):
        self.assertTrue(is_stop_command('Herr bitte auf.'))
        for text in ('Herr Müller wartet auf den Bus.','Der Herr bitte auf die Bühne.','Herr bitte auf der Straße.'):
            self.assertFalse(is_stop_command(text),text)
    def test_only_single_imperatives_get_brief_speaker_context(self):
        for text in ('Stopp.','Hey Robby, Stop!','Halt, bitte.'):
            self.assertTrue(is_brief_stop_command(text))
        for text in ('Stopp, bitte hör auf.','Bitte hör auf.','Bitte sei mal leise.','What does stop mean?'):
            self.assertFalse(is_brief_stop_command(text))
    def test_direct_requests(self):
        for text in ('Stopp!','Stop.','Halt.','Pause, bitte.','Bitte hör auf.','Bitte, bitte sei mal leise.',
                     'Halt, bitte hör auf.','Hey Robby, Stopp!','Robby, bitte sei mal leise.',
                     'Please stop talking.','Be quiet, please.','Hör auf zu reden.',
                     'Bitte hör aufzusprechen.','Bitte höre auf zu sprechen.'):
            with self.subTest(text=text):self.assertTrue(is_stop_command(text))
    def test_repeated_stops_and_stop_before_new_request(self):
        self.assertTrue(is_stop_command('Stopp, stopp, bitte hör jetzt auf!'))
        self.assertTrue(is_brief_stop_command('Stopp, stopp!'))
        for text in ('Stopp, erkläre mir lieber den anderen Weg.','Könntest du bitte kurz aufhören?',
                     'Could you please stop for a second?', 'Hör bitte auf, ich möchte etwas fragen.'):
            self.assertTrue(has_stop_intent(text),text)
        for text in ('Stop motion ist toll.', 'Was bedeutet Stopp?', 'Sag bitte das Wort Stopp.', 'Bitte erzähl weiter.'):
            self.assertFalse(has_stop_intent(text),text)
    def test_mentions_and_new_requests_are_not_silenced(self):
        for text in ('Das ist halt so.','Was bedeutet Stopp?','Sag bitte das Wort Stopp.','Stop motion ist toll.',
                     'Stopp, erkläre mir lieber den anderen Weg.','Wir machen später eine Pause.','Bitte erzähl weiter.'):
            with self.subTest(text=text):self.assertFalse(is_stop_command(text))
