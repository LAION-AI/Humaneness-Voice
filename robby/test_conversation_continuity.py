import unittest
from conversation_tasks import joke_task,memory_content,acknowledgement_only

class ConversationTaskTests(unittest.TestCase):
    def test_promise_is_not_a_story_and_a_short_story_is_not_a_promise(self):
        for text in ('Das ist eine wunderbare Bitte, die ich sehr gerne erfülle.','Natürlich, Christoph.','Ich kann dir gerne eine Geschichte erzählen.'):
            self.assertTrue(acknowledgement_only(text),text)
        for text in ('Ein Roboter fand einen Stern und brachte ihn nach Hause.',
                     'Ja, sagte der Roboter, und lachte über seinen eigenen Witz.',
                     'Warum ging der Roboter zur Schule? Er brauchte mehr Bits.'):
            self.assertFalse(acknowledgement_only(text),text)
    def test_elliptical_jokes_keep_latest_task_without_turning_every_question_into_joke(self):
        history=[{'role':'user','content':'Erzähl einen Witz über Roboter.'},
                 {'role':'assistant','content':'Generated joke.'},
                 {'role':'user','content':'Jetzt einen Witz über eine Katze.'}]
        for text in ('Noch einen, bitte.','Und noch einen.','Erzähl mir einen weiteren.','Den bitte ausführlicher.'):
            self.assertIsNotNone(joke_task(text,history),text)
        self.assertIsNone(joke_task('Wie wird das Wetter morgen?',history))
        self.assertIsNone(joke_task('Schreib eine Geschichte.',history))
        self.assertIsNone(joke_task('Keinen Witz, bitte.',history))
        self.assertIsNone(joke_task('Noch einen.',[{'role':'user','content':'Ein Gedicht bitte.'}]))
    def test_new_task_stops_older_joke_and_expansion_is_distinct_from_new_joke(self):
        old=[{'role':'user','content':'Erzähl einen Witz.'}]
        self.assertTrue(joke_task('Den bitte länger.',old)['retell'])
        self.assertFalse(joke_task('Noch einen.',old)['retell'])
        self.assertIsNone(joke_task('Noch einen.',old+[{'role':'user','content':'Ein Gedicht bitte.'}]))
        self.assertIsNone(joke_task('Noch einen.',[]))
    def test_interrupt_preserves_actual_generated_context_with_hearing_warning(self):
        record={'reply':'Ein Roboter stand am Bahnhof. Er fragte nach seinem Zug.'}
        text=memory_content(record,True)
        self.assertIn(record['reply'],text);self.assertIn('only a prefix may have been heard',text)
        self.assertEqual(memory_content(record),record['reply'])

if __name__=='__main__':unittest.main()
