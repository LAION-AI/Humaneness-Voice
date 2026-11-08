import unittest,json,tempfile
from pathlib import Path
from speaker_run import arguments
from inputs import load_inputs
from recipe import HERE,specification
class InputTests(unittest.TestCase):
    def test_default_is_listener_approved_spontaneous(self):
        a=arguments(['--model','unused']);self.assertEqual(a.prompt_style,'spontaneous')
        refs,scripts,jobs=load_inputs(a)
        self.assertEqual(len(jobs),40)
        for lang,t in jobs:self.assertEqual(' '.join(x['text'] for x in t['delivery_chunks']),t['text'])
    def test_invalid_rewritten_spoken_text_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'script.json';p.write_text(json.dumps(dict(language='en',turns=[dict(speaker='S1',text='Hello there.',delivery_chunks=[dict(text='Different words.',cue='gentle')])])) )
            a=arguments(['--model','unused','--script',str(p)])
            with self.assertRaises(ValueError):load_inputs(a)
    def test_single_turn_adds_style_but_preserves_words(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'script.json';p.write_text(json.dumps(dict(language='de',turns=[dict(speaker='S1',text='Ein kleiner Moment reicht.')])) )
            a=arguments(['--model','unused','--script',str(p)])
            _,_,jobs=load_inputs(a);t=jobs[0][1]
            self.assertEqual(t['text'],'Ein kleiner Moment reicht.')
            self.assertIn('relaxed, spontaneous, gently spoken',specification(t,'de')['prompt'])

class MultilingualTests(unittest.TestCase):
    def test_five_language_conditioning_and_preserved_chunks(self):
        from languages import language_name
        for lang,name in [('nl','Dutch'),('fr','French'),('es','Spanish')]:
            a=arguments(['--model','unused','--script',str(HERE/f'script-spontaneous-{lang}.json')]);_,_,jobs=load_inputs(a)
            self.assertEqual(language_name(lang),name);self.assertEqual(len(jobs),20)
            for code,t in jobs:
                self.assertEqual(code,lang);self.assertEqual(' '.join(c['text'] for c in t['delivery_chunks']),t['text'])
    def test_boundary_fallback_rejects_missing_end(self):
        from audit_alignment import transcript_supports_edges
        self.assertTrue(transcript_supports_edges('Dit is een mooie dag.','Dit is een mooie dag.'))
        self.assertFalse(transcript_supports_edges('Dit is een mooie dag.','Dit is een mooie'))
        self.assertFalse(transcript_supports_edges('Dit is een mooie dag.',''))

if __name__=='__main__':unittest.main()
