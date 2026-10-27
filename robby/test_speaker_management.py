import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock
import numpy as np
from sentence_session import SentenceSession,unit
from speaker_management import request

class ManagementTests(unittest.TestCase):
    def session(self):
        c=SimpleNamespace(speaking=False,ws=SimpleNamespace(send_json=AsyncMock()),session=SimpleNamespace(reset=Mock()))
        s=SentenceSession(c);s.focus.enabled=True;v=np.eye(192,dtype=np.float32)
        s.memory.assign(v[0],'Ich bin Christoph.');s.memory.users['speaker-1']['level']=.06;s.focus.accept('speaker-1')
        return s,v

    def test_explicit_german_and_english_invitations(self):
        for text,name in [('Das ist hier mein Kumpel Bob, red mal auch mit dem.','Bob'),
                          ('Das ist Alice. Reagiere bitte auch auf sie.','Alice'),
                          ('This is my friend Roman, please talk to him too.','Roman'),
                          ('Reagier mal auf die.',None)]:
            self.assertEqual(request(text),{'action':'invite','name':name},text)
        for text in ['Ignorier bitte die Hintergrundstimmen.','Ignore the background speakers.','Rede nur mit mir.']:
            self.assertEqual(request(text)['action'],'ignore_background')
        for text in ['Sprich bitte nicht mit Bob.','Wenn Bob kommt, rede mit ihm.','Sag "Rede auch mit Bob".','Bob redet im Hintergrund.']:
            self.assertIsNone(request(text),text)

    def test_new_guest_joins_without_replacing_original_speaker(self):
        s,v=self.session();p=s.permissions
        result=p.command('Das ist mein Kumpel Bob, rede bitte auch mit ihm.',{'speaker_id':'speaker-1'})
        self.assertTrue(result['waiting_for_next_voice']);self.assertEqual(result['name'],'Bob')
        self.assertFalse(p.candidate(v[0],v[0],.06,'Ich rede noch.'))
        self.assertFalse(p.candidate(v[1],v[1],.005,'Leiser Fernseher'))
        self.assertTrue(p.candidate(v[1],v[1],.06,'Guten Morgen!'))
        guest=p.enroll(v[1],'Guten Morgen!',v[1],.06)
        self.assertEqual(guest['display_name'],'Bob');self.assertEqual(len(s.focus.members),2)
        self.assertEqual(s.focus.check(v[0],s.memory.users)['speaker_id'],'speaker-1')
        self.assertTrue(s.focus.check(v[0],s.memory.users)['accepted'])
        self.assertTrue(s.focus.check(v[1],s.memory.users)['accepted'])
        self.assertFalse(s.focus.check(v[2],s.memory.users)['accepted'])

    def test_higher_similar_voice_guest_is_not_merged_into_main_profile(self):
        s,v=self.session();p=s.permissions;p.command('Das ist mein Kumpel Bob, rede auch mit ihm.',{'speaker_id':'speaker-1'})
        q=unit(.5*v[0]+np.sqrt(.75)*v[1])
        guest=p.enroll(q,'Hallo!',q,.06)
        self.assertNotEqual(guest['speaker_id'],'speaker-1');self.assertEqual(s.memory.users['speaker-1']['name'],'Christoph')

    def test_known_guest_is_authorized_immediately(self):
        s,v=self.session();bob=s.memory.assign(v[1],'Ich bin Bob.')['speaker_id']
        self.assertFalse(s.focus.check(v[1],s.memory.users)['accepted'])
        result=s.permissions.command('Rede auch mit Bob.',{'speaker_id':'speaker-1'})
        self.assertEqual(result['speaker_id'],bob);self.assertTrue(s.focus.check(v[1],s.memory.users)['accepted'])

    def test_invited_name_reuses_matching_unnamed_guest_instead_of_duplicate(self):
        s,v=self.session();guest=s.memory.assign(v[1],'Guten Morgen.')['speaker_id'];s.focus.members.add(guest)
        s.permissions.command('Das ist meine Freundin Alice, rede bitte auch mit ihr.',{'speaker_id':'speaker-1'})
        self.assertTrue(s.permissions.candidate(v[1],v[1],.06,'Wie geht es dir?'))
        context=s.permissions.enroll(v[1],'Wie geht es dir?',v[1],.06)
        self.assertEqual(context['speaker_id'],guest);self.assertEqual(context['display_name'],'Alice')
        self.assertEqual(len(s.memory.users),2)

    def test_unadmitted_background_cannot_grant_permissions(self):
        s,v=self.session()
        self.assertIsNone(s.permissions.command('Das ist Alice, rede auch mit ihr.',{'speaker_id':'outside'}))
        self.assertIsNone(s.permissions.pending)

    def test_ignore_background_stops_questions_and_only_me_revokes_guests(self):
        s,v=self.session();bob=s.memory.assign(v[1],'Ich bin Bob.')['speaker_id'];s.focus.members.add(bob)
        s.permissions.command('Ignorier die Stimmen im Hintergrund.',{'speaker_id':'speaker-1'})
        self.assertIn(bob,s.focus.members);self.assertFalse(s.permissions.questions)
        s.permissions.command('Rede nur mit mir.',{'speaker_id':'speaker-1'})
        self.assertEqual(s.focus.members,{'speaker-1'});self.assertFalse(s.focus.check(v[1],s.memory.users)['accepted'])

    def test_split_introduction_retains_guest_name_without_early_permission(self):
        s,v=self.session();p=s.permissions;actor={'speaker_id':'speaker-1'}
        self.assertIsNone(p.command('Das ist mein Kumpel Roman.',actor));self.assertIsNone(p.pending)
        self.assertEqual(p.command('Rede auch mit ihm.',actor)['name'],'Roman')

    def test_question_cooldown_and_expiring_invitation(self):
        s,v=self.session();now=[1000.];s.permissions.clock=lambda:now[0]
        p=s.permissions;p.observe_background();p.observe_background()
        self.assertTrue(p.context()['may_ask_about_background'])
        p.note_reply('Wer redet da im Hintergrund?');self.assertFalse(p.context()['may_ask_about_background'])
        p.command('Rede auch mit Bob.',{'speaker_id':'speaker-1'});now[0]+=46
        self.assertFalse(p.candidate(v[1],v[1],.06,'Hallo'))
        self.assertIsNone(p.context()['pending_invitation'])

if __name__=='__main__':unittest.main()
