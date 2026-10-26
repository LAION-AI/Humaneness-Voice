import unittest
from laya_policy import prepare,interpret,MODEL,REVISION

class LocalPolicyTests(unittest.TestCase):
    def answers(self,p=.1,q=.1):
        return {'stop_forward':{'probabilities':{'A':p,'B':1-p},'action':{'act_probability':1.}},
                'stop_reverse':{'probabilities':{'A':1-q,'B':q},'action':{'act_probability':1.}}}
    def test_explicit_commands_are_hybrid_but_mentions_and_negations_are_not(self):
        for text in ('Stopp!','Bitte sei mal leise.','Könntest du bitte kurz aufhören?'):
            self.assertEqual(interpret({'agent_speaking':True,'user_partial':text},self.answers(),[])[0]['action'],'stop_reply')
        for text in ('Was bedeutet Stopp?', 'Bitte hör nicht auf.', 'Do not stop speaking.', 'Mhm, ja.'):
            self.assertEqual(interpret({'agent_speaking':True,'user_partial':text},self.answers(),[])[0]['action'],'listen')
    def test_counterbalanced_score_not_unusable_act_probability_controls_ambiguous_requests(self):
        context={'agent_speaking':True,'user_partial':'Abends stop'}
        self.assertEqual(interpret(context,self.answers(.95,.62),[])[0]['action'],'stop_reply')
        self.assertEqual(interpret(context,self.answers(.95,.20),[])[0]['action'],'listen')
    def test_explicit_continue_request_overrides_high_stop_probability(self):
        for text in ('Bitte hör nicht auf.','Do not stop speaking.','Keep talking, please.'):
            self.assertEqual(interpret({'agent_speaking':True,'user_partial':text},self.answers(.99,.99),[])[0]['action'],'listen')
    def test_no_rules_cannot_invent_rule_ids_and_context_is_bounded(self):
        self.assertEqual(interpret({'agent_speaking':False}, {},[])[0],{'action':'listen','rule_id':None})
        state,questions,ids=prepare({'agent_speaking':True,'agent_full_reply':'x'*10000,'user_partial':'y'*10000})
        self.assertLess(len(state),2000);self.assertIn('actually played',state);self.assertEqual(len(questions),2);self.assertFalse(ids)
        self.assertTrue(MODEL.endswith('laya-multilingual'));self.assertEqual(len(REVISION),40)

if __name__=='__main__':unittest.main()
