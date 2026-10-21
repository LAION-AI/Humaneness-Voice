"""Real configured semantic decision API, synthetic dialogue only, no mic."""
import asyncio,hashlib,json,time
from pathlib import Path
from live_decisions import DecisionClient,HELPFUL_ID,HELPFUL_CONDITION

async def main():
    client=DecisionClient();report={'started':time.time(),'passed':False,'model':client.label(),
        'synthetic_text_only':True,'identity_authorization_tested_separately':True,'cases':[]}
    target=Path(__file__).parent/'logs/semantic-decision-checks.json'
    if client.base.endswith(':8797'):
        import httpx
        from laya_policy import MODEL,REVISION,SOURCE_COMMIT
        async with httpx.AsyncClient() as http:
            health=(await http.get(client.base+'/health')).json()
        report['local_model_health']=health
        report['local_model_verified']=bool(health.get('ready') and health.get('model')==MODEL and health.get('revision')==REVISION
            and health.get('source_commit')==SOURCE_COMMIT and health.get('policy_sha256')==hashlib.sha256((Path(__file__).parent/'laya_policy.py').read_bytes()).hexdigest())
    for label,user,speaking,rules,expected in [
        ('polite_stop','Könntest du bitte kurz aufhören? Ich möchte etwas fragen.',True,[],'stop_reply'),
        ('stop_then_request','Stopp, erzähle mir lieber von Äpfeln.',True,[],'stop_reply'),
        ('echo_prefix_stop','Abends stop',True,[],'stop_reply'),
        ('quoted_mention','Was bedeutet das Wort Stopp?',True,[],'listen'),
        ('agreed_semantic','Jetzt erzähle ich von gelben Bananen und esse sie jeden Tag.',False,
         [{'id':'banana-game','condition':'Interrupt when the user talks about bananas.','reply':'Stopp, stopp!'}],'interrupt_user'),
        ('no_agreement','Bananen sind gesund.',False,[],'listen'),
        ('useful_invitation_de','Warum unterbrichst du mich nicht? Ich habe dich doch gebeten, mich zu unterbrechen. Sag doch jetzt etwas.',False,
         [{'id':HELPFUL_ID,'condition':HELPFUL_CONDITION}],'interrupt_user'),
        ('useful_invitation_en','I am going round in circles. Please jump in now and help me decide what to do next.',False,
         [{'id':HELPFUL_ID,'condition':HELPFUL_CONDITION}],'interrupt_user'),
        ('ordinary_question','Kannst du mir bitte erklären, wie ein Elektromotor funktioniert und warum er sich dreht?',False,
         [{'id':HELPFUL_ID,'condition':HELPFUL_CONDITION}],'listen'),
        ('let_me_finish','Darf ich bitte weiterreden? Ich bin noch nicht fertig mit meiner Erklärung.',False,
         [{'id':HELPFUL_ID,'condition':HELPFUL_CONDITION}],'listen')]:
        context={'user_partial':user,'agent_speaking':speaking,'agent_full_reply':'Here is a long explanation of apples and how they grow.',
            'agent_played_seconds':1.25,'agent_generated_seconds':4.,'history':[],
            'rules':rules,'proactive_enabled':True}
        start=time.perf_counter()
        try:
            result=await client.choose(context)
            passed=result['action']==expected and (expected!='interrupt_user' or result['rule_id']==rules[0]['id'])
            row={'case':label,'result':result,'expected':expected,'passed':passed}
        except Exception as exc:row={'case':label,'passed':False,'error':type(exc).__name__+': '+str(exc)}
        row['ms']=round((time.perf_counter()-start)*1000,1);report['cases'].append(row)
        target.write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
    report.update(passed=all(r['passed'] for r in report['cases']),finished=time.time(),
        explicit_stop_and_controls_passed=all(r['passed'] for r in report['cases'] if r['case']!='agreed_semantic'))
    target.write_text(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit('Semantic decision checks failed; retained results')

if __name__=='__main__':asyncio.run(main())
