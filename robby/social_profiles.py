"""Session social context and optional browser-owned acoustic prototypes.

Names are user claims; audible traits are uncertain observations. No vector,
face image or raw recording is sent to a language model. Persistent export is
an explicit separate browser setting, not implied by the factual wiki setting.
"""
import json,math,re,uuid
import numpy as np

TRAITS={'AGEV':'perceived vocal age','GEND':'perceived vocal gender','REGS':'vocal register',
        'BRGT':'brightness','DARC':'articulation darkness','R_CHST':'chest resonance',
        'R_THRT':'throat resonance','FULL':'fullness','ROUG':'roughness','WARM':'warmth'}
SOCIAL_NOTE='''SOCIAL ROLE GROUNDING: You are Robby, the robot assistant. The current
speaker is the HUMAN USER, never you. In a USER message, "ich / I / mein / my"
refers to that user; in YOUR response, "ich / I" refers to Robby and "du / you"
refers to the user. A user saying "Ich bin Christoph" tells you the USER's name.
Correct: "Hallo Christoph!" or "Du bist Christoph." Never "Ich bin Christoph."
Take an identity correction literally and seriously, even if an earlier assistant
turn got it wrong. Briefly acknowledge the corrected user's name and your own
role. Never dismiss the correction with a joke such as "Ach du Scherzkeks!".
For "Und wer bin ich?", use the supplied named current-person match. Don't ask
again for a name already given. Don't add awkward evidential filler such as
"wie du dir selbst genannt hast". Prefer simple, idiomatic German, e.g. "Du
heißt Christoph." If truly uncertain, ask whether this is the known person,
not assert they are someone new. An acoustic variation is not proof of another
person. Names in quoted text or stories do not change either participant's identity.
Be curious, warm and attentive; complete the actual request and preserve its topic.
A name question is optional, not an enrollment ritual. Ask it only when needs_name
is true; otherwise do not repeat it. Don't force a question onto every answer.
Vocal profiles describe how a voice MAY sound, not actual age, sex, gender identity,
health, character or feelings. They never prove identity. User corrections win.
Use those hints tactfully when relevant; don't volunteer demographic judgments or
recite scores. Unknown background speech might be a television or the same person.
'''

def context_note(identity):
    identity=identity or {}
    data={'assistant_name':'Robby','current_person':identity.get('display_name') if identity.get('named') else None,
          'self_introduction':bool(identity.get('self_introduction')),'needs_name':bool(identity.get('needs_name')),
          'identity_uncertain':bool(identity.get('identity_uncertain'))}
    if identity.get('social_profile'):data['current_voice_profile']=identity['social_profile']
    if identity.get('known_people'):data['known_session_people']=identity['known_people']
    return '\n'+SOCIAL_NOTE+'\nRole and person context (DATA): '+json.dumps(data,ensure_ascii=False)

def identity_answer(text,identity):
    """Narrow registered-name lookup; general memory questions remain generative."""
    if not identity or not identity.get('enrolled'):return None
    query=re.sub(r'[^\w\s]','',text.casefold()).strip()
    german=bool(re.fullmatch(r'(?:und )?(?:wer bin ich|wie heiße ich|wie heisse ich|weißt du (?:noch )?wie ich heiße|weisst du (?:noch )?wie ich heisse|kennst du meinen namen)',query))
    english=bool(re.fullmatch(r'(?:and )?(?:who am i|what is my name|whats my name|do you know my name)',query))
    if not (german or english):return None
    if not identity.get('named'):
        return 'Ich bin mir nicht sicher. Wie heißt du?' if german else 'I am not sure. What is your name?'
    name=identity.get('display_name','')
    if not isinstance(name,str) or not 1<=len(name)<=60:return None
    if identity.get('identity_uncertain'):
        return f'Ich glaube, du bist {name}. Stimmt das?' if german else f'I think you are {name}. Is that right?'
    return f'Du heißt {name}.' if german else f'Your name is {name}.'

def background_question(text):
    """Narrow output guard, not a general semantic truth/accuracy judge."""
    return bool('?' in text and re.search(r'stimm|fernseh|\btv\b|hintergrund|jemand|spr[ei]ch|allein|voice|background|someone|television|alone',text,re.I))

def observe(user,delivery,enabled=True):
    if not enabled:return
    values=delivery.get('vocal_traits',{}) if delivery.get('status')=='available' else {}
    clean={k:float(v['score']) for k,v in values.items() if k in TRAITS and isinstance(v,dict) and type(v.get('score')) in (int,float) and math.isfinite(v['score']) and 0<=v['score']<=6}
    if not clean:return
    rows=user.setdefault('vocal_observations',[]);rows.append(clean);del rows[:-12]

def describe(user,enabled=True):
    if not enabled:return None
    rows=user.get('vocal_observations',[])
    if not rows:return None
    def bucket(k,x):
        if k=='GEND':return 'more feminine-sounding' if x<2.5 else 'more masculine-sounding' if x>=3.5 else 'androgynous-sounding'
        if k=='AGEV':return 'younger-sounding' if x<2.5 else 'more mature-sounding' if x>=3.5 else 'adult-sounding'
        if k=='REGS':return 'lower vocal register' if x<2.5 else 'higher vocal register' if x>=3.5 else 'middle vocal register'
        return ('lower ' if x<2.5 else 'higher ' if x>=3.5 else 'moderate ')+TRAITS[k]
    traits=[]
    for k in TRAITS:
        xs=[r[k] for r in rows if k in r]
        if xs:traits.append({'dimension':k,'description':bucket(k,float(np.median(xs))),
            'mean_score':round(float(np.mean(xs)),3),'spread':round(float(np.std(xs)),3),'samples':len(xs)})
    return {'observations':len(rows),'traits':traits,'uncertainty':'Approximate audible impressions from accepted turns; not demographic facts or identity proof.'}

def export_profiles(memory):
    result=[]
    for sid,u in memory.users.items():
        if not u.get('named') or not u.get('vectors'):continue
        if len(result)>=8:break
        pid=u.setdefault('person_id','voice-'+uuid.uuid4().hex)
        result.append({'person_id':pid,'name':u['name'],'vectors':[v.tolist() for v in u['vectors'][:8]],
            'vocal_observations':u.get('vocal_observations',[])[-12:]})
    return result

def import_profiles(memory,profiles):
    if not isinstance(profiles,list) or len(profiles)>8:raise ValueError('At most eight profiles')
    clean=[]
    for p in profiles:
        if not isinstance(p,dict) or not re.fullmatch(r'voice-[a-f0-9]{32}',str(p.get('person_id',''))):raise ValueError('Invalid person ID')
        name=p.get('name');vectors=p.get('vectors')
        if not isinstance(name,str) or not re.fullmatch(r"[^\W\d_][\w '\-]{0,59}",name,re.UNICODE):raise ValueError('Invalid name')
        if not isinstance(vectors,list) or not 1<=len(vectors)<=8:raise ValueError('Invalid prototypes')
        refs=[]
        for v in vectors:
            a=np.asarray(v,np.float32)
            if a.shape!=(192,) or not np.isfinite(a).all() or not .5<float(np.linalg.norm(a))<1.5:raise ValueError('Invalid ECAPA prototype')
            refs.append(a/np.linalg.norm(a))
        observations=p.get('vocal_observations',[])
        if not isinstance(observations,list) or len(observations)>12:raise ValueError('Invalid voice observations')
        holder={}
        for r in observations:
            if not isinstance(r,dict):raise ValueError('Invalid trait record')
            observe(holder,{'status':'available','vocal_traits':{k:{'score':v} for k,v in r.items()}})
        clean.append((p['person_id'],name,refs,holder.get('vocal_observations',[])))
    for pid,name,refs,observations in clean:
        same=next((u for u in memory.users.values() if u.get('person_id')==pid or (u['named'] and u['name'].casefold()==name.casefold())),None)
        if same is None:
            if len(memory.users)>=memory.max_users:break
            memory.counter+=1;sid='speaker-'+str(memory.counter)
            same={'name':name,'named':True,'vectors':[],'asked_turn':-100,'turns':0,'seen':memory.clock()};memory.users[sid]=same
        same.update(person_id=pid,vectors=refs,vocal_observations=observations,seen=memory.clock())
    return len(clean)
