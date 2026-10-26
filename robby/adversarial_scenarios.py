"""Predeclared stateful test expectations; never derived from model outputs."""

def scenarios():
    cases=[{
        'id':'conversation_thread','speaker':'Christoph','intro':'christoph-intro',
        'description':'Noisy spoken jokes, anaphoric follow-up, real Wikipedia lookup, then return to the earlier topic.',
        'steps':[
            {'id':'joke','clip':'task-joke','snr_db':12,'expect':{'min_words':25,'required':['(?:F|Ph)ip+s','bahnhof|zug|bahnsteig']}},
            {'id':'another','clip':'task-another','snr_db':12,'expect':{'min_words':25,'required':['(?:F|Ph)ip+s','bahnhof|zug|bahnsteig'],'different_from_previous':True}},
            {'id':'wikipedia','clip':'task-wiki','snr_db':12,'max_submissions':2,'expect':{'min_words':20,'required':['pinguin'],'source_host':'wikipedia.org'}},
            {'id':'recall_after_tool','clip':'task-recall','snr_db':12,'expect':{'min_words':7,'required':['(?:F|Ph)ip+s','bahnhof|zug|bahnsteig']}},
        ]}, {
        'id':'long_tasks_and_live_tools','speaker':'Christoph','intro':'christoph-intro',
        'description':'Eight-line poem, weather and current news; actual provider/tool/TTS outputs are retained.',
        'steps':[
            {'id':'eight_line_poem','clip':'task-poem','snr_db':15,'expect':{'min_words':35,'required':['robot|blech|robby','mond'],'min_poem_lines':8}},
            {'id':'weather','clip':'task-weather','snr_db':15,'max_submissions':2,'expect':{'min_words':12,'required':['berlin','grad|temperatur|regen|sonne|bewölk|wolke'],'source_host':'open-meteo.com'}},
            {'id':'news','clip':'task-news','snr_db':15,'expect':{'min_words':25,'source_host':'tagesschau.de'}},
        ]}, {
        'id':'noisy_group_permissions','speaker':'Christoph','intro':'christoph-intro',
        'description':'Spoken guest invitation, real guest, owner return, revocation and rejected guest under background speech.',
        'steps':[
            {'id':'invite','clip':'focus-invite-alice','snr_db':15,'max_submissions':2,'expect':{'permission':'invite','min_words':3}},
            {'id':'alice_joins','clip':'selma-followup-clear','snr_db':15,'speaker':'Alice','expect':{'min_words':5,'participants':['Christoph','Alice']}},
            {'id':'owner_returns','clip':'christoph-followup','snr_db':15,'speaker':'Christoph','expect':{'min_words':20}},
            {'id':'revoke','clip':'focus-ignore-onlyme','snr_db':15,'max_submissions':2,'expect':{'min_words':3,'participants_exact':['Christoph']}},
            {'id':'revoked_guest','kind':'reject','clip':'selma-followup-clear','gain':1.,'seconds_after':3.},
            {'id':'tv_cannot_invite','kind':'reject','clip':'focus-background-invite','gain':.12,'seconds_after':3.},
        ]}, {
        'id':'interruptions','speaker':'Christoph','intro':'christoph-intro',
        'description':'Stop/Wait at four encoder-hop offsets, background spoof control and negated Stop.',
        'steps':[
            {'id':'tv_spoof_during_reply','kind':'negative_control','clip':'tv-spoof','gain':.12},
            {'id':'negated_stop','kind':'negative_control','clip':'keep-speaking','gain':1.},
            *[{'id':f'{cue}_phase_{phase}','kind':'stop','clip':cue,'phase_ms':phase,'snr_db':12}
                for cue in ('christoph-stop','christoph-wait') for phase in (0,80,160,240)],
            {'id':'resume_after_stops','clip':'task-continue','snr_db':15,'max_submissions':2,'expect':{'min_words':25,'required':['lotti|kätz|katze']}},
        ]}, {
        'id':'background_only','speaker':'Christoph','intro':'christoph-intro',
        'description':'No main-speaker input: television-style speech with names, commands, bananas and an unauthorized invitation.',
        'steps':[{'id':'spoof_no_reply','kind':'reject','clip':'tv-spoof','gain':.12,'seconds_after':4.},
                 {'id':'background_invitation','kind':'reject','clip':'focus-background-invite','gain':.12,'seconds_after':4.}],
        },
    ]
    for speaker, intro, clip, word in [('Christoph','christoph-intro','christoph-followup','geschichte'),
                                     ('Selma','selma-intro','task-selma','katze|katz')]:
        cases.append({'id':'acoustic_'+speaker.casefold(),'speaker':speaker,'intro':intro,
            'description':'Foreground over two other voices, continuous music/noise/impacts; stronger overlap is a diagnostic boundary.',
            'steps':[{'id':f'snr_{snr}','clip':clip,'snr_db':snr,'speed':1.04 if snr==6 else 1.,
                      'reverb_s':.18 if snr==6 else 0.,'required':snr>=12,
                      'expect':{'min_words':18,'required':[word]}}
                     for snr in (20,12,6,0)]})
    return cases


def validate_plan(plan):
    ids=set()
    for scenario in plan:
        if scenario['id'] in ids:raise ValueError('Duplicate scenario ID')
        ids.add(scenario['id']); steps=set()
        for step in scenario['steps']:
            if step['id'] in steps:raise ValueError('Duplicate step ID')
            steps.add(step['id'])
            if step.get('kind','speak') not in ('speak','stop','negative_control','reject'):
                raise ValueError('Unknown step kind')
            if step.get('phase_ms',0) not in (0,80,160,240):raise ValueError('Invalid hop phase')
    return plan
