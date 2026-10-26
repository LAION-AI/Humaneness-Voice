"""Anchored short stop imperatives, not any occurrence of a stop-related word."""
import re
from speaker_turns import words

def has_stop_cue(text):
    """Admission to a semantic check, NOT permission to stop playback.

    Do not ask a stochastic classifier on every ordinary fragment: repeated
    guesses eventually turn fluent continuing speech into a false Stop.
    """
    return bool(re.search(r'\b(?:stopp?\w*|halte?|pause|warte\w*|aufhör\w*|auf|leise|still|ruhe|reicht|genügt|klappe|schnauze|unterbr\w*|ausreden|'
        r'wait|quiet|silence|enough|interrupt\w*|hold|speak|question)\b',' '.join(words(text))))

def is_stop_command(text):
    value=' '.join(words(text))
    value=re.sub(r'^(?:(?:hey|hallo|hi)\s+)?r[oa]b{1,2}(?:y|ie|i|o)\s+','',value)
    value=re.sub(r'^(?:(?:bitte|please)\s+){1,3}','',value)
    value=re.sub(r'\s+(?:bitte|please)$','',value)
    # ASR often joins the separable infinitive; keep this exact and anchored,
    # rather than fuzzy-matching arbitrary sentences containing "stop".
    value=re.sub(r'\baufzu(reden|sprechen)\b',r'auf zu \1',value)
    # Bounded ASR homophone, not fuzzy substring matching. A literal person
    # mention ("Herr Müller ...") or ordinary sentence must never qualify.
    value=re.sub(r'^(?:herr|her) ((?:bitte |doch |mal |jetzt ){0,3})auf$',r'hör \1auf',value)
    # Closing the floor is a stop request too. Keep complete, anchored
    # utterances: "Danke" alone and "das reicht nicht" must keep listening.
    if re.fullmatch(r'(?:(?:danke(?: schön)?|dankeschön|vielen dank) )?(?:das|es|cas) (?:reicht|genügt)(?: (?:mir|jetzt|erstmal|erst mal|schon|völlig|danke)){0,2}',value):return True
    if re.fullmatch(r'halte? (?:(?:bitte|doch|mal|jetzt|endlich) ){0,3}(?:die|deine|deinen) (?:klappe|schnauze|mund)',value):return True
    if re.fullmatch(r'(?:stopp?\s*){1,5}',value):return True
    if re.fullmatch(r'(?:(?:stopp?\s+){1,5})?(?:bitte\s+)?höre?\s+(?:(?:bitte|doch|mal|jetzt|endlich)\s+){0,3}auf(?:\s+zu\s+(?:reden|sprechen))?',value):return True
    return bool(re.fullmatch(r'(?:stopp?|halt|pause|ruhe|warte(?:\s+(?:bitte|kurz|mal|einen moment|einen augenblick)){0,3}|wait(?: a moment| a second)?|hold on|hör auf|höre auf|hör bitte auf|'
        r'höre? auf zu (?:reden|sprechen)|sei (?:mal )?(?:leise|still)|sei bitte (?:leise|still)|'
        r'sei mal bitte leise|stopp? bitte hör auf|halt bitte hör auf|'
        r'stop (?:talking|speaking)|be quiet|stop please|enough)',value))

def has_stop_intent(text):
    """A direct stop may precede a new request; mentions/questions are not stops."""
    if is_stop_command(text):return True
    value=' '.join(words(text))
    value=re.sub(r'^(?:(?:hey|hallo|hi)\s+)?r[oa]b{1,2}(?:y|ie|i|o)\s+','',value)
    value=re.sub(r'^(?:(?:bitte|please)\s+){1,3}','',value)
    if re.match(r'^(?:stopp?|halte?|pause)\s+(?:bitte\s+)?nicht\b',value):return False
    if re.match(r'^halte?\s+.*\b(?:klappe|schnauze|mund)\s+nicht\b',value):return False
    if re.match(r'^(?:stopp?|halt|pause)\b',value):
        return not re.match(r'^(?:stopp?|halt|pause)\s+(?:motion|ist|is|heißt|bedeutet|means|word|wort|zeichen|schild)\b',value)
    return bool(re.match(r'^(?:(?:nein\s+)?lass\s+mich\s+(?:(?:bitte|kurz|doch|jetzt)\s+){0,3}(?:ausreden|sprechen|zu\s+wort\s+kommen)\b|warte\s+(?:(?:bitte|kurz|mal)\s+)*(?:ich|lass|einen moment|einen augenblick)\b|höre?\s+(?:(?:bitte|doch|mal|jetzt|endlich)\s+){0,3}auf\b|'
        r'sei\s+(?:(?:bitte|doch|mal|jetzt)\s+){0,3}(?:leise|still)\b|'
        r'(?:kannst|könntest|würdest)\s+du\s+(?:bitte\s+)?(?:kurz\s+)?aufhören\b|'
        r'(?:can|could|would)\s+you\s+(?:please\s+)?stop\b|be quiet\b|(?:wait|hold on)\s+(?:please\s+)?(?:i|let me)\b)',value))

def is_brief_stop_command(text):
    """Only a single imperative, not a clipped tail of a longer stop phrase."""
    value=' '.join(words(text))
    value=re.sub(r'^(?:(?:hey|hallo|hi)\s+)?r[oa]b{1,2}(?:y|ie|i|o)\s+','',value)
    value=re.sub(r'^(?:(?:bitte|please)\s+){1,3}','',value)
    value=re.sub(r'\s+(?:bitte|please)$','',value)
    return value in ('stop','stopp','halt','pause','ruhe','enough','warte','wait','hold on') or bool(re.fullmatch(r'(?:stopp?\s*){1,5}',value))
