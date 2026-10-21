"""Independent same-winner consensus for an existing enrolled identity only."""

def known_voice_consensus(teacher_id,score,margin,fast_vote,*,teacher_threshold=.5,
                          teacher_margin=.08,fast_threshold=.55):
    """A fast predictor may corroborate the teacher, never choose another ID.

    Mirrors the final-turn consensus policy, without a spoken-name shortcut or
    reducing the absolute teacher floor. Scores are not identity probabilities.
    """
    return bool(teacher_id and score>=max(.55,teacher_threshold) and
        margin>=max(.04,teacher_margin/2) and fast_vote.get('id')==teacher_id and
        fast_vote.get('similarity',-1)>=max(.60,fast_threshold) and
        fast_vote.get('margin',0)>=.10)
