"""Actual worker schedule: ten seeds per identical condition, paired arms."""
from collections import defaultdict
import inspect

from catalogue_steering_study import tasks, make_rows, lifespan


def test_ten_actual_seeds_per_identical_prompt_and_matched_reference_arms():
    with_ref=make_rows('emolia_c0542','collection')
    without_ref=make_rows('none','collection')
    groups=defaultdict(list)
    for a,b in zip(with_ref,without_ref):
        assert a['prompt']==b['prompt'] and a['seed']==b['seed']
        assert a['reference_present'] and not b['reference_present']
        assert not a['reference_identity'] and not b['reference_identity']
        assert 'speaker' not in a['prompt'].lower()
        groups[a['condition_id']].append(a)
    assert len(groups)==10
    for group in groups.values():
        assert len(group)==10 and len({r['seed'] for r in group})==10
        assert len({r['prompt'] for r in group})==1
    for repeat in range(10):
        batchrows=[r for r in with_ref if r['repeat']==repeat]
        for start in (0,4,8):
            chunk=batchrows[start:start+4]
            assert len({r['seed'] for r in chunk})==1
            assert len({r['condition_id'] for r in chunk})==len(chunk)


def test_collection_and_validation_disjoint_prompts_and_actual_seeds():
    collection=tasks('collection');validation=tasks('validation')
    assert not {r['seed'] for r in collection}&{r['seed'] for r in validation}
    assert not {r['text'] for r in collection}&{r['text'] for r in validation}
    assert len(validation)==10


def test_async_initializer_passes_callable_instead_of_calling_it():
    assert 'to_thread(initialize_scores)' in inspect.getsource(lifespan.__wrapped__)
