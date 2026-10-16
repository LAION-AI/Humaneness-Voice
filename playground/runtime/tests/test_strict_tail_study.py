import strict_tail_study as study


def test_collection_cell_repetition_and_actual_seed_counts():
    rows=study.blocks('collection')
    assert len(rows)==4000
    cells={r['condition_id'] for r in rows}
    assert len(cells)==20
    assert len({r['block'] for r in rows})==4000
    for cell in cells:
        data=[r for r in rows if r['condition_id']==cell]
        assert len({r['seed'] for r in data})==200
        assert {r['repeat'] for r in data}==set(range(200))
        assert len({r['prompt'] for r in data})==1


def test_heldout_text_and_seed_sets_are_disjoint():
    phases=[study.blocks(p) for p in ('collection','screen','confirm')]
    for a,b in ((0,1),(0,2),(1,2)):
        assert not ({r['text'] for r in phases[a]} & {r['text'] for r in phases[b]})
        assert not ({r['seed'] for r in phases[a]} & {r['seed'] for r in phases[b]})
    assert len(phases[1])==len(phases[2])==40


def test_fifty_one_percent_alias():
    assert study.actual(50,1)==study.actual(50,2)
    assert study.actual(200,1)!=study.actual(200,2)
