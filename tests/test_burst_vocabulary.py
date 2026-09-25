#!/usr/bin/env python3
"""The two burst vocabularies must not drift apart.

`skills.py` decides which sounds the director is OFFERED, from the wikiskills
tables.  `timed_script.py` decides which round brackets are READ as sounds.  If
the second is narrower than the first, the server offers a label, the director
writes it, and the bracket is silently re-read as a delivery direction: the round
bracket becomes an instruction about how to speak and no sound is produced.  No
error, no warning, just a recipe that cannot fire.

That is exactly what had happened.  `BURST_LABELS` was a hard-coded list of 22
while the wiki carried 117 pages; of the 36 classes `Skills.offerable` actually
offered, **9 were unrecognised** — including `guffaw` (measured hit 0.633, the
fourth-best recipe in the bank) and `clears_throat` (0.48).

Run directly (`python3 tests/test_burst_vocabulary.py`) or under pytest.  Parses
under 3.9, which is what the login node has.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# Point the config at the wikiskills in this checkout before anything imports it,
# so the test is about this repo and not about whatever is on the server's disk.
WIKI = os.path.join(ROOT, "wikiskills")
os.environ.setdefault("MOSS_SKILLS_DIR", WIKI)

import config          # noqa: E402
import skills          # noqa: E402
import timed_script    # noqa: E402


def _wiki_labels():
    """Every label with a pattern page, i.e. everything a caller may ask for."""
    pat = os.path.join(config.SKILLS_DIR, "patterns")
    if not os.path.isdir(pat):
        return []
    return sorted(f[3:-3] for f in os.listdir(pat)
                  if f.startswith("vb-") and f.endswith(".md"))


def test_every_wiki_label_is_recognised():
    """No label the wiki knows about may be read as a delivery direction."""
    labels = _wiki_labels()
    assert labels, "no wikiskills pattern pages found -- test is not measuring anything"
    missed = [l for l in labels
              if not (timed_script._is_burst_label(l)
                      and timed_script._is_burst_label(l.replace("_", " ")))]
    assert not missed, (
        "%d of %d wikiskills burst labels are not recognised by "
        "_is_burst_label and would be performed as delivery directions: %s"
        % (len(missed), len(labels), ", ".join(missed)))


def test_every_offered_class_is_recognised():
    """The load-bearing half: what the director is actually told it may use.

    `Skills.offerable` is what reaches the system prompt.  Anything in there that
    `timed_script` does not recognise is a recipe the director will write and the
    server will silently discard.
    """
    s = skills.Skills()
    assert s.ok, "wikiskills did not parse"
    offered = [c for c, _ in s.offerable(_wiki_labels())]
    assert offered, "nothing offerable -- test is not measuring anything"
    missed = [c for c in offered
              if not timed_script._is_burst_label(c.replace("_", " "))]
    assert not missed, (
        "%d of %d offered burst classes would be read as delivery directions: %s"
        % (len(missed), len(offered), ", ".join(missed)))


def test_core_labels_survive():
    """The hard-coded fallback list still resolves, wiki present or not."""
    for lab in timed_script.BURST_LABELS:
        assert timed_script._is_burst_label(lab), lab
    # and the fuzzy fragment rule the core list exists for
    for frag in ("a soft chuckle", "a raw, tearing scream", "chuckle."):
        assert timed_script._is_burst_label(frag), frag


def test_fallback_without_wikiskills():
    """With no skills directory the core list is still the vocabulary."""
    vocab = timed_script.burst_vocabulary(os.path.join(ROOT, "no-such-dir"))
    assert set(timed_script.BURST_LABELS) <= vocab
    assert "chuckle" in vocab


def test_directions_are_not_read_as_bursts():
    """Widening the vocabulary must not turn ordinary directions into sounds.

    These are real delivery cues taken from the system prompt and docs.  Each is
    a round bracket WITHOUT a number that must stay a direction.
    """
    directions = [
        "amused", "casual", "conversational", "clearly amused",
        "dropping to a whisper", "voice tightening, barely holding it",
        "spitting the words out", "intensely amused, letting it out",
        "quietly, almost flat", "warmly, taking their time",
        "clearly amused, with a small chuckle",
    ]
    bad = [d for d in directions if timed_script._is_burst_label(d)]
    # "clearly amused, with a small chuckle" is the documented trap and is
    # matched by the pre-existing fuzzy rule; it is six words, so it is not.
    assert not bad, "these delivery directions would be performed as sounds: %s" % bad


def test_parse_makes_a_burst_of_an_offered_label():
    """End to end: an offered label in its own bracket becomes a burst item."""
    items = timed_script.parse("well now. (guffaw) that is the funniest thing "
                               "i have heard all week long.")
    kinds = [i[0] for i in items]
    assert "burst" in kinds, items
    burst = [i for i in items if i[0] == "burst"][0]
    assert burst[1] == "guffaw"
    assert burst[2] == timed_script.BURST_DEFAULT


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for t in tests:
        try:
            t()
            print("ok    %s" % t.__name__)
        except AssertionError as e:
            bad += 1
            print("FAIL  %s\n      %s" % (t.__name__, e))
    print("\n%d/%d passed" % (len(tests) - bad, len(tests)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main())


def test_a_burst_written_with_its_length_is_not_inserted_twice():
    """The regression that put two screams at the top of a line.

    The duplicate guard matched only brackets with no comma and no digit, so it
    saw "(scream)" and missed "(scream, 0.7 seconds)" — which is the form the
    prompt now asks for.  A director who wrote the burst correctly and then
    mentioned it in the following direction got a second one inserted.
    """
    import re
    import skills
    sk = skills.load()
    assert sk is not None and sk.ok
    line = ("(scream, 0.7 seconds) (intensely horrified, letting it out, voice "
            "breaking at the edge of another scream) Get away from the window!")
    out, added = sk.repair_script(line, ["scream"])
    assert added == []
    assert len(re.findall(r"\(scream[^)]*\)", out)) == 1
    # the old form still must not double either
    out2, added2 = sk.repair_script(line.replace("(scream, 0.7 seconds)",
                                                 "(scream)"), ["scream"])
    assert added2 == []
    assert len(re.findall(r"\(scream[^)]*\)", out2)) == 1
    # and a direction that only NAMES a burst still gets one, which is the
    # whole point of the repair
    bare = ("(a raw, tearing scream, completely overwhelmed) Get away from the "
            "window!")
    out3, added3 = sk.repair_script(bare, ["scream"])
    assert added3 == ["scream"]
    assert len(re.findall(r"\(scream\)", out3)) == 1
