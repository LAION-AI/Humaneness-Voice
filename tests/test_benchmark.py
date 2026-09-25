"""The benchmark-item path: detection, verbatim safety, breath."""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import benchmark as B

ITEM = {
    "id": "EMO-001-MODERATE-01",
    "target": {"label": "Amusement", "intensity": "moderate",
               "valence": "positive", "arousal": "medium-high",
               "descriptors": ["mirth", "playfulness"]},
    "instruction": {"context": "A coworker discovers the elevator plays their "
                               "old voicemail greeting as music.",
                    "performance_direction": "Keep the amusement tucked behind "
                                             "a courteous office voice.",
                    "vocal_burst": None},
    "script": {"text": "The elevator is playing your old voicemail greeting "
                       "between floors. It says, “Leave a message,” and then the "
                       "doors open as if nothing happened. I promise I will not "
                       "tell anyone until after lunch while the lobby is this "
                       "crowded.", "sentence_count": 3},
}


def test_detect_plain_bare_and_fenced():
    raw = json.dumps(ITEM)
    assert B.detect(raw) is not None
    assert B.detect("```json\n" + raw + "\n```") is not None
    # pasted out of an array, so with a trailing comma, and with a lead-in
    assert B.detect("perform this:\n" + raw + ",") is not None


def test_detect_ignores_ordinary_chat():
    assert B.detect("What do you think happens after we die?") is None
    assert B.detect("") is None
    # JSON that is not an item
    assert B.detect('{"a": 1, "b": [2, 3]}') is None


def test_brief_carries_the_words_and_never_the_json():
    b = B.brief(ITEM)
    assert B.script_text(ITEM) in b
    assert '"target"' not in b and "descriptors" not in b
    assert "clearly amused" in b          # noun mapped to the adjective
    assert "PERFORMANCE, NOT A CONVERSATION" in b


def test_verbatim_ok_accepts_annotation_only():
    good = ("(clearly amused) The elevator is playing your old voicemail "
            "greeting between floors. [0.4 seconds pause] It says, “Leave a "
            "message,” and then the doors open as if nothing happened. "
            "(chuckle, 0.3 seconds) I promise I will not tell anyone until "
            "after lunch while the lobby is this crowded.")
    assert B.verbatim_ok(good, ITEM)
    assert not B.verbatim_ok(good.replace("elevator", "lift"), ITEM)
    assert not B.verbatim_ok("Oh no, that is so embarrassing for them!", ITEM)


def test_annotate_is_verbatim_and_directed():
    a = B.annotate(ITEM)
    assert B.verbatim_ok(a, ITEM)
    assert a.count("(") == 3                      # one direction per sentence
    assert "amusement" not in a                   # never the noun form
    assert B._has_inner_pause(a)


def test_breathe_only_fires_when_there_is_no_inner_pause():
    flat = ("(clearly amused) I promise I will not tell anyone until after "
            "lunch, while the lobby is this crowded and loud.")
    out, n = B.breathe(flat)
    assert n == 1 and B._has_inner_pause(out)
    # a pause the model chose is left exactly as it is
    chosen = ("(clearly amused) I promise I will not tell anyone "
              "[0.7 seconds pause] until after lunch today.")
    assert B.breathe(chosen) == (chosen, 0)


def test_breathe_never_touches_the_words_or_the_brackets():
    flat = ("(clearly amused) I promise I will not tell anyone until after "
            "lunch, while the lobby is this crowded and loud.")
    out, _ = B.breathe(flat)
    assert B._norm(out) == B._norm(flat)
    assert out.count("(clearly amused)") == 1


def test_a_breathed_script_still_renders_with_a_correct_token_sum():
    import re
    import timed_script
    out, _ = B.breathe("(clearly amused) I promise I will not tell anyone "
                       "until after lunch, while the lobby is this crowded.")
    tagged, frames, _ = timed_script.render(out)
    secs = sum(float(x) for x in
               re.findall(r"\[([0-9.]+) seconds (?:pause|duration)\]", tagged))
    assert abs(secs * 12.5 - frames) < 1
