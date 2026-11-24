"""Rule wording, alert trigger and the reference rule engine (CPU only, no model):  python -m pytest tests"""
import pytest

from glance_stream.rules import Rule, Trigger, violations


def test_templates_match_training_wording():
    # The model was trained and evaluated with exactly these sentences (plus " Answer yes or no.").
    assert Rule.now("someone is cutting vegetables").question() == \
        "Is this happening right now: someone is cutting vegetables? Answer yes or no."
    assert Rule.must_follow("open fridge", "close fridge", within=30).template() == \
        "Rule: whenever 'open fridge' happens, 'close fridge' must follow within 30 seconds. Is the rule being violated right now?"
    assert Rule.after("A", "B").template() == \
        "Rule: after 'A' has happened, 'B' is not allowed. Is the rule being violated right now?"
    assert Rule.requires_before("B", "A", within=15).template() == \
        "Rule: 'B' is only allowed if 'A' happened in the 15 seconds before it. Is the rule being violated right now?"
    assert Rule.at_most("X", times=2, within=60).template() == \
        "Rule: 'X' may happen at most 2 times in any 60 seconds. Is the rule being violated right now?"
    assert Rule.forbidden_between("A", "C", "B").template() == \
        "Rule: between 'A' and the next 'C', 'B' is not allowed. Is the rule being violated right now?"
    assert Rule.custom("Is the stove on?").question() == "Is the stove on? Answer yes or no."


def test_rule_validation():
    with pytest.raises(ValueError):
        Rule("now")
    with pytest.raises(ValueError):
        Rule("sometimes", X="a")
    assert Rule.now("x", name="n").name == "n"


def test_trigger_k_and_cooldown():
    tr = Trigger(threshold=0.5, k=2, cooldown_s=5.0)
    fired = [t for t, p in [(0.5, .9), (1.0, .9), (1.5, .9), (2.0, .9), (6.0, .9), (6.5, .9), (7.0, .2)] if tr.update(t, p)]
    assert fired == [1.0, 6.0]  # the run keeps counting during the cooldown


def test_reference_engine():
    ev = [("A", 10, 12), ("B", 5, 6), ("B", 20, 22), ("C", 30, 31), ("B", 40, 41), ("A", 100, 101),
          ("X", 0, 1), ("X", 50, 51), ("X", 60, 61), ("X", 200, 201)]
    T = 300
    assert violations(Rule.now("A"), ev, T) == [[10, 12], [100, 101]]
    assert violations(Rule.after("A", "B"), ev, T) == [[20, 22], [40, 41]]
    assert violations(Rule.requires_before("B", "A", within=15), ev, T) == [[5, 6], [40, 41]]
    assert violations(Rule.must_follow("A", "B", within=5), ev, T) == [[15, 20], [105, 300]]
    assert violations(Rule.must_follow("A", "B", within=15), ev, T) == [[115, 300]]
    ev2 = [("A", 0, 1), ("A", 50, 51), ("B", 80, 81)]  # a second A resets the first deadline
    assert violations(Rule.must_follow("A", "B", within=10), ev2, 200) == [[10, 50], [60, 80]]
    assert violations(Rule.now("Z"), [("Z", 3.0, 3.02)], 10) == []  # < 0.5 s dropped
    assert violations(Rule.at_most("X", times=2, within=70), ev, T) == [[60, 70]]
    assert violations(Rule.at_most("X", times=1, within=20), ev, T) == [[60, 70]]
    assert violations(Rule.forbidden_between("A", "C", "B"), ev, T) == [[20, 22]]
    assert violations(Rule.forbidden_between("A", "C", "X"), ev, T) == [[200, 201]]
