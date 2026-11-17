import numpy as np

from possession.external import rules
from possession.external.dfl import AWAY, DEAD, HOME


def _labels(*runs):
    return np.concatenate([np.full(n, lab, dtype=np.int8) for lab, n in runs])


def test_a_short_flip_without_control_is_merged_back():
    lab = _labels((HOME, 200), (AWAY, 20), (HOME, 200))            # 0.8 s blip for away
    out = rules.rule_labels(lab, [(0, len(lab) - 1)], [])
    assert (out == HOME).all()


def test_a_long_flip_is_kept():
    lab = _labels((HOME, 200), (AWAY, 100), (HOME, 200))           # 4 s: keeps the ball
    out = rules.rule_labels(lab, [(0, len(lab) - 1)], [])
    assert (out[200:300] == AWAY).all()


def test_a_short_flip_with_two_ball_events_is_kept():
    lab = _labels((HOME, 200), (AWAY, 30), (HOME, 200))
    ev = [(205, AWAY, False), (225, AWAY, False)]
    out = rules.rule_labels(lab, [(0, len(lab) - 1)], ev)
    assert (out[200:230] == AWAY).all()


def test_a_single_event_is_not_enough_but_a_strong_one_is():
    lab = _labels((HOME, 200), (AWAY, 30), (HOME, 200))
    one = rules.rule_labels(lab, [(0, len(lab) - 1)], [(210, AWAY, False)])
    strong = rules.rule_labels(lab, [(0, len(lab) - 1)], [(210, AWAY, True)])
    assert (one == HOME).all() and (strong[200:230] == AWAY).all()


def test_first_possessor_after_dead_ball_is_kept_and_dead_stays():
    lab = _labels((HOME, 100), (DEAD, 50), (AWAY, 10), (HOME, 100))
    out = rules.rule_labels(lab, [(0, len(lab) - 1)], [])
    assert (out[150:160] == AWAY).all() and (out[100:150] == DEAD).all()


def test_event_rows_maps_time_to_frames_and_drops_outside_halves():
    ev = [(100.0, 0, False), (101.0, 1, True), (5000.0, 0, False)]
    rows = rules.event_rows(ev, [100.0], [(10, 1000)])
    assert rows == [(10, 0, False), (35, 1, True)]
