from possession import tagging
from possession.params import Params
from possession.pipeline import decode_match
from synth import make_match


def test_side_of_fuzzy_names_and_na():
    assert tagging.side_of("Barcelona", "FC Barcelona", "Athletic Club") == 1
    assert tagging.side_of("Atletic Club", "FC Barcelona", "Athletic Club") == -1
    assert tagging.side_of("N/A", "FC Barcelona", "Athletic Club") == 0
    assert tagging.side_of(None, "FC Barcelona", "Athletic Club") == 0


def test_factors_set_piece_and_phase():
    inst = [
        {"code": "set pieces", "start": 100, "end": 200, "tags": {"Team": "Barcelona", "Type": "Corner Kick"}},
        {"code": "build up", "start": 300, "end": 400, "tags": {"Team": "Atletic Club"}},
        {"code": "set pieces", "start": 500, "end": 600, "tags": {"Team": "N/A", "Type": "Throw-in"}},
    ]
    ev, dead = tagging.factors(inst, 700, "FC Barcelona", "Athletic Club", Params())
    assert (dead[100:150] == Params().w_tag_dead).all() and (dead[150:200] == 0).all()
    assert (ev[150:200] > 0).all() and (ev[100:150] == 0).all()      # hint after the restart
    assert (ev[300:400] < 0).all()                                   # away phase
    assert (ev[500:600] == 0).all() and (dead[500:550] > 0).all()    # throw-in: D only, no team


def test_tracking_can_overrule_a_false_tag():
    # tag says away for a stretch where the home team clearly has the ball
    tr, truth = make_match([("play", 0, 400, [])])
    ev, dead = tagging.factors(
        [{"code": "build up", "start": 100, "end": 300, "tags": {"Team": "Athletic Club"}}],
        tr.n_frames, "FC Barcelona", "Athletic Club")
    labels = decode_match(tr, [(0, tr.n_frames - 1)], Params(), tag_factors=(ev, dead))
    assert (labels == "H").mean() > 0.95


def test_tag_decides_only_when_tracking_is_silent():
    # no ball for 200 frames: without a tag the state carries (H); a long enough team hint flips it
    tr, _ = make_match([("play", 0, 100, []), ("play", 0, 200, [(0, 200)])])
    ev, dead = tagging.factors(
        [{"code": "build up", "start": 100, "end": 300, "tags": {"Team": "Athletic Club"}}],
        tr.n_frames, "FC Barcelona", "Athletic Club")
    no_tag = decode_match(tr, [(0, tr.n_frames - 1)], Params())
    with_tag = decode_match(tr, [(0, tr.n_frames - 1)], Params(), tag_factors=(ev, dead))
    assert (no_tag[110:290] == "H").all()
    assert (with_tag[280:290] == "A").all()