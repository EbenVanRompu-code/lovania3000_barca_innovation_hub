import numpy as np
import pytest

from possession import ball as ball_mod
from possession import classifier
from possession.params import Params
from possession.pipeline import decode_match
from synth import STANDARD, make_match


def test_roll_mean_past_and_future_exclude_the_frame():
    x = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    assert np.allclose(classifier.roll_mean(x, 2, -1)[[1, 2, 4]], [1.0, 1.5, 3.0])
    assert np.allclose(classifier.roll_mean(x, 2, +1)[[0, 1]], [2.5, 3.0])
    assert np.isnan(classifier.roll_mean(x, 2, -1)[0]) and np.isnan(classifier.roll_mean(x, 2, +1)[4])


def test_features_shape_and_names():
    tr, _ = make_match(STANDARD)
    b = ball_mod.clean(tr, Params())
    X = classifier.compute_features(b, tr)
    assert X.shape == (tr.n_frames, len(classifier.FEATURES))
    assert "d_diff" in classifier.FEATURES and "gap_len" in classifier.FEATURES


def test_variant_c_requires_a_model():
    classifier.ACTIVE = None
    tr, _ = make_match(STANDARD)
    with pytest.raises(RuntimeError):
        decode_match(tr, [(0, tr.n_frames - 1)], Params(), variant="C")


def test_variant_c_learns_the_scripted_match():
    tr, truth = make_match(STANDARD, seed=1)
    b = ball_mod.clean(tr, Params())
    X = classifier.compute_features(b, tr)
    y = np.array([{"H": 0, "A": 1, "D": 2}[t] for t in truth])
    clf = classifier.fit(X, y, max_iter=30)
    classifier.ACTIVE = classifier.Model(clf, classifier.fit_temperature(clf, X, y))
    try:
        tr2, truth2 = make_match(STANDARD, seed=2)     # a different random match
        labels = decode_match(tr2, [(0, tr2.n_frames - 1)], Params(), variant="C")
        assert (labels == truth2).mean() > 0.9
    finally:
        classifier.ACTIVE = None
