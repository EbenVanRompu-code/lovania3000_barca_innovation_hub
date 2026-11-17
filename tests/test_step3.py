from possession import ball as ball_mod
from possession import run
from possession.params import Params
from possession.pipeline import decode_match
from synth import STANDARD, make_match


def _setup():
    tr, _ = make_match(STANDARD, lead=20)
    n = tr.n_frames
    blocks = [(20, n - 1)]
    labels = decode_match(tr, blocks, Params(), (50, 149))
    return tr, labels, blocks, n


def test_hard_checks_pass_on_pipeline_output():
    tr, labels, blocks, n = _setup()
    assert all(run.hard_checks(labels, blocks, n, (50, 149)).values())


def test_hard_checks_catch_x_inside_and_bad_forced_block():
    tr, labels, blocks, n = _setup()
    bad = labels.copy()
    bad[100] = "X"
    checks = run.hard_checks(bad, blocks, n, (50, 149))
    assert not checks["no X inside the half blocks"]
    bad2 = labels.copy()
    bad2[60] = "H"
    assert not run.hard_checks(bad2, blocks, n, (50, 149))["2nd-half pre-kickoff frames all D"]
    assert not run.hard_checks(labels[:-1], blocks, n, (50, 149))["row count"]


def test_statistics_and_report_render():
    tr, labels, blocks, n = _setup()
    b = ball_mod.clean(tr, Params())
    stats = run.statistics(labels, tr, b, blocks, Params(), None)
    assert abs(sum(stats["share " + c] for c in "HAD") - 1) < 1e-9
    assert stats["H<->A switches"] >= 1
    text = run.format_report(run.hard_checks(labels, blocks, n, None), stats)
    assert "OVERALL: PASS" in text


def test_csv_roundtrip_matches_submission_format(tmp_path):
    tr, labels, blocks, n = _setup()
    path = tmp_path / "x.csv"
    run.write_csv(str(path), labels)
    lines = path.read_text().splitlines()
    assert lines[0] == "frame,label" and len(lines) == n + 1
    assert lines[1] == "0,X" and lines[n] == "{},{}".format(n - 1, labels[-1])
