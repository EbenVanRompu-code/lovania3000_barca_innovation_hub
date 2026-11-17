"""Run the final pipeline twice and compare file hashes (PLAN.md section 9). Run by the team.

    uv run python -m possession.check_repro
"""

import hashlib
import os

from . import run

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


if __name__ == "__main__":
    paths = []
    for i in (1, 2):
        path = os.path.join(OUT, "repro_{}.csv".format(i))
        run.main(["--out", path])
        paths.append(path)
    a, b = sha(paths[0]), sha(paths[1])
    print("run 1:", a)
    print("run 2:", b)
    print("IDENTICAL" if a == b else "DIFFERENT -- something is not deterministic")
    raise SystemExit(0 if a == b else 1)
