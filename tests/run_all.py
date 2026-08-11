#!/usr/bin/env python
"""Run the test suite.

    uv run tests/run_all.py             fast suite (correctness, target < 60s)
    uv run tests/run_all.py --slow      everything: training runs, brute force, large n
    uv run tests/run_all.py -k flex     any other pytest argument is passed through

Individual files stay directly runnable:

    uv run pytest tests/test_flex.py -v
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    os.chdir(ROOT)                       # module imports and ./environments resolve here

    slow = "--slow" in argv
    args = [HERE,
            "-v",              # one line per test: name and outcome
            "--tb=short",      # what failed and where
            "-ra",             # closing recap of every non-pass, with its reason
            "--no-header",
            "--color=yes",
            "-W", "ignore::DeprecationWarning"]
    args += argv

    banner = "FULL SUITE (including slow)" if slow else "FAST SUITE (--slow adds training runs)"
    print(f"\n{'=' * 72}\n  {banner}\n{'=' * 72}\n")
    code = pytest.main(args)
    if not slow and code == 0:
        print("\n  slow tests were skipped -- run with --slow before trusting a release\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
