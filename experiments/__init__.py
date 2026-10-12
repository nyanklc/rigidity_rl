"""Runnable experiments. One script per question; `common.py` is their shared setup.

The drivers and `common.py` import the libraries at the repository root, so importing
this package puts the root on `sys.path` if it is not already there. Without it,
`import experiments.common` works only from the root, and every driver would have to
get its own path handling right before it could import anything shared.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
