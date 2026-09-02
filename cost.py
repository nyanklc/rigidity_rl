"""Counts what each method spends, so the compute comparison is measured rather than argued.

The primitives that own the linear algebra carry a `@counted` decorator, so a call site
stays exactly as it reads. `Meter` brackets one method and reports the deltas.

Counts are per *call to a named primitive*, not per `np.linalg` call: `nullspace` is one
`eigh(6n)`, `edge_block_ranks` is `m` ranks of a 3 x 6n block, and `candidate_gain` is one
batched eigvalsh over n(n-1) 3x3 Grams. The map from a name to what it does belongs in the
report legend, and the wall time is what weighs them against each other.

Nested primitives are counted at every level (`is_MBR` calls `edge_block_ranks`), so a
single headline number sums LEAVES only.
"""
import collections
import contextlib
import functools
import time

# Every counted call lands here. Cleared by nothing -- read differences, not absolutes.
COUNTS = collections.Counter()

# Where measurement work is tallied while `reporting()` is open, so per-step tracing does
# not land on the method being measured.
_REPORTING = "_reporting"
_in_reporting = 0

# The primitives that call no other counted one. A total over these double counts
# nothing. `eigenvalues` and `solve_shape` are deliberately absent: both build B
# themselves, so their work is already in `extended_bearing_rigidity_matrix`.
# tests/test_cost.py checks the property rather than trusting this list.
LEAVES = (
    "extended_bearing_rigidity_matrix",
    "rigidity_decomposition",
    "nullspace",
    "nullspace_and_softest",
    "error_covariance",
    "estimation_error_blocks",
    "removal_costs",
    "candidate_gain",
    "edge_block_ranks",
    "flex_space",
    "nullspace_in_scaled_units",
    "is_IBR_explicit",
)

# What one call of each primitive actually does, for the report legend.
OPERATION = {
    "extended_bearing_rigidity_matrix": "build B (3m x 6n)",
    "rigidity_decomposition": "svd(B)",
    "nullspace": "eigh(6n)",
    "nullspace_and_softest": "eigh(6n)",
    "error_covariance": "eigh(6n)",
    "estimation_error_blocks": "eigh(6n)",
    "removal_costs": "m x eigvalsh(3), plus eigvalsh(6n) per redundant edge",
    "candidate_gain": "one batched eigvalsh over n(n-1) 3x3 Grams",
    "edge_block_ranks": "m x rank(3 x 6n)",
    "eigenvalues": "eigvalsh(6n)",
    "flex_space": "svd(6n x k)",
    "nullspace_in_scaled_units": "qr(6n x k)",
    "solve_shape": "iters x lstsq(3m x 6n)",
    "is_IBR_explicit": "rank(B)",
    "is_MBR": "rank(B), then the per-edge blocks",
    "max_edge_rank": "n(n-1) x rank(3 x 6n), once per episode",
    "required_edge_count": "the block ranks of the complete graph, once per episode",
    "repair_edge_count": "rank(B), then the absent-pair marginals",
    "greedy_rigid_construction": "one construction from the empty graph",
    "greedy_rigid_repair": "one repair by marginal rank gain",
    "score_network": "one phi evaluation",
    "deterministic_action": "one policy forward pass",
    "forward": "one policy forward pass",
}


def tally(name, k=1):
    """Count `k` calls to `name`. The hook for work no decorator can reach."""
    COUNTS[_REPORTING if _in_reporting else name] += k


def counted(fn):
    """Tally each call under the function's own name."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        COUNTS[_REPORTING if _in_reporting else fn.__name__] += 1
        return fn(*args, **kwargs)
    return wrapper


def measurement(fn):
    """Tally this function's work as overhead. For helpers that only exist to report."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with reporting():
            return fn(*args, **kwargs)
    return wrapper


@contextlib.contextmanager
def reporting():
    """Work inside is tallied as measurement overhead, not against the method.

    Per-step tracing costs an eigendecomposition per step, which would otherwise
    swamp the difference between the methods it is measuring.
    """
    global _in_reporting
    _in_reporting += 1
    try:
        yield
    finally:
        _in_reporting -= 1


class Meter:
    """Bracket one method. `counts` are the deltas and `ms` the wall time."""

    def __init__(self):
        self.counts = {}
        self.ms = 0.0

    def __enter__(self):
        self._before = COUNTS.copy()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self._t0) * 1e3
        self.counts = {k: COUNTS[k] - self._before.get(k, 0) for k in COUNTS
                       if COUNTS[k] - self._before.get(k, 0) > 0}
        return False

    def total(self):
        """Calls to the leaf primitives. The one number that sums honestly."""
        return sum(self.counts.get(k, 0) for k in LEAVES)
