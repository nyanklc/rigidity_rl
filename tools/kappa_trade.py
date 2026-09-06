"""How many edges does raising kappa actually buy?

    OMP_NUM_THREADS=1 PYTHONPATH=.:tools uv run tools/kappa_trade.py

Over rigid graphs phi = 100 - one_edge*(m - kappa*q), q in (0,1), so the phi-optimal
graph at kappa minimises (m - kappa*q). Storing (m, q) per graph gives the exact kappa
curve with no search. The optimum moves from m to m+1 exactly when
kappa*(q*(m+1) - q*(m)) > 1, so the kappa that buys one edge is 1/Delta q.

Both the sigmoid gap Delta q and the underlying physical gain
Delta log10(shape_err) are reported: q can be flat because the graph is already far
above the sigmoid's fixed centre, which is a different statement from the geometry
having nothing left to give.

Exhaustive to n=5. Above that the best graph at each edge count is approximated by
random restarts plus a single-edge-swap hill climb on shape_err, which is a LOWER
bound on q*(m) and so an UPPER bound on the kappa needed.
"""
import sys, itertools; sys.path.insert(0, ".")
import numpy as np
from scenario import random_scenario
import rigidity as R
from domain_closed_forms import predict

KAPPAS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]


def score(net, rank_K, n):
    """(is_rigid, q, log10 shape_err) for net.edges as it stands."""
    B = R.extended_bearing_rigidity_matrix(net)
    rank, _, _ = R.rigidity_decomposition(B, rank_K)
    if rank < rank_K:
        return False, None, None
    _, s, _ = R.rigidity_decomposition(R.scaled_rigidity_matrix(net, brmat=B), rank_K)
    a = float((1.0 / (s[:rank_K] ** 2)).sum())
    e = float(np.sqrt(a / n))
    return True, R.shape_error_quality(e, n), float(np.log10(e))


def best_exhaustive(net, domains, extra):
    n = net.n
    rank_K, _, m_req = predict(domains)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    out = {}
    for m in range(m_req, min(m_req + extra, len(pairs)) + 1):
        bq = be = None
        for combo in itertools.combinations(range(len(pairs)), m):
            E = np.zeros((n, n), dtype=bool)
            for k in combo:
                E[pairs[k]] = True
            net.edges = E
            ok, q, lg = score(net, rank_K, n)
            if ok and (bq is None or q > bq):
                bq, be = q, lg
        if bq is not None:
            out[m] = (bq, be)
    return out


def best_sampled(net, domains, extra, rng, restarts=6, sweeps=3):
    n = net.n
    rank_K, _, m_req = predict(domains)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    out = {}
    for m in range(m_req, min(m_req + extra, len(pairs)) + 1):
        bq = be = None
        for _ in range(restarts):
            idx = list(rng.choice(len(pairs), size=m, replace=False))
            E = np.zeros((n, n), dtype=bool)
            for k in idx:
                E[pairs[k]] = True
            net.edges = E.copy()
            ok, q, lg = score(net, rank_K, n)
            cur = (q, lg) if ok else None
            for _ in range(sweeps):                      # swap one edge, keep gains
                improved = False
                for a_pos in range(m):
                    for k in range(len(pairs)):
                        if k in idx:
                            continue
                        trial = list(idx); trial[a_pos] = k
                        E2 = np.zeros((n, n), dtype=bool)
                        for kk in trial:
                            E2[pairs[kk]] = True
                        net.edges = E2
                        ok2, q2, lg2 = score(net, rank_K, n)
                        if ok2 and (cur is None or q2 > cur[0]):
                            cur, idx, improved = (q2, lg2), trial, True
                if not improved:
                    break
            if cur is not None and (bq is None or cur[0] > bq):
                bq, be = cur
        if bq is not None:
            out[m] = (bq, be)
    return out


def report(label, best):
    if len(best) < 2:
        print(f"{label:38s}  not enough rigid edge counts")
        return None
    ms = sorted(best)
    q = {m: best[m][0] for m in ms}
    lg = {m: best[m][1] for m in ms}
    dq = q[ms[1]] - q[ms[0]]
    dlg = lg[ms[0]] - lg[ms[1]]                          # decades of error removed
    k_one = (1.0 / dq) if dq > 1e-12 else float("inf")
    opt = {k: min(ms, key=lambda m: m - k * q[m]) for k in KAPPAS}
    gains = "".join(f"{opt[k] - opt[0.0]:>4d}" for k in KAPPAS)
    print(f"{label:38s} {q[ms[0]]:6.3f} {dq:+7.3f} {dlg:+7.3f} {k_one:9.1f}  {gains}")
    return opt


HEAD = (f"{'configuration':38s} {'q(m_req)':>6s} {'dq/edge':>7s} "
        f"{'ddec':>7s} {'k for 1 edge':>9s}  " +
        "".join(f"k={k:g}".rjust(4) for k in KAPPAS))
print("\nextra edges the phi-optimum spends, against kappa (0 = spends none)\n")
print(HEAD); print("-" * len(HEAD))

# --- n=4, exhaustive, five compositions x 3 networks
for doms, name in ((["R^2"] * 4, "R^2"), (["R^3"] * 4, "R^3"),
                   (["R^2xS^1"] * 4, "R^2xS^1"), (["R^3xS^1"] * 4, "R^3xS^1"),
                   (["SE(3)"] * 4, "SE(3)"),
                   (["R^2", "R^3", "R^2xS^1", "SE(3)"], "mixed")):
    for t in range(3):
        np.random.seed(100 + t)
        net, _ = random_scenario(4, doms, edge_count=0)
        report(f"n=4  {name:9s} net {t}", best_exhaustive(net, doms, 6))

print()
for doms, name in ((["R^2"] * 5, "R^2"), (["R^3"] * 5, "R^3"),
                   (["R^3xS^1"] * 5, "R^3xS^1"), (["SE(3)"] * 5, "SE(3)"),
                   (["R^2", "R^3", "R^2xS^1", "R^3xS^1", "SE(3)"], "mixed")):
    for t in range(2):
        np.random.seed(200 + t)
        net, _ = random_scenario(5, doms, edge_count=0)
        report(f"n=5  {name:9s} net {t}  (exhaustive)", best_exhaustive(net, doms, 2))

print()
rng = np.random.default_rng(0)
for n in (6, 8, 10):
    for doms, name in (([("R^3")] * n, "R^3"), (["SE(3)"] * n, "SE(3)"),
                       ([["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)"][i % 5]
                         for i in range(n)], "mixed")):
        for t in range(2):
            np.random.seed(300 + t)
            net, _ = random_scenario(n, doms, edge_count=0)
            report(f"n={n:<2d} {name:9s} net {t}  (search)",
                   best_sampled(net, doms, 2, rng))
