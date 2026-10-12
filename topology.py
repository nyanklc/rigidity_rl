"""What the graphs each method builds actually look like, against what the theory predicts.

The comparison reports how *well* a method does -- edge count, rigidity, shape error --
and says nothing about the structure of the answer. Chapter 4's results are statements
about structure, so they can be checked directly on the topologies the methods produce:

  reciprocity      the reverse of an edge adds rank 0 in R^2/R^3 and 1/1/2 in the
                   oriented domains, so a minimal frameless topology should carry no
                   reciprocal pair at all
  out-degree       every rigid framework gives agent i at least r_i outgoing edges,
                   r_i = 0/0/1/1/2 by domain. That is proven, so a violation is a bug
                   here and the informative quantity is the overshoot
  pair classes     an edge between two planar agents carries rank 1 where every other
                   edge carries 2, so a minimum-edge topology should under-use that class
  geometry         edge length in formation radii, and how degenerate the bearings one
                   agent measures are
  known answers    in homogeneous R^2 a minimal topology's underlying undirected graph
                   must be Laman, and `spectral` must reproduce `greedy` edge for edge
                   at stiffness_kappa 0

Nothing here is metered. What is measured is the graph, not what it cost to find, and
`experiments/comparison.py` is where the spend is reported.

The library, not the script: `experiments/topology.py` is what runs it.
"""
import copy
import itertools
from types import SimpleNamespace

import numpy as np

import repair
from network import DOMAINS
from rigidity import (candidate_block, characteristic_length,
                      extended_bearing_rigidity_matrix, shape_error_of)

# Outgoing edges agent i cannot do without: its own attitude is unobservable below this,
# whatever the rest of the graph does. Zero where the agent carries no frame.
R_MIN = {"R^2": 0, "R^3": 0, "R^2xS^1": 1, "R^3xS^1": 1, "SE(3)": 2}

# A bearing between two planar agents is one angle, so its row block has rank 1; every
# other ordered pair gives 2. Measured per class rather than assumed, by `class_ranks`.
PLANAR = frozenset({"R^2", "R^2xS^1"})

# An agent with no frame of its own. The reverse of an edge between two of these adds no
# rank at all, which is the only place reciprocity is predicted to vanish; once either
# end carries a frame the attitude columns differ between the two directions and it does
# not. Pooling the two reads the prediction at the wrong grain.
FRAMELESS = frozenset({"R^2", "R^3"})

# Laman by brute force is 2^n subsets. Above this the question is not worth a pebble game.
LAMAN_MAX_N = 20

ROW_FIELDS = (
    "episode", "method", "edges_are",
    "n", "m", "m_req", "m_over_m_req", "rank_K", "rank", "is_rigid", "shape_err",
    # reciprocity
    "recip", "recip_edges", "recip_pairs", "recip_random", "m_undirected",
    "edges_free", "recip_free_edges", "recip_free",
    "edges_framed", "recip_framed_edges", "recip_framed",
    # out-degree against the per-node floor
    "outdeg_mean", "outdeg_min", "floor_ok", "floor_slack",
    # which ordered pairs get an edge, collapsed to the rank class
    "avail_c1", "edges_c1", "share_c1", "avail_c2", "edges_c2", "share_c2",
    # geometry
    "len_mean", "len_total", "len_min", "len_max",
    "spread_min", "spread_median", "spread2_median", "spread_nodes",
    # the known answer, where it is defined
    "laman",
)

NODE_FIELDS = ("episode", "method", "agent", "domain", "r_i", "out_deg", "in_deg",
               "meets_floor", "recip_out", "spread", "spread2", "len_mean_out")

PAIR_FIELDS = ("episode", "method", "domain_i", "domain_j", "c_k",
               "avail", "edges", "share", "recip_edges", "len_mean")

JACCARD_FIELDS = ("episode", "method_a", "method_b", "jaccard", "shared", "union",
                  "m_a", "m_b")

# the null: a rank-greedy construction in random order, pruned back to minimality
NULL_METHOD = "random_rigid"


# ---------------------------------------------------------------- primitives

def out_degrees(edges):
    """How many bearings each agent measures. Row = measuring agent.

    Not `Network.get_degree_features`, whose column 0 is the IN-degree despite the
    name order; `tests/test_topology.py` pins the two against each other.
    """
    return np.asarray(edges, dtype=bool).sum(axis=1)


def in_degrees(edges):
    """How many agents measure each agent."""
    return np.asarray(edges, dtype=bool).sum(axis=0)


def reciprocity(edges):
    """(m, reciprocated edges, reciprocal pairs, fraction of edges reciprocated).

    The two counts differ by a factor of two and mean different things. `edges & edges.T`
    marks both directions of a reciprocal pair, so its sum over m is the fraction of
    EDGES whose reverse also exists, which is 1 on a fully reciprocal graph.
    """
    e = np.asarray(edges, dtype=bool)
    m = int(e.sum())
    both = int((e & e.T).sum())
    return m, both, both // 2, (both / m if m else 0.0)


def expected_reciprocity(n, m):
    """Fraction of edges reciprocated in a uniform directed graph on n nodes, m edges.

    m edges placed uniformly over the n(n-1) ordered pairs, so conditioned on one edge
    being present its reverse is one of the remaining m-1 among the remaining
    n(n-1)-1 pairs. Closed form, no sampling.
    """
    pairs = n * (n - 1)
    if m <= 0 or pairs <= 1:
        return 0.0
    return float(max(m - 1, 0) / (pairs - 1))


def edge_lengths(net, length_scale=None):
    """{(i, j): ||p_j - p_i|| / L} over the edges that exist, L the formation's radius."""
    L = characteristic_length(net) if length_scale is None else length_scale
    p = np.array([a.pose.position for a in net.agents], dtype=float)
    return {(i, j): float(np.linalg.norm(p[j] - p[i]) / L)
            for i, j in net.get_edge_list()}


def spanned_axes(bearings, tol=1e-12):
    """Coordinate axes the formation's bearings have any component along.

    A planar formation sits in z = 0, so every bearing's third component is exactly
    zero and any matrix of bearings is singular by construction. The spread has to be
    read in the axes the formation actually occupies.
    """
    flat = np.asarray(bearings, dtype=float).reshape(-1, 3)
    return np.abs(flat).max(axis=0) > tol


def out_bearings(net, i, bearings=None, axes=None):
    """The bearings agent i measures, as columns. World frame.

    World frame, not `get_all_pairs_bearings`: the body-frame measurement differs by
    R_i for an oriented agent, so a statistic read off it would move when one SE(3)
    agent turns and not when an R^3 agent does.
    """
    b = net.get_all_pairs_bearings_world() if bearings is None else bearings
    js = np.nonzero(np.asarray(net.edges, dtype=bool)[i])[0]
    out = np.asarray(b, dtype=float)[i, js, :].T
    return out if axes is None else out[axes, :]


def bearing_spread(net, i, bearings=None, axes=None):
    """(sigma_min, sigma_2) of agent i's outgoing bearings. (None, None) below two.

    sigma_min is zero when the bearings are degenerate: collinear at out-degree 2,
    coplanar above it. It therefore means a different thing at each out-degree, which
    is why sigma_2 is reported beside it -- that one is the collinearity test alone and
    is comparable across out-degrees.
    """
    B = out_bearings(net, i, bearings=bearings, axes=axes)
    if B.shape[1] < 2:
        return None, None
    s = np.linalg.svd(B, compute_uv=False)
    # a collinear formation spans one axis, so there is no second singular value to read
    return float(s[-1]), (float(s[1]) if s.size > 1 else 0.0)


def undirected(edges):
    """The underlying undirected graph, as a set of sorted pairs."""
    e = np.asarray(edges, dtype=bool)
    u = e | e.T
    return {(i, j) for i in range(u.shape[0]) for j in range(i + 1, u.shape[0]) if u[i, j]}


def is_laman(edges):
    """Is the underlying undirected graph Laman? None when n is too large to brute force.

    Laman: |E| = 2n - 3 and no subset of k vertices spans more than 2k - 3 edges. At
    n <= 20 that is 2^n subset tests of a pair list, which is cheaper to write and to
    trust than a pebble game.
    """
    n = int(np.asarray(edges).shape[0])
    if n > LAMAN_MAX_N:
        return None
    if n < 2:
        return False
    U = undirected(edges)
    if len(U) != 2 * n - 3:
        return False
    for size in range(2, n + 1):
        for S in itertools.combinations(range(n), size):
            T = set(S)
            inside = sum(1 for i, j in U if i in T and j in T)
            if inside > 2 * size - 3:
                return False
    return True


def jaccard(a, b):
    """(|a & b| / |a | b|, shared, union) over two edge sets."""
    shared, union = len(a & b), len(a | b)
    return (shared / union if union else 1.0), shared, union


def class_ranks(net):
    """{(domain_i, domain_j): rank of that edge's row block}, one representative each.

    Measured from the matrix routine itself rather than from the both-planar rule, so
    the rule is a check on this and not an assumption behind it.
    """
    doms = [a.domain for a in net.agents]
    reps = {}
    for i in range(net.n):
        for j in range(net.n):
            if i != j:
                reps.setdefault((doms[i], doms[j]), (i, j))
    return {cls: int(np.linalg.matrix_rank(candidate_block(net, i, j)))
            for cls, (i, j) in reps.items()}


# ---------------------------------------------------------------- one graph

def describe(net, *, rank_K, m_req, ranks=None):
    """One graph, one pass. -> (row, nodes, pairs), each missing the run keys.

    `ranks` is `class_ranks(net)`, which depends on the poses and domains and not on
    the edges, so the caller computes it once per instance.
    """
    n = net.n
    edges = np.asarray(net.edges, dtype=bool)
    doms = [a.domain for a in net.agents]
    ranks = class_ranks(net) if ranks is None else ranks

    m, recip_edges, recip_pairs, recip = reciprocity(edges)
    outd, ind = out_degrees(edges), in_degrees(edges)
    both = edges & edges.T

    brmat = extended_bearing_rigidity_matrix(net) if m else np.zeros((0, 6 * n))
    rank = int(np.linalg.matrix_rank(brmat)) if brmat.size else 0
    is_rigid = rank == rank_K
    shape_err = shape_error_of(net, rank_K, brmat=brmat) if is_rigid else np.inf

    L = characteristic_length(net)
    lengths = edge_lengths(net, L)
    bearings = net.get_all_pairs_bearings_world()
    axes = spanned_axes(bearings)

    # ---- per node
    nodes, spreads, spreads2 = [], [], []
    for i in range(n):
        sig, sig2 = bearing_spread(net, i, bearings=bearings, axes=axes)
        if sig is not None:
            spreads.append(sig)
            spreads2.append(sig2)
        mine = [lengths[(i, j)] for j in range(n) if edges[i, j]]
        nodes.append({
            "agent": i, "domain": doms[i], "r_i": R_MIN[doms[i]],
            "out_deg": int(outd[i]), "in_deg": int(ind[i]),
            "meets_floor": bool(outd[i] >= R_MIN[doms[i]]),
            "recip_out": int(both[i].sum()),
            "spread": sig, "spread2": sig2,
            "len_mean_out": float(np.mean(mine)) if mine else None,
        })

    slack = [int(outd[i]) - R_MIN[doms[i]] for i in range(n)]

    # ---- per ordered domain class. avail excludes self loops, so (A, A) is p(p-1).
    counts = {d: doms.count(d) for d in set(doms)}
    pairs = []
    for di in DOMAINS:
        for dj in DOMAINS:
            if di not in counts or dj not in counts:
                continue
            avail = counts[di] * (counts[dj] - 1) if di == dj else counts[di] * counts[dj]
            if avail <= 0:
                continue
            members = [(i, j) for i in range(n) for j in range(n)
                       if i != j and doms[i] == di and doms[j] == dj]
            here = [(i, j) for i, j in members if edges[i, j]]
            lens = [lengths[e] for e in here]
            pairs.append({
                "domain_i": di, "domain_j": dj, "c_k": ranks.get((di, dj)),
                "avail": avail, "edges": len(here),
                "share": len(here) / avail,
                "recip_edges": sum(1 for i, j in here if edges[j, i]),
                "len_mean": float(np.mean(lens)) if lens else None,
            })

    def total(key, which):
        return sum(p[key] for p in pairs if p["c_k"] == which)

    def over_frameless(key, want):
        return sum(p[key] for p in pairs
                   if (p["domain_i"] in FRAMELESS and p["domain_j"] in FRAMELESS) is want)

    lens = list(lengths.values())
    row = {
        "n": n, "m": m, "m_req": m_req, "rank_K": rank_K, "rank": rank,
        "m_over_m_req": (m / m_req if m_req else None),
        "is_rigid": is_rigid,
        "shape_err": (float(shape_err) if np.isfinite(shape_err) else None),
        "recip": recip, "recip_edges": recip_edges, "recip_pairs": recip_pairs,
        "recip_random": expected_reciprocity(n, m),
        "m_undirected": m - recip_pairs,
        "edges_free": over_frameless("edges", True),
        "recip_free_edges": over_frameless("recip_edges", True),
        "edges_framed": over_frameless("edges", False),
        "recip_framed_edges": over_frameless("recip_edges", False),
        "outdeg_mean": float(outd.mean()), "outdeg_min": int(outd.min()),
        "floor_ok": all(s >= 0 for s in slack), "floor_slack": int(min(slack)),
        "avail_c1": total("avail", 1), "edges_c1": total("edges", 1),
        "avail_c2": total("avail", 2), "edges_c2": total("edges", 2),
        "len_mean": float(np.mean(lens)) if lens else None,
        "len_total": float(np.sum(lens)) if lens else 0.0,
        "len_min": float(np.min(lens)) if lens else None,
        "len_max": float(np.max(lens)) if lens else None,
        "spread_min": float(np.min(spreads)) if spreads else None,
        "spread_median": float(np.median(spreads)) if spreads else None,
        "spread2_median": float(np.median(spreads2)) if spreads2 else None,
        "spread_nodes": len(spreads),
        # the Laman characterisation is a homogeneous R^2 statement and nothing else
        "laman": (is_laman(edges) if set(doms) == {"R^2"} else None),
    }
    row["recip_free"] = ((row["recip_free_edges"] / row["edges_free"])
                         if row["edges_free"] else None)
    row["recip_framed"] = ((row["recip_framed_edges"] / row["edges_framed"])
                           if row["edges_framed"] else None)
    row["share_c1"] = (row["edges_c1"] / row["avail_c1"]) if row["avail_c1"] else None
    row["share_c2"] = (row["edges_c2"] / row["avail_c2"]) if row["avail_c2"] else None
    return row, nodes, pairs


def consistency(row, pairs):
    """[complaint] for anything the bookkeeping cannot be right about. Empty is good."""
    bad = []
    n, m = row["n"], row["m"]
    if sum(p["avail"] for p in pairs) != n * (n - 1):
        bad.append(f"availability sums to {sum(p['avail'] for p in pairs)}, "
                   f"not n(n-1) = {n * (n - 1)}")
    if sum(p["edges"] for p in pairs) != m:
        bad.append(f"class edge counts sum to {sum(p['edges'] for p in pairs)}, not m = {m}")
    # a reciprocal pair has one edge in (A, B) and one in (B, A), so the two agree
    by_class = {(p["domain_i"], p["domain_j"]): p["recip_edges"] for p in pairs}
    for (di, dj), v in by_class.items():
        if by_class.get((dj, di), v) != v:
            bad.append(f"reciprocal edges disagree between ({di},{dj}) and ({dj},{di})")
    if row["is_rigid"] and not row["floor_ok"]:
        bad.append(f"a rigid graph left an agent {-row['floor_slack']} short of its "
                   f"out-degree floor, which Chapter 4 says cannot happen")
    return bad


# ---------------------------------------------------------------- the run

def measure(*, env, instances, methods, models=None, wrapped=None, steps=None,
            policy_mode="greedy", spectral_shortlist=5, restarts=20,
            seed=0, progress=None):
    """One row per (instance, method). -> (rows, nodes, pairs, jac).

    Every method faces the same instance, restored by deepcopy between them, which is
    what makes the Jaccard block and the shares comparable across methods.
    """
    import baselines

    rows, nodes, pairs, jac = [], [], [], []
    models = models or {}
    # baselines.run_method reads these three off ctx["args"]; argparse stays in the driver
    args = SimpleNamespace(spectral_shortlist=spectral_shortlist, restarts=restarts,
                           policy_mode=policy_mode)
    # private to the methods that use randomness, so adding one does not move another
    construct_rng = np.random.default_rng(seed)
    anneal_rng = np.random.default_rng(seed)
    degree_rng = np.random.default_rng(seed)

    for ep, pristine in enumerate(instances):
        if progress:
            progress(f"instance {ep + 1}/{len(instances)}")
        env.freeze_network = False
        env.network = copy.deepcopy(pristine)
        env.freeze_network = True
        env.reset()                              # bookkeeping only
        instance = copy.deepcopy(env.network)
        rank_K, m_req = int(env.rank_K), int(env.m_req)
        ranks = class_ranks(instance)

        def restore():
            env.network = copy.deepcopy(instance)
            env.reset()

        ctx = dict(env=env, args=args, traces=None, episode=ep,
                   steps=steps or int(env.max_steps), models=models, wrapped=wrapped,
                   decisions=None, construct_rng=construct_rng,
                   anneal_rng=anneal_rng, degree_rng=degree_rng, anneal_budget=None)

        chosen = {}
        for name in methods:
            restore()
            if name == NULL_METHOD:
                # the null for the share and reciprocity columns, keyed on the instance
                # so it does not move when the method list changes
                net = repair.converge(copy.deepcopy(instance), rank_K,
                                      np.random.default_rng([seed, ep, 983]))
                edges, are = net.edges.copy(), "final"
            else:
                out = baselines.run_method(name, ctx)
                if out is None or out.get("edges") is None:
                    continue                     # no model, or n too large for optimal
                edges, are = np.asarray(out["edges"], dtype=bool), out["edges_are"]

            net = copy.deepcopy(instance)
            net.edges = edges.copy()
            row, nd, pr = describe(net, rank_K=rank_K, m_req=m_req, ranks=ranks)
            row |= {"episode": ep, "method": name, "edges_are": are}
            for complaint in consistency(row, pr):
                print(f"  instance {ep} {name}: {complaint}")
            rows.append(row)
            nodes.extend(dict(r, episode=ep, method=name) for r in nd)
            pairs.extend(dict(r, episode=ep, method=name) for r in pr)
            chosen[name] = repair._edge_set(edges)

        names = [m for m in methods if m in chosen]
        for a, b in itertools.combinations(names, 2):
            value, shared, union = jaccard(chosen[a], chosen[b])
            jac.append({"episode": ep, "method_a": a, "method_b": b,
                        "jaccard": value, "shared": shared, "union": union,
                        "m_a": len(chosen[a]), "m_b": len(chosen[b])})

    return rows, nodes, pairs, jac


# ---------------------------------------------------------------- aggregation

def _methods_in(records):
    """The methods present, in report's drawing order, then anything unregistered."""
    import report
    seen = list(dict.fromkeys(r["method"] for r in records))
    order = [m for m in report.METHOD_ORDER if m in seen]
    return order + [m for m in seen if m not in order]


def _mean(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _sd(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.std(vals)) if vals else None


def summarize(rows):
    """One record per method. Shares and reciprocity are count-weighted, not means of
    per-instance fractions: the domain mix is redrawn per instance, so a mean of
    fractions over-weights whichever instance happened to draw one high-DOF agent."""
    out = []
    for m in _methods_in(rows):
        sel = [r for r in rows if r["method"] == m]
        m_tot = sum(r["m"] for r in sel)
        laman = [r["laman"] for r in sel if r["laman"] is not None]
        out.append({
            "method": m, "instances": len(sel),
            "m_mean": _mean([r["m"] for r in sel]), "m_sd": _sd([r["m"] for r in sel]),
            "m_over_m_req": _mean([r["m_over_m_req"] for r in sel]),
            "recip": (sum(r["recip_edges"] for r in sel) / m_tot) if m_tot else None,
            "recip_free": _ratio(sel, "recip_free_edges", "edges_free"),
            "recip_framed": _ratio(sel, "recip_framed_edges", "edges_framed"),
            "edges_free": sum(r["edges_free"] for r in sel),
            # the same quantity for a uniform directed graph at each row's own m
            "recip_random": ((sum(r["recip_random"] * r["m"] for r in sel) / m_tot)
                             if m_tot else None),
            "outdeg_mean": _mean([r["outdeg_mean"] for r in sel]),
            "outdeg_sd": _sd([r["outdeg_mean"] for r in sel]),
            "floor_pct": 100.0 * _mean([float(r["floor_ok"]) for r in sel]),
            "floor_slack": min((r["floor_slack"] for r in sel), default=None),
            "share_c1": _share(sel, 1), "share_c2": _share(sel, 2),
            "len_mean": _mean([r["len_mean"] for r in sel]),
            "len_sd": _sd([r["len_mean"] for r in sel]),
            "len_total": _mean([r["len_total"] for r in sel]),
            "spread_median": _mean([r["spread_median"] for r in sel]),
            "spread2_median": _mean([r["spread2_median"] for r in sel]),
            "rigid_pct": 100.0 * _mean([float(r["is_rigid"]) for r in sel]),
            "laman_pct": (100.0 * float(np.mean([float(x) for x in laman]))
                          if laman else None),
        })
    return out


def _ratio(sel, numerator, denominator):
    bottom = sum(r[denominator] for r in sel)
    return (sum(r[numerator] for r in sel) / bottom) if bottom else None


def _share(sel, which):
    avail = sum(r[f"avail_c{which}"] for r in sel)
    return (sum(r[f"edges_c{which}"] for r in sel) / avail) if avail else None


def by_domain(nodes):
    """One record per (method, domain), count-weighted over agents."""
    out = []
    for m in _methods_in(nodes):
        for d in DOMAINS:
            sel = [r for r in nodes if r["method"] == m and r["domain"] == d]
            if not sel:
                continue
            out.append({
                "method": m, "domain": d, "r_i": R_MIN[d], "agents": len(sel),
                "out_deg_mean": float(np.mean([r["out_deg"] for r in sel])),
                "out_deg_min": int(min(r["out_deg"] for r in sel)),
                "meets_floor_pct": 100.0 * float(np.mean(
                    [float(r["meets_floor"]) for r in sel])),
            })
    return out


def by_class(pairs):
    """One record per (method, domain_i, domain_j), count-weighted over pairs."""
    out = []
    for m in _methods_in(pairs):
        classes = dict.fromkeys((r["domain_i"], r["domain_j"])
                                for r in pairs if r["method"] == m)
        for di, dj in classes:
            sel = [r for r in pairs if r["method"] == m
                   and r["domain_i"] == di and r["domain_j"] == dj]
            avail = sum(r["avail"] for r in sel)
            out.append({
                "method": m, "domain_i": di, "domain_j": dj,
                "c_k": sel[0]["c_k"], "avail": avail,
                "edges": sum(r["edges"] for r in sel),
                "share": (sum(r["edges"] for r in sel) / avail) if avail else None,
                "recip_edges": sum(r["recip_edges"] for r in sel),
                "len_mean": _mean([r["len_mean"] for r in sel]),
            })
    return out


def jaccard_matrix(jac):
    """{(method_a, method_b): (mean, min, n)} over the instances, both orders keyed."""
    out = {}
    for a, b in dict.fromkeys((r["method_a"], r["method_b"]) for r in jac):
        vals = [r["jaccard"] for r in jac if r["method_a"] == a and r["method_b"] == b]
        cell = (float(np.mean(vals)), float(np.min(vals)), len(vals))
        out[(a, b)] = out[(b, a)] = cell
    return out


# ---------------------------------------------------------------- the table

def _cell(v, spec=".2f"):
    return "-" if v is None else f"{v:{spec}}"


def _pm(mean, sd, spec=".2f"):
    if mean is None:
        return "-"
    return f"{mean:{spec}}" if sd is None else f"{mean:{spec}}+-{sd:{spec}}"


def format_table(rows, nodes, pairs, jac, context=None, brief=False, kappa=None):
    """The five blocks and their legend. `kappa` is the run's stiffness_kappa."""
    agg = summarize(rows)
    if not agg:
        return "no topologies measured"
    has_laman = any(v["laman_pct"] is not None for v in agg)

    head1 = (f"  {'method':<14}{'m':>11}{'m/m_req':>9}{'recip':>8}{'rand':>7}"
             f"{'recip free':>12}{'framed':>8}"
             f"{'outdeg':>13}{'slack':>7}{'floor%':>8}{'c1 share':>10}{'c2 share':>10}"
             f"{'len':>13}{'len tot':>9}{'spread':>8}{'rigid%':>8}")
    head2 = (f"  {'':<14}{'edges':>11}{'(ratio)':>9}{'(frac)':>8}{'(null)':>7}"
             f"{'(frac)':>12}{'(frac)':>8}"
             f"{'mean+-sd':>13}{'worst':>7}{'%':>8}{'(frac)':>10}{'(frac)':>10}"
             f"{'radii':>13}{'radii':>9}{'median':>8}{'%':>8}")
    if has_laman:
        head1 += f"{'laman%':>8}"
        head2 += f"{'%':>8}"
    w = max(len(head1), 100)

    lines = ["=" * w, "TOPOLOGY PROPERTIES", "=" * w]
    for k, v in (context or {}).items():
        lines.append(f"  {k:<12}: {v}")
    lines.append("")
    lines.append(head1)
    lines.append(head2)
    lines.append("  " + "-" * (len(head1) - 2))
    for v in agg:
        row = (f"  {v['method']:<14}"
               f"{_pm(v['m_mean'], v['m_sd'], '.1f'):>11}"
               f"{_cell(v['m_over_m_req']):>9}"
               f"{_cell(v['recip']):>8}{_cell(v['recip_random']):>7}"
               f"{_cell(v['recip_free']):>12}{_cell(v['recip_framed']):>8}"
               f"{_pm(v['outdeg_mean'], v['outdeg_sd'], '.2f'):>13}"
               f"{_cell(v['floor_slack'], '+.0f'):>7}{v['floor_pct']:>8.0f}"
               f"{_cell(v['share_c1']):>10}{_cell(v['share_c2']):>10}"
               f"{_pm(v['len_mean'], v['len_sd']):>13}"
               f"{_cell(v['len_total'], '.1f'):>9}"
               f"{_cell(v['spread_median']):>8}{v['rigid_pct']:>8.0f}")
        if has_laman:
            row += ("-" if v["laman_pct"] is None else f"{v['laman_pct']:.0f}").rjust(8)
        lines.append(row)
    lines.append("")

    # ---- out-degree by domain
    dom = by_domain(nodes)
    if dom:
        methods = _methods_in(rows)
        present = [d for d in DOMAINS if any(r["domain"] == d for r in dom)]
        lines.append("OUT-DEGREE BY AGENT DOMAIN   mean/min, count-weighted over agents")
        lines.append(f"  {'domain':<10}{'r_i':>5}{'agents':>8}"
                     + "".join(f"{m:>16}" for m in methods))
        for d in present:
            cells = ""
            agents = 0
            for m in methods:
                rec = next((r for r in dom if r["method"] == m and r["domain"] == d), None)
                agents = max(agents, rec["agents"] if rec else 0)
                cells += ("-" if rec is None
                          else f"{rec['out_deg_mean']:.2f}/{rec['out_deg_min']}").rjust(16)
            lines.append(f"  {d:<10}{R_MIN[d]:>5}{agents:>8}{cells}")
        lines.append("")

    # ---- the contingency table
    cls = by_class(pairs)
    if cls:
        methods = _methods_in(rows)
        keys = sorted({(r["domain_i"], r["domain_j"]) for r in cls},
                      key=lambda k: (DOMAINS.index(k[0]), DOMAINS.index(k[1])))
        lines.append("EDGES BY ORDERED DOMAIN CLASS   share of the available pairs in it")
        lines.append(f"  {'measures ->':<24}{'c_k':>4}{'avail':>7}"
                     + "".join(f"{m:>16}" for m in methods))
        for di, dj in keys:
            cells, c_k, avail = "", None, 0
            for m in methods:
                rec = next((r for r in cls if r["method"] == m
                            and r["domain_i"] == di and r["domain_j"] == dj), None)
                if rec is not None:
                    c_k, avail = rec["c_k"], rec["avail"]
                cells += ("-" if rec is None
                          else f"{100 * rec['share']:.0f}%  ({rec['edges']})").rjust(16)
            lines.append(f"  {di + ' -> ' + dj:<24}{_cell(c_k, '.0f'):>4}{avail:>7}{cells}")
        # the prediction is that a reciprocal pair never pays off between two frameless
        # agents, so naming the classes where one appeared is shorter than a column
        lines.append("  reciprocal edges between two frameless agents, the classes where")
        lines.append("  the reverse adds no rank at all:")
        for m in methods:
            broke = sorted(f"{r['domain_i']}->{r['domain_j']} ({r['recip_edges']})"
                           for r in cls if r["method"] == m and r["recip_edges"]
                           and r["domain_i"] in FRAMELESS and r["domain_j"] in FRAMELESS)
            lines.append(f"    {m:<14}" + (", ".join(broke) if broke else "none"))
        lines.append("")

    # ---- Jaccard
    mat = jaccard_matrix(jac)
    if mat:
        methods = [m for m in _methods_in(rows) if any(k[0] == m for k in mat)]
        lines.append("EDGE-SET OVERLAP BETWEEN METHODS   Jaccard on the same instance, "
                     "mean (worst)")
        lines.append(f"  {'':<14}" + "".join(f"{m:>16}" for m in methods))
        for a in methods:
            cells = ""
            for b in methods:
                if a == b:
                    cells += "-".rjust(16)
                    continue
                cell = mat.get((a, b))
                cells += ("-" if cell is None
                          else f"{cell[0]:.2f} ({cell[1]:.2f})").rjust(16)
            lines.append(f"  {a:<14}{cells}")
        cal = mat.get(("greedy", "spectral"))
        if cal is not None:
            lines.append(f"  greedy against spectral: {cal[0]:.3f} mean, "
                         f"{cal[1]:.3f} worst over {cal[2]} instances")
            if kappa:
                lines.append(f"  this run is at stiffness_kappa {kappa}, where the closed "
                             f"form spectral reads is a ranking prior rather than the true")
                lines.append("  lambda, so below 1.00 is expected. It is exact, and that "
                             "cell must read 1.00, only at kappa 0.")
        lines.append("")

    if brief:
        return "\n".join(lines)

    import report
    lines.append("-" * w)
    lines.append("WHAT THE METHODS ARE")
    for v in agg:
        blurb = (report.METHOD_BLURB.get(v["method"]) or
                 ("a rank-greedy construction in random order, pruned back to minimality, "
                  "which is the reference the share and reciprocity columns are read "
                  "against" if v["method"] == NULL_METHOD else ""))
        lines.append(f"  {v['method']:<14} {blurb}")
    lines.append("")
    lines.append("WHAT THE COLUMNS MEAN")
    lines.append("  m         directed bearing measurements in the graph, and m/m_req is")
    lines.append("            that over the fewest these poses admit.")
    lines.append("  recip     fraction of EDGES whose reverse also exists, so a fully")
    lines.append("            reciprocal graph reads 1.00 and a graph with one reciprocal")
    lines.append("            pair out of ten edges reads 0.20. It is not pairs over pairs.")
    lines.append("  rand      the same fraction for a uniform random directed graph on the")
    lines.append("            same number of nodes and edges, in closed form. It is the")
    lines.append("            level reciprocity reaches by accident at that density.")
    lines.append("  recip     the same fraction split by whether both ends of the edge are")
    lines.append("   free /    agents with no frame of their own, where the reverse adds no")
    lines.append("   framed    rank whatsoever, or at least one end carries a frame, where it")
    lines.append("            adds 1 or 2. Only the first is predicted to be 0, so the")
    lines.append("            pooled recip column above reads the prediction at the wrong")
    lines.append("            grain and is kept only for comparison against rand.")
    lines.append("  outdeg    bearings one agent measures, averaged over agents and then")
    lines.append("            over networks, and min is the smallest seen anywhere.")
    lines.append("  slack     the smallest out-degree minus r_i seen on any agent, so a")
    lines.append("            negative value is a violated floor and 0 means some agent sits")
    lines.append("            exactly on it.")
    lines.append("  floor%    networks where every agent met its own out-degree floor r_i,")
    lines.append("            0 for an agent with no frame, 1 for one turning about a fixed")
    lines.append("            axis and 2 for a freely turning one. A rigid network must")
    lines.append("            meet it, so anything below 100 on a rigid row is a bug.")
    lines.append("  c1/c2     edges placed over ordered pairs available, split by how much")
    lines.append("            rank one such edge carries. c1 is the class where both ends")
    lines.append("            are planar agents, where a bearing is a single angle and the")
    lines.append("            edge carries rank 1; every other class carries 2. A")
    lines.append("            minimum-edge graph is expected to under-use c1.")
    lines.append("  len       edge length in formation radii, so it does not depend on how")
    lines.append("            far apart the agents happen to be, and len tot is the sum over")
    lines.append("            the graph. Compare it only at a similar m, which is why m is")
    lines.append("            in the same row.")
    lines.append("  spread    per-agent smallest singular value of the matrix of bearings")
    lines.append("            that agent measures, median over agents. It is zero when")
    lines.append("            those bearings are degenerate, meaning collinear for an agent")
    lines.append("            measuring two and coplanar for one measuring three or more,")
    lines.append("            so it means a different thing at each out-degree. The node")
    lines.append("            file carries out-degree beside it, and a second column there")
    lines.append("            holds the second singular value, which is the collinearity")
    lines.append("            test alone and is comparable across out-degrees.")
    if has_laman:
        lines.append("  laman%    networks whose underlying undirected graph is Laman, which")
        lines.append("            in homogeneous R^2 is exactly minimal bearing rigidity. It")
        lines.append("            is a check against a known theorem, not a method score.")
    lines.append("")
    lines.append("HOW TO READ IT")
    lines.append("  Four predictions, each with the column that tests it.")
    lines.append("  1  The reverse of an edge adds no rank when neither agent carries a")
    lines.append("     frame, so recip free should be 0 and the classes listed under the")
    lines.append("     contingency table should say none. recip framed has no prediction")
    lines.append("     attached, and the pooled recip is a mixture of the two, which is why")
    lines.append("     it sits near rand whatever the method does.")
    lines.append("  2  Every rigid framework already satisfies the out-degree floor, so")
    lines.append("     floor% carries no information beyond being 100. What does is how far")
    lines.append("     above the floor each domain sits, which is the per-domain block.")
    lines.append("  3  A planar-to-planar edge buys half the rank of any other, so a method")
    lines.append("     minimising edges should show a lower c1 share than c2 share, and a")
    lines.append("     lower c1 share than the random rigid graph at the same size.")
    lines.append("  4  A policy trained with the conditioning term should prefer shorter")
    lines.append("     edges than one trained without it at the same m. len is the column;")
    lines.append("     the kappa each policy was trained at is in the header.")
    lines.append("")
    lines.append("  Shares, reciprocity and the per-domain means are count-weighted rather")
    lines.append("  than averages of per-network fractions, because the domain mix is")
    lines.append("  redrawn per network and an average of fractions would over-weight")
    lines.append("  whichever network drew one high-DOF agent.")
    lines.append("  Nothing here is metered. What is measured is the graph and not the work")
    lines.append("  that found it, so anneal gets its fallback budget rather than the one")
    lines.append("  greedy spent; experiments/comparison.py is where the spend is reported.")
    lines.append("=" * w)
    return "\n".join(lines)
