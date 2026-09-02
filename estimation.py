"""Shape estimation from noisy bearings: the measured counterpart of
rigidity.estimation_error. Evaluation only -- nothing here runs in step().

sigma is an angle in radians. Position error comes back in units of the
formation's RMS radius, attitude error in radians; the two are never summed.
"""
import copy

import numpy as np

from rigidity import (characteristic_length, extended_bearing_rigidity_matrix,
                      node_dof_projectors, nullspace, rigidity_decomposition,
                      scaled_rigidity_matrix)
from util import skew_symmetric
from cost import counted


# ---------------------------------------------------------------- the bearing map
def edge_list(network):
    """Directed edges in B's row-block order."""
    i_idx, j_idx = np.nonzero(network.edges)
    return list(zip(i_idx.tolist(), j_idx.tolist()))


def true_bearings(network):
    """(3m,) stacked R_i^T p_hat_ij, in B's row-block order.

    Built the way extended_bearing_rigidity_matrix builds them, not through
    Agent.get_bearing, which returns the world vector for R^2/R^3.
    """
    out = []
    for i, j in edge_list(network):
        pij = network.agents[j].pose.position - network.agents[i].pose.position
        out.append(network.agents[i].pose.rotation_mat().T
                   @ (pij / np.linalg.norm(pij)))
    return np.concatenate(out) if out else np.zeros(0)


def perturb_bearings(network, sigma, rng, bearings=None):
    """Noisy unit bearings: z = normalize(b + sigma * (I - b b^T) eps).

    Small-angle von Mises-Fisher, so sigma is an angle in radians. Full 2-DOF
    tangent in every domain: the DOF restriction is on an agent's motion, not on
    its camera.
    """
    b = true_bearings(network) if bearings is None else np.asarray(bearings)
    if b.size == 0:
        return b
    B3 = b.reshape(-1, 3)
    eps = rng.normal(size=B3.shape)
    tangent = eps - (np.einsum("kd,kd->k", eps, B3)[:, None] * B3)
    z = B3 + sigma * tangent
    return (z / np.linalg.norm(z, axis=1, keepdims=True)).reshape(-1)


# ---------------------------------------------------------------- the state update
def so3_exp(w):
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    K = skew_symmetric(np.asarray(w, dtype=float) / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def so3_log(R):
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(c))
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-9:
        return 0.5 * axis
    return (theta / (2.0 * np.sin(theta))) * axis


def restrict_to_dofs(network, delta):
    """S_i on each position increment, P_i on each attitude increment.

    B zeroes the same columns, so lstsq's minimum-norm solution already lands
    here; explicit so it does not depend on the solver's tolerance.
    """
    n = network.n
    out = np.asarray(delta, dtype=float).copy()
    for i, agent in enumerate(network.agents):
        S, P = node_dof_projectors(agent)
        out[3 * i:3 * i + 3] = S @ out[3 * i:3 * i + 3]
        out[3 * n + 3 * i:3 * n + 3 * i + 3] = P @ out[3 * n + 3 * i:3 * n + 3 * i + 3]
    return out


def apply_variation(network, delta, step=1.0):
    """chi <- chi + step*delta on a copy, in the parameterisation B differentiates.

    delta is (6n,): position increments, then world-frame axis-angle attitude
    increments applied on the left, R_i <- exp(skew(w_i)) R_i.
    """
    n = network.n
    out = copy.deepcopy(network)
    for i, agent in enumerate(out.agents):
        agent.pose.position = agent.pose.position + step * delta[3 * i:3 * i + 3]
        w = step * delta[3 * n + 3 * i:3 * n + 3 * i + 3]
        if np.linalg.norm(w) > 0:
            agent.pose.set_rotation_mat(so3_exp(w) @ agent.pose.rotation_mat())
    return out


# ---------------------------------------------------------------- the solver
@counted
def solve_shape(network, z, init=None, iters=30, tol=1e-12, backtracks=6):
    """Gauss-Newton on sum ||b_ij(chi) - z_ij||^2. Returns (estimate, info).

    B is the Jacobian, so the step solves B dchi = z - b(chi). lstsq's
    minimum-norm solution puts zero in ker(B), which holds the iterate inside
    each agent's DOFs and off the gauge.

    `init` defaults to the true poses, so this measures local accuracy.
    """
    est = copy.deepcopy(network if init is None else init)
    z = np.asarray(z, dtype=float)
    if z.size == 0:
        return est, {"iters": 0, "residual": 0.0, "converged": True}

    residual = float(np.linalg.norm(z - true_bearings(est)))
    it, settled = 0, False
    for it in range(1, iters + 1):
        r = z - true_bearings(est)
        J = extended_bearing_rigidity_matrix(est)
        delta = restrict_to_dofs(est, np.linalg.lstsq(J, r, rcond=None)[0])

        if not np.all(np.isfinite(delta)) or np.linalg.norm(delta) < tol:
            settled = True
            break

        step, improved = 1.0, False
        for _ in range(backtracks):
            trial = apply_variation(est, delta, step)
            trial_residual = float(np.linalg.norm(z - true_bearings(trial)))
            if trial_residual <= residual:
                est, residual, improved = trial, trial_residual, True
                break
            step *= 0.5
        if not improved:
            settled = True
            break

    # stopped on its own rather than running out of budget; says nothing about
    # which minimum it stopped at
    return est, {"iters": it, "residual": residual, "converged": bool(settled)}


# ---------------------------------------------------------------- the error metric
def gauge_basis(network, rank_K, length_scale=None):
    """ker(B_K), the variations no bearing can detect, in length-normalised units.

    Translation and uniform scaling everywhere, plus a global rotation only in
    the domains that carry a frame. Static: Schiano and Tron show the scale
    leaves this set once the agents move with known inputs.
    """
    K = network.fully_connected()
    if length_scale is None:
        length_scale = characteristic_length(network)
    BK = scaled_rigidity_matrix(K, length_scale=length_scale)
    return nullspace(BK, int(rank_K))


def shape_error(net_true, net_est, rank_K, Z_K=None, length_scale=None,
                max_scale_ratio=5.0):
    """(rms_position, rms_attitude) with the unobservable gauge projected out.

    The projection off ker(B_K) is exact only to first order, matching the
    linearisation of the Cramer-Rao prediction, so this is a small-error metric.
    A scaling by zero is a gauge direction, so a collapsed formation would
    otherwise project to no error at all; max_scale_ratio returns inf instead.

    The gauge mixes the two blocks, so the split reports the quotiented error's
    components rather than an orthogonal decomposition.
    """
    n = net_true.n
    if length_scale is None:
        length_scale = characteristic_length(net_true)
    if Z_K is None:
        Z_K = gauge_basis(net_true, rank_K, length_scale)

    ratio = characteristic_length(net_est) / max(length_scale, 1e-30)
    if not (1.0 / max_scale_ratio < ratio < max_scale_ratio):
        return np.inf, np.inf

    e = np.zeros(6 * n)
    for i in range(n):
        a, b = net_true.agents[i], net_est.agents[i]
        e[3 * i:3 * i + 3] = (b.pose.position - a.pose.position) / length_scale
        e[3 * n + 3 * i:3 * n + 3 * i + 3] = so3_log(
            b.pose.rotation_mat() @ a.pose.rotation_mat().T)

    if Z_K.shape[1]:
        e = e - Z_K @ (Z_K.T @ e)

    return (float(np.linalg.norm(e[:3 * n]) / np.sqrt(n)),
            float(np.linalg.norm(e[3 * n:]) / np.sqrt(n)))


def monte_carlo_error(network, sigma, trials=30, rng=None, rank_K=None,
                      length_scale=None, init=None):
    """{'position': {rms, mean, median, p90}, 'attitude': {...}, 'converged': f}."""
    rng = np.random.default_rng(0) if rng is None else rng
    if rank_K is None:
        rank_K = int(np.linalg.matrix_rank(
            extended_bearing_rigidity_matrix(network.fully_connected())))
    if length_scale is None:
        length_scale = characteristic_length(network)
    Z_K = gauge_basis(network, rank_K, length_scale)

    b0 = true_bearings(network)
    pos, att, ok = [], [], 0
    for _ in range(int(trials)):
        z = perturb_bearings(network, sigma, rng, bearings=b0)
        est, info = solve_shape(network, z, init=init)
        dp, da = shape_error(network, est, rank_K, Z_K=Z_K,
                             length_scale=length_scale)
        pos.append(dp)
        att.append(da)
        ok += int(info["converged"])

    # rms, not mean, is what the Cramer-Rao bound predicts: it bounds E[||x||^2].
    # A trial whose estimate collapsed has infinite error and cannot enter any of
    # these, so it is dropped and counted instead -- `failed` is the honest report.
    def summary(v):
        v = np.asarray(v, dtype=float)
        good = v[np.isfinite(v)]
        if not len(good):
            return {"rms": np.inf, "mean": np.inf, "median": np.inf, "p90": np.inf,
                    "failed": 1.0}
        return {"rms": float(np.sqrt((good ** 2).mean())), "mean": float(good.mean()),
                "median": float(np.median(good)),
                "p90": float(np.percentile(good, 90)),
                "failed": float(1.0 - len(good) / len(v))}

    return {"position": summary(pos), "attitude": summary(att),
            "converged": ok / max(int(trials), 1)}


def predicted_error(network, rank_K, brmat=None, length_scale=None):
    """(position, attitude) RMS error predicted per unit sigma. Multiply by sigma.

    The two blocks of (B^T B)^+ separately: the whole trace predicts a number
    mixing lengths and radians.
    """
    from rigidity import estimation_error_blocks
    Bs = scaled_rigidity_matrix(network, brmat, length_scale)
    a_pos, a_att = estimation_error_blocks(Bs, rank_K, network.n)
    n = network.n
    return (float(np.sqrt(a_pos / n)) if np.isfinite(a_pos) else np.inf,
            float(np.sqrt(a_att / n)) if np.isfinite(a_att) else np.inf)
