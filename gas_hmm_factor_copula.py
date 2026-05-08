from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm



@dataclass
class GASParams:
    omega_M: float
    A_M: float
    B_M: float
    omega_C: float
    A_C: float
    B_C: float
    delta: float


@dataclass
class GASHMMResult:
    params: GASParams
    f_M: pd.DataFrame
    f_C: pd.DataFrame
    lambda_M: pd.DataFrame
    lambda_C: pd.DataFrame
    gamma_time: pd.DataFrame
    posterior_probs: dict
    transition_matrices: list
    loglik: float
    n_iter: int
    converged: bool
    static_assignments: pd.Series


def _clip_u(U, eps=1e-6):
    if isinstance(U, pd.DataFrame):
        return U.clip(eps, 1.0 - eps)
    return np.clip(np.asarray(U, dtype=float), eps, 1.0 - eps)


def _to_gaussian_scores(U, eps=1e-6):
    if isinstance(U, pd.DataFrame):
        Uc = U.clip(eps, 1.0 - eps)
        return norm.ppf(Uc.values), U.index, U.columns
    Uc = np.clip(np.asarray(U, dtype=float), eps, 1.0 - eps)
    return norm.ppf(Uc), None, None


def _safe_inverse_and_logdet(R, jitter=1e-8, max_tries=6):
    R = np.asarray(R, dtype=float)
    N = R.shape[0]
    for k in range(max_tries):
        try:
            Rj = R + (10 ** k) * jitter * np.eye(N)
            sign, logdet = np.linalg.slogdet(Rj)
            if sign > 0 and np.isfinite(logdet):
                invR = np.linalg.inv(Rj)
                return invR, logdet
        except np.linalg.LinAlgError:
            pass
    Rj = R + 1e-3 * np.eye(N)
    sign, logdet = np.linalg.slogdet(Rj)
    if sign <= 0:
        raise np.linalg.LinAlgError("Could not regularize correlation matrix")
    return np.linalg.pinv(Rj), logdet


def transform_states_to_loadings(f_M_t, f_C_t, max_total_loading=0.98):
    """
    Maps unconstrained GAS states to admissible factor loadings
    """
    f_M_t = np.asarray(f_M_t, dtype=float)
    f_C_t = np.asarray(f_C_t, dtype=float)

    lambda_M = max_total_loading * np.tanh(f_M_t)
    remaining = np.sqrt(np.maximum(max_total_loading**2 - lambda_M**2, 1e-12))
    lambda_C = remaining * np.tanh(f_C_t)
    return lambda_M, lambda_C


def build_R_from_assignments(assignments, lambda_M_t, lambda_C_t, jitter=1e-8):
    """
    Builds the Gaussian factor-copula correlation matrix for a given date t
    """
    assignments = np.asarray(assignments, dtype=int)
    lambda_M_t = np.asarray(lambda_M_t, dtype=float)
    lambda_C_t = np.asarray(lambda_C_t, dtype=float)
    N = len(assignments)

    R = np.eye(N)
    for i in range(N):
        gi = assignments[i]
        for j in range(i + 1, N):
            gj = assignments[j]
            rho = lambda_M_t[gi] * lambda_M_t[gj]
            if gi == gj:
                rho += lambda_C_t[gi] * lambda_C_t[gj]
            rho = float(np.clip(rho, -0.995, 0.995))
            R[i, j] = rho
            R[j, i] = rho
    return R + jitter * np.eye(N)


def gaussian_copula_loglik_t(x_t, assignments, lambda_M_t, lambda_C_t):
    """
    Gaussian copula log density at one date
    """
    R = build_R_from_assignments(assignments, lambda_M_t, lambda_C_t)
    invR, logdet = _safe_inverse_and_logdet(R)
    x = np.asarray(x_t, dtype=float)
    q = x @ (invR - np.eye(len(x))) @ x
    return float(-0.5 * logdet - 0.5 * q)


def gaussian_copula_loglik_path(X, gamma_time, lambda_M_path, lambda_C_path):
    T, N = X.shape
    total = 0.0
    for t in range(T):
        total += gaussian_copula_loglik_t(
            X[t], gamma_time[t], lambda_M_path[t], lambda_C_path[t]
        )
    return total


# -----------------------------------------------------------------------------
# GAS score
# -----------------------------------------------------------------------------

def numerical_score_t(x_t, assignments, f_M_t, f_C_t, eps=1e-4):
    """
    Slower than an analytical score, but faithful to the GAS definition: update parameters using the score of the conditional copula log-likelihood
    """
    f_M_t = np.asarray(f_M_t, dtype=float)
    f_C_t = np.asarray(f_C_t, dtype=float)
    G = len(f_M_t)

    def ll_from_states(a, b):
        lm, lc = transform_states_to_loadings(a, b)
        return gaussian_copula_loglik_t(x_t, assignments, lm, lc)

    score_M = np.zeros(G)
    score_C = np.zeros(G)

    for g in range(G):
        fp = f_M_t.copy(); fm = f_M_t.copy()
        fp[g] += eps; fm[g] -= eps
        score_M[g] = (ll_from_states(fp, f_C_t) - ll_from_states(fm, f_C_t)) / (2 * eps)

        fp = f_C_t.copy(); fm = f_C_t.copy()
        fp[g] += eps; fm[g] -= eps
        score_C[g] = (ll_from_states(f_M_t, fp) - ll_from_states(f_M_t, fm)) / (2 * eps)

    return score_M, score_C


def gas_recursion(X, gamma_time, params, f0_M, f0_C, score_scale=1.0):
    """
    Computes the GAS paths for f and lambda given hard cluster assignments
    """
    T, N = X.shape
    G = len(f0_M)

    f_M = np.zeros((T, G))
    f_C = np.zeros((T, G))
    lambda_M = np.zeros((T, G))
    lambda_C = np.zeros((T, G))

    f_M[0] = f0_M
    f_C[0] = f0_C

    for t in range(T):
        lambda_M[t], lambda_C[t] = transform_states_to_loadings(f_M[t], f_C[t])

        if t < T - 1:
            sM, sC = numerical_score_t(X[t], gamma_time[t], f_M[t], f_C[t])
            sM = np.clip(score_scale * sM, -25.0, 25.0)
            sC = np.clip(score_scale * sC, -25.0, 25.0)

            f_M[t + 1] = params.omega_M + params.A_M * sM + params.B_M * f_M[t]
            f_C[t + 1] = params.omega_C + params.A_C * sC + params.B_C * f_C[t]

            f_M[t + 1] = np.clip(f_M[t + 1], -4.0, 4.0)
            f_C[t + 1] = np.clip(f_C[t + 1], -4.0, 4.0)

    return f_M, f_C, lambda_M, lambda_C


def _unpack_theta(theta):
    """
    Constrained parametrization for stable optimization. A coefficients positive, B in (0, .999), delta positive.
    """
    omega_M = theta[0]
    A_M = np.exp(theta[1])
    B_M = 0.999 / (1.0 + np.exp(-theta[2]))
    omega_C = theta[3]
    A_C = np.exp(theta[4])
    B_C = 0.999 / (1.0 + np.exp(-theta[5]))
    delta = np.exp(theta[6])
    return GASParams(omega_M, A_M, B_M, omega_C, A_C, B_C, delta)


def _pack_initial_theta():
    # conservative starting values: weak score response and persistent dynamics
    return np.array([
        0.0,
        np.log(0.01),
        np.log(0.95 / (0.999 - 0.95)),
        0.0,
        np.log(0.01),
        np.log(0.95 / (0.999 - 0.95)),
        np.log(20.0),
    ], dtype=float)


def initial_states_from_static_loadings(static_result, G):
    """
    Builds initial f states from static loadings.If inversion is numerically difficult, uses small positive states.
    """
    lm = np.asarray(static_result.loadings.lambda_M, dtype=float)
    lc = np.asarray(static_result.loadings.lambda_C, dtype=float)
    lm = np.clip(lm, -0.95, 0.95)

    f0_M = np.arctanh(np.clip(lm / 0.98, -0.999, 0.999))

    remaining = np.sqrt(np.maximum(0.98**2 - lm**2, 1e-10))
    ratio_c = np.clip(lc / remaining, -0.999, 0.999)
    f0_C = np.arctanh(ratio_c)

    if len(f0_M) != G:
        f0_M = np.resize(f0_M, G)
        f0_C = np.resize(f0_C, G)
    return f0_M, f0_C


def fit_gas_parameters_fixed_gamma(
    U,
    gamma_time,
    static_result,
    maxiter=80,
    method="Nelder-Mead",
    verbose=True,
):
    """
    Estimates GAS parameters for a fixed gamma path. This is the dynamic-loading counterpart of the static factor copula.
    """
    X, index, columns = _to_gaussian_scores(U)
    gamma_time = np.asarray(gamma_time, dtype=int)
    G = int(gamma_time.max()) + 1
    f0_M, f0_C = initial_states_from_static_loadings(static_result, G)

    def objective(theta):
        params = _unpack_theta(theta)
        try:
            _, _, lm_path, lc_path = gas_recursion(X, gamma_time, params, f0_M, f0_C)
            ll = gaussian_copula_loglik_path(X, gamma_time, lm_path, lc_path)
            if not np.isfinite(ll):
                return 1e12
            return -ll
        except Exception:
            return 1e12

    theta0 = _pack_initial_theta()
    opt = minimize(objective, theta0, method=method, options={"maxiter": maxiter, "disp": verbose})
    params = _unpack_theta(opt.x)
    f_M, f_C, lm, lc = gas_recursion(X, gamma_time, params, f0_M, f0_C)
    ll = gaussian_copula_loglik_path(X, gamma_time, lm, lc)
    return params, f_M, f_C, lm, lc, ll, opt


def transition_matrix_t(lambda_M_t, delta):

    lm = np.asarray(lambda_M_t, dtype=float)
    d = np.abs(lm[:, None] - lm[None, :])
    logits = -float(delta) * d
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    return P


def conditional_loglik_asset_all_g(X, t, asset_idx, base_assignments_t, lambda_M_t, lambda_C_t):

    T, N = X.shape
    G = len(lambda_M_t)
    other_idx = np.array([j for j in range(N) if j != asset_idx])
    x_i = X[t, asset_idx]
    X_o = X[t, other_idx]

    out = np.zeros(G)
    for g in range(G):
        a = base_assignments_t.copy()
        a[asset_idx] = g
        R = build_R_from_assignments(a, lambda_M_t, lambda_C_t)
        R_oo = R[np.ix_(other_idx, other_idx)]
        r_io = R[asset_idx, other_idx]
        inv_R_oo, _ = _safe_inverse_and_logdet(R_oo)
        beta = r_io @ inv_R_oo
        cond_var = max(float(1.0 - beta @ r_io.T), 1e-8)
        cond_mean = float(X_o @ beta)
        resid = x_i - cond_mean
        # log phi_cond(x_i|x_-i) - log phi(x_i)
        out[g] = -0.5 * (np.log(cond_var) + resid**2 / cond_var - x_i**2)
    return out


def hmm_filter_dynamic_loadings(
    U,
    lambda_M_path,
    lambda_C_path,
    initial_gamma,
    delta,
    init_strength=0.98,
):
    """
    Runs the HMM forward filter for all assets using dynamic GAS loadings.

    Other firms' cluster paths are kept fixed at their current values when
    computing each firm's conditional likelihood, matching the conditional
    update logic used for high-dimensional tractability.
    """
    X, index, columns = _to_gaussian_scores(U)
    T, N = X.shape
    lambda_M_path = np.asarray(lambda_M_path, dtype=float)
    lambda_C_path = np.asarray(lambda_C_path, dtype=float)
    current_gamma = np.asarray(initial_gamma, dtype=int)
    if current_gamma.ndim == 1:
        current_gamma = np.tile(current_gamma[None, :], (T, 1))

    G = lambda_M_path.shape[1]
    gamma_new = np.zeros((T, N), dtype=int)
    posterior_probs = {}
    transition_matrices = [transition_matrix_t(lambda_M_path[t], delta) for t in range(T)]

    for i in range(N):
        posterior_i = np.zeros((T, G))

        init_probs = np.ones(G) * ((1.0 - init_strength) / max(G - 1, 1))
        init_probs[current_gamma[0, i]] = init_strength

        # t=0
        loglik0 = conditional_loglik_asset_all_g(
            X, 0, i, current_gamma[0], lambda_M_path[0], lambda_C_path[0]
        )
        log_joint = np.log(init_probs + 1e-300) + loglik0
        posterior_i[0] = np.exp(log_joint - logsumexp(log_joint))

        for t in range(1, T):
            Pprev = transition_matrices[t - 1]
            pred = posterior_i[t - 1] @ Pprev
            loglik_t = conditional_loglik_asset_all_g(
                X, t, i, current_gamma[t], lambda_M_path[t], lambda_C_path[t]
            )
            log_joint = np.log(pred + 1e-300) + loglik_t
            posterior_i[t] = np.exp(log_joint - logsumexp(log_joint))

        gamma_new[:, i] = posterior_i.argmax(axis=1)
        name = columns[i] if columns is not None else f"asset_{i}"
        posterior_probs[name] = pd.DataFrame(
            posterior_i,
            index=index,
            columns=[f"cluster_{g}" for g in range(G)],
        )

    gamma_df = pd.DataFrame(gamma_new, index=index, columns=columns)
    return gamma_df, posterior_probs, transition_matrices


def fit_gas_hmm_dynamic_clusters(
    U,
    static_result,
    max_outer_iter=5,
    gas_maxiter=60,
    init_strength=0.98,
    tol_assignments=0.001,
    verbose=True,
):
    """
    Full Gaussian version of the Rensen-style dynamic clustering model:

    1. Initialize gamma_i using static clusters.
    2. Estimate GAS dynamic loadings lambda M(g,t), lambda C(g,t).
    3. Build time-varying HMM transition matrices using lambda^M_{g,t} distances.
    4. Forward-filter cluster probabilities for every asset.
    5. Update gamma_(i,t) = argmax posterior.
    6. Iterate until assignments stabilize.

    This is the exact next modelling block after the static Gaussian factor copula.
    Student-t and skew-t copulas can be added once this Gaussian block is stable.
    """
    X, index, columns = _to_gaussian_scores(U)
    T, N = X.shape
    static_assignments = np.asarray(static_result.assignments, dtype=int)
    gamma = np.tile(static_assignments[None, :], (T, 1))

    last_change_rate = np.inf
    final = None

    for outer in range(max_outer_iter):
        if verbose:
            print(f"\n[GAS-HMM] Outer iteration {outer + 1}/{max_outer_iter}")

        params, f_M, f_C, lm, lc, ll, opt = fit_gas_parameters_fixed_gamma(
            U=U,
            gamma_time=gamma,
            static_result=static_result,
            maxiter=gas_maxiter,
            verbose=False,
        )

        gamma_df, posterior_probs, P_list = hmm_filter_dynamic_loadings(
            U=U,
            lambda_M_path=lm,
            lambda_C_path=lc,
            initial_gamma=gamma,
            delta=params.delta,
            init_strength=init_strength,
        )

        gamma_new = gamma_df.values.astype(int)
        change_rate = np.mean(gamma_new != gamma)

        if verbose:
            print(f"  loglik={ll:.2f}")
            print(f"  params={params}")
            print(f"  assignment change rate={change_rate:.4%}")

        gamma = gamma_new
        last_change_rate = change_rate

        final = (params, f_M, f_C, lm, lc, ll, posterior_probs, P_list, outer + 1)

        if change_rate <= tol_assignments:
            break

    params, f_M, f_C, lm, lc, ll, posterior_probs, P_list, n_iter = final

    cols_g = [f"cluster_{g}" for g in range(lm.shape[1])]
    f_M_df = pd.DataFrame(f_M, index=index, columns=cols_g)
    f_C_df = pd.DataFrame(f_C, index=index, columns=cols_g)
    lm_df = pd.DataFrame(lm, index=index, columns=cols_g)
    lc_df = pd.DataFrame(lc, index=index, columns=cols_g)
    gamma_df = pd.DataFrame(gamma, index=index, columns=columns)

    static_ser = pd.Series(static_assignments, index=columns, name="static_cluster")

    return GASHMMResult(
        params=params,
        f_M=f_M_df,
        f_C=f_C_df,
        lambda_M=lm_df,
        lambda_C=lc_df,
        gamma_time=gamma_df,
        posterior_probs=posterior_probs,
        transition_matrices=P_list,
        loglik=ll,
        n_iter=n_iter,
        converged=(last_change_rate <= tol_assignments),
        static_assignments=static_ser,
    )


def gas_hmm_transition_summary(result):
    gamma = result.gamma_time
    switches = (gamma.diff().fillna(0) != 0).sum(axis=0)
    modal = gamma.mode(axis=0).iloc[0].astype(int)
    out = pd.DataFrame({
        "static_cluster": result.static_assignments,
        "modal_dynamic_cluster": modal,
        "n_switches": switches,
        "switch_rate": switches / max(len(gamma) - 1, 1),
    })
    return out.sort_values("n_switches", ascending=False)


def gas_hmm_cluster_occupancy(result):
    gamma = result.gamma_time
    G = int(gamma.values.max()) + 1
    occ = pd.DataFrame(index=gamma.index)
    for g in range(G):
        occ[f"cluster_{g}"] = (gamma == g).sum(axis=1)
    return occ


def posterior_for_ticker(result, ticker):
    return result.posterior_probs[ticker]

