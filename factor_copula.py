"""
Static Gaussian factor copula with k-means cluster estimation.
Stage 1 of Appendix B (Rensen 2025).

References:
    - Oh & Patton (2023), Sections 2-3, Appendix A.3
    - Rensen (2025), Section 2.3, Appendix B Stage 1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FactorLoadings:
    lambda_M: np.ndarray  # (G,) scaled market loadings λ̃^M_g
    lambda_C: np.ndarray  # (G,) scaled cluster loadings λ̃^C_g


@dataclass
class ClusterResult:
    assignments: np.ndarray
    loadings: FactorLoadings
    loglik: float
    bic: float
    n_groups: int
    n_iter: int
    tickers: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normal_scores(U: np.ndarray) -> np.ndarray:
    return norm.ppf(np.clip(U, 1e-8, 1 - 1e-8))


def _block_corr(S: np.ndarray, gamma: np.ndarray, G: int) -> np.ndarray:
    """G×G matrix of average within- and between-group correlations (Ω̂)."""
    Omega = np.zeros((G, G))
    groups = [np.where(gamma == g)[0] for g in range(G)]

    for g in range(G):
        ig = groups[g]
        for h in range(g, G):
            ih = groups[h]
            if g == h:
                if len(ig) >= 2:
                    block = S[np.ix_(ig, ig)]
                    mask = ~np.eye(len(ig), dtype=bool)
                    Omega[g, g] = block[mask].mean()
                else:
                    Omega[g, g] = 0.3
            else:
                Omega[g, h] = S[np.ix_(ig, ih)].mean()
                Omega[h, g] = Omega[g, h]
    return Omega


def _extract_loadings(Omega: np.ndarray) -> FactorLoadings:
    """
    Variance targeting: extract loadings from Ω̂ via eigendecomposition.

    Ω̂ ≈ λ̃^M (λ̃^M)' + diag((λ̃^C)²)
    Leading eigenvector → market loadings.
    Residual diagonal  → cluster loadings.
    """
    eigvals, eigvecs = np.linalg.eigh(Omega)
    k = np.argmax(eigvals)
    lam1 = max(eigvals[k], 1e-8)
    v1 = eigvecs[:, k]

    lam_M = np.sqrt(lam1) * v1
    if lam_M.sum() < 0:
        lam_M = -lam_M

    lam_C_sq = np.maximum(np.diag(Omega) - lam_M ** 2, 1e-8)
    return FactorLoadings(lambda_M=lam_M, lambda_C=np.sqrt(lam_C_sq))


def _distance_matrix(
    S: np.ndarray,
    gamma: np.ndarray,
    loadings: FactorLoadings,
    G: int,
) -> np.ndarray:
    """
    N×G matrix: d[i,g] = Σ_j (S[i,j] − R_model(i→g, j))².

    Measures how well variable i fits into group g.
    """
    N = S.shape[0]
    D = np.zeros((N, G))
    lM, lC = loadings.lambda_M, loadings.lambda_C

    for g in range(G):
        model = lM[g] * lM[gamma]
        model[gamma == g] += lC[g] ** 2
        diff = S - model[np.newaxis, :]
        np.fill_diagonal(diff, 0.0)
        D[:, g] = np.sum(diff ** 2, axis=1)

    return D


def _reassign(
    S: np.ndarray,
    gamma: np.ndarray,
    loadings: FactorLoadings,
    G: int,
    min_size: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """k-means step: move each variable to its closest group."""
    D = _distance_matrix(S, gamma, loadings, G)
    new_gamma = gamma.copy()
    order = rng.permutation(len(gamma)) if rng else np.arange(len(gamma))

    for i in order:
        old_g = new_gamma[i]
        old_size = np.sum(new_gamma == old_g)
        for g in np.argsort(D[i]):
            if g == old_g or old_size > min_size:
                new_gamma[i] = g
                break
    return new_gamma


def _build_R(gamma: np.ndarray, loadings: FactorLoadings) -> np.ndarray:
    """N×N model-implied correlation matrix from factor structure."""
    lM, lC = loadings.lambda_M, loadings.lambda_C
    R = np.outer(lM[gamma], lM[gamma])
    G = len(lM)
    for g in range(G):
        mask = gamma == g
        R[np.ix_(mask, mask)] += lC[g] ** 2
    np.fill_diagonal(R, 1.0)
    return R


def _gaussian_copula_ll(X: np.ndarray, R: np.ndarray) -> float:
    """
    Gaussian copula log-likelihood.
    L = Σ_t [ -½ log|R| - ½ x_t'(R⁻¹ − I)x_t ]
      = -T/2 log|R| - T/2 tr((R⁻¹ − I) S)
    where S = X'X / T.
    """
    T, N = X.shape
    try:
        cho = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        return -1e15
    log_det = 2.0 * np.sum(np.log(np.diag(cho)))
    R_inv = np.linalg.solve(R, np.eye(N))
    S = (X.T @ X) / T
    return -T / 2 * log_det - T / 2 * np.trace((R_inv - np.eye(N)) @ S)


def _random_init(N: int, G: int, rng: np.random.Generator) -> np.ndarray:
    gamma = np.zeros(N, dtype=int)
    perm = rng.permutation(N)
    for g in range(G):
        gamma[perm[2 * g]] = g
        gamma[perm[2 * g + 1]] = g
    rest = perm[2 * G :]
    gamma[rest] = rng.integers(0, G, size=len(rest))
    return gamma


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_static_clusters(
    U: pd.DataFrame,
    G: int,
    n_starts: int = 10,
    max_iter: int = 100,
    seed: int = 42,
) -> ClusterResult:
    """
    EM/k-means algorithm for static Gaussian factor copula.

    Parameters
    ----------
    U : (T, N) uniform PIT values from marginal model
    G : number of clusters
    n_starts : random restarts (Oh & Patton use 10)
    max_iter : max EM iterations per start
    seed : reproducibility

    Returns
    -------
    ClusterResult with best assignment, loadings, log-likelihood, BIC.
    """
    X = _normal_scores(U.values)
    T_, N = X.shape
    S = np.corrcoef(X.T)
    tickers = list(U.columns)
    rng = np.random.default_rng(seed)

    if 2 * G > N:
        raise ValueError(f"Need N ≥ 2G for identification. N={N}, G={G}.")

    best_ll = -np.inf
    best: Optional[ClusterResult] = None

    for _ in range(n_starts):
        gamma = _random_init(N, G, rng)

        for it in range(max_iter):
            Omega = _block_corr(S, gamma, G)
            loadings = _extract_loadings(Omega)
            gamma_new = _reassign(S, gamma, loadings, G, rng=rng)
            if np.array_equal(gamma, gamma_new):
                break
            gamma = gamma_new

        R = _build_R(gamma, loadings)
        ll = _gaussian_copula_ll(X, R)

        if ll > best_ll:
            best_ll = ll
            best = ClusterResult(
                assignments=gamma.copy(),
                loadings=loadings,
                loglik=ll,
                bic=-2 * ll + 2 * G * np.log(T_),
                n_groups=G,
                n_iter=it + 1,
                tickers=tickers,
            )

    return best  # type: ignore[return-value]


def select_n_groups(
    U: pd.DataFrame,
    G_range: range = range(3, 26),
    n_starts: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> dict[int, ClusterResult]:
    """Run fit_static_clusters for multiple G values, select by BIC."""
    results: dict[int, ClusterResult] = {}

    for G in G_range:
        if verbose:
            print(f"  G={G:>2}", end="", flush=True)
        res = fit_static_clusters(U, G, n_starts=n_starts, seed=seed)
        results[G] = res
        if verbose:
            print(f"  BIC={res.bic:>12.0f}  LL={res.loglik:>10.0f}  iter={res.n_iter}")

    best_G = min(results, key=lambda g: results[g].bic)
    if verbose:
        print(f"\n  → Optimal G = {best_G}  (BIC = {results[best_G].bic:.0f})")

    return results


def cluster_table(result: ClusterResult) -> pd.DataFrame:
    """Readable summary of cluster composition."""
    rows = []
    for g in range(result.n_groups):
        members = [
            result.tickers[i]
            for i in range(len(result.assignments))
            if result.assignments[i] == g
        ]
        rows.append({
            "Group": g + 1,
            "Size": len(members),
            "λ̃_M": f"{result.loadings.lambda_M[g]:.3f}",
            "λ̃_C": f"{result.loadings.lambda_C[g]:.3f}",
            "Members": ", ".join(sorted(members)),
        })
    df = pd.DataFrame(rows).sort_values("Size", ascending=False).reset_index(drop=True)
    return df