"""
Marginal model: AR(1) + GJR-GARCH(1,1) + Hansen (1994) skewed-t
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import t as student_t


@dataclass
class MarginalParams:
    phi0: float
    phi1: float
    omega: float
    alpha: float
    kappa: float
    beta: float
    eta: float
    lam: float


@dataclass
class MarginalResult:
    ticker: str
    params: MarginalParams
    std_resid: np.ndarray
    cond_vol: np.ndarray
    uniform: np.ndarray
    index: pd.DatetimeIndex


class HansenSkewT:
    """
    Standardized skewed Student-t from Hansen 1994

    Parameters
    ----------
    eta : float, > 2
        Degrees of freedom (tail thickness).
    lam : float, in (-1, 1)
        Asymmetry.  lam < 0 => left tail heavier.
    """

    @staticmethod
    def _constants(eta: float, lam: float) -> tuple[float, float, float]:
        c = np.exp(
            gammaln((eta + 1) / 2) - gammaln(eta / 2)
        ) / np.sqrt(np.pi * (eta - 2))
        a = 4 * lam * c * (eta - 2) / (eta - 1)
        b = np.sqrt(1 + 3 * lam ** 2 - a ** 2)
        return a, b, c

    @staticmethod
    def logpdf(x: np.ndarray, eta: float, lam: float) -> np.ndarray:
        a, b, c = HansenSkewT._constants(eta, lam)
        y = b * x + a
        s = np.where(x < -a / b, 1 - lam, 1 + lam)
        return np.log(b * c) - ((eta + 1) / 2) * np.log1p((y / s) ** 2 / (eta - 2))

    @staticmethod
    def cdf(x: np.ndarray, eta: float, lam: float) -> np.ndarray:
        a, b, _ = HansenSkewT._constants(eta, lam)
        y = b * x + a
        neg = x < -a / b
        s = np.where(neg, 1 - lam, 1 + lam)
        xi = y / s * np.sqrt(eta / (eta - 2))
        F = student_t.cdf(xi, eta)
        return np.where(neg, (1 - lam) * F, (1 - lam) / 2 + (1 + lam) * (F - 0.5))

    @staticmethod
    def fit(x: np.ndarray) -> tuple[float, float]:
        def neg_ll(params: np.ndarray) -> float:
            eta, lam = float(params[0]), float(params[1])
            if eta <= 2.1 or np.abs(lam) >= 0.98:
                return 1e12
            if 1 + 3 * lam ** 2 - (4 * lam * np.exp(
                gammaln((eta + 1) / 2) - gammaln(eta / 2)
            ) / np.sqrt(np.pi * (eta - 2)) * (eta - 2) / (eta - 1)) ** 2 <= 0:
                return 1e12
            return -float(np.sum(HansenSkewT.logpdf(x, eta, lam)))

        best_val, best_x = 1e12, np.array([6.0, 0.0])
        for x0 in ([6.0, 0.0], [4.0, -0.05], [10.0, 0.05], [3.0, -0.1]):
            res = minimize(neg_ll, x0, method="Nelder-Mead",
                           options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8})
            if res.fun < best_val:
                best_val, best_x = res.fun, res.x
        return float(best_x[0]), float(best_x[1])


####### AR(1) + GJR-GARCH(1,1)   â€”   Gaussian quasi-MLE ===> Change to full MLE not in 2 steps

def _garch_recursion(
    eps: np.ndarray,
    omega: float,
    alpha: float,
    kappa: float,
    beta: float,
    backcast: float,
) -> np.ndarray:
    T = len(eps)
    h = np.empty(T)
    h[0] = backcast
    for t in range(1, T):
        e2 = eps[t - 1] ** 2
        h[t] = omega + alpha * e2 + kappa * e2 * (eps[t - 1] < 0) + beta * h[t - 1]
    return h


def _gaussian_qml(params: np.ndarray, returns: np.ndarray) -> float:
    """Negative Gaussian log-likelihood (to minimize)"""
    phi0, phi1, log_omega, alpha, kappa, beta = params
    omega = np.exp(log_omega)

    if alpha < 0 or beta < 0 or alpha + kappa < 0 or alpha + kappa / 2 + beta >= 1:
        return 1e12

    T = len(returns)
    eps = np.empty(T)
    eps[0] = 0.0
    for t in range(1, T):
        eps[t] = returns[t] - phi0 - phi1 * returns[t - 1]

    backcast = np.mean(eps[1:] ** 2)
    h = _garch_recursion(eps, omega, alpha, kappa, beta, backcast)

    if np.any(h[1:] <= 0):
        return 1e12

    nll = 0.5 * np.sum(np.log(h[1:]) + eps[1:] ** 2 / h[1:])
    return float(nll)


def _fit_garch(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (params_array, standardized_residuals, conditional_volatility).
    params_array = [phi0, phi1, omega, alpha, kappa, beta].
    """
    r = returns
    T = len(r)

    phi1_init = np.corrcoef(r[1:], r[:-1])[0, 1]
    phi0_init = np.mean(r) * (1 - phi1_init)
    var_r = np.var(r)

    x0 = np.array([
        phi0_init,
        phi1_init,
        np.log(var_r * 0.05),    # log(omega)
        0.04,                    # alpha
        0.06,                    # kappa
        0.88,                    # beta
    ])

    best_val, best_x = 1e12, x0.copy()
    for perturbation in (
        np.zeros(6),
        np.array([0, 0, 0.5, 0.02, 0.02, -0.05]),
        np.array([0, 0, -0.5, -0.02, -0.02, 0.05]),
    ):
        res = minimize(
            _gaussian_qml, x0 + perturbation, args=(r,),
            method="Nelder-Mead",
            options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-10},
        )
        if res.fun < best_val:
            best_val, best_x = res.fun, res.x

    phi0, phi1, log_omega, alpha, kappa, beta = best_x
    omega = np.exp(log_omega)

    eps = np.empty(T)
    eps[0] = 0.0
    for t in range(1, T):
        eps[t] = r[t] - phi0 - phi1 * r[t - 1]

    backcast = np.mean(eps[1:] ** 2)
    h = _garch_recursion(eps, omega, alpha, kappa, beta, backcast)

    z = eps[1:] / np.sqrt(h[1:])
    sigma = np.sqrt(h[1:])
    params_out = np.array([phi0, phi1, omega, alpha, kappa, beta])
    return params_out, z, sigma


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def fit_single(returns: pd.Series) -> MarginalResult:
    ticker = str(returns.name or "unknown")
    r = returns.values.astype(np.float64)

    garch_params, z, sigma = _fit_garch(r)
    phi0, phi1, omega, alpha, kappa, beta = garch_params

    eta, lam = HansenSkewT.fit(z)
    u = np.clip(HansenSkewT.cdf(z, eta, lam), 1e-10, 1 - 1e-10)

    return MarginalResult(
        ticker=ticker,
        params=MarginalParams(
            phi0=phi0, phi1=phi1, omega=omega,
            alpha=alpha, kappa=kappa, beta=beta,
            eta=eta, lam=lam,
        ),
        std_resid=z,
        cond_vol=sigma,
        uniform=u,
        index=returns.index[1:],
    )


def fit_all(
    prices: pd.DataFrame,
    verbose: bool = True,
) -> dict[str, MarginalResult]:
    log_rets = compute_log_returns(prices)
    results: dict[str, MarginalResult] = {}
    n = len(log_rets.columns)
    for i, ticker in enumerate(log_rets.columns):
        if verbose:
            print(f"  [{i + 1:>3}/{n}] {ticker:<6}", end="")
        results[ticker] = fit_single(log_rets[ticker])
        if verbose:
            p = results[ticker].params
            print(f"  aplpha={p.alpha:.3f}  kappa={p.kappa:.3f}  beta={p.beta:.3f}"
                  f"  eta={p.eta:.1f}  lambda={p.lam:+.3f}")
    return results


def get_uniform_matrix(results: dict[str, MarginalResult]) -> pd.DataFrame:
    series = [pd.Series(r.uniform, index=r.index, name=t) for t, r in results.items()]
    return pd.concat(series, axis=1).dropna()


def summary_table(results: dict[str, MarginalResult]) -> pd.DataFrame:
    """Cross-sectional summary to compare with Appendix Table 3"""
    rows = []
    for r in results.values():
        p = r.params
        rows.append({
            "Constant": p.phi0,
            "AR(1)": p.phi1,
            "omegaÃ—1e4": p.omega * 1e4,
            "alpha": p.alpha,
            "kappa": p.kappa,
            "beta": p.beta,
            "eta": p.eta,
            "psi": p.lam,
        })
    df = pd.DataFrame(rows, index=[r.ticker for r in results.values()])
    return df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).loc[
        ["mean", "5%", "25%", "50%", "75%", "95%"]
    ]

