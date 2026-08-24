"""
four_pl.py
----------
Four-parameter logistic (4PL) power curve model with multiple
fitting strategies: 4PL MLE and the Kusiak power-distribution MLE.

Also provides Weibull parameter estimation and utilities for
computing the analytical power PDF derived from a Weibull wind
speed distribution.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def four_pl(
    v: np.ndarray,
    a: float,
    m: float,
    n: float,
    tau: float,
) -> np.ndarray:
    """
    Four-parameter logistic (4PL) power curve.

        P(v) = a * (1 + m * exp(-v / tau)) / (1 + n * exp(-v / tau))

    Parameters
    ----------
    v : np.ndarray
        Wind speed values (m/s).
    a : float
        Asymptotic power parameter (kW).
    m : float
        Lower asymptote shape parameter.
    n : float
        Upper asymptote shape parameter.
    tau : float
        Scale parameter (m/s).

    Returns
    -------
    np.ndarray
        Predicted power values (kW).
    """
    return a * (1 + m * np.exp(-v / tau)) / (1 + n * np.exp(-v / tau))


# ---------------------------------------------------------------------------
# Likelihood helpers
# ---------------------------------------------------------------------------

def pdf_P(
    x: np.ndarray,
    p: np.ndarray,
    a: float,
    m: float,
    n: float,
    tau: float,
    sigma: float,
) -> np.ndarray:
    """
    Gaussian likelihood of observed power ``p`` given wind speed ``x``
    and 4PL parameters.

    Parameters
    ----------
    x : np.ndarray
        Wind speed values (m/s).
    p : np.ndarray
        Observed power values (kW).
    a, m, n, tau : float
        4PL curve parameters.
    sigma : float
        Standard deviation of the Gaussian noise.

    Returns
    -------
    np.ndarray
        Likelihood values for each observation.
    """
    numer = np.sqrt(2 * np.pi * sigma ** 2)
    exponent = -1 / (2 * sigma ** 2) * \
        np.pow((p - four_pl(x, a, m, n, tau)), 2)
    return (1 / numer) * np.exp(exponent)


# ---------------------------------------------------------------------------
# Fitting — Evolutionary Algorithm (Differential Evolution) + local polish
# ---------------------------------------------------------------------------

def fourpl_mle(
    x: np.ndarray,
    y: np.ndarray,
    a0: float = None,
    maxiter: int = 500,
    popsize: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """
    Fit the 4PL curve by maximum likelihood using a differential
    evolution (DE) evolutionary algorithm with local Powell polishing.

    The parameter space is explored globally by a population-based
    evolutionary algorithm (DE/best/1/bin strategy), which applies
    selection, crossover, and mutation operators without requiring
    gradient information. This makes it well-suited to the
    multi-modal, non-convex log-likelihood surface of the 4PL model.
    A final local Powell refinement is applied once the evolutionary
    search has converged (``polish=True``).

    The initial population is drawn from a Latin hypercube design to
    ensure uniform coverage of the search space.

    Parameters
    ----------
    x : np.ndarray
        Wind speed observations (m/s).
    y : np.ndarray
        Power observations (kW).
    a0 : float, optional
        Upper bound hint for the asymptotic power. Defaults to
        ``max(y)``.
    maxiter : int
        Maximum number of evolutionary generations.
    popsize : int
        Population size multiplier (total population = popsize *
        number of parameters).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Optimised parameters ``[a, m, n, tau, sigma]``.

    Raises
    ------
    ValueError
        If no valid observations are provided.
    RuntimeError
        If the optimisation fails to converge.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if y.size == 0:
        raise ValueError("No valid power observations provided.")

    if a0 is None:
        a0 = float(np.max(y))

    bounds = [
        (a0-10, a0+10),  # a
        (-0.002, -0.001),     # m
        (0.01, 100),     # n
        (0.05, 100),    # tau
        (0.01, 1000),     # sigma
    ]

    def neg_loglikelihood(theta):
        return -float(np.sum(np.log(pdf_P(x, y, *theta))))

    result = differential_evolution(
        neg_loglikelihood,
        bounds=bounds,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-7,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        polish=True,
        updating="deferred",
        workers=1,
        init="latinhypercube",
    )

    if not result.success:
        raise RuntimeError(f"MLE optimisation failed: {result.message}")

    return result.x


# ---------------------------------------------------------------------------
# Weibull helpers
# ---------------------------------------------------------------------------

def estimate_weibull_from_x(
    x: np.ndarray,
    k0: float = 1.5,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> tuple[float, float]:
    """
    Estimate Weibull shape ``k`` and scale ``c`` by the fixed-point MLE:

        k = [ sum(x^k ln x) / sum(x^k) - mean(ln x) ]^{-1}
        c = [ mean(x^k) ]^{1/k}

    If the fixed-point iteration does not converge (e.g. due to a poor
    starting point or heavy clustering of values), falls back to a
    numerical MLE via ``scipy.optimize.minimize``.

    Returns
    -------
    tuple[float, float]
        ``(k_hat, c_hat)``
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.where(x <= 0, 1e-9, x)

    if x.size == 0:
        raise ValueError("x contains no valid observations.")

    logx = np.log(x)
    mean_logx = np.mean(logx)
    k = float(k0)

    for _ in range(1, max_iter + 1):
        with np.errstate(over='ignore', invalid='ignore'):
            xk = x ** k
            bracket = np.sum(xk * logx) / np.sum(xk) - mean_logx

        if bracket <= 0 or not np.isfinite(bracket):
            break  # fall through to numerical fallback

        k_new = 1.0 / bracket
        if not np.isfinite(k_new):
            break  # fall through to numerical fallback

        if abs(k_new - k) < tol:
            k = k_new
            return float(k), float((np.mean(x ** k)) ** (1.0 / k))

        k = k_new
    else:
        # Loop completed max_iter times without break — use last k
        return float(k), float((np.mean(x ** k)) ** (1.0 / k))

    # ----------------------------------------------------------------
    # Numerical MLE fallback (negative log-likelihood of Weibull)
    # ----------------------------------------------------------------
    def neg_ll(params):
        k_, c_ = params
        if k_ <= 0 or c_ <= 0:
            return 1e12
        return -float(np.sum(
            np.log(k_ / c_) + (k_ - 1) * np.log(x / c_) - (x / c_) ** k_
        ))

    result = minimize(
        neg_ll,
        x0=[max(k0, 0.5), float(np.mean(x))],
        bounds=[(1e-4, 50.0), (1e-6, None)],
        method="L-BFGS-B",
    )

    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(
            "Weibull estimation failed: both fixed-point iteration and "
            "numerical MLE did not converge."
        )

    return float(result.x[0]), float(result.x[1])


# ---------------------------------------------------------------------------
# Power distribution utilities (Kusiak method)
# ---------------------------------------------------------------------------

def power_pdf_from_weibull(
    p: np.ndarray,
    a: float,
    m: float,
    n: float,
    tau: float,
    k_weib: float,
    lam_weib: float,
) -> np.ndarray:
    """
    Analytical PDF of power output derived by applying the
    change-of-variables formula to a Weibull wind speed distribution
    and the 4PL power curve (Kusiak et al., 2009).

    Parameters
    ----------
    p : np.ndarray
        Power values (kW).
    a, m, n, tau : float
        4PL curve parameters.
    k_weib : float
        Weibull shape parameter of the wind speed distribution.
    lam_weib : float
        Weibull scale parameter of the wind speed distribution.

    Returns
    -------
    np.ndarray
        PDF of power evaluated at each value in ``p``.
    """
    x1 = np.abs(
        (k_weib * tau * a * (n - m)) /
        (lam_weib * (a - p) * (n * p - a * m))
    )
    x2 = np.pow(
        (-tau / lam_weib * np.log((a - p) / (n * p - a * m))),
        k_weib - 1,
    )
    x3 = np.exp(
        -np.pow((-tau * np.log((a - p) / (n * p - a * m))) / lam_weib, k_weib)
    )
    res = x1 * x2 * x3
    res[np.isnan(res)] = 1e-9
    return res


def fourpl_mle_powerdist(
    x: np.ndarray,
    y: np.ndarray,
    a0: float = None,
    m0: float = -0.5,
    n0: float = 5.0,
    tau0: float = 2.0,
    maxiter: int = 1000,
) -> np.ndarray:
    """
    Fit the 4PL curve by maximum likelihood using the analytical
    power distribution derived from a Weibull wind speed model
    (Kusiak method).

    Only observations with power > 10 kW are used, since the
    power PDF is undefined near zero.

    Parameters
    ----------
    x : np.ndarray
        Wind speed observations (m/s).
    y : np.ndarray
        Power observations (kW).
    a0 : float, optional
        Initial guess for the asymptotic power. Defaults to ``max(y)``.
    m0, n0, tau0 : float, optional
        Initial guesses for the remaining 4PL parameters.
    maxiter : int
        Maximum number of Powell iterations.

    Returns
    -------
    np.ndarray
        Optimised parameters ``[a, m, n, tau]``.

    Raises
    ------
    ValueError
        If no valid observations remain after filtering.
    RuntimeError
        If the optimisation fails to converge.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = y > 10
    x, y = x[mask], y[mask]

    if x.size == 0:
        raise ValueError("No valid wind speed observations provided.")
    if y.size == 0:
        raise ValueError("No valid power observations provided.")

    k_weib, lam_weib = estimate_weibull_from_x(x)

    if a0 is None:
        a0 = np.max(y)

    theta0 = np.array([a0, m0, n0, tau0], dtype=float)

    def neg_loglikelihood(theta, x, y):
        a, m, n, tau = theta
        f_y = power_pdf_from_weibull(y, a=a, m=m, n=n, tau=tau,
                                     k_weib=k_weib, lam_weib=lam_weib)
        return -np.sum(np.log(f_y))

    result = minimize(
        neg_loglikelihood,
        theta0,
        args=(x, y),
        bounds=[
            (a0 - 100, a0 + 100),
            (-0.002, -0.001),
            (0.01, None),
            (0.05, None),
        ],
        method="Powell",
        options={"maxiter": maxiter},
    )

    if not result.success:
        raise RuntimeError(
            f"Power-distribution MLE failed: {result.message}"
        )

    return result.x
