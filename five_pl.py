"""
five_pl.py
----------
Five-parameter logistic (5PL) power curve model: 5PL MLE
"""

import numpy as np
from scipy.optimize import differential_evolution


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def five_pl(
    v: np.ndarray,
    A: float,
    B: float,
    C: float,
    D: float,
    G: float,
) -> np.ndarray:
    """
    Five-parameter logistic (5PL) power curve.

        P(v) = D + (A - D) / (1 + (v / C)^B)^G

    Parameters
    ----------
    v : np.ndarray
        Wind speed values (m/s). Non-positive values are replaced
        by a small positive constant.
    A : float
        Upper asymptote (kW).
    B : float
        Slope parameter.
    C : float
        Inflection-point scale parameter (m/s).
    D : float
        Lower asymptote (kW).
    G : float
        Asymmetry parameter.

    Returns
    -------
    np.ndarray
        Predicted power values (kW).
    """
    v = np.asarray(v, dtype=float).copy()
    v[v <= 0] = 1e-9
    return D + (A - D) / ((1 + (v / C) ** B) ** G)


# ---------------------------------------------------------------------------
# Likelihood helper
# ---------------------------------------------------------------------------

def pdf_P(
    x: np.ndarray,
    p: np.ndarray,
    A: float,
    B: float,
    C: float,
    D: float,
    G: float,
    sigma: float,
) -> np.ndarray:
    """
    Gaussian likelihood of observed power ``p`` given wind speed ``x``
    and 5PL parameters.

    Parameters
    ----------
    x : np.ndarray
        Wind speed values (m/s).
    p : np.ndarray
        Observed power values (kW).
    A, B, C, D, G : float
        5PL curve parameters.
    sigma : float
        Standard deviation of the Gaussian noise.

    Returns
    -------
    np.ndarray
        Likelihood values for each observation.
    """
    numer = np.sqrt(2 * np.pi * sigma ** 2)
    exponent = -1 / (2 * sigma ** 2) * np.pow((p - five_pl(x, A, B, C, D, G)), 2)
    return (1 / numer) * np.exp(exponent)


# ---------------------------------------------------------------------------
# Fitting — Evolutionary Algorithm (Differential Evolution) + local polish
# ---------------------------------------------------------------------------

def fivepl_mle(
    x: np.ndarray,
    y: np.ndarray,
    A0: float = None,
    C0: float = None,
    D0: float = None,
    sigma0: float = None,
    maxiter: int = 500,
    popsize: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """
    Fit the 5PL curve by maximum likelihood using a differential
    evolution (DE) evolutionary algorithm with local Powell polishing.

    The parameter space is explored globally by a population-based
    evolutionary algorithm (DE/best/1/bin strategy), which applies
    selection, crossover, and mutation operators without requiring
    gradient information. This makes it well-suited to the
    multi-modal, non-convex log-likelihood surface of the 5PL model,
    where the additional asymmetry parameter G further complicates
    the optimization landscape compared to the 4PL model.
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
    A0 : float, optional
        Upper bound hint for the upper asymptote. Defaults to
        ``max(y)``.
    C0 : float, optional
        Hint for the inflection scale. Defaults to ``median(x)``.
    D0 : float, optional
        Lower bound hint for the lower asymptote. Defaults to
        ``min(y)``.
    sigma0 : float, optional
        Hint for noise standard deviation. Defaults to
        ``std(y - median(y))``.
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
        Optimised parameters ``[A, B, C, D, G, sigma]``.

    Raises
    ------
    RuntimeError
        If the optimisation fails to converge.
    """
    x = np.asarray(x, dtype=float)
    x[np.where(x <= 0)] = 1e-9
    y = np.asarray(y, dtype=float)
    y[np.where(y <= 0)] = 1e-9

    if A0 is None:
        A0 = float(np.max(y))
    if D0 is None:
        D0 = float(np.min(y))
    if C0 is None:
        C0 = float(np.median(x))

    bounds = [
        (A0 - 10, A0 + 10),                 # A
        (-100, -0.01),             # B
        (0.01, 100),            # C
        (0, D0+10),           # D
        (0.01, 100),             # G
        (0.01, 1000),            # sigma
    ]

    def neg_loglikelihood(theta):
        A, B, C, D, G, sigma = theta
        return -np.sum(np.log(pdf_P(x, y, A, B, C, D, G, sigma)))

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
