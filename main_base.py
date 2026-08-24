"""
main_base.py
------------
Shared experiment logic for wind turbine power curve model comparison.

Both site-specific runners (``main_fr.py`` and ``main_jp.py``) import
from this module. Site-specific behaviour is injected via a
:class:`SiteConfig` dataclass and, where needed, through optional
arguments (e.g. ``turbine_id`` for per-turbine manufacturer curves).

Do not run this file directly — use ``main_fr.py`` or ``main_jp.py``.
"""

import itertools
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from filtering import clean
from four_pl import four_pl, fourpl_mle, fourpl_mle_powerdist
from five_pl import five_pl, fivepl_mle


# ---------------------------------------------------------------------------
# Site configuration
# ---------------------------------------------------------------------------

@dataclass
class SiteConfig:
    """
    All site-specific parameters for one wind farm experiment.

    Parameters
    ----------
    site_label : str
        Short label used in scenario names (e.g. ``"FR"`` or ``"JP"``).
    turbine_ids : list of int
        IDs of turbines to evaluate in real-data mode.
    scada_csv : str
        Path to the SCADA data CSV file.
    p_rated_real : float
        Rated power of the real turbines (kW), used to normalise nRMSE.
    p_rated_simulation : float
        Rated power used as the Gompertz asymptote in simulation mode (kW).
    manufacture_power_curve : Callable
        Function with signature ``(v, turbine_id=None) -> np.ndarray``
        that returns interpolated manufacturer power (kW).
    save_dir : str
        Directory for saving figures.
    dropna_before_clean : bool
        If ``True``, drop NaN rows before passing data to the MAD
        filter. Required for the Abukuma (JP) dataset.
    real_data_loader : Callable, optional
        Function with signature ``(scada_csv, turbine_id) -> pd.DataFrame``
        returning a frame with ``wind_speed`` and ``power`` columns.
        Supply this for sites whose SCADA export is not in the default
        long format (``turbine_id``/``variable_name``/``value``), e.g.
        the wide-format Türkiye dataset. If ``None``, the long-format
        reader is used.
    """
    site_label: str
    turbine_ids: list
    scada_csv: str
    p_rated_real: float
    p_rated_simulation: float
    manufacture_power_curve: Callable
    save_dir: str = './figures'
    dropna_before_clean: bool = True
    real_data_loader: Optional[Callable] = None


# ---------------------------------------------------------------------------
# Module-level cache (populated at runtime by get_real_dataset)
# ---------------------------------------------------------------------------

_CACHED_REAL_DATA: dict = {}

logging.basicConfig(
    format='%(asctime)-15s\t%(levelname)s:%(name)s: %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def nrmse(y_true: np.ndarray, y_pred: np.ndarray, rated: float) -> float:
    """
    Normalised Root Mean Squared Error (nRMSE).

    RMSE is divided by ``rated`` power so that results are
    dimensionless and comparable across turbine types.

    Parameters
    ----------
    y_true : np.ndarray
        Observed power values (kW).
    y_pred : np.ndarray
        Predicted power values (kW).
    rated : float
        Rated (nameplate) power of the turbine (kW).

    Returns
    -------
    float
        nRMSE value.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) / rated


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------

def weibull_sample(
    n: int,
    k: float,
    c: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw ``n`` samples from a Weibull(k, c) distribution via the
    inverse-transform method.

    Parameters
    ----------
    n : int
        Number of samples.
    k : float
        Shape parameter (k > 0).
    c : float
        Scale parameter (c > 0).
    rng : np.random.Generator
        NumPy random generator instance.

    Returns
    -------
    np.ndarray
        Array of ``n`` Weibull-distributed wind speed samples (m/s).
    """
    u = rng.uniform(0.0, 1.0, size=n)
    return c * (-np.log(1.0 - u)) ** (1.0 / k)


def gompertz_power(v: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    """
    Gompertz mean power curve used to generate synthetic data.

        P(v) = A * exp(-B * exp(-C * v))

    Parameters
    ----------
    v : np.ndarray
        Wind speed values (m/s).
    A : float
        Asymptotic (rated) power (kW).
    B : float
        Displacement parameter.
    C : float
        Growth-rate parameter.

    Returns
    -------
    np.ndarray
        Mean power values (kW).
    """
    return A * np.exp(-B * np.exp(-C * v))


def build_scenarios(P_rated: float, sigma_eps: float) -> pd.DataFrame:
    """
    Build the full simulation scenario grid.

    Scenarios are the Cartesian product of:

    * Sample sizes: 2000, 4000, 8000
    * Weibull wind regimes: low (~4 m/s mean), normal (~7 m/s),
      high (~10 m/s)
    * Gompertz power-curve shapes: early, balanced, late inflection

    Parameters
    ----------
    P_rated : float
        Rated power used as the Gompertz asymptote (kW).
    sigma_eps : float
        Standard deviation of Gaussian observation noise (kW).

    Returns
    -------
    pd.DataFrame
        One row per scenario with columns for all parameters.
    """
    n_samples_list = [2000, 4000, 8000]

    weibull_regimes = [
        {"name": "low",    "k": 2.0, "c": 4.64},
        {"name": "normal", "k": 2.0, "c": 7.90},
        {"name": "high",   "k": 2.0, "c": 11.16},
    ]

    gompertz_shapes = [
        {"name": "early",    "B": 6.0, "C": 0.50},
        {"name": "balanced", "B": 6.0, "C": 0.40},
        {"name": "late",     "B": 6.0, "C": 0.30},
    ]

    rows = []
    scenario_id = 1
    for n, w, g in itertools.product(n_samples_list, weibull_regimes, gompertz_shapes):
        rows.append({
            "Scenario":       f"S{scenario_id}",
            "n_samples":      n,
            "wind_regime":    w["name"],
            "k":              w["k"],
            "c":              w["c"],
            "gompertz_shape": g["name"],
            "A":              P_rated,
            "B":              g["B"],
            "C":              g["C"],
            "sigma":          sigma_eps,
            "turbine_id":     float('nan'),
        })
        scenario_id += 1

    return pd.DataFrame(rows)


def build_real_cases(cfg: SiteConfig) -> pd.DataFrame:
    """
    Build the scenario table for real-data evaluation.

    Returns one row per turbine in ``cfg.turbine_ids``, with
    NaN-filled simulation parameters.

    Parameters
    ----------
    cfg : SiteConfig
        Site configuration supplying turbine IDs and site label.

    Returns
    -------
    pd.DataFrame
        One row per turbine with a scenario label
        ``REAL-Tx-<site_label>``.
    """
    rows = []
    for td_id in cfg.turbine_ids:
        rows.append({
            "Scenario":       f"REAL-T{int(td_id) - min(cfg.turbine_ids) + 1}-{cfg.site_label}",
            "n_samples":      float('nan'),
            "wind_regime":    '',
            "k":              float('nan'),
            "c":              float('nan'),
            "gompertz_shape": '',
            "A":              float('nan'),
            "B":              float('nan'),
            "C":              float('nan'),
            "sigma":          float('nan'),
            "turbine_id":     td_id,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dataset generation / loading
# ---------------------------------------------------------------------------

def simulate_dataset_for_scenario(
    scenario_row: pd.Series,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate one synthetic (wind speed, power) dataset from a
    scenario row.

    Parameters
    ----------
    scenario_row : pd.Series
        Must contain: ``n_samples``, ``k``, ``c``, ``A``, ``B``,
        ``C``, ``sigma``.
    rng : np.random.Generator
        NumPy random generator instance.

    Returns
    -------
    v : np.ndarray
        Simulated wind speed samples (m/s).
    p : np.ndarray
        Simulated power samples (kW).
    """
    n = int(scenario_row["n_samples"])
    k = float(scenario_row["k"])
    c = float(scenario_row["c"])
    A = float(scenario_row["A"])
    B = float(scenario_row["B"])
    C = float(scenario_row["C"])
    sigma = float(scenario_row["sigma"])

    v = weibull_sample(n, k, c, rng)
    p = gompertz_power(v, A, B, C) + rng.normal(loc=0.0, scale=sigma, size=n)
    return v, p


def get_real_dataset(
    cfg: SiteConfig,
    turbine_id: Optional[int] = None,
    draw_plot: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load, clean, and optionally visualise SCADA data for one turbine.

    Results are cached to avoid repeated file I/O across replications.

    Parameters
    ----------
    cfg : SiteConfig
        Site configuration supplying the CSV path and cleaning options.
    turbine_id : int, optional
        Turbine identifier. If ``None``, all turbines are included.
    draw_plot : bool
        If ``True``, display a scatter plot distinguishing inliers
        from outliers removed by the cleaning step.

    Returns
    -------
    wind_speed : np.ndarray
        Cleaned wind speed array (m/s).
    power : np.ndarray
        Cleaned power array (kW).
    """
    cache_key = (cfg.site_label, turbine_id)
    if cache_key in _CACHED_REAL_DATA:
        return _CACHED_REAL_DATA[cache_key]

    if cfg.real_data_loader is not None:
        df = cfg.real_data_loader(cfg.scada_csv, turbine_id)
    else:
        df = pd.read_csv(cfg.scada_csv)
        if turbine_id:
            df = df[df['turbine_id'] == turbine_id]

        df = df.sort_values('update_time', ascending=False)
        df = df.drop_duplicates(
            subset=['turbine_id', 'variable_id', 'time'], keep='first'
        )

        wind_speed_series = (
            df[df['variable_name'] == 'wind_speed_actual'][['turbine_id', 'time', 'value']]
            .groupby('time')['value'].mean()
        )
        power_series = (
            df[df['variable_name'] == 'power_kw_actual'][['turbine_id', 'time', 'value']]
            .groupby('time')['value'].sum()
        )

        ts_index = np.union1d(wind_speed_series.index, power_series.index)
        df = pd.DataFrame(index=ts_index)
        df.loc[wind_speed_series.index, 'wind_speed'] = wind_speed_series
        df.loc[power_series.index, 'power'] = power_series

    if cfg.dropna_before_clean:
        df = df.dropna()

    df_clean = clean(df.copy())

    if draw_plot:
        outlier_idx = df.index.difference(df_clean.index)
        df_outliers = df.loc[outlier_idx]

        plt.figure(figsize=(8, 5))
        plt.scatter(
            df_clean["wind_speed"], df_clean["power"],
            facecolors='none', edgecolors='gray',
            s=25, linewidths=0.7, label='Normal',
        )
        plt.scatter(
            df_outliers["wind_speed"], df_outliers["power"],
            color='black', s=35, marker='x',
            linewidths=1.2, label='Outlier',
        )
        plt.xlabel("Wind Speed (m/s)")
        plt.ylabel("Power (kW)")
        title = "Cleaned Data and Outliers"
        if turbine_id:
            title += f' — Turbine {turbine_id - min(cfg.turbine_ids) + 1}'
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    _CACHED_REAL_DATA[cache_key] = (
        df_clean['wind_speed'].to_numpy(),
        df_clean['power'].to_numpy(),
    )
    return _CACHED_REAL_DATA[cache_key]


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def compare_models_on_dataset(
    v: np.ndarray,
    p: np.ndarray,
    cfg: SiteConfig,
    is_real: bool = False,
    turbine_id: Optional[int] = None,
) -> dict:
    """
    Fit all models on an 80/20 train/test split and return the nRMSE
    of each model on the held-out test set.

    Parameters
    ----------
    v : np.ndarray
        Wind speed array (m/s).
    p : np.ndarray
        Power array (kW).
    cfg : SiteConfig
        Site configuration for rated power and manufacturer curve.
    is_real : bool
        If ``True``, also evaluate the manufacturer curve baseline
        and use ``cfg.p_rated_real`` for normalisation; otherwise
        uses ``cfg.p_rated_simulation``.
    turbine_id : int, optional
        Turbine identifier, passed to the manufacturer curve function
        when ``is_real=True``.

    Returns
    -------
    dict
        Keys are model names; values are nRMSE scores (dimensionless).
        Failed fits are recorded as ``np.inf``.
    """
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(v))
    split_idx = int(0.8 * len(v))

    v_train = v[idx[:split_idx]][np.argsort(v[idx[:split_idx]])]
    p_train = p[idx[:split_idx]][np.argsort(v[idx[:split_idx]])]
    v_test = v[idx[split_idx:]][np.argsort(v[idx[split_idx:]])]
    p_test = p[idx[split_idx:]][np.argsort(v[idx[split_idx:]])]

    rated = cfg.p_rated_real if is_real else cfg.p_rated_simulation

    try:
        params_4pl_mle = fourpl_mle(v_train, p_train)
        nrmse_4pl_mle = nrmse(p_test, four_pl(v_test, *params_4pl_mle[:4]), rated)
    except Exception as e:
        logger.exception(e)
        nrmse_4pl_mle = np.inf

    try:
        params_4pl_kusiak = fourpl_mle_powerdist(v_train, p_train)
        nrmse_4pl_kusiak = nrmse(
            p_test, four_pl(v_test, *params_4pl_kusiak[:4]), rated
        )
    except Exception as e:
        logger.exception(e)
        nrmse_4pl_kusiak = np.inf

    try:
        params_5pl_mle = fivepl_mle(v_train, p_train)
        nrmse_5pl_mle = nrmse(p_test, five_pl(v_test, *params_5pl_mle[:5]), rated)
    except Exception as e:
        logger.exception(e)
        nrmse_5pl_mle = np.inf

    results = {
        "4PL (Kusiak)": nrmse_4pl_kusiak,
        "4PL (MLE)":    nrmse_4pl_mle,
        "5PL (MLE)":    nrmse_5pl_mle,
    }

    if is_real:
        try:
            nrmse_manufacture = nrmse(
                p_test,
                cfg.manufacture_power_curve(v_test, turbine_id),
                rated,
            )
        except Exception as e:
            logger.exception(e)
            nrmse_manufacture = np.inf
        results["MANUFACTURER"] = nrmse_manufacture

    return results


# ---------------------------------------------------------------------------
# Parallel Monte Carlo runner
# ---------------------------------------------------------------------------

def _run_one_replication(args: tuple) -> dict:
    """
    Worker function for a single Monte Carlo replication.

    Must be a top-level function so it can be pickled by
    ``ProcessPoolExecutor``.

    Parameters
    ----------
    args : tuple
        ``(scenario_row_dict, replication_idx, seed, cfg)``

    Returns
    -------
    dict
        ``{"Replication": int, "nRMSEs": dict}``
    """
    scenario_row_dict, replication_idx, seed, cfg = args
    rng = np.random.default_rng(seed)
    row = pd.Series(scenario_row_dict)
    is_real = not scenario_row_dict['Scenario'].startswith('S')

    if is_real:
        v, p = get_real_dataset(cfg, row.get('turbine_id'))
    else:
        v, p = simulate_dataset_for_scenario(row, rng)

    nrmses = compare_models_on_dataset(
        v, p, cfg, is_real, row.get('turbine_id')
    )
    return {"Replication": replication_idx, "nRMSEs": nrmses}


def _run_one_scenario_parallel(
    row_dict: dict,
    cfg: SiteConfig,
    n_replications: int,
    base_seed: int,
) -> tuple[dict, list]:
    """
    Run all replications for one scenario in parallel and aggregate
    the results.

    Parameters
    ----------
    row_dict : dict
        Scenario parameters (one row from the scenarios DataFrame).
    cfg : SiteConfig
        Site configuration.
    n_replications : int
        Number of Monte Carlo replications.
    base_seed : int
        Base random seed; each replication uses ``base_seed + r``.

    Returns
    -------
    summary_row : dict
        Scenario-level summary (mean nRMSE, success rates, ranks).
    long_rows : list of dict
        Per-replication records for the long-format results DataFrame.
    """
    scenario_name = row_dict["Scenario"]
    wins: dict = defaultdict(int)
    nrmses_all: dict = defaultdict(list)
    long_rows: list = []

    max_workers = os.cpu_count() or 1
    tasks = [(row_dict, r, base_seed + r, cfg) for r in range(n_replications)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_one_replication, task) for task in tasks]

        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            r = result["Replication"]
            nrmse_scores = result["nRMSEs"]

            for model_name, score in nrmse_scores.items():
                nrmses_all[model_name].append(score)
                long_rows.append({
                    "Scenario":    scenario_name,
                    "Replication": r,
                    "Model":       model_name,
                    "nRMSE":       score,
                })

            wins[min(nrmse_scores, key=nrmse_scores.get)] += 1

            if i % 50 == 0:
                print(f"ITER: {i}/{n_replications} | Scenario: {scenario_name}")

    mean_nrmses = {k: float(np.nanmean(v)) for k, v in nrmses_all.items()}
    success_rates = {k: 100.0 * wins[k] / n_replications for k in mean_nrmses}

    mean_ranks = (
        pd.Series(mean_nrmses)
        .rank(method="min", ascending=True)
        .astype(int)
        .to_dict()
    )
    success_ranks = (
        pd.Series(success_rates)
        .rank(method="min", ascending=False)
        .astype(int)
        .to_dict()
    )

    summary_row = {
        "Scenario":       scenario_name,
        "n_samples":      row_dict["n_samples"],
        "wind_regime":    row_dict["wind_regime"],
        "gompertz_shape": row_dict["gompertz_shape"],
        **{f"Mean_{k}":          f"{v * 100:.2f}%" for k, v in mean_nrmses.items()},
        **{f"SuccessRate_{k}_%":  v                for k, v in success_rates.items()},
        **{f"RankMean_{k}":       v                for k, v in mean_ranks.items()},
        **{f"RankSuccess_{k}":    v                for k, v in success_ranks.items()},
    }

    return summary_row, long_rows


def run_model_comparison(
    df_scenarios: pd.DataFrame,
    cfg: SiteConfig,
    n_replications: int = 100,
    seed: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full model comparison experiment over all scenarios.

    Parameters
    ----------
    df_scenarios : pd.DataFrame
        Scenario table from :func:`build_scenarios` or
        :func:`build_real_cases`.
    cfg : SiteConfig
        Site configuration.
    n_replications : int
        Number of Monte Carlo replications per scenario.
    seed : int
        Base random seed.

    Returns
    -------
    df_summary : pd.DataFrame
        One row per scenario: mean nRMSE, success rates, and ranks.
    df_long : pd.DataFrame
        One row per (scenario, replication, model): individual nRMSE
        values suitable for box plots.
    """
    summary_rows = []
    long_rows = []

    for scenario_idx, (_, row) in enumerate(df_scenarios.iterrows()):
        row_dict = row.to_dict()
        summary_row, scenario_long_rows = _run_one_scenario_parallel(
            row_dict=row_dict,
            cfg=cfg,
            n_replications=n_replications,
            base_seed=seed + scenario_idx * 100_000,
        )
        summary_rows.append(summary_row)
        long_rows.extend(scenario_long_rows)
        print(f"Scenario {row_dict['Scenario']} completed.")

    return pd.DataFrame(summary_rows), pd.DataFrame(long_rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_one_replication_per_scenario(
    df_scenarios: pd.DataFrame,
    cfg: SiteConfig,
    seed: int = 123,
) -> None:
    """
    For each scenario, fit all models on one dataset (simulated or
    real) and produce a publication-ready scatter + curve plot.

    Parameters
    ----------
    df_scenarios : pd.DataFrame
        Scenario table from :func:`build_scenarios` or
        :func:`build_real_cases`.
    cfg : SiteConfig
        Site configuration.
    seed : int
        Random seed for reproducible simulation.
    """
    rng = np.random.default_rng(seed)

    for _, row in df_scenarios.iterrows():
        scenario_name = row["Scenario"]
        is_real = not scenario_name.startswith('S')

        if is_real:
            v, p = get_real_dataset(cfg, row.get('turbine_id'))
            rated = cfg.p_rated_real
        else:
            v, p = simulate_dataset_for_scenario(row, rng)
            rated = cfg.p_rated_simulation

        rng_split = np.random.default_rng(42)
        idx = rng_split.permutation(len(v))
        split_idx = int(0.8 * len(v))

        v_train = v[idx[:split_idx]][np.argsort(v[idx[:split_idx]])]
        p_train = p[idx[:split_idx]][np.argsort(v[idx[:split_idx]])]
        v_test = v[idx[split_idx:]][np.argsort(v[idx[split_idx:]])]
        p_test = p[idx[split_idx:]][np.argsort(v[idx[split_idx:]])]

        p_hat_4pl, nrmse_4pl = None, 0.0
        try:
            params_4pl = fourpl_mle(v_train, p_train)
            print(f"scenario_name: {scenario_name} params_4pl: {params_4pl}")
            p_hat_4pl = four_pl(v_test, *params_4pl[:4])
            nrmse_4pl = nrmse(p_test, p_hat_4pl, rated)
        except Exception as e:
            logger.exception(f"{scenario_name} 4PL fit failed: {e}")

        p_hat_4pl_kusiak, nrmse_4pl_kusiak = None, 0.0
        try:
            params_4pl_kusiak = fourpl_mle_powerdist(v_train, p_train)
            print(f"scenario_name: {scenario_name} params_4pl_kusiak: {params_4pl_kusiak}")
            p_hat_4pl_kusiak = four_pl(v_test, *params_4pl_kusiak[:4])
            nrmse_4pl_kusiak = nrmse(p_test, p_hat_4pl_kusiak, rated)
        except Exception as e:
            logger.exception(f"{scenario_name} 4PL Kusiak fit failed: {e}")

        p_hat_5pl, nrmse_5pl = None, 0.0
        try:
            params_5pl = fivepl_mle(v_train, p_train)
            print(f"scenario_name: {scenario_name} params_5pl: {params_5pl}")
            p_hat_5pl = five_pl(v_test, *params_5pl[:5])
            nrmse_5pl = nrmse(p_test, p_hat_5pl, rated)
        except Exception as e:
            logger.exception(f"{scenario_name} 5PL fit failed: {e}")

        p_hat_manufacture, nrmse_manufacture = None, 0.0
        if is_real:
            try:
                p_hat_manufacture = cfg.manufacture_power_curve(
                    v_test, row.get('turbine_id')
                )
                nrmse_manufacture = nrmse(p_test, p_hat_manufacture, rated)
            except Exception as e:
                logger.exception(f"{scenario_name} manufacturer curve failed: {e}")

        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.scatter(v_test, p_test, s=8, color='0.5', alpha=0.8,
                   marker='o', linewidths=0, label="Data")

        if p_hat_4pl_kusiak is not None:
            ax.plot(v_test, p_hat_4pl_kusiak, color='black', linestyle='--',
                    linewidth=2,
                    label=f"4PL (Kusiak), nRMSE: {nrmse_4pl_kusiak * 100:.2f}%")

        if p_hat_4pl is not None:
            ax.plot(v_test, p_hat_4pl, color='0.35', linestyle='-.',
                    linewidth=2,
                    label=f"4PL (MLE), nRMSE: {nrmse_4pl * 100:.2f}%")

        if p_hat_5pl is not None:
            ax.plot(v_test, p_hat_5pl, color='black', linestyle='-',
                    linewidth=3,
                    label=f"5PL (MLE), nRMSE: {nrmse_5pl * 100:.2f}%")

        if p_hat_manufacture is not None:
            ax.plot(v_test, p_hat_manufacture, color='0.15', linestyle=':',
                    linewidth=2.5,
                    label=f"Manufacturer, nRMSE: {nrmse_manufacture * 100:.2f}%")

        if not is_real:
            ax.set_title(
                f"{scenario_name} | n={int(row['n_samples'])}, "
                f"wind={row['wind_regime']}, "
                f"gompertz={row['gompertz_shape']}"
            )
        else:
            ax.set_title(scenario_name)

        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Power (kW)")
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        ax.legend(loc='lower right', frameon=True,
                  facecolor='white', edgecolor='black')
        plt.tight_layout()

        if cfg.save_dir:
            os.makedirs(cfg.save_dir, exist_ok=True)
            plt.savefig(f"{cfg.save_dir}/{scenario_name}_fits.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
