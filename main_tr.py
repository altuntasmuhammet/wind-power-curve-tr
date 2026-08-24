"""
main_tr.py
----------
Experiment runner for Wind Farm C (Türkiye, rated power 3600 kW).

Data source: "Wind Turbine SCADA Dataset — 2018 SCADA Data of a Wind
Turbine in Turkey" (``data/turkiye_scada_2018.csv``, originally ``T1.csv``),
50 530 records at 10-minute intervals covering the whole of 2018 for a
single turbine.

Unlike the FR and JP sites, this export is in **wide** format — one row
per timestamp with the wind speed and power in their own columns —
rather than the long ``variable_name``/``value`` format the other sites
use. The site therefore supplies its own ``real_data_loader`` to
``SiteConfig``; everything downstream (MAD cleaning, train/test split,
model fitting, nRMSE) is the shared code in ``main_base.py``.

The manufacturer curve comes from the dataset's own
``Theoretical_Power_Curve`` column — see ``manufacture_curve_tr.py``.

Supports two modes controlled by ``USE_SIMULATION``:

* **Simulation mode**: synthetic datasets are generated from a
  Gompertz power curve with Weibull wind speeds and Gaussian noise
  across a grid of scenarios. Each scenario is replicated
  ``n_replications`` times (Monte Carlo).
* **Real-data mode**: SCADA data are loaded, cleaned, and evaluated
  once for the turbine.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from filtering import clean
from main_base import (
    SiteConfig,
    build_real_cases,
    build_scenarios,
    plot_one_replication_per_scenario,
    run_model_comparison,
)
from manufacture_curve_tr import (
    SCADA_FILE,
    manufacture_power_curve as _mpc_tr,
    plot_power_curve_with_characteristics,
)


# Output directory is resolved relative to this file, not the working
# directory, so the runner works from anywhere.
FIGURE_DIR = str(Path(__file__).resolve().parent / 'figures_tr')


# ---------------------------------------------------------------------------
# Site-specific data loading (wide-format SCADA export)
# ---------------------------------------------------------------------------

COLUMN_MAP = {
    'Wind Speed (m/s)': 'wind_speed',
    'LV ActivePower (kW)': 'power',
    'Theoretical_Power_Curve (KWh)': 'theoretical_power',
    'Wind Direction (°)': 'wind_direction',
}

DATETIME_FORMAT = '%d %m %Y %H:%M'


def load_tr_scada(scada_csv: str, turbine_id=None) -> pd.DataFrame:
    """
    Load the wide-format Türkiye SCADA export.

    The file holds a single turbine, so ``turbine_id`` is accepted for
    signature compatibility with ``SiteConfig.real_data_loader`` and
    otherwise ignored.

    Parameters
    ----------
    scada_csv : str
        Path to the Türkiye SCADA CSV.
    turbine_id : int, optional
        Unused; present for interface compatibility.

    Returns
    -------
    pd.DataFrame
        Frame indexed by timestamp with ``wind_speed`` (m/s) and
        ``power`` (kW) columns, ready for the shared MAD filter.
    """
    df = pd.read_csv(scada_csv, encoding='utf-8-sig')
    df = df.rename(columns=COLUMN_MAP)

    df['time'] = pd.to_datetime(df['Date/Time'], format=DATETIME_FORMAT)
    df = df.sort_values('time').drop_duplicates(subset='time', keep='last')
    df = df.set_index('time')

    return df[['wind_speed', 'power']]


# ---------------------------------------------------------------------------
# Site configuration
# ---------------------------------------------------------------------------

def _manufacture_power_curve_tr(v, turbine_id=None):
    """Wrapper so the TR curve matches the (v, turbine_id) signature."""
    return _mpc_tr(v)


CFG = SiteConfig(
    site_label='TR',
    turbine_ids=[1],                      # single turbine in this dataset
    scada_csv=str(SCADA_FILE),
    p_rated_real=3600,                    # kW — from the theoretical curve
    p_rated_simulation=3200,              # kW — simulated turbine
    manufacture_power_curve=_manufacture_power_curve_tr,
    save_dir=FIGURE_DIR,
    dropna_before_clean=True,
    real_data_loader=load_tr_scada,
)

USE_SIMULATION = False
SIGMA_EPS = 100.0   # kW — Gaussian noise std for simulation mode


# ---------------------------------------------------------------------------
# Figure: cleaned data and outliers
# ---------------------------------------------------------------------------

def plot_cleaned_data_and_outliers(save_dir: str = None) -> None:
    """
    Scatter the cleaned SCADA data against the points the MAD filter
    removed.

    Parameters
    ----------
    save_dir : str, optional
        Directory in which to save the figure. If ``None``, the plot is
        displayed interactively.
    """
    df = load_tr_scada(CFG.scada_csv)
    if CFG.dropna_before_clean:
        df = df.dropna()

    df_clean = clean(df.copy())
    df_outliers = df.loc[df.index.difference(df_clean.index)]

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
    plt.title("Cleaned Data and Outliers — Türkiye Turbine")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/CleanedFigure_TR.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    print(
        f"Raw records: {len(df)} | cleaned: {len(df_clean)} | "
        f"removed: {len(df_outliers)} "
        f"({100.0 * len(df_outliers) / len(df):.2f}%)"
    )


# ---------------------------------------------------------------------------
# Figure: simulation box plots
# ---------------------------------------------------------------------------

def boxplot_nrmse(df: pd.DataFrame, save_dir: str = None) -> None:
    """
    Draw a faceted box plot of nRMSE values by model, wind regime,
    and Gompertz shape (simulation mode only).

    Parameters
    ----------
    df : pd.DataFrame
        Long-format results DataFrame with columns ``Model``,
        ``nRMSE``, ``wind_regime``, and ``gompertz_shape``.
    save_dir : str, optional
        Directory in which to save the figure. If ``None``, the plot is
        displayed interactively.
    """
    sns.set_theme(style="ticks")
    df = df[~df['nRMSE'].isna()]

    g = sns.catplot(
        data=df,
        x="Model",
        y="nRMSE",
        kind="box",
        col="wind_regime",
        row="gompertz_shape",
        height=3.5,
        aspect=1.1,
        color="white",
        linewidth=1.2,
        fliersize=2,
        showfliers=False,
    )

    hatches = ["", "//", "xx", "\\\\", "++"]
    for ax in g.axes.flat:
        for i, artist in enumerate(ax.artists):
            artist.set_edgecolor("black")
            artist.set_facecolor("white")
            artist.set_hatch(hatches[i % len(hatches)])
        for line in ax.lines:
            line.set_color("black")
            line.set_linewidth(1)

    g.set_axis_labels("Model", "nRMSE")
    g.set_titles(
        col_template="Wind Regime: {col_name}",
        row_template="Gompertz Shape = {row_name}",
    )
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        g.savefig(f"{save_dir}/Boxplot_TR.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if USE_SIMULATION:
        df_scenarios = build_scenarios(
            P_rated=CFG.p_rated_simulation, sigma_eps=SIGMA_EPS
        )
    else:
        df_scenarios = build_real_cases(CFG)
        plot_cleaned_data_and_outliers(save_dir=CFG.save_dir)
        plot_power_curve_with_characteristics(save_dir=CFG.save_dir)

    plot_one_replication_per_scenario(df_scenarios, CFG)

    df_summary, df_long = run_model_comparison(
        df_scenarios,
        CFG,
        n_replications=1000 if USE_SIMULATION else 1,
        seed=123,
    )

    if USE_SIMULATION:
        df_long = df_long.merge(
            df_scenarios[["Scenario", "wind_regime", "gompertz_shape"]],
            on="Scenario", how="left",
        )
        boxplot_nrmse(df_long, save_dir=CFG.save_dir)

    os.makedirs(CFG.save_dir, exist_ok=True)
    df_summary.to_csv(f"{CFG.save_dir}/summary_TR.csv", sep=';', index=False)
    df_long.to_csv(f"{CFG.save_dir}/long_TR.csv", index=False)

    print("Summary results:")
    print(df_summary.to_csv(sep=';', index=False))
