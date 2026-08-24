"""
manufacture_curve_tr.py
-----------------------
Manufacturer power curve for Wind Farm C (Türkiye, rated power 3600 kW).

The manufacturer curve is not published as a separate document: the
Türkiye SCADA export carries a ``Theoretical_Power_Curve (KWh)`` column
giving the manufacturer power for the wind speed of every 10-minute
record. ``data/turkiye_manufacture_curve.csv`` was derived from that
column by :func:`build_curve_from_scada` on a 0.5 m/s grid, with the
cut-off drop to zero appended at 26 m/s.

Provides a look-up interpolation function and a plotting utility
that annotates cut-in speed, cut-off speed, and rated power.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Paths are resolved relative to this file, not the working directory,
# so the runner works from anywhere.
DATA_DIR = Path(__file__).resolve().parent / 'data'
CURVE_FILE = DATA_DIR / 'turkiye_manufacture_curve.csv'
SCADA_FILE = DATA_DIR / 'turkiye_scada_2018.csv'

POWER_CURVE = pd.read_csv(CURVE_FILE, sep=',')


def build_curve_from_scada(
    scada_csv=SCADA_FILE,
    step: float = 0.5,
    max_speed: float = 25.0,
    out_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Rebuild the manufacturer curve table from the SCADA file's
    ``Theoretical_Power_Curve`` column.

    The theoretical column is a deterministic function of wind speed,
    so the curve is recovered by averaging within each unique wind
    speed and interpolating onto a regular grid. A zero-power tail is
    appended above ``max_speed`` to encode the cut-off.

    Parameters
    ----------
    scada_csv : str
        Path to the Türkiye SCADA CSV.
    step : float
        Grid spacing in m/s.
    max_speed : float
        Highest wind speed retained at rated power (cut-off speed).
    out_csv : str, optional
        If given, the resulting table is written to this path.

    Returns
    -------
    pd.DataFrame
        Table with ``wind_speed`` and ``power`` columns.
    """
    df = pd.read_csv(scada_csv)
    df = df.rename(columns={
        'Wind Speed (m/s)': 'wind_speed',
        'Theoretical_Power_Curve (KWh)': 'power',
    })[['wind_speed', 'power']].dropna()

    lookup = df.groupby('wind_speed')['power'].mean().sort_index()

    grid = np.arange(0.0, max_speed + step, step)
    curve = pd.DataFrame({
        'wind_speed': grid,
        'power': np.round(
            np.interp(grid, lookup.index.to_numpy(), lookup.to_numpy()), 4
        ),
    })

    cutoff_tail = pd.DataFrame({
        'wind_speed': [26.0, 27.0, 28.0, 29.0, 30.0],
        'power': [0.0] * 5,
    })
    curve = pd.concat([curve, cutoff_tail], ignore_index=True)

    if out_csv:
        curve.to_csv(out_csv, index=False, quoting=1)

    return curve


def manufacture_power_curve(v: np.ndarray) -> np.ndarray:
    """
    Interpolate the manufacturer power curve at arbitrary wind speeds.

    Parameters
    ----------
    v : np.ndarray
        Wind speed values (m/s).

    Returns
    -------
    np.ndarray
        Interpolated power values (kW).
    """
    return np.interp(
        v,
        POWER_CURVE['wind_speed'].to_numpy(),
        POWER_CURVE['power'].to_numpy(),
    )


def plot_power_curve_with_characteristics(
    power_positive_threshold: float = 1.0,
    rated_power_tolerance_ratio: float = 0.01,
    save_dir: str = None,
) -> None:
    """
    Plot the manufacturer power curve and annotate its key
    operational characteristics.

    Cut-in speed is defined as the first wind speed at which power
    exceeds ``power_positive_threshold``. The rated region is where
    power reaches within ``rated_power_tolerance_ratio`` of the
    maximum. Cut-off speed is the last wind speed with positive power
    after the rated region begins.

    Parameters
    ----------
    power_positive_threshold : float
        Minimum power (kW) to consider the turbine as producing.
    rated_power_tolerance_ratio : float
        Fraction below the maximum power that still counts as rated.
        For example, 0.01 means values within 1 % of peak power are
        treated as rated.
    save_dir : str, optional
        Directory in which to save the figure. If ``None``, the plot
        is displayed interactively.
    """
    turbine_df = POWER_CURVE.copy()
    turbine_df = turbine_df.sort_values("wind_speed").reset_index(drop=True)
    turbine_df["wind_speed"] = pd.to_numeric(
        turbine_df["wind_speed"], errors="coerce"
    )
    turbine_df["power"] = pd.to_numeric(
        turbine_df["power"], errors="coerce"
    )
    turbine_df = turbine_df.dropna(
        subset=["wind_speed", "power"]
    ).reset_index(drop=True)

    if turbine_df.empty:
        raise ValueError("No valid numeric data found in the power curve.")

    rated_power = turbine_df["power"].max()

    cut_in_candidates = turbine_df[turbine_df["power"] > power_positive_threshold]
    cut_in_speed: Optional[float] = (
        None if cut_in_candidates.empty
        else float(cut_in_candidates.iloc[0]["wind_speed"])
    )

    rated_threshold = rated_power * (1 - rated_power_tolerance_ratio)
    rated_region = turbine_df[turbine_df["power"] >= rated_threshold]

    cut_off_speed: Optional[float] = None
    if not rated_region.empty:
        first_rated_speed = float(rated_region.iloc[0]["wind_speed"])
        after_rated = turbine_df[turbine_df["wind_speed"] > first_rated_speed]
        cut_off_speed = float(
            after_rated[after_rated['power'] > 0].iloc[-1]["wind_speed"]
        )

    plt.figure(figsize=(10, 6))
    plt.plot(
        turbine_df["wind_speed"],
        turbine_df["power"],
        marker="o",
        linewidth=2,
        color="black",
        markersize=4,
        label="Power Curve",
    )
    plt.axhline(
        rated_power, linestyle="--", linewidth=1.5, color="black",
        label=f"Rated Power = {rated_power:.2f} kW",
    )

    if cut_in_speed is not None:
        cut_in_power = float(
            turbine_df.loc[
                turbine_df["wind_speed"] == cut_in_speed, "power"
            ].iloc[0]
        )
        plt.axvline(
            cut_in_speed, linestyle="--", linewidth=1.5, color="0.35",
            label=f"Cut-in = {cut_in_speed:.2f} m/s",
        )
        plt.scatter([cut_in_speed], [cut_in_power], s=80, zorder=5,
                    color="black")

    if cut_off_speed is not None:
        cut_off_power = float(
            turbine_df.loc[
                turbine_df["wind_speed"] == cut_off_speed, "power"
            ].iloc[0]
        )
        plt.axvline(
            cut_off_speed, linestyle="-.", linewidth=1.5, color="0.35",
            label=f"Cut-off = {cut_off_speed:.2f} m/s",
        )
        plt.scatter([cut_off_speed], [cut_off_power], s=80, zorder=5,
                    color="black")

    plt.title("Manufacturer Power Curve — Türkiye Turbine (3600 kW)")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Power (kW)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/Manufacturer_Power_Curve_TR_3600kW.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
