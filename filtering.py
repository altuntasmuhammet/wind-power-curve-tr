"""
filtering.py
------------
Robust outlier removal for wind turbine SCADA data using a
sliding-window MAD (Median Absolute Deviation) filter.
"""

import numpy as np
from numpy import absolute, median


# Half-width of each wind-speed bin (m/s)
TOL = 0.5


def mad(data: np.ndarray, axis=None) -> np.ndarray:
    """
    Compute the Median Absolute Deviation (MAD).

    Parameters
    ----------
    data : np.ndarray
        Input array.
    axis : int or None
        Axis along which the MAD is computed. None computes over
        the flattened array.

    Returns
    -------
    np.ndarray
        MAD of ``data``.
    """
    return median(absolute(data - median(data, axis)), axis)


def clean(df):
    """
    Remove outliers from a wind-power DataFrame using a sliding
    MAD filter applied bin-by-bin across wind speed.

    For each bin centred at every 0.5 m/s step the function
    computes the robust distance

        d_i = |P_i - median(P)| / MAD(P)

    and drops observations with d_i > 2.706 (the 99th percentile
    of the chi-squared distribution with one degree of freedom).
    Negative power values are clipped to zero before filtering.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``wind_speed`` (m/s) and
        ``power`` (kW).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with outlier rows removed.
    """
    df.loc[df['power'] < 0, 'power'] = 0

    for ws in np.arange(0, np.ceil(df['wind_speed'].max()), TOL):
        partial_df = df[
            (df['wind_speed'] >= (ws - TOL)) &
            (df['wind_speed'] <= (ws + TOL))
        ]
        power_arr = partial_df['power'].values

        mean_val = median(power_arr)
        scale_val = mad(power_arr)

        robust_distance = absolute(power_arr - mean_val) / scale_val
        outlier_df = partial_df[robust_distance > 2.706]
        df = df.drop(outlier_df.index)

    return df
