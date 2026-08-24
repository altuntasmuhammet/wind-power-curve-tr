# Wind Turbine Power Curve Modelling — Türkiye Case Study

Reproducible code and data for the Türkiye case study: comparing parametric power-curve
models against a wind turbine's published manufacturer curve on one year of real SCADA
data.

Four models are evaluated on a held-out test set and scored by normalised RMSE (nRMSE):

| Model | Description |
|---|---|
| 4PL (MLE) | Four-parameter logistic curve, Gaussian maximum likelihood |
| 4PL (Kusiak) | Four-parameter logistic curve, maximum likelihood on the analytical power distribution derived from a Weibull wind speed model (Kusiak et al., 2009) |
| 5PL (MLE) | Five-parameter logistic curve, Gaussian maximum likelihood |
| Manufacturer | The turbine's published theoretical power curve, used as a baseline |

Both logistic models are fitted by differential evolution (a population-based evolutionary
algorithm) with a local Powell polish, since the log-likelihood surface is non-convex and
multi-modal.

## Results

Running `main_tr.py` reproduces the following, evaluated on the 20 % held-out test set and
normalised by the 3600 kW rated power:

| Model | nRMSE | Rank |
|---|---|---|
| **5PL (MLE)** | **2.65 %** | 1 |
| 4PL (MLE) | 4.69 % | 2 |
| Manufacturer | 5.19 % | 3 |
| 4PL (Kusiak) | 10.76 % | 4 |

The five-parameter logistic model fits the observed data more closely than the
manufacturer's own published curve.

Reference copies of every figure and results table are in [`reference_outputs/`](reference_outputs/),
so the expected result can be checked without running anything.

## Requirements

Python 3.10 or newer. The pinned dependency versions in `requirements.txt` are the ones the
published results were produced with (Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3,
pandas 2.3.3, matplotlib 3.10.8, seaborn 0.13.2).

NumPy 2.0 or newer is required — the model modules use `np.pow`, which does not exist in
NumPy 1.x.

## Running

```bash
git clone <repository-url>
cd wind-power-curve-tr

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python main_tr.py
```

The run takes about one minute on a modern laptop and needs no arguments, no network
access, and no configuration — the dataset is included in `data/`. All paths resolve
relative to the source files, so the working directory does not matter.

Output is written to `figures_tr/`:

| File | Content |
|---|---|
| `REAL-T1-TR_fits.png` | All four fitted curves over the test data, labelled with each model's nRMSE |
| `CleanedFigure_TR.png` | Retained measurements against the points removed by the outlier filter |
| `Manufacturer_Power_Curve_TR_3600kW.png` | The manufacturer curve, annotated with cut-in speed, rated power, and cut-off speed |
| `summary_TR.csv` | Mean nRMSE, success rate, and rank per model |
| `long_TR.csv` | One row per (scenario, replication, model) |

Figures are deliberately greyscale — line style and hatching carry the distinction rather
than colour — because they are prepared for black-and-white print.

## Data

`data/turkiye_scada_2018.csv` is the public Kaggle dataset *Wind Turbine SCADA Dataset —
2018 SCADA Data of a Wind Turbine in Turkey*, distributed there as `T1.csv`:

> https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset

It holds 50 530 records at 10-minute intervals covering all of 2018 for a single turbine,
with five columns: `Date/Time`, `LV ActivePower (kW)`, `Wind Speed (m/s)`,
`Theoretical_Power_Curve (KWh)`, and `Wind Direction (°)`. The file is unmodified apart
from being renamed for portability.

`data/turkiye_manufacture_curve.csv` is derived, not independently sourced. This turbine's
manufacturer curve is not published as a separate document — the SCADA export's
`Theoretical_Power_Curve` column already gives the manufacturer power for each record's
wind speed. The lookup table was recovered from that column on a 0.5 m/s grid by
`manufacture_curve_tr.build_curve_from_scada()`, with a zero-power tail appended above
25 m/s to encode the cut-off. Call that function to regenerate it. The resulting
characteristics are a 3.0 m/s cut-in speed, 3600 kW rated power reached at 13.5 m/s, and a
25 m/s cut-off speed.

## Method

1. **Load.** Read the wide-format SCADA export into wind speed and power series indexed by
   timestamp.
2. **Clean.** Remove outliers with a sliding-window filter based on the median absolute
   deviation: within each 0.5 m/s wind-speed bin, drop points whose robust distance
   `|P - median(P)| / MAD(P)` exceeds 2.706, the 99th percentile of the chi-squared
   distribution with one degree of freedom. Negative power readings are clipped to zero
   first. This removes 10 345 of 50 530 records (20.47 %), leaving 40 185.
3. **Split.** Partition 80/20 into training and test sets under a fixed random seed.
4. **Fit.** Estimate each model's parameters on the training set.
5. **Score.** Compute nRMSE on the held-out test set, normalised by the 3600 kW rated
   power so the figure is dimensionless and comparable across turbines.

Every step is seeded, so repeated runs give identical numbers.

## Repository layout

| Path | Role |
|---|---|
| `main_tr.py` | Entry point: site configuration, data loader, figures |
| `main_base.py` | Shared experiment engine — scenarios, split, model comparison, plotting |
| `four_pl.py` | 4PL curve, Gaussian MLE, Kusiak power-distribution MLE, Weibull estimation |
| `five_pl.py` | 5PL curve and its Gaussian MLE |
| `filtering.py` | Median-absolute-deviation outlier filter |
| `manufacture_curve_tr.py` | Manufacturer curve lookup, derivation, and plot |
| `data/` | Input dataset and derived manufacturer curve |
| `reference_outputs/` | Committed copies of the figures and tables from a verified run |

`main_base.py` is the shared engine of a larger multi-site study and is included here
unmodified, so that the code in this repository is demonstrably the code that produced the
reported numbers. Two consequences are worth knowing:

- It carries a reader for a different, long-format SCADA schema used by the other sites.
  That path is unused here — this study supplies its own loader through
  `SiteConfig.real_data_loader`.
- It supports a Monte Carlo simulation mode over synthetic Gompertz/Weibull scenarios,
  reachable by setting `USE_SIMULATION = True` in `main_tr.py`. That mode is **not** part
  of the Türkiye case study and is not needed to reproduce the results above. It is
  site-independent and takes many hours, since it fits every model 1000 times across
  27 scenarios.

## Licence

**No licence is set yet.** Without one, default copyright applies and others may not reuse
this code. Add a licence file before making the repository public or citing it as a
reusable artifact.

The dataset in `data/` is redistributed from Kaggle, where its licence is listed as
*Unknown*. Confirm the terms with the dataset author before publishing this repository or
redistributing the data further; otherwise, remove `data/turkiye_scada_2018.csv` and have
users download it from Kaggle themselves.

## Reference

Kusiak, A., Zheng, H., & Song, Z. (2009). On-line monitoring of power curves.
*Renewable Energy*, 34(6), 1487–1493.
