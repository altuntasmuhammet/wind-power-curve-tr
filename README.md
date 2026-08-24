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

Committed copies of every figure and results table are in [`results/`](results/), so the
expected output can be inspected without running anything. Each figure is provided as PNG,
JPG, and TIFF at 300 DPI; `results/figure_index.csv` lists the figures with their captions.

### Figure 1 — Cleaned data and outliers

Measurements retained by the filter against the points it removed. Curtailment and
downtime show up as the band of readings at zero power across all wind speeds.

![Figure 1](results/Figure%201.png)

### Figure 2 — Manufacturer power curve

The turbine's published curve, annotated with its operating characteristics: 3.0 m/s
cut-in, 3600 kW rated power from 13.5 m/s, and 25 m/s cut-off.

![Figure 2](results/Figure%202.png)

### Figure 3 — Fitted models

All four models over the held-out test data, each labelled with its nRMSE. The 5PL curve
tracks the measurements through the knee of the curve, where the 4PL variants and the
manufacturer curve deviate most.

![Figure 3](results/Figure%203.png)

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

Output is written to `figures_tr/`, which mirrors the committed `results/` directory:

| File | Content |
|---|---|
| `Figure 1.png` / `.jpg` / `.tiff` | Retained measurements against the points removed by the outlier filter |
| `Figure 2.png` / `.jpg` / `.tiff` | The manufacturer curve, annotated with cut-in speed, rated power, and cut-off speed |
| `Figure 3.png` / `.jpg` / `.tiff` | All four fitted curves over the test data, labelled with each model's nRMSE |
| `figure_index.csv` | Figure numbers with their captions |
| `summary_TR.csv` | Mean nRMSE, success rate, and rank per model |
| `long_TR.csv` | One row per (scenario, replication, model) |

Figures are numbered sequentially and rendered at 300 DPI, then written in all three
formats so they can go straight into a manuscript — journals typically ask for TIFF or
high-quality JPG. To change the numbering or captions, edit `FIGURE_ORDER` in `main_tr.py`.

Figures are deliberately greyscale — line style and hatching carry the distinction rather
than colour — because they are prepared for black-and-white print.

`figures_tr/` is git-ignored so that re-running does not leave the repository dirty; the
committed copies in `results/` are what the tables and images above refer to.

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
| `results/` | Committed figures and tables from a verified run |

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

**No licence file is set yet.** This repository is public, but public visibility is not a
licence: without one, default copyright applies and readers have no right to reuse the
code. Add a licence — MIT and CC BY are the usual choices for a paper artifact — so that
citing readers can actually run and build on it.

**The dataset in `data/` needs a licence check.** `turkiye_scada_2018.csv` is redistributed
from Kaggle, where the licence is listed as *Unknown*. Redistributing it from a public
repository may not be permitted. Either confirm the terms with the dataset author, or
remove `data/turkiye_scada_2018.csv` from the repository and its history and have users
download `T1.csv` from Kaggle themselves — the loader reads it unchanged apart from the
file name.

## Reference

Kusiak, A., Zheng, H., & Song, Z. (2009). On-line monitoring of power curves.
*Renewable Energy*, 34(6), 1487–1493.
