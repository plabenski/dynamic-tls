# Heatwave experiment — 2026

This folder contains measurements of leaf temperature and maximum photosystem II efficiency (*Fv/Fm*), thermal tolerance thresholds and scorching observations for ten *Fagus sylvatica* seedlings during an experimental five-day heatwave and four-day recovery period. Air temperatures during the heatwave reached daytime maxima between 43 and 50 °C; night-time temperatures were around 21 °C. The recovery period had daytime air temperatures between 25–30 °C and night-time temperatures around 17 °C. Seedlings were watered to field capacity to avoid soil-water limitation. 

## Files

| File | Contents |
| --- | --- |
| `leaf_temp.csv` | Leaf temperatures derived from infrared imagery |
| `leaf_temp_means_per_tree.csv` | Average leaf temperatures per tree |
| `leaf_fvfm.csv` | Leaf-level *Fv/Fm* measurements during the experiment |
| `leaf_fvfm_means_per_tree.csv` | Mean *Fv/Fm* and relative *Fv/Fm* per tree, with standard deviations |
| `leaf_scorching.csv`| Leaf-level observations of scorched leaf area |
| `thermal_thresholds_per_tree.csv` | Individual thermal tolerance thresholds (T5, T50 and T95) estimated before the experiment |

## 1. Leaf temperatures

### `leaf_temp.csv`

Leaf temperatures at 5-min intervals from infrared cameras (Optris Xi 410), corrected for camera offsets using reference plates and for local warming from metal leaf clips during *Fv/Fm* measurements; evaluated against thermocouple measurements (R² = 0.98, RMSE = 1.17 °C).

| Column | Description | Unit or format |
| --- | --- | --- |
| `Datetime` | Measurement timestamp | `DD.MM.YYYY HH:MM` |
| `Tree_ID` | Individual tree identifier | Integer from 1 to 10 |
| `Leaf_ID` | Individual leaf identifier | Integer from 1 to 3 |
| `Temperature` | Corrected leaf temperature | °C |

### `leaf_temp_means_per_tree.csv`

Tree-level mean temperatures calculated from the leaf-level temperature records.

| Column | Description | Unit or format |
| --- | --- | --- |
| `Datetime` | Measurement timestamp | `DD.MM.YYYY HH:MM` |
| `Tree_ID` | Individual tree identifier | Integer from 1 to 10 |
| `Temperature` | Average leaf temperature per tree | °C |

## 2. PSII efficiency
 
Maximum PSII quantum yield (*Fv/Fm*) was measured on the same three leaves per tree in morning, midday and evening campaigns during the heatwave, and at midday during recovery. Measurements followed 30 min of dark adaptation.

### `leaf_fvfm.csv`

| Column | Description | Unit or format |
| --- | --- | --- |
| `Date` | Measurement date | `DD.MM.YYYY` |
| `Time` | Time of the *Fv/Fm* measurement | `HH:MM:SS` |
| `Campaign` | Measurement campaign | `morning`, `midday` or `evening` |
| `Tree_ID` | Individual tree identifier | Integer from 1 to 10 |
| `Leaf_ID` | Individual leaf identifier | Integer from 1 to 3 |
| `Fv/Fm` | Measured maximum PSII quantum yield | Dimensionless |
| `Datetime_group` | First actual measurement time of each tree within each measurement campaign | `DD.MM.YYYY HH:MM` |
| `Baseline_fvfm` | Initial *Fv/Fm* for each leaf, from the first morning measurement before the start of the heatwave | Dimensionless |
| `FvFm_rel` | Leaf-level relative *Fv/Fm*: `Fv/Fm` divided by `Baseline_fvfm` | Dimensionless |

### `leaf_fvfm_means_per_tree.csv`

Tree-level summaries for each measurement campaign. Relative values are calculated for each leaf before averaging across leaves. Standard deviations describe variation among leaves, not uncertainty in the mean.

| Column | Description | Unit or format |
| --- | --- | --- |
| `Date` | Measurement date | `DD.MM.YYYY` |
| `Campaign` | Measurement campaign | `morning`, `midday` or `evening` |
| `Tree_ID` | Individual tree identifier | Integer from 1 to 10 |
| `Fv/Fm_mean` | Mean measured *Fv/Fm* across the three leaves per tree | Dimensionless |
| `Fv/Fm_sd` | Standard deviation of measured *Fv/Fm* across the three leaves per tree | Dimensionless |
| `Fv/Fm_rel_mean` | Mean of relative (normalised) *Fv/Fm* across three leaves per tree | Dimensionless | 
| `Fv/Fm_rel_sd` | Standard deviation of relative (normalised) *Fv/Fm* across three leaves per tree | Dimensionless |
| `Datetime_group` | First actual measurement time of each tree within each measurement campaign | `DD.MM.YYYY HH:MM` |

## 3. Leaf scorching

The fraction of scorched leaf area was estimated from photographs collected after each measurement campaign. This file contains only records for scorched leaves; zero-scorching observations are not included. Absent records should therefore be interpreted as zero scorching, as all leaves were assessed each measurement time.

### `leaf_scorching.csv`

| Column | Description | Unit or format |
| --- | --- | --- |
| `Date` | Measurement date | `DD.MM.YYYY` |
| `Campaign` | Measurement campaign | `morning`, `midday` or `evening` |
| `Tree_ID` | Individual tree identifier | Integer from 1 to 10 |
| `Leaf_ID` | Individual leaf identifier | Integer from 1 to 3 |
| `Scorch` | Fraction of scorched leaf area | Dimensionless fraction from 0 to 1 | 

## 4. Thermal tolerance thresholds

Individual thermal tolerance thresholds determined before the start of the experimental heatwave. Thermal thresholds were estimated by fitting Weibull temperature–response curves to individual assay measurements of *Fv/Fm* for three leaves per tree. Measurements were taken after an overnight recovery period following 15 min exposures to nine different water-bath temperatures. `Tx` denotes the temperature associated with an *x*% decline in *Fv/Fm* relative to the reference response at 25 °C. For example, `T50` represents the temperature associated with a 50% decline in *Fv/Fm*.

### `thermal_thresholds_per_tree.csv`

| Column or column pattern | Description | Unit |
| --- | --- | --- |
| `Tree_ID` | Individual tree identifier | Integer from 1 to 10 |
| `Tx` | Estimated temperature associated with an *x*% decline in *Fv/Fm*, for `x = 5, 50, 95` | °C |

## Linking the files

Use `Tree_ID` and `Leaf_ID` together to identify a leaf; `Leaf_ID` alone is not unique across trees. Tree-level summary files can be linked by `Tree_ID`. Physiological measurements are associated with a `Date` and `Campaign`; `Datetime_group` records the first measurement time for each tree within a campaign, rather than the actual measurement time of every leaf. Temperature records have a separate 5 min time axis.

