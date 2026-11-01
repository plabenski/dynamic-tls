# Thermal tolerance assay — 2025

This folder contains measurements of maximum photosystem II quantum yield (*Fv/Fm*) after controlled thermal exposure following the water-bath method described by Curtis et al. (2014), together with the duration-specific thermal tolerance thresholds derived from those measurements.

Detached leaf discs from four temperate tree species were exposed to nine different water-bath temperatures for 0.25, 0.5, 1, or 4 hours. *Fv/Fm* was measured after 24 hours of recovery in darkness.

## Files

| File | Contents |
| --- | --- |
| `assay_fvfm_measurements.csv` | Individual-associated *Fv/Fm* measurements following thermal exposure and recovery |
| `thermal_thresholds_by_duration.csv` | Species-level thermal tolerance thresholds for each exposure duration, with bootstrap confidence intervals |

## Species and identifiers

| Code | Scientific name |
| --- | --- |
| AP | *Acer platanoides* |
| CS | *Castanea sativa* |
| FS | *Fagus sylvatica* |
| QP | *Quercus petraea* |

`Tree_ID` identifies an individual tree using a two-letter species code followed by an individual number. Measurements sharing a `Tree_ID` refer to the same individual.

## 1. Assay measurements

### `assay_fvfm_measurements.csv`

Maximum PSII quantum yield (*Fv/Fm*) measured after thermal exposure and 24 hours of dark recovery.

| Column | Description | Unit or format |
| --- | --- | --- |
| `Date` | Date of the *Fv/Fm* measurement | `DD.MM.YYYY` |
| `Time` | Time of the *Fv/Fm* measurement | `HH:MM:SS` |
| `Tree_ID` | Individual tree identifier, combining species code and individual number | Text |
| `Duration` | Duration of thermal exposure in the water bath | Hours: 0.25, 0.5, 1, or 4 |
| `Temperature` | Water-bath temperature used for the assay | °C |
| `FvFm` | Measured maximum PSII quantum yield after recovery | Dimensionless |

## 2. Derived thermal tolerance thresholds

### `thermal_thresholds_by_duration.csv`

Duration-specific thermal tolerance thresholds estimated by fitting a Weibull temperature–response curve to pooled assay measurements from 5–8 individuals per species. `Tx` denotes the temperature at which *Fv/Fm* has declined by *x* percent relative to the reference response at 25°C. For example, `T50` is the temperature associated with a 50% decline in *Fv/Fm* at the specified exposure duration. 

| Column or column pattern | Description | Unit |
| --- | --- | --- |
| `Species_short` | Two-letter species code defined above | — |
| `Species` | Full scientific species name | — |
| `Duration` | Duration of thermal exposure | Minutes |
| `Tx` | Mean temperature threshold for an *x*% decline in *Fv/Fm*, for the following decline levels: 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95 % | °C |
| `Tx_lower` | Lower endpoint of the bootstrap 95% confidence interval for the corresponding threshold, using individual trees as resampling units | °C |
| `Tx_upper` | Upper endpoint of the bootstrap 95% confidence interval for the corresponding threshold, using individual trees as resampling unit | °C |

## References
Curtis, E. M., Knight, C. A., Petrou, K., & Leigh, A. (2014). A comparative analysis of photosynthetic recovery from thermal stress: a desert plant case study. Oecologia, 175(4), 1051-1061. https://doi.org/10.1007/s00442-014-2988-5
