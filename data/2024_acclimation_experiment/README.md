# Acclimation experiment — 2024

This folder contains repeated measurements of thermal tolerance in *Fagus sylvatica* seedlings, together with greenhouse environmental measurements and estimated leaf temperatures, before, during and after two consecutive experimental heatwaves in 2024.

## Experimental design

Seedlings were assigned to two treatment groups, each comprising eight individuals:

| Treatment | Temperature conditions | Soil water content (SWC) |
| --- | --- | --- |
| Control | Mean daytime air temperature of 25 °C | Maintained at 25% |
| Heat | Two consecutive heatwaves (HW1 and HW2), separated by a recovery period | Maintained at 25% |

Each heatwave lasted six days, during which mean daytime air temperature increased from 30 to 40 °C. Recovery periods lasted 14 days at a daytime air temperature of 25 °C. Nighttime temperatures were maintained at 16 °C.

Thermal tolerance was assessed before the first heatwave, on the last day of each heatwave, and on the last day of each recovery period. Assays followed the water-bath method of Curtis et al. (2014), with a 15 min exposure duration.

## Files

| File | Contents |
| --- | --- |
| `greenhouse_conditions_leaf_temp.csv` | Environmental measurements for the heat treatment and estimated leaf temperatures |
| `tolerance_thresholds_acclimation.csv` | Individual thermal tolerance thresholds at each measurement time point, with bootstrap confidence intervals |

## 1. Environmental conditions and estimated leaf temperature

### `greenhouse_conditions_leaf_temp.csv`

This file contains greenhouse air temperature, photosynthetically active radiation and relative humidity for the heat treatment at 10 min intervals. Leaf temperatures were estimated from these environmental variables using an empirical model developed in a separate experiment with well-watered *Fagus sylvatica* seedlings (R²=0.91, RMSE=2.3°C). `Tleaf` therefore represents modelled rather than directly measured leaf temperature.

| Column | Description | Unit or format |
| --- | --- | --- |
| `Datetime` | Measurement timestamp | `DD.MM.YYYY HH:MM` |
| `PAR` | Photosynthetically active radiation | µmol m⁻² s⁻¹ |
| `RH` | Relative humidity | % |
| `Tair` | Air temperature | °C |
| `Tleaf` | Estimated leaf temperature for the heat treatment | °C |

## 2. Thermal tolerance thresholds

### `tolerance_thresholds_acclimation.csv`

Thermal tolerance thresholds were estimated by fitting Weibull temperature–response curves to individual assay measurements of *Fv/Fm* after an overnight recovery period following 15 min exposures to nine different water-bath temperatures. `Tx` denotes the temperature associated with an *x*% decline in *Fv/Fm* relative to the reference response at 25 °C. For example, `T50` represents the temperature associated with a 50% decline in *Fv/Fm*.


| Column or column pattern | Description | Unit or format |
| --- | --- | --- |
| `Code` | Individual seedling identifier | Identifier |
| `Date` | Date of the thermal tolerance assay | `DD.MM.YYYY` |
| `Treatment` | Treatment group: `Control` or `Heat` | Category |
| `Period` | Measurement period; see definitions below | Category |
| `Tx` | Estimated temperature associated with an *x*% decline in *Fv/Fm*, for `x = 5, 50, 95` | °C |
| `Tx_lower` | Lower endpoint of the bootstrap 95% confidence interval for the corresponding threshold | °C |
| `Tx_upper` | Upper endpoint of the bootstrap 95% confidence interval for the corresponding threshold | °C |

Measurement periods are defined as follows:

| `Period` | Definition |
| --- | --- |
| `Pre` | Before the first heatwave |
| `HW1` | Last day of the first heatwave |
| `PostHW1` | Last day of the recovery period following the first heatwave |
| `HW2` | Last day of the second heatwave |
| `PostHW2` | Last day of the recovery period following the second heatwave |

## References
Curtis, E. M., Knight, C. A., Petrou, K., & Leigh, A. (2014). A comparative analysis of photosynthetic recovery from thermal stress: a desert plant case study. Oecologia, 175(4), 1051-1061. https://doi.org/10.1007/s00442-014-2988-5

