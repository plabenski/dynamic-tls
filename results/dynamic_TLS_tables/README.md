# Dynamic TLS model: exported tables

This README documents the tables exported from `run_dynamic_TLS.py` for the 2026 *Fagus sylvatica* heatwave experiment to the projects' `results/dynamic_TLS_tables/` folder. Simulations are run separately for each leaf and summarised by tree at 5 min intervals. Predicted responses are relative to the initial leaf-level Fv/Fm baseline.

## Files

| File | Model components |
| --- | --- |
| `simulation_damage.csv` | Thermal damage |
| `simulation_damage_repair.csv` | Thermal damage and repair |
| `simulation_damage_repair_acclimation.csv` | Thermal damage, repair and acclimation |
| `variant_metrics.csv` | Predictive performance of the three variants |

## Simulation columns

All three simulation files share the following columns. Model states and rates are arithmetic means across leaves; minimum, maximum and standard deviation refer to leaf-level predictions. These describe variation among leaves, not parameter uncertainty or confidence intervals.

| Column | Description | Unit |
| --- | --- | --- |
| `Datetime` | Simulation timestamp (`YYYY-MM-DD HH:MM:SS`) | — |
| `Tree_ID` | Individual tree identifier | Integer between 1 and 10 |
| `Fvfm_rel_pred` | Mean predicted relative Fv/Fm after damage and any enabled repair | Dimensionless |
| `Fvfm_pred_min`, `Fvfm_pred_max` | Minimum and maximum predicted relative Fv/Fm across leaves | Dimensionless |
| `Fvfm_pred_sd` | Sample standard deviation of predicted relative Fv/Fm across leaves | Dimensionless |
| `Temperature` | Mean leaf temperature used to drive the model | °C |
| `A` | Acclimation-induced increase in the TDT intercept | °C |
| `b_dyn` | Dynamic TDT intercept: initial intercept plus `A` | °C |
| `dD` | Thermal load (dose) added during the current timestep | Dimensionless |
| `D_cum` | Cumulative thermal load added since simulation start, without subtracting repair | Dimensionless |
| `D_equiv` | Equivalent dose inferred from relative Fv/Fm at the start of the timestep, before adding `dD` and applying repair | Dimensionless |
| `r_repair_base` | Temperature-dependent repair rate before adjustment for repair capacity | min⁻¹ |
| `repair_capacity` | Multiplier accounting for impairment-dependent repair capacity | Value between 0 and 1 |
| `r_repair` | Effective repair rate: `r_repair_base × repair_capacity` | min⁻¹ |
| `F_after_heat` | Relative Fv/Fm after the thermal-load increment, before repair | Dimensionless |
| `F` | Duplicate of `Fvfm_rel_pred`, retained as a model-state variable | Dimensionless |

Thermal load is scaled so that a load of 1 corresponds to the T50 endpoint under the underlying TDT model; it is not itself the fraction of PSII impairment. Disabled acclimation gives `A = 0`; disabled repair gives zero repair rates and `repair_capacity = 1`.

## Performance metrics

Each row of `variant_metrics.csv` represents one model variant; the first column contains the variant name. Metrics compare predicted and observed tree-level relative Fv/Fm, matching each observation to the nearest simulation timestamp within its tree.

| Column | Description |
| --- | --- |
| `RMSE` | Root mean squared prediction error |
| `R2` | Coefficient of determination; may be negative |
| `Bias` | Mean predicted minus observed relative Fv/Fm |
| `n` | Number of matched observation–prediction pairs |

RMSE and bias use the dimensionless relative Fv/Fm scale. These are fit/evaluation statistics, not independent validation unless evaluated on withheld observations.
