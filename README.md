# Modelling tree responses to heatwaves using a dynamic TLS approach

Data, analysis code and model outputs accompanying *From thermal load to heat injury: Dynamic exposure, acclimation and recovery predict tree responses to heatwaves* by Labenski et al. (submitted). The analyses quantify exposure-duration-dependent thermal tolerance, compare static thermal safety margins with safety metrics based on cumulative thermal load, and model changes in photosystem II (PSII) efficiency during heat exposure and recovery, including repair and acclimation processes.

**Manuscript:** submitted, no DOI available yet\
**Code:** to be added\
**Data:** to be added

## Folder structure

| Folder | Contents |
| --- | --- |
| [`data/2024_acclimation_experiment/`](data/2024_acclimation_experiment/README.md) | Repeated thermal tolerance measurements in *Fagus sylvatica* during two heatwaves and recovery periods; environmental measurements and estimated leaf temperatures |
| [`data/2025_tolerance_assay/`](data/2025_tolerance_assay/README.md) | Fv/Fm measurements and duration-specific thermal tolerance thresholds for *Acer platanoides*, *Castanea sativa*, *Fagus sylvatica* and *Quercus petraea* |
| [`data/2026_heatwave_experiment/`](data/2026_heatwave_experiment/README.md) | Leaf temperatures, Fv/Fm, scorching observations and initial thermal tolerance thresholds for ten *Fagus sylvatica* seedlings |
| `scripts/` | Numbered scripts for running the analyses |
| `src/` | Shared model and fitting functions in `dynamic_tls.py` |
| `results/manuscript_figures/` | Figure outputs used to prepare manuscript figures, in PDF and PNG format |
| `results/diagnostic_figures/` | Additional plots supporting model assessment and interpretation |
| [`results/thermal_tolerance_tables/`](results/thermal_tolerance_tables/README.md) | Thermal death time (TDT) parameters, parameter draws and thermal transition breadth estimates |
| [`results/dynamic_TLS_tables/`](results/dynamic_TLS_tables/README.md) | Dynamic TLS simulations and model performance metrics |

The linked READMEs define file contents, columns, units and processing details. Files in `data/` include both measurements and derived inputs; files in `results/` are analysis outputs.

## Running the analyses

Use a dedicated Python environment. The scripts were developed and tested with Python version 3.14.6. The required packages can be installed by running:

```bash
conda env create -f environment.yml
conda activate dynamic-tls
```

Open a terminal in the publication folder and run the scripts in order:

```bash
python scripts/01_thermal_tolerance_landscapes.py
python scripts/02_thermal_safety_margins.py
python scripts/03_run_dynamic_tls.py
```

| Script | Purpose |
| --- | --- |
| `01_thermal_tolerance_landscapes.py` | Fit duration-dependent thermal tolerance relationships, quantify thermal transition breadth and export TDT parameter estimates and draws |
| `02_thermal_safety_margins.py` | Compare static thermal safety margins and dynamic thermal-load metrics using synthetic temperature scenarios |
| `03_run_dynamic_tls.py` | Fit and evaluate thermal damage, repair and acclimation components using the experimental data; compare different model variants |

Review the configuration section of each script for model settings, random seeds and plot-display options (e.g. disable interactive figures). Re-running analyses will overwrite existing outputs.

## Citation and reuse
Code in scripts/ and src/ is licensed under the MIT License. Data, results and documentation are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). Please cite the accompanying manuscript and archived code and data releases when using this work.

## Contact
Pia Labenski (pia.labenski@kit.edu)\
Karlsruhe Institute of Technology (KIT), Institute of Meteorology and Climate Research Atmospheric Environmental Research (IMKIFU)