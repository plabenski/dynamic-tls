"""
Workflow for TLS-based PSII predictions in Fagus sylvatica during a heatwave.
It uses three experiments:
the 2025 tolerance assay for the TDT dose-response,
the 2024 acclimation experiment to constrain acclimation and deacclimation kinetics,
the 2026 heatwave experiment to constrain recovery and refine acclimation estimates,
and to evaluate all predictions against measured PSII responses.

This script:
1. loads and prepares all datasets from the different experiments;
2. reconstructs the species-mean TDT dose-response dataset;
3. fits the independent acclimation experiment;
4. compares dose-scale and Fv/Fm-scale recovery;
5. fits experiment-specific A_max and T_acclim;
6. simulates the three enabled model variants;
7. plots fitted components and predicted Fv/Fm responses;
8. saves manuscript/diagnostic figures and exports model comparison tables.

This script uses functions from src/thermal_tls_model_leaf.py.
Outputs are written to: results/manuscript_figures/, results/diagnostic_figures/,
and results/dynamic_TLS_tables/.
"""
# ------- #
# Imports #
# ------- #

import sys
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from dynamic_tls import (
    thermal_load_rate,
    fit_species_mean_dose_response,
    fit_reference_acclimation,
    fit_recovery,
    fit_experiment_acclimation,
    make_variant_params,
    simulate_variant,
    match_simulation_to_observations,
    evaluate_predictions,
    plot_variant_fvfm,
    plot_intercept_acclimation,
    plot_recovery_rate,
    plot_recovery_diagnostics,
    plot_acclimation_target,
    plot_reference_acclimation,
    plot_tree_panels,
    plot_observed_vs_predicted_leaf_range,
    plot_observed_vs_predicted_mean,
)

# ----------------------------- #
# 0. Project paths and settings #
# ----------------------------- #

data_dir, results_dir = project_root / "data", project_root / "results"
figures_dir = results_dir / "manuscript_figures"
diagnostics_dir = results_dir / "diagnostic_figures"
tables_dir = results_dir / "dynamic_TLS_tables"
for directory in (figures_dir, diagnostics_dir, tables_dir):
    directory.mkdir(parents=True, exist_ok=True)

SHOW_PLOTS = False  # Set False for non-interactive runs; figures are saved in both modes.

# Data from the 2026 heatwave experiment
TEMP_LEAF_FILE = data_dir / "2026_heatwave_experiment" / "leaf_temp.csv"
TEMP_TREE_FILE = data_dir / "2026_heatwave_experiment" / "leaf_temp_means_per_tree.csv"
T50_TREE_FILE  = data_dir / "2026_heatwave_experiment" / "thermal_thresholds_per_tree.csv"
FVFM_LEAF_FILE = data_dir / "2026_heatwave_experiment" / "leaf_fvfm.csv"
FVFM_TREE_FILE = data_dir / "2026_heatwave_experiment" / "leaf_fvfm_means_per_tree.csv"
SCORCHING_FILE = data_dir / "2026_heatwave_experiment" / "leaf_scorching.csv"

# Species-level TDT parameters from the 2025 tolerance assay
TDT_FVFM_FILE = data_dir / "2025_tolerance_assay" / "assay_fvfm_measurements.csv"
TDT_PARAM_FILE = results_dir / "thermal_tolerance_tables" / "tdt_parameter_draws.csv"

# Independent 2024 acclimation experiment
ACCLIMATION_T50_FILE = data_dir / "2024_acclimation_experiment" / "tolerance_thresholds_acclimation.csv"
ACCLIMATION_TEMP_FILE = data_dir / "2024_acclimation_experiment" / "greenhouse_conditions_leaf_temp.csv"

# 2026 Heatwave experiment-specific parameters
SPECIES = "Fagus sylvatica"
SPECIES_CODE = "FS"
DT_MIN = 5
START_CURRENT_EXPERIMENT = pd.Timestamp("2026-07-13 08:00:00")

# Current experiment pooled T50 at 15 min.
BASELINE_T50_15 = 50.6
USE_TREE_SPECIFIC_T50 = False

# Acclimation model parameters
BETA_ACCLIM = 0.3
REGULARIZE_ACCLIMATION = False
SIGMA_A = 1.0
SIGMA_T = 3.0
A_MAX_UPPER = 4.0

# Recovery model parameters
RECOVERY_MODEL = "fvfm" # alternative "dose"
MIN_FVFM_RECOVERY = None
REPAIR_TH = 30.0
REPAIR_T_REF = 25.0
REPAIR_EA = 0.8e5
REPAIR_EH = 3.8e5

# Temperature resolution used for different model components.
# Full dynamic simulation:
#   "leaf" is recommended because thermal load is nonlinear in temperature.
# Recovery fitting:
#   "leaf" propagates recovery independently through the three measured leaf
#   temperature trajectories and averages predicted Fv/Fm afterwards.
#   "tree" uses the precomputed tree-mean temperature series.
MODEL_TEMP_LEVEL = "leaf"
RECOVERY_TEMP_LEVEL = "leaf"

# Switch impairment-dependent repair capacity on/off in the dynamic model.
USE_REDUCED_REPAIR_CAPACITY = True
P_REPAIR = 8.0

# Figure style  
sns.set_theme(style='ticks', font_scale=1.1, rc={'axes.linewidth': 1.0, 'xtick.major.size': 5, 'xtick.major.width': 1, 'ytick.major.size': 5, 'ytick.major.width': 1})

# Figure saving for plot functions imported from src/
@contextmanager
def save_plot_outputs(directory, stem, labels=None):
    """Save new figures before display, including figures made inside imported functions.

    Optional labels name figures produced in a known order (e.g. model variants).
    The temporary show hook is restored on exit, including if plotting raises an error.
   .
    """
    directory.mkdir(parents=True, exist_ok=True)
    original_show = plt.show
    existing_figures = {plt.figure(number) for number in plt.get_fignums()}
    saved_figures = set()

    def save_pending_figures():
        """Save each newly created figure once, restoring the active figure afterwards."""
        active_figure = plt.gcf() if plt.get_fignums() else None
        for number in plt.get_fignums():
            figure = plt.figure(number)
            if figure in existing_figures or figure in saved_figures:
                continue
            index = len(saved_figures)
            if labels is not None and index < len(labels):
                suffix = f"_{labels[index]}"
            else:
                suffix = "" if index == 0 else f"_{index + 1:02d}"
            for extension in ("pdf", "png"):
                figure.savefig(directory / f"{stem}{suffix}.{extension}", dpi=300, bbox_inches="tight")
            saved_figures.add(figure)
        if active_figure is not None:
            plt.figure(active_figure.number)

    def show_and_save(*args, **kwargs):
        """Save before a blocking show can close the figure."""
        save_pending_figures()
        if SHOW_PLOTS:
            return original_show(*args, **kwargs)

    plt.show = show_and_save
    try:
        yield
        save_pending_figures()  # Also support plotting functions that do not call show().
    finally:
        plt.show = original_show
        if not SHOW_PLOTS:
            for figure in saved_figures:
                plt.close(figure)


# ----------------------------------- #
# 1. Load current heatwave experiment #
# ----------------------------------- #

# 1.1 Resample leaf temperatures; exclude grouping IDs from numeric averages.
df_temp_leaf = pd.read_csv(TEMP_LEAF_FILE)
df_temp_leaf["Datetime"] = pd.to_datetime(df_temp_leaf["Datetime"])
df_temp_leaf = df_temp_leaf[df_temp_leaf["Datetime"] >= START_CURRENT_EXPERIMENT].copy()

df_temp_leaf = (
    df_temp_leaf
    .set_index("Datetime")
    .groupby(["Tree_ID", "Leaf_ID"])
    .resample(f"{DT_MIN}min", origin=START_CURRENT_EXPERIMENT)
    .mean(numeric_only=True)
    .drop(columns=["Tree_ID", "Leaf_ID"], errors="ignore")
    .reset_index()
    .set_index(["Datetime", "Tree_ID", "Leaf_ID"])
    .sort_index()
)

# 1.2 Prepare tree-mean temperatures for comparison and optional recovery fitting
df_temp_tree = pd.read_csv(TEMP_TREE_FILE)
df_temp_tree["Datetime"] = pd.to_datetime(df_temp_tree["Datetime"])
df_temp_tree = df_temp_tree[df_temp_tree["Datetime"] >= START_CURRENT_EXPERIMENT].copy()

df_temp_tree = (
    df_temp_tree
    .set_index("Datetime")
    .groupby("Tree_ID")
    .resample(f"{DT_MIN}min", origin=START_CURRENT_EXPERIMENT)
    .mean(numeric_only=True)
    .drop(columns="Tree_ID", errors="ignore")
    .reset_index()
    .set_index(["Datetime", "Tree_ID"])
    .sort_index()
)

# Temperature data used by the full model
df_temp_model = (
    df_temp_leaf
    if MODEL_TEMP_LEVEL == "leaf"
    else df_temp_tree
)

tree_ids = (
    df_temp_model.index.get_level_values("Tree_ID")
    .unique()
    .tolist()
)

# Temperature data used to estimate the recovery rate and impairment-dependent capacity.
df_temp_recovery = (
    df_temp_leaf
    if RECOVERY_TEMP_LEVEL == "leaf"
    else df_temp_tree
)

# Prepare Fv/Fm data
df_fvfm = pd.read_csv(FVFM_TREE_FILE)
df_fvfm["Datetime"] = pd.to_datetime(df_fvfm["Datetime_group"])
df_fvfm = (
    df_fvfm[df_fvfm["Datetime"] >= START_CURRENT_EXPERIMENT.normalize()]
    .set_index(["Datetime", "Tree_ID"])
    .sort_index()
)

df_fvfm_leaf = pd.read_csv(FVFM_LEAF_FILE)
df_fvfm_leaf["Datetime"] = pd.to_datetime(df_fvfm_leaf["Datetime_group"])

# Load T50 data
df_t50_tree = pd.read_csv(T50_TREE_FILE, sep=",")
df_t50_tree["Tree_ID"] = df_t50_tree["Tree_ID"].astype(int)

# Prepare scorching data
df_scorch = pd.read_csv(SCORCHING_FILE)
df_scorch["Date"] = pd.to_datetime(
    df_scorch["Date"], format="%d.%m.%Y"
).dt.normalize()
df_scorch = (
    df_scorch.groupby(["Date", "Campaign", "Tree_ID"])
    .mean(numeric_only=True)
)


print(f"Current experiment: {len(tree_ids)} trees")
print(f"Full-model temperature level: {MODEL_TEMP_LEVEL}")
print(f"Recovery-fit temperature level: {RECOVERY_TEMP_LEVEL}")

# ---------------------------------------------------------- #
# 2. TDT parameters and empirical species-mean dose-response #
# ---------------------------------------------------------- #
# Load TDT parameters from thermal tolerance assay
df_params = pd.read_csv(TDT_PARAM_FILE)

# Species-level TDT slope from 2025 assay
a_species = df_params.loc[df_params["Species"] == SPECIES, "a"].mean()

# Convert the initial pooled 15-min T50 of the 2026 trees
# to the intercept b of T50(t) = a ln(t) + b
b_species = BASELINE_T50_15 - a_species * np.log(15.0)

# Create tree-specific T50 intercepts for the 2026 trees
df_t50_tree["b0"] = df_t50_tree["T50"] - a_species * np.log(15.0)

print("\nTDT parameters")
print("--------------")
print(f"a = {a_species:.5f}")
print(f"b = {b_species:.5f}")

# 2.1 Rebuild the dose-response data from the independent tolerance assay.
species_dict = {
    "AP": "Acer platanoides",
    "CS": "Castanea sativa",
    "FS": "Fagus sylvatica",
    "QP": "Quercus petraea",
}

df_tdt = pd.read_csv(TDT_FVFM_FILE)
df_tdt = df_tdt[df_tdt["Tree_ID"].str[:2].isin(species_dict)].copy()
df_tdt["species"] = df_tdt["Tree_ID"].str[:2].map(species_dict)

# 2.2 Normalise Fv/Fm to each tree's mean response in the 25 °C control treatment.
fvfm_max = (
    df_tdt[df_tdt["Temperature"] == 25]
    .groupby("Tree_ID")["FvFm"]
    .mean()
)

# 2.3 Pair estimated thermal load in the 2025 assay with the measured relative Fv/Fm.
response_rows = []

for _, row in df_tdt.iterrows():
    species = row["species"]
    p = df_params.loc[df_params["Species"] == species]

    a_samples = np.asarray(p["a"])
    b_samples = np.asarray(p["b"])
    a = a_samples.mean()
    b = b_samples.mean()

    duration_min = row["Duration"] * 60.0
    dose = thermal_load_rate(row["Temperature"], a, b) * duration_min

    dose_samples = (thermal_load_rate(row["Temperature"], a_samples, b_samples) * duration_min)

    response_rows.append({
        "tree_id": row["Tree_ID"],
        "species": species,
        "temperature": row["Temperature"],
        "duration_min": duration_min,
        "dose": dose,
        "dose_lower": np.percentile(dose_samples, 2.5),
        "dose_upper": np.percentile(dose_samples, 97.5),
        "response_rel": row["FvFm"] / fvfm_max.loc[row["Tree_ID"]],
    })

df_response = pd.DataFrame(response_rows)
df_response_mean = (
    df_response
    .groupby([
        "species",
        "temperature",
        "duration_min",
    ])
    .agg(
        dose=("dose", "first"),
        dose_lower=("dose_lower", "first"),
        dose_upper=("dose_upper", "first"),

        response_rel=("response_rel", "mean"),
        response_sd=("response_rel", "std"),

        n_trees=("tree_id", "nunique"),
    )
    .reset_index()
)

# 2.4 Fit the Weibull response to species-mean relative Fv/Fm.
dose_fit = fit_species_mean_dose_response(
    df_response_mean,
    species=SPECIES,
)

# Resulting parameters of the Weibull fit
lam = dose_fit["lam"]
k = dose_fit["k"]

print("\nSpecies-mean dose-response")
print("--------------------------")
print(f"lambda = {lam:.6f}")
print(f"k      = {k:.6f}")

# Plot species-mean dose-response
D_plot = np.linspace(0, 10, 300)
data = dose_fit["data"].copy()
xerr = np.vstack([
    data["dose"] - data["dose_lower"],
    data["dose_upper"] - data["dose"],
])
yerr = data["response_sd"].values

fig, ax = plt.subplots(figsize=(4.5, 4.2), dpi=300)
ax.errorbar(
    data["dose"],
    data["response_rel"],

    # parameter uncertainty in thermal load
    xerr=xerr,

    # biological variation in Fv/Fm
    yerr=yerr,

    fmt="o",
    color="darkgrey",
    ecolor="darkgrey",
    markeredgecolor="white",
    markeredgewidth=0.6,
    markersize=9,
    alpha=0.7,
    elinewidth=0.8,
    capsize=2,
    zorder=3,
)
ax.plot(
    D_plot,
    np.exp(-((D_plot / lam) ** k)),
    label=f"Weibull fit\n R² = {dose_fit['r2']:.2f}",
    c="black",
    lw=1
)
ax.axhline(0.5, ls="--", lw=1, c='black')
ax.set_xlabel("Cumulative thermal load [-]")
ax.set_ylabel("Relative $F_{v}/F_{m}$ [-]")
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(-0.3, 6.3)
ax.grid(alpha=0.25)
ax.legend()
ax.text(
    -0.18, 1.08, "(a)",
    transform=ax.transAxes,
    ha="left", va="top",
    fontweight="bold")
fig.tight_layout()
for extension in ("pdf", "png"):
    fig.savefig(figures_dir / f"dose_response.{extension}", dpi=300, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)


# ------------------------------------------ #
# 3. Independent 2024 acclimation experiment #
# ------------------------------------------ #
df_acc = pd.read_csv(ACCLIMATION_T50_FILE)
df_acc["Date"] = pd.to_datetime(df_acc["Date"], format="%d.%m.%Y")

# 3.1 Aggregate tree responses by treatment and sampling period.
df_acc_mean = (
    df_acc.groupby(["Date", "Treatment", "Period"])["T50"]
    .mean()
    .reset_index(name="T50")
)

df_acc_std = (
    df_acc.groupby(["Date", "Treatment", "Period"])["T50"]
    .std()
    .reset_index(name="T50_std")
)

df_acc_wide = df_acc_mean.pivot(
    index="Treatment",
    columns="Period",
    values="T50",
)

# 3.2 Express tolerance changes relative to the experiment's sampling periods.
df_acc_diff = pd.DataFrame({
    "Treatment": df_acc_wide.index,
    "HW1-Pre": df_acc_wide["HW1"] - df_acc_wide["Pre"],
    "PostHW1-HW1": df_acc_wide["PostHW1"] - df_acc_wide["HW1"],
    "PostHW1-Pre": df_acc_wide["PostHW1"] - df_acc_wide["Pre"],
    "HW2-PostHW1": df_acc_wide["HW2"] - df_acc_wide["PostHW1"],
    "PostHW2-HW2": df_acc_wide["PostHW2"] - df_acc_wide["HW2"],
}).reset_index(drop=True)

# Read temperature file for acclimation experiment
df_temp_acc = pd.read_csv(ACCLIMATION_TEMP_FILE)
df_temp_acc["Datetime"] = pd.to_datetime(df_temp_acc["Datetime"])
df_temp_acc = (
    df_temp_acc
    .drop_duplicates(subset="Datetime")
    .set_index("Datetime")
    .sort_index()
)

# Put observation dates in chronological period order
period_order = ["Pre", "HW1", "PostHW1", "HW2", "PostHW2"]
period_dates = (
    df_acc_mean.groupby("Period")["Date"]
    .first()
    .reindex(period_order)
)

# Acclimation is defined relative to pre-heatwave tolerance, so fit HW1 ... PostHW2
obs_dates = pd.DatetimeIndex(period_dates.iloc[1:].values)

acclimation_ref, fitted_acc = fit_reference_acclimation(
    df_temp_acc=df_temp_acc,
    df_diff=df_acc_diff,
    obs_dates=obs_dates,
    treatment="Heat",
    temp_col="Tleaf",
    beta=BETA_ACCLIM,
)

print("\nReference acclimation")
print("---------------------")
print(f'A_max    = {acclimation_ref["A_max_ref"]:.3f} °C')
print(f'T_acclim = {acclimation_ref["T_acclim_ref"]:.3f} °C')
print(f'tau_up   = {acclimation_ref["tau_up_days"]:.3f} d')
print(f'tau_down = {acclimation_ref["tau_down_days"]:.3f} d')

with save_plot_outputs(diagnostics_dir, "reference_A_target"):
    plot_acclimation_target(
        acclimation_ref["A_max_ref"],
        acclimation_ref["T_acclim_ref"],
        acclimation_ref["beta"],
    )

with save_plot_outputs(figures_dir, "reference_acclimation_T50"):
    plot_reference_acclimation(
        fitted_acc,
        df_temp_acc,
        df_acc_mean,
        df_acc_std
    )

# -------------------------- #
# 4. Fit recovery separately #
# -------------------------- #

# 4.1 Fit the Fv/Fm-scale recovery formulation.
recovery_fvfm = fit_recovery(
    df_fvfm=df_fvfm,
    df_temp=df_temp_recovery,
    lam=lam,
    k=k,
    model="fvfm",
    dt_min=DT_MIN,
    min_fvfm=MIN_FVFM_RECOVERY,
    Th=REPAIR_TH,
    p_repair=P_REPAIR, T_ref=REPAIR_T_REF, Ea=REPAIR_EA, Eh=REPAIR_EH,
    fit_repair_capacity=True
)

# 4.2 Fit the dose-scale alternative using the same temperature resolution.
recovery_dose = fit_recovery(
    df_fvfm=df_fvfm,
    df_temp=df_temp_recovery,
    lam=lam,
    k=k,
    model="dose",
    dt_min=DT_MIN,
    min_fvfm=MIN_FVFM_RECOVERY,
    Th=REPAIR_TH,
    p_repair=P_REPAIR, T_ref=REPAIR_T_REF, Ea=REPAIR_EA, Eh=REPAIR_EH,
    fit_repair_capacity=True
)

print("\nRecovery comparison")
print("-------------------")
print(
    f'Fv/Fm-scale: r_ref={recovery_fvfm["r_ref"]:.6f}, '
    f'RMSE={recovery_fvfm["rmse"]:.4f}',
    f'I_50={recovery_fvfm["I_scale"]:.3f}'
    )

print(
    f'Dose-scale:  r_ref={recovery_dose["r_ref"]:.6f}, '
    f'RMSE={recovery_dose["rmse"]:.4f}'
)

recovery_fit = (
    recovery_fvfm if RECOVERY_MODEL == "fvfm"
    else recovery_dose
)

with save_plot_outputs(figures_dir, "recovery_rate_and_capacity"):
    plot_recovery_rate(recovery_fit, Th=REPAIR_TH)
with save_plot_outputs(diagnostics_dir, "recovery_diagnostics"):
    plot_recovery_diagnostics(recovery_fit)


# ------------------------- #
# 5. Shared base parameters #
# ------------------------- #

base_params = {
    "a": a_species,
    "b0": b_species,
    "lam": lam,
    "k": k,
    "T_ref": REPAIR_T_REF,
    "Th": REPAIR_TH,
    "Ea": REPAIR_EA,
    "Eh": REPAIR_EH,
    "use_repair_capacity": USE_REDUCED_REPAIR_CAPACITY,
    "I_scale": recovery_fit["I_scale"],
    "p_repair": P_REPAIR,
    "use_individual_T50": USE_TREE_SPECIFIC_T50
}


# ------------------------------------------------- #
# 6. Adapt A_max and T_acclim to current experiment #
# ------------------------------------------------- #

acclimation_current = fit_experiment_acclimation(
    df_temp=df_temp_model,
    df_fvfm=df_fvfm,
    df_t50=df_t50_tree,
    tree_ids=tree_ids,
    base_params=base_params,
    acclimation_reference=acclimation_ref,
    recovery_fit=recovery_fit,
    dt_min=DT_MIN,
    regularize=REGULARIZE_ACCLIMATION,
    sigma_A=SIGMA_A,
    sigma_T=SIGMA_T,
    A_max_upper=A_MAX_UPPER
)

print("\nCurrent-experiment acclimation")
print("-----------------------------")
print(
    f'A_max: {acclimation_ref["A_max_ref"]:.3f} -> '
    f'{acclimation_current["A_max"]:.3f} °C'
)
print(
    f'T_acclim: {acclimation_ref["T_acclim_ref"]:.3f} -> '
    f'{acclimation_current["T_acclim"]:.3f} °C'
)

with save_plot_outputs(diagnostics_dir, "current_A_target"):
    plot_acclimation_target(
        acclimation_current["A_max"],
        acclimation_current["T_acclim"],
        acclimation_current["beta"],
    )

# -------------------------------- #
# 7. Simulate three model variants #
# -------------------------------- #
# 1) "damage"
#    Baseline TDT + empirical dose-response only.
#
# 2) "damage_repair"
#    Tests whether dynamic repair explains deviations from pure accumulation.
#
# 3) "damage_repair_acclimation"
#    Adds dynamic thermal tolerance. A_max and T_acclim are allowed to differ
#    from the independent acclimation experiment; kinetics are transferred.
#
variant_names = [
    "damage",
    "damage_repair",
    "damage_repair_acclimation",
]

simulations = {}
combined = {}
metrics = {}

for variant in variant_names:

    params = make_variant_params(
        base_params=base_params,
        variant=variant,
        acclimation_params=(
            acclimation_current
            if variant in {"damage_repair_acclimation",
                           "damage_acclimation"}
            else None
        ),
        recovery_fit=(
            recovery_fit
            if variant in {"damage_repair",
                           "damage_repair_acclimation"}
            else None
        ),
        )

    sim = simulate_variant(
        df_temp=df_temp_model,
        df_t50=df_t50_tree,
        tree_ids=tree_ids,
        params=params,
        dt_min=DT_MIN,
    )

    simulations[variant] = sim
    combined[variant] = match_simulation_to_observations(sim, df_fvfm)
    metrics[variant] = evaluate_predictions(combined[variant])

metrics_table = pd.DataFrame(metrics).T

print("\nNested model comparison")
print("-----------------------")
print(metrics_table)

# Plot predicted Fv/Fm response for every model variant
with save_plot_outputs(figures_dir, "fvfm", labels=variant_names):
    plot_variant_fvfm(simulations, df_fvfm, subplot_no='(d)')

# Plot intercept evolution for each enabled variant (constant without acclimation).
with save_plot_outputs(figures_dir, "intercept_evolution", labels='acclimation'):
    plot_intercept_acclimation(simulations, variant="damage_repair_acclimation", subplot_no="(d)")

# Plot observed versus predicted Fv/Fm for each model variant
with save_plot_outputs(figures_dir, "observed_vs_predicted_mean", labels=variant_names):
    plot_observed_vs_predicted_mean(simulations, df_fvfm) # Means per timestep
with save_plot_outputs(figures_dir, "observed_vs_predicted", labels=variant_names):
    plot_observed_vs_predicted_leaf_range(simulations, df_fvfm_leaf, subplot_no='(b)')

# Plot individual trees
with save_plot_outputs(figures_dir, "fvfm_tree_panels", labels=variant_names):
    plot_tree_panels(simulations, df_fvfm_leaf)

# ------------------------------------------------ #
# 8. Export model comparison and simulation tables #
# ------------------------------------------------ #

metrics_table.to_csv(
    tables_dir / "variant_metrics.csv"
)

for name, sim in simulations.items():
    sim.to_csv(
        tables_dir / f"simulation_{name}.csv",
        index=False,
    )
