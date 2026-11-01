"""Fit thermal tolerance relationships based on 2025 thermal tolerance assay
 and reproduce manuscript/supporting figures.

Tables go to results/thermal_tolerance_tables/; manuscript figures to results/manuscript_figures/; supporting figures go to results/diagnostic_figures/.
Dependencies: numpy, pandas, scipy, matplotlib, seaborn.

TDT model: T_x(t) = a_x * ln(t) + b_x = CT_max,x - z_x * log10(t).
Durations are in minutes; temperatures and thermal transition breadth are in °C.

"""

# ------------------------ #
# Imports and run settings #
# ------------------------ #
import warnings
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
from scipy.stats import linregress, pearsonr

RANDOM_SEED = 42
N_T50_DRAWS = 500
N_BREADTH_DRAWS = 1000
SHOW_PLOTS = True  # Set False for non-interactive runs.

# ---------------------------------------- #
# Model relationships and workflow helpers #
# ---------------------------------------- #

def tdt_temperature(duration_min, a, b):
    """Return the threshold (°C): T(t) = a * ln(t) + b, with duration in minutes."""
    return a * np.log(duration_min) + b


def inverse_threshold_time(temperature, a, b):
    """Return inverse time to the threshold (min⁻¹) at temperature (°C), i.e. the thermal load accumulation rate.

    Inverting T(t) gives t(T) = exp((T - b) / a); this function returns 1 / t(T).
    """
    return np.exp(-(temperature - b) / a)


def curve_label_angle(ax, x, log_slope, log_intercept, half_step=0.5):
    """Return the display angle of duration = 10**(log_slope * temperature + log_intercept)."""
    x0, x1 = x - half_step, x + half_step
    y0 = 10 ** (log_slope * x0 + log_intercept)
    y1 = 10 ** (log_slope * x1 + log_intercept)

    display_start = ax.transData.transform((x0, y0))
    display_end = ax.transData.transform((x1, y1))

    angle = np.degrees(np.arctan2(display_end[1] - display_start[1], display_end[0] - display_start[0]))

    # Fold the angle into [-90°, 90°] so labels remain upright.
    if angle < -90:
        angle += 180
    if angle > 90:
        angle -= 180

    return angle

def validate_assay_data(assay_data):
    """Check required columns and finite, ordered threshold intervals before fitting."""
    # All thresholds are fitted; only T10, T50 and T90 need uncertainty bounds.
    thresholds = [f"T{i}" for i in range(10, 100, 10)]
    bounds = [f"{threshold}_{bound}" for threshold in ("T10", "T50", "T90")
              for bound in ("lower", "upper")]
    numeric_columns = ["Duration"] + thresholds + bounds
    missing_columns = {"Species", *numeric_columns} - set(assay_data.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")
    measurements = assay_data[numeric_columns].to_numpy(dtype=float)
    if assay_data.empty or assay_data["Species"].isna().any() or not np.isfinite(measurements).all():
        raise ValueError("Input must contain species names and finite numeric measurements.")
    expected_species = {"Acer platanoides", "Castanea sativa", "Fagus sylvatica", "Quercus petraea"}
    if set(assay_data["Species"]) != expected_species:
        raise ValueError(f"Expected these four species: {sorted(expected_species)}")
    if (assay_data["Duration"] <= 0).any() or (assay_data.groupby("Species")["Duration"].nunique() < 2).any():
        raise ValueError("Each species needs at least two distinct, positive exposure durations.")
    # Triangular sampling needs ordered bounds with a non-zero interval width.
    for threshold in ("T10", "T50", "T90"):
        interval_columns = (f"{threshold}_lower", threshold, f"{threshold}_upper")
        lower, estimate, upper = (assay_data[column].to_numpy() for column in interval_columns)
        if ((lower > estimate) | (estimate > upper) | (lower >= upper)).any():
            raise ValueError(f"Invalid triangular bounds for {threshold}: require lower <= estimate <= upper and lower < upper.")

def summarize_draws(values, prefix):
    """Summarise draws along axis 0; remaining dimensions are retained.

    Output keys use prefix_mean/lower/upper; bounds are the 2.5th/97.5th percentiles.
    """
    mean = np.mean(values, axis=0)
    lower, upper = np.percentile(values, [2.5, 97.5], axis=0)
    return {f"{prefix}_mean": mean, f"{prefix}_lower": lower, f"{prefix}_upper": upper}


def style_diagnostic(ax):
    """Apply a common axes style to supporting plots without changing global settings."""
    ax.grid(which="both", alpha=0.3)
    ax.tick_params(direction="out", labelsize=10)
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)
    for spine in ax.spines.values():
        spine.set_linewidth(1)


def save_and_show_figure(fig, name, directory):
    """Save PDF and 300-dpi PNG copies before optionally displaying the figure."""
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ------------- #
# Main workflow #
# ------------- #

def main():
    """Load assay data, fit TDT relationships, and save figures and parameter tables."""
    # -------------------------------------- #
    # 1. Input data and plotting conventions #
    # -------------------------------------- #
    # 1.1 Locate inputs, load all thresholds, and initialise output folders.
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    results_dir = project_root / "results" / "thermal_tolerance_tables"
    figures_dir = project_root / "results" / "manuscript_figures"
    diagnostics_dir = project_root / "results" / "diagnostic_figures"
    assay_file = data_dir / "2025_tolerance_assay" / "thermal_thresholds_by_duration.csv"
    assay_data = pd.read_csv(assay_file)
    validate_assay_data(assay_data)
    results_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    # A single generator preserves the draw sequence across the two sampling analyses.
    rng = np.random.default_rng(RANDOM_SEED)

    # 1.2 Use the same species colours and markers throughout the workflow.
    species_palette = ["darkblue", "darkmagenta", "goldenrod", "crimson"]
    species_order = ["Acer platanoides", "Castanea sativa", "Fagus sylvatica", "Quercus petraea"]
    species_colors = dict(zip(species_order, species_palette))
    species_markers = dict(zip(species_order, ["o", "X", "s", "P"]))
    theme_settings = {
        "axes.linewidth": 1.0, "xtick.major.size": 5, "xtick.major.width": 1,
        "ytick.major.size": 5, "ytick.major.width": 1,
    }

    # ---------------------------------------------- #
    # 2. T50 relationships and parameter uncertainty #
    # ---------------------------------------------- #
    # 2.1 Fit T50(t) = a * ln(t) + b to independent triangular draws of the input estimates.
    # The reported estimate is the mode; interval endpoints bound the sampling distribution.
    threshold = "T50"
    species_groups = assay_data.groupby("Species")

    t50_curve_frames = []
    t50_parameter_records = []
    t50_parameter_frames = []
    rate_curve_frames = []

    for species, group in species_groups:
        duration_min = group["Duration"].to_numpy()
        threshold_estimates = group[threshold].to_numpy()
        threshold_lower = group[f"{threshold}_lower"].to_numpy()
        threshold_upper = group[f"{threshold}_upper"].to_numpy()

        duration_grid = np.arange(5, 300, 1)
        temperature_grid = np.arange(40, 60, 0.1)

        temperature_draws = []
        slope_draws = []
        intercept_draws = []
        rate_draws = []

        for draw_index in range(N_T50_DRAWS):
            # Approximate uncertainty with a triangular distribution bounded by the reported confidence interval.
            sampled_thresholds = rng.triangular(
                left=threshold_lower, mode=threshold_estimates, right=threshold_upper,
            )
            try:
                fitted_parameters, _ = curve_fit(tdt_temperature, duration_min, sampled_thresholds)
                a, b = fitted_parameters
                fitted_temperatures = tdt_temperature(duration_grid, *fitted_parameters)

                temperature_draws.append(fitted_temperatures)
                slope_draws.append(a)
                intercept_draws.append(b)

                fitted_rates = inverse_threshold_time(temperature_grid, a, b)
                rate_draws.append(fitted_rates)

            except RuntimeError as exc:
                warnings.warn(f"{species}: fit failed in iteration {draw_index + 1}: {exc}", RuntimeWarning)

        if not temperature_draws:
            raise RuntimeError(f"No successful {threshold} fits for {species}.")
        temperature_draws = np.array(temperature_draws)
        rate_draws = np.array(rate_draws)

        # 2.2 Summarise fitted curves pointwise across successful draws.
        t50_curve_frames.append(pd.DataFrame({
            "Species": species, "Duration": duration_grid, **summarize_draws(temperature_draws, threshold),
        }))
        rate_curve_frames.append(pd.DataFrame({
            "Species": species, "Temperature": temperature_grid, **summarize_draws(rate_draws, "rate"),
        }))

        # 2.3 Retain paired a/b draws and summarise their marginal uncertainty.
        t50_parameter_frames.append(
            pd.DataFrame({"Species": species, "a": slope_draws, "b": intercept_draws}),
        )
        t50_parameter_records.append(
            {"Species": species, **summarize_draws(slope_draws, "a"), **summarize_draws(intercept_draws, "b")},
        )

    t50_curves = pd.concat(t50_curve_frames)
    rate_curves = pd.concat(rate_curve_frames)
    t50_parameters = pd.DataFrame(t50_parameter_records)
    t50_parameter_draws = pd.concat(t50_parameter_frames, ignore_index=True)

    # -------------------------------------------------------- #
    # 3. Supporting plots: T50 fits and inverse threshold time #
    # -------------------------------------------------------- #
    # 3.1 Compare fitted T50 curves with estimates at the experimental durations.
    sns.set_theme(style="ticks", font_scale=1.1, rc=theme_settings)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)

    for species in t50_curves["Species"].unique():
        species_curve = t50_curves[t50_curves["Species"] == species]

        ax.plot(
            species_curve["Duration"], species_curve[f"{threshold}_mean"], label=species,
            c=species_colors[species], lw=1,
        )
        ax.fill_between(
            species_curve["Duration"], species_curve[f"{threshold}_lower"],
            species_curve[f"{threshold}_upper"], alpha=0.1, color=species_colors[species],
        )

        species_observations = assay_data[assay_data["Species"] == species]
        ax.errorbar(species_observations["Duration"], species_observations[f"{threshold}"],
                    yerr=[species_observations[f"{threshold}"] - species_observations[f"{threshold}_lower"],
                    species_observations[f"{threshold}_upper"] - species_observations[f"{threshold}"]],
                    fmt="none", alpha=0.3, color=species_colors[species])
    sns.scatterplot(
        data=assay_data, x="Duration", y=threshold, hue="Species", style="Species", ax=ax,
        palette=species_colors, markers=species_markers, s=60,
    )
    ax.grid(which="both", alpha=0.3)
    ax.set_xlabel("Exposure duration [min]")
    ax.set_ylabel(f"${threshold}$ [°C]")
    ax.text(0.01, 0.94, "(a)", weight="demibold", transform=ax.transAxes)
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(
        legend_handles[:4], legend_labels[:4], title="", bbox_to_anchor=(1.02, 1), loc="upper left",
        frameon=False,
    )
    style_diagnostic(ax)
    save_and_show_figure(fig, "t50_duration_fit", diagnostics_dir)

    # 3.2 Plot thermal load accumulation rate in [% min-1]
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)

    for species in rate_curves["Species"].unique():
        species_curve = rate_curves[rate_curves["Species"] == species]

        ax.plot(
            species_curve["Temperature"], species_curve["rate_mean"] * 100, label=species,
            c=species_colors[species], lw=1,
        )
        ax.fill_between(
            species_curve["Temperature"], species_curve["rate_lower"] * 100, species_curve["rate_upper"] * 100,
            alpha=0.1, color=species_colors[species],
        )

    ax.grid(which="both", alpha=0.3)
    ax.set_xlabel("Temperature [°C]")
    ax.text(0.01, 0.94, "(b)", weight="demibold", transform=ax.transAxes)
    ax.set_ylim(0, 20)
    ax.set_xlim(40, 54)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    style_diagnostic(ax)
    # This is inverse time to T50, not a measured linear percentage-damage rate.
    ax.set_ylabel("Inverse time to $T_{50}$ [% min$^{-1}$]")
    save_and_show_figure(fig, "inverse_time_to_t50", diagnostics_dir)

    # ----------------------------------------------------------- #
    # 4. Threshold-specific fits and thermal tolerance landscapes #
    # ----------------------------------------------------------- #
    # 4.1 Fit each impairment threshold to the reported estimates without resampling.
    thresholds = ["T10", "T20", "T30", "T40", "T50", "T60", "T70", "T80", "T90"]

    species_groups = assay_data.groupby("Species")

    threshold_parameter_records = []
    landscape_curve_frames = []

    for species, group in species_groups:
        for threshold in thresholds:
            duration_min = group["Duration"].to_numpy()
            threshold_estimates = group[threshold].to_numpy()
            temperature_grid = np.arange(40, 65, 0.1)

            fitted_parameters, _ = curve_fit(tdt_temperature, duration_min, threshold_estimates)
            a, b = fitted_parameters
            fitted_rates = inverse_threshold_time(temperature_grid, a, b)

            # Invert the rate to obtain exposure duration at each temperature.
            landscape_curve_frames.append(pd.DataFrame({
                "Species": species,
                "Temperature": temperature_grid,
                "duration_min": 1/fitted_rates,
                "Threshold": threshold
            }))

            # Retain a and b for the threshold-sensitivity diagnostic.
            threshold_parameter_records.append({"Species": species, "a": a, "b": b, "Threshold": threshold})

    landscape_curves = pd.concat(landscape_curve_frames)
    threshold_parameters = pd.DataFrame(threshold_parameter_records)

    # 4.2 Draw manuscript plot: thermal tolerance landscapes
    sns.set_theme(style="ticks", font_scale=1, rc=theme_settings)
    fig, panel_axes = plt.subplots(
        2, 2, figsize=(6.5, 6.5), dpi=300, sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.1, "hspace": 0.15},
    )
    landscape_axes = panel_axes.flatten()
    label_durations = np.logspace(2.1, 0.3, len(thresholds))
    label_fits = []

    # Shared log-duration grid aligns the shaded bands between neighbouring thresholds.
    fill_durations = np.logspace(0, 3, 400)
    band_alphas = np.linspace(0.08, 0.6, len(thresholds) - 1)

    for panel_index, species in enumerate(landscape_curves["Species"].unique()):
        threshold_temperatures = {}
        for threshold_index, threshold in enumerate(thresholds):
            selected_curve = (landscape_curves["Species"] == species) & (landscape_curves["Threshold"] == threshold)
            threshold_curve = landscape_curves.loc[selected_curve].sort_values("duration_min")

            # Interpolate temperature as a function of log-duration for the band boundaries.
            interpolated_temperature = np.interp(
                np.log10(fill_durations), np.log10(threshold_curve["duration_min"]),
                threshold_curve["Temperature"],
            )
            threshold_temperatures[threshold] = interpolated_temperature

            # Draw the threshold curve and retain its log-space slope for label rotation.
            landscape_axes[panel_index].plot(
                threshold_curve["Temperature"], threshold_curve["duration_min"], color=species_colors[species],
                lw=1,
            )

            label_slope, label_intercept = np.polyfit(
                threshold_curve["Temperature"], np.log10(threshold_curve["duration_min"]), 1,
            )
            label_fits.append((panel_index, threshold_index, threshold, label_slope, label_intercept))

        for threshold_index in range(len(thresholds)-1):
            landscape_axes[panel_index].fill_betweenx(
                fill_durations, threshold_temperatures[thresholds[threshold_index]],
                threshold_temperatures[thresholds[threshold_index + 1]], color=species_colors[species],
                alpha=band_alphas[threshold_index], linewidth=0, zorder=1,
            )

        # Shade the region beyond the highest impairment threshold (T90).
        landscape_axes[panel_index].fill_betweenx(
            fill_durations, threshold_temperatures[thresholds[-1]], landscape_axes[panel_index].get_xlim()[1],
            color=species_colors[species], alpha=0.65, linewidth=0, zorder=1,
        )

        landscape_axes[panel_index].set_yscale("log")
        landscape_axes[panel_index].grid(which="both", alpha=0.3)
        landscape_axes[panel_index].set_xlim(40, 65)
        landscape_axes[panel_index].set_ylim(1, 300)
        landscape_axes[panel_index].set_title(f"{species}", fontsize=11, fontstyle="italic")
        landscape_axes[panel_index].xaxis.set_major_locator(MultipleLocator(5))

    landscape_axes[0].text(-0.3, 1.0, "(a)", weight="bold", transform=landscape_axes[0].transAxes)
    fig.supxlabel("Temperature [°C]", y=0.03, fontsize=11)
    fig.supylabel("Exposure duration [min]", x=0.03, fontsize=11)
    # Render the axes before calculating display-space angles for the curve labels.
    fig.canvas.draw()

    for panel_index, threshold_index, threshold, label_slope, label_intercept in label_fits:
        label_duration = label_durations[threshold_index]
        label_temperature = (np.log10(label_duration) - label_intercept) / label_slope

        if not (30 < label_temperature < 65):
            continue

        angle = curve_label_angle(landscape_axes[panel_index], label_temperature, label_slope, label_intercept)

        landscape_axes[panel_index].annotate(
            f"{threshold}",
            xy=(label_temperature, label_duration),
            rotation=angle,
            rotation_mode="anchor",
            ha="center",
            va="center",
            color=species_colors[landscape_curves["Species"].unique()[panel_index]],
            fontsize=6.5,
            path_effects=[
                pe.Stroke(linewidth=3, foreground="white"),
                pe.Normal()
            ],
            clip_on=True
        )
    plt.tight_layout()
    save_and_show_figure(fig, "thermal_tolerance_landscapes", figures_dir)

    # ------------------------------------------------- #
    # 5. Thermal transition breadth and its uncertainty #
    # ------------------------------------------------- #
    # 5.1 Propagate T10/T90 uncertainty through separate TDT fits.
    duration_grid = np.logspace(0, 2.5, 8)

    breadth_curve_frames = []
    breadth_parameter_records = []

    for species, group in assay_data.groupby("Species"):
        duration_min = group["Duration"].to_numpy()

        # Independent triangular draws approximate marginal threshold uncertainty.
        # They do not account for covariance among thresholds or exposure durations.
        t10_estimates = group["T10"].to_numpy()
        t10_lower = group["T10_lower"].to_numpy()
        t10_upper = group["T10_upper"].to_numpy()

        t90_estimates = group["T90"].to_numpy()
        t90_lower = group["T90_lower"].to_numpy()
        t90_upper = group["T90_upper"].to_numpy()

        breadth_draws = []
        t10_slope_draws = []
        t90_slope_draws = []
        slope_difference_draws = []

        for draw_index in range(N_BREADTH_DRAWS):
            # Sample each threshold independently using its estimate and interval endpoints.
            sampled_t10 = rng.triangular(left=t10_lower, mode=t10_estimates, right=t10_upper)
            sampled_t90 = rng.triangular(left=t90_lower, mode=t90_estimates, right=t90_upper)

            try:
                # Fit the lower and upper impairment thresholds to the same duration values.
                t10_fit_parameters, _ = curve_fit(tdt_temperature, duration_min, sampled_t10)
                t90_fit_parameters, _ = curve_fit(tdt_temperature, duration_min, sampled_t90)

                a10, b10 = t10_fit_parameters
                a90, b90 = t90_fit_parameters

                # Evaluate both relationships on the same duration grid.
                fitted_t10 = tdt_temperature(duration_grid, a10, b10)
                fitted_t90 = tdt_temperature(duration_grid, a90, b90)

                # Thermal transition breadth is the temperature difference T90(t) - T10(t).
                fitted_breadth = fitted_t90 - fitted_t10

                breadth_draws.append(fitted_breadth)

                t10_slope_draws.append(a10)
                t90_slope_draws.append(a90)
                slope_difference_draws.append(a90 - a10)

            except RuntimeError as exc:
                warnings.warn(f"{species}: fit failed in iteration {draw_index + 1}: {exc}", RuntimeWarning)

        if not breadth_draws:
            raise RuntimeError(f"No successful transition-breadth fits for {species}.")
        breadth_draws = np.array(breadth_draws)

        # 5.2 Summarise breadth at each duration and the slope difference across draws.
        breadth_curve_frames.append(pd.DataFrame({
            "Species": species, "Duration": duration_grid, **summarize_draws(breadth_draws, "DW"),
        }))

        # The slope difference is the slope of breadth versus ln(duration).
        breadth_parameter_records.append({
            "Species": species, **summarize_draws(t10_slope_draws, "a10"),
            **summarize_draws(t90_slope_draws, "a90"), **summarize_draws(slope_difference_draws, "delta_a"),
        })

    breadth_curves = pd.concat(breadth_curve_frames, ignore_index=True)
    breadth_parameters = pd.DataFrame(breadth_parameter_records)

    # 5.3 Calculate observed breadth and endpoint-based uncertainty bounds.
    assay_data["DW"] = assay_data["T90"] - assay_data["T10"]

    # Combine opposite interval endpoints; these are not a joint 95% confidence interval.
    assay_data["DW_lower"] = assay_data["T90_lower"] - assay_data["T10_upper"]
    assay_data["DW_upper"] = assay_data["T90_upper"] - assay_data["T10_lower"]

    # --------------------------------------------------------------- #
    # 6. Manuscript comparison: transition breadth and TDT parameters #
    # --------------------------------------------------------------- #
    # 6.1 Plot fitted breadth and estimates at the experimental durations.

    fig, panel_axes = plt.subplots(
        2, 1, figsize=(3.5, 7), dpi=300, gridspec_kw={"wspace": 0.1, "hspace": 0.3},
    )
    breadth_ax, tdt_ax = panel_axes
    for species in assay_data["Species"].unique():
        # Fitted breadth and its central 95% Monte Carlo interval.
        species_breadth_curve = breadth_curves[breadth_curves["Species"] == species]

        breadth_ax.plot(
            species_breadth_curve["Duration"], species_breadth_curve["DW_mean"], color=species_colors[species],
            lw=1.3,
        )

        breadth_ax.fill_between(
            species_breadth_curve["Duration"], species_breadth_curve["DW_lower"],
            species_breadth_curve["DW_upper"], color=species_colors[species], alpha=0.1, lw=0,
        )

        # Observed breadth with endpoint bounds (a different uncertainty construction).
        species_observations = assay_data[assay_data["Species"] == species]

        breadth_ax.errorbar(
            species_observations["Duration"],
            species_observations["DW"],
            yerr=[
                species_observations["DW"] - species_observations["DW_lower"],
                species_observations["DW_upper"] - species_observations["DW"]
            ],
            fmt="none",
            color=species_colors[species],
            alpha=0.4,
            lw=1
        )
    sns.scatterplot(
        data=assay_data, x="Duration", y="DW", hue="Species", style="Species", ax=breadth_ax,
        palette=species_palette, s=60,
    )
    legend_handles, legend_labels = breadth_ax.get_legend_handles_labels()
    breadth_ax.legend(
        legend_handles, ["AP", "CS", "FS", "QP"], handletextpad=0.4, title="", bbox_to_anchor=(1.0, 1.0),
    )
    breadth_ax.grid(which="both", alpha=0.3)
    breadth_ax.set_ylabel("$ΔT_{10}$-$T_{90}$ [°C]")
    breadth_ax.set_xlabel("Exposure duration [min]")
    breadth_ax.set_xlim(1, 300)
    breadth_ax.set_xscale("log")

    # 6.2 Add literature conifer parameters from Hauck & Dulamsuren.
    # Source supplied with the study: https://doi.org/10.64898/2026.07.24.740464
    # Convert published z values to a = -z / ln(10) to match the fitted parameterisation.
    literature_records = [{"Species": "Abies alba", "a_mean": 6.2/-np.log(10), "b_mean": 56.3},
                          {"Species": "Larix decidua", "a_mean": 5.8/-np.log(10), "b_mean": 52.0},
                          {"Species": "Picea abies", "a_mean": 5.45/-np.log(10), "b_mean": 55.17},
                          {"Species": "Pinus sylvestris", "a_mean": 4.145/-np.log(10), "b_mean": 52.6},
                          {"Species": "Pseudotsuga menziesii", "a_mean": 6.43/-np.log(10), "b_mean": 55.5}]

    literature_parameters = pd.DataFrame(literature_records, columns=["Species", "a_mean", "b_mean"])
    combined_parameters = pd.concat(
        [literature_parameters, t50_parameters], axis=0, join="inner", ignore_index=True,
    )
    combined_parameters["CT_max"] = combined_parameters["b_mean"]
    combined_parameters["z"] = combined_parameters["a_mean"] * - np.log(10)

    # 6.3 Convert a/b to conventional TDT parameters: z = -a * ln(10), CT_max = b.
    # CT_max is the fitted threshold at one-minute exposure; converting a to z swaps the bounds.
    t50_parameters["z_lower"] = t50_parameters["a_upper"] * - np.log(10)
    t50_parameters["z_upper"] = t50_parameters["a_lower"] * - np.log(10)
    t50_parameters["z"] = t50_parameters["a_mean"] * - np.log(10)
    t50_parameters["CT_max"] = t50_parameters["b_mean"]
    literature_parameters["z"] = literature_parameters["a_mean"] * - np.log(10)
    literature_parameters["CT_max"] = literature_parameters["b_mean"]

    # 6.4 Fit a descriptive regression across study and literature species, then draw the panel.
    tdt_regression = linregress(combined_parameters["z"], combined_parameters["CT_max"])
    z_grid = np.linspace(combined_parameters["z"].min(), combined_parameters["z"].max(), 100)
    fitted_ct_max = tdt_regression.slope * z_grid + tdt_regression.intercept

    for _, row in t50_parameters.iterrows():
        species = row["Species"]
        tdt_ax.errorbar(
            row["z"], row["CT_max"], xerr=[[row["z"] - row["z_lower"]], [row["z_upper"] - row["z"]]],
            yerr=[[row["b_mean"] - row["b_lower"]], [row["b_upper"] - row["b_mean"]]], fmt="",
            color=species_colors[species], alpha=0.5, lw=1, zorder=-10,
        )
    sns.scatterplot(
        data=t50_parameters, x="z", y="CT_max", hue="Species", style="Species", ax=tdt_ax,
        palette=species_palette, s=60,
    )
    sns.scatterplot(
        data=literature_parameters, x="z", y="CT_max", hue="Species", style="Species", ax=tdt_ax,
        palette="Greys", markers=["v", "^", "p", ">", "h"], s=60,
    )
    tdt_ax.plot(
        z_grid, fitted_ct_max, color="black", lw=1, ls="--", label=f"$R^2$ = {tdt_regression.rvalue ** 2:.2f}",
    )

    # Fixed label offsets reproduce the manuscript layout.
    literature_annotations = {
        "Abies alba": ("A. alba", (-44, 5)), "Picea abies": ("P. abies", (-49, 0)),
        "Larix decidua": ("L. decidua", (-63, -7)), "Pinus sylvestris": ("P. sylvestris", (7, -2)),
        "Pseudotsuga menziesii": ("P. menziesii", (-60, 8)),
    }
    for name, (label, offset) in literature_annotations.items():
        row = literature_parameters.set_index("Species").loc[name]
        tdt_ax.annotate(
            label, (row["z"], row["CT_max"]), xytext=offset, textcoords="offset points", fontstyle="italic",
        )
    tdt_ax.grid(which="both", alpha=0.3)
    tdt_ax.set_ylabel(r"$CT_{\max,50}$ [°C]")
    tdt_ax.set_xlabel(r"$z_{50}$ [°C]")
    breadth_ax.text(-0.22, 1.0, "(b)", weight="bold", transform=breadth_ax.transAxes)

    regression_handles = [Line2D([0], [0], color="black", linestyle="--", lw=1, label=f"$R^2$ = {tdt_regression.rvalue ** 2:.2f}")]
    tdt_ax.legend(handles=regression_handles, handlelength=2, loc="best", frameon=True)
    tdt_ax.text(-0.22, 1.0, "(c)", weight="bold", transform=tdt_ax.transAxes)
    plt.tight_layout()
    save_and_show_figure(fig, "thermal_transition_breadth_and_tdt_parameters", figures_dir)

    # ------------------------------------------------------------------- #
    # 7. Supporting plots: threshold sensitivity and breadth associations #
    # ------------------------------------------------------------------- #
    # 7.1 Compare z across impairment thresholds using the deterministic fits from section 4.
    # A larger positive z means a greater temperature decline per tenfold increase in duration.
    z_thresholds = ["T10", "T50", "T90"]
    threshold_sensitivity = threshold_parameters[threshold_parameters["Threshold"].isin(z_thresholds)].copy()
    threshold_sensitivity["z"] = -threshold_sensitivity["a"] * np.log(10)
    threshold_sensitivity["Threshold_num"] = threshold_sensitivity["Threshold"].str[1:].astype(int)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    for species in threshold_sensitivity["Species"].unique():
        selected_species = threshold_sensitivity["Species"] == species
        species_sensitivity = threshold_sensitivity.loc[selected_species].sort_values("Threshold_num")

        ax.plot(
            species_sensitivity["Threshold_num"], species_sensitivity["z"], color=species_colors[species],
            lw=1,
        )

    sns.scatterplot(
        data=threshold_sensitivity, x="Threshold_num", y="z", hue="Species", style="Species",
        palette=species_colors, markers=species_markers, s=60, ax=ax,
    )
    style_diagnostic(ax)
    ax.set_xticks([10, 50, 90])
    ax.set_xticklabels(["T10", "T50", "T90"])
    ax.set_xlabel("Thermal tolerance threshold")
    ax.set_ylabel("Duration sensitivity, $z$ [°C decade$^{-1}$]")
    ax.grid(which="both", alpha=0.3)
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(legend_handles, legend_labels, title="", bbox_to_anchor=(1.05, 1.0))
    plt.tight_layout()
    save_and_show_figure(fig, "duration_sensitivity_by_threshold", diagnostics_dir)

    # 7.2 Inspect breadth versus T10 for individual assay estimates.
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    sns.scatterplot(
        data=assay_data, x="T10", y="DW", hue="Species", style="Species", palette=species_colors,
        markers=species_markers, s=60, ax=ax,
    )
    ax.set_xlabel("$T_{10}$ [°C]")
    ax.set_ylabel("Thermal transition breadth, $T_{90} - T_{10}$ [°C]")
    style_diagnostic(ax)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    save_and_show_figure(fig, "breadth_vs_t10_observations", diagnostics_dir)

    # 7.3 Repeat the comparison for species means (four points only).
    species_means = assay_data.groupby("Species", as_index=False)[["T10", "DW"]].mean()
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    sns.scatterplot(
        data=species_means, x="T10", y="DW", hue="Species", style="Species", palette=species_colors,
        markers=species_markers, s=60, ax=ax,
    )
    ax.set_xlabel("Mean $T_{10}$ [°C]")
    ax.set_ylabel("Mean thermal transition breadth, $T_{90} - T_{10}$ [°C]")
    style_diagnostic(ax)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    save_and_show_figure(fig, "breadth_vs_t10_species_means", diagnostics_dir)

    # -------------------------------------------------------------- #
    # 8. Export parameter tables and report exploratory correlations #
    # -------------------------------------------------------------- #
    # 8.1 Export after calculating z/CT_max so the T50 table includes these columns.
    # Keep existing filenames and column names for compatibility with downstream analyses.
    t50_parameter_draws.to_csv(results_dir / "tdt_parameter_draws.csv", index=False)
    t50_parameters.to_csv(results_dir / "tdt_parameters.csv", index=False)
    threshold_parameters.to_csv(results_dir / "tdt_parameters_by_threshold.csv", index=False)
    breadth_curves.to_csv(results_dir / "thermal_transition_breadth.csv", index=False)
    breadth_parameters.to_csv(results_dir / "thermal_transition_breadth_parameters.csv", index=False)
    print(breadth_parameters.to_string(index=False))

    # 8.2 Exploratory only: DW = T90 - T10 shares T10 with the predictor (mathematical coupling).
    # Pooled observations also repeat species; the species-mean correlation has only four points.
    print("Pooled T10-breadth correlation:", pearsonr(assay_data["T10"], assay_data["DW"]))
    print("Species-mean T10-breadth correlation:", pearsonr(species_means["T10"], species_means["DW"]))


if __name__ == "__main__":
    main()
